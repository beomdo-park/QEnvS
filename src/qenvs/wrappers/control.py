"""Control errors: miscalibrated and noisy control amplitudes.

:class:`ControlErrors` applies, at step ``k`` and on channel ``c``,

    u = clip((a (1 + epsilon) + o) (1 + xi_k), -B, B)

where ``a`` is the commanded amplitude, ``epsilon`` the relative gain error (fixed for the
episode), ``o`` an absolute offset in the units of the action (fixed for the episode),
``xi_k`` a relative noise redrawn every step and ``B`` the ``action_bound`` of the wrapped
environment. This is an explicitly defined simulation model; it does not claim to be the
order in which every device produces its errors. The relative errors have the whole real
line as support (a gain below ``-1`` inverts the pulse; the simulation is defined for any
real gain), the clip at ``B`` keeps the amplitude in the allowed range, and a positive gain
can be enforced with a :class:`~qenvs.distributions.TruncatedNormal`.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from qenvs import keys
from qenvs.distributions import Distribution
from qenvs.envs.base import Environment, TimeStep
from qenvs.wrappers.core import WrappedState, Wrapper

CLIP_FRACTION = "clip_fraction"


def apply_control_errors(
    action: Array,
    gain: Array | None,
    offset: Array | None,
    noise: Array | None,
    bound: float,
) -> tuple[Array, Array]:
    """``u = clip((a (1 + gain) + offset) (1 + noise), -bound, bound)`` and the clipped fraction.

    ``None`` switches a term off (no operation at all, so the output equals ``clip(a)``
    bit for bit when everything is off). Returns ``(u, fraction)`` where ``fraction`` is the
    share of channels with ``|u| > bound`` before clipping.
    """
    u = jnp.asarray(action)
    if gain is not None:
        u = u * (1.0 + gain)
    if offset is not None:
        u = u + offset
    if noise is not None:
        u = u * (1.0 + noise)
    fraction = jnp.mean((jnp.abs(u) > bound).astype(jnp.result_type(float)))
    return jnp.clip(u, -bound, bound), fraction


class ControlErrors(Wrapper):
    """Gain error, offset and per-step noise on the control amplitudes (see module docstring).

    Args:
        env: environment to wrap.
        amplitude_scale: distribution of the relative gain error ``epsilon`` (episode fixed).
        offset: distribution of the absolute offset ``o`` in action units (episode fixed).
        step_noise: distribution of the relative per-step noise ``xi_k``.
        correlated: if true, one value is drawn and shared by all channels (for example a
            common AWG gain error); if false (default) every channel gets its own draw.
            It applies to every enabled error source.

    A source that is ``None`` is off. With all three off the wrapper only reports
    ``info["clip_fraction"]``.

    Info: every ``reset`` and ``step`` adds ``info["clip_fraction"]``, the share of channels
    clipped at ``+-action_bound`` after the errors were applied (``0`` at ``reset``). Noise
    followed by clipping piles probability up at the bounds, so report it with results.

    Key use: the wrapper passes the episode key unchanged to the inner environment; gain and
    offset come from ``keys.calibration_key(episode_key, keys.GAIN / keys.OFFSET)`` and the
    noise of step ``k`` from ``keys.step_noise_key(episode_key, k)``, so switching one source
    on or off, changing a distribution, or ``correlated`` never changes another source.
    """

    env: Environment
    amplitude_scale: Distribution | None = eqx.field(static=True)
    offset: Distribution | None = eqx.field(static=True)
    step_noise: Distribution | None = eqx.field(static=True)
    correlated: bool = eqx.field(static=True)

    def __init__(
        self,
        env: Environment,
        amplitude_scale: Distribution | None = None,
        offset: Distribution | None = None,
        step_noise: Distribution | None = None,
        *,
        correlated: bool = False,
    ) -> None:
        for name, dist in (
            ("amplitude_scale", amplitude_scale),
            ("offset", offset),
            ("step_noise", step_noise),
        ):
            if dist is not None and not isinstance(dist, Distribution):
                msg = f"{name} must be a Distribution (Normal, Uniform, TruncatedNormal) or None"
                raise TypeError(msg)
        self.env = env
        self.amplitude_scale = amplitude_scale
        self.offset = offset
        self.step_noise = step_noise
        self.correlated = bool(correlated)

    @property
    def _shape(self) -> tuple[int, ...]:
        return () if self.correlated else (self.env.action_dim,)

    def _episode_data(self, key: Array) -> dict[str, Array]:
        data: dict[str, Array] = {}
        dtype = jnp.result_type(float)
        if self.amplitude_scale is not None:
            k = keys.calibration_key(key, keys.GAIN)
            data["gain"] = self.amplitude_scale.sample(k, 1.0, self._shape, dtype)
        if self.offset is not None:
            k = keys.calibration_key(key, keys.OFFSET)
            data["offset"] = self.offset.sample(k, 1.0, self._shape, dtype)
        return data

    def _transform_timestep(self, state: WrappedState, ts: TimeStep) -> TimeStep:  # noqa: ARG002
        """Used by ``reset``: no action has been applied yet, so nothing was clipped."""
        zero = jnp.zeros((), jnp.result_type(float))
        return TimeStep(ts.observation, ts.reward, ts.done, {**ts.info, CLIP_FRACTION: zero})

    def applied_action(self, state: WrappedState, action: Array) -> tuple[Array, Array]:
        """Amplitude ``u`` sent to the inner environment at the next step, and the clip fraction."""
        noise = None
        if self.step_noise is not None:
            k = keys.step_noise_key(state.key, state.step)
            noise = self.step_noise.sample(k, 1.0, self._shape, jnp.result_type(float))
        return apply_control_errors(
            action,
            state.data.get("gain"),
            state.data.get("offset"),
            noise,
            self.env.action_bound,
        )

    def step(self, state: WrappedState, action: Array) -> tuple[WrappedState, TimeStep]:
        applied, fraction = self.applied_action(state, action)
        inner, ts = self._inner_env(state.data).step(state.inner, applied)
        new = WrappedState(inner=inner, key=state.key, step=state.step + 1, data=state.data)
        ts = TimeStep(ts.observation, ts.reward, ts.done, {**ts.info, CLIP_FRACTION: fraction})
        return new, ts
