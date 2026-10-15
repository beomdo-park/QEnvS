"""Model-parameter randomisation: a different device (parameter draw) every episode.

:class:`ParameterRandomization` wraps any environment whose ``model`` declares randomisable
parameters (``Model.parameter_specs``). At ``reset`` it draws one offset per randomised
parameter from the episode key (leaf ``keys.parameter_key(episode_key, name)``), keeps it
fixed for the whole episode, and runs the inner environment on
``model.with_parameters(...)`` swapped in with ``equinox.tree_at``.

The episode data are ``"offset/<name>"`` (the drawn offset ``delta``, in the unit of the
parameter) and ``"value/<name>"`` (``nominal + delta``). Samples never leave the support of
a parameter: a distribution whose range could is rejected at construction (never clipped).
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from qenvs import keys
from qenvs.distributions import Distribution, check_strength
from qenvs.envs.base import Environment, TimeStep
from qenvs.models.base import Model
from qenvs.wrappers.core import WrappedState, Wrapper


def _model_of(env: Environment) -> Model:
    """The model of the innermost environment of a (possibly wrapped) environment."""
    while isinstance(env, Wrapper):
        env = env.env
    return env.model  # type: ignore[attr-defined]


class ParameterRandomization(Wrapper):
    """Randomise model parameters per episode (fixed within the episode).

    Args:
        env: environment to wrap; its model must declare the named parameters
            (:meth:`~qenvs.models.base.Model.parameter_specs`). Several wrappers may be
            nested; the model of the innermost environment is the one randomised.
        distributions: mapping parameter name to an offset distribution
            (:class:`~qenvs.distributions.Normal`, ``Uniform``, ``TruncatedNormal``). The
            sampled value is ``nominal + offset`` in the unit of the parameter. A
            distribution that states a ``unit`` must agree with the model's unit.
        strength: finite factor ``>= 0`` multiplying the configured widths (see
            :mod:`qenvs.distributions`); ``0`` gives the nominal model exactly.
        oracle: if true, append ``(value - nominal) / w_ref`` of every randomised
            parameter with ``w_ref > 0`` to the observation, in the order of
            ``distributions``. ``w_ref`` is the model reference width, independent of the
            configured widths and of ``strength``, so the scale is the same for a training
            and an evaluation distribution. The oracle tells the policy the true parameters,
            which a real device would have to estimate with a separate calibration; with
            the ``"state"`` observation it makes the observation Markov again.

    Raises:
        ValueError: unknown or non-randomisable parameter (the message gives the reason for
            dissipation rates and structural sizes), a distribution that conflicts with the
            support or the unit, two names with the same key tag, or an invalid ``strength``.
        TypeError: ``env`` has no :class:`~qenvs.models.base.Model`.

    Key use: the wrapper passes the episode key it receives in ``reset`` unchanged to the
    inner environment, and draws parameter ``name`` from
    ``keys.parameter_key(episode_key, name)``, so adding, removing or reconfiguring one
    parameter (or any other wrapper) never changes the samples of the others.
    """

    env: Environment
    distributions: tuple[tuple[str, Distribution], ...] = eqx.field(static=True)
    strength: float = eqx.field(static=True)
    oracle: bool = eqx.field(static=True)

    def __init__(
        self,
        env: Environment,
        distributions: dict[str, Distribution],
        *,
        strength: float = 1.0,
        oracle: bool = False,
    ) -> None:
        model = getattr(env, "model", None)
        if not isinstance(model, Model):
            msg = f"ParameterRandomization needs an environment with a Model, got {type(env)!r}"
            raise TypeError(msg)
        strength = check_strength(strength)
        specs = model.parameter_specs()
        nominal = model.parameter_values()
        for name, dist in distributions.items():
            if name not in specs:
                raise model._unknown_parameter_error(name)
            if not isinstance(dist, Distribution):
                msg = f"{name}: expected a Distribution (Normal, Uniform, ...), got {dist!r}"
                raise TypeError(msg)
            if dist.unit and dist.unit != specs[name].unit:
                msg = (
                    f"{name}: the distribution is given in {dist.unit!r} but the model "
                    f"declares the unit {specs[name].unit!r}"
                )
                raise ValueError(msg)
            dist.check_support(name, float(nominal[name]), specs[name].support, strength)
        keys.check_name_tags(distributions)
        self.env = env
        self.distributions = tuple(distributions.items())
        self.strength = strength
        self.oracle = bool(oracle)

    # ------------------------------------------------------------------ derived
    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Names of the randomised parameters, in the order of ``distributions``."""
        return tuple(name for name, _ in self.distributions)

    @property
    def oracle_names(self) -> tuple[str, ...]:
        """Parameters of the oracle observation: randomised and with a reference width ``> 0``."""
        specs = _model_of(self.env).parameter_specs()
        return tuple(n for n in self.parameter_names if specs[n].w_ref > 0.0)

    @property
    def observation_dim(self) -> int:
        return self.env.observation_dim + (len(self.oracle_names) if self.oracle else 0)

    def episode_model(self, state: WrappedState) -> Model:
        """The model of this episode (nominal model with the sampled parameters)."""
        return self._model(state.data)

    # ------------------------------------------------------------------ wrapper hooks
    def _model(self, data: dict[str, Array]) -> Model:
        values = {n: data[f"value/{n}"] for n in self.parameter_names}
        return _model_of(self.env).with_parameters(**values)

    def _episode_data(self, key: Array) -> dict[str, Array]:
        nominal = _model_of(self.env).parameter_values()
        dtype = jnp.result_type(float)
        data: dict[str, Array] = {}
        for name, dist in self.distributions:
            delta = dist.sample(keys.parameter_key(key, name), self.strength, (), dtype)
            data[f"offset/{name}"] = delta
            data[f"value/{name}"] = jnp.asarray(nominal[name], dtype) + delta
        return data

    def _inner_env(self, data: dict[str, Array]) -> Environment:
        if not self.distributions:
            return self.env
        return eqx.tree_at(_model_of, self.env, self._model(data))

    def _transform_timestep(self, state: WrappedState, ts: TimeStep) -> TimeStep:
        if not self.oracle or not self.oracle_names:
            return ts
        specs = _model_of(self.env).parameter_specs()
        extra = jnp.stack([state.data[f"offset/{n}"] / specs[n].w_ref for n in self.oracle_names])
        obs = jnp.concatenate([ts.observation, extra.astype(ts.observation.dtype)])
        return TimeStep(observation=obs, reward=ts.reward, done=ts.done, info=ts.info)
