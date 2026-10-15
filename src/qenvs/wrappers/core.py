"""Base class of functional environment wrappers (no third-party dependencies).

A wrapper is itself an :class:`~qenvs.envs.base.Environment`. Its state is a
:class:`WrappedState` that holds the inner state, the episode key and the wrapper's own
episode data (sampled parameters, calibration errors). Wrappers nest: every wrapper
passes the key it receives in ``reset`` unchanged to the inner environment and derives
its own random numbers from that episode key through :mod:`qenvs.keys`, so the
samples of one wrapper never depend on which other wrappers are present.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from qenvs.envs.base import Environment, EnvState, TimeStep


class WrappedState(eqx.Module):
    """State of a wrapped environment.

    Attributes:
        inner: state of the wrapped environment (an ``EnvState`` or another ``WrappedState``).
        key: episode key received in ``reset``.
        step: number of steps taken in this episode.
        data: the wrapper's episode data (for example sampled parameters).
    """

    inner: Any
    key: Array
    step: Array
    data: dict[str, Array]

    @property
    def base(self) -> EnvState:
        """The innermost ``EnvState``."""
        s = self.inner
        while isinstance(s, WrappedState):
            s = s.inner
        return s

    @property
    def qstate(self) -> Any:
        """Quantum state of the innermost environment."""
        return self.base.qstate

    @property
    def t(self) -> Array:
        """Time of the innermost environment."""
        return self.base.t


class Wrapper(Environment):
    """Environment that delegates to ``env``; subclasses override what they change.

    Attributes not defined on the wrapper (``model``, ``dt``, ``target``, ...) are read
    from the wrapped environment.
    """

    env: Environment

    # ------------------------------------------------------------------ delegation
    def __getattr__(self, name: str) -> Any:
        if name == "env" or name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.env, name)

    @property
    def unwrapped(self) -> Environment:
        """The innermost (unwrapped) environment."""
        e = self.env
        while isinstance(e, Wrapper):
            e = e.env
        return e

    @property
    def action_dim(self) -> int:
        return self.env.action_dim

    @property
    def observation_dim(self) -> int:
        return self.env.observation_dim

    @property
    def action_bound(self) -> float:
        return self.env.action_bound

    @property
    def horizon(self) -> int:
        return self.env.horizon

    # ------------------------------------------------------------------ hooks
    def _episode_data(self, key: Array) -> dict[str, Array]:  # noqa: ARG002
        """Episode data drawn at ``reset`` (default: none)."""
        return {}

    def _inner_env(self, data: dict[str, Array]) -> Environment:  # noqa: ARG002
        """Environment to step this episode (default: ``env``; randomisers swap the model)."""
        return self.env

    def _transform_action(self, state: WrappedState, action: Array) -> Array:  # noqa: ARG002
        """Action actually applied to the inner environment (default: unchanged)."""
        return action

    def _transform_timestep(self, state: WrappedState, ts: TimeStep) -> TimeStep:  # noqa: ARG002
        """Timestep returned to the caller (default: unchanged); ``state`` is the new state."""
        return ts

    # ------------------------------------------------------------------ API
    def reset(self, key: Array) -> tuple[WrappedState, TimeStep]:
        data = self._episode_data(key)
        inner, ts = self._inner_env(data).reset(key)
        state = WrappedState(inner=inner, key=key, step=jnp.asarray(0), data=data)
        return state, self._transform_timestep(state, ts)

    def step(self, state: WrappedState, action: Array) -> tuple[WrappedState, TimeStep]:
        applied = self._transform_action(state, action)
        inner, ts = self._inner_env(state.data).step(state.inner, applied)
        new = WrappedState(inner=inner, key=state.key, step=state.step + 1, data=state.data)
        return new, self._transform_timestep(new, ts)

    def info(self, state: WrappedState) -> dict[str, Array]:
        """Exact diagnostics of ``state`` from the inner environment (not a policy input)."""
        return self._inner_env(state.data).info(state.inner)

    def diagnostics(self, state: WrappedState) -> dict[str, Array]:
        """Exact diagnostics of ``state``; for evaluation and debugging, never a policy input."""
        inner_env = self._inner_env(state.data)
        if isinstance(inner_env, Wrapper):
            return inner_env.diagnostics(state.inner)
        return inner_env.info(state.inner)

    def rollout(self, actions: Array, key: Array | None = None) -> tuple[WrappedState, Array]:
        """Open-loop rollout of ``actions`` (horizon, action_dim); returns final state, rewards."""
        if key is None:
            key = jax.random.PRNGKey(0)
        state0, _ = self.reset(key)

        def body(state: WrappedState, a: Array) -> tuple[WrappedState, Array]:
            state, ts = self.step(state, a)
            return state, ts.reward

        return jax.lax.scan(body, state0, actions)
