"""Gymnasium adapter for functional :class:`qenvs.Environment` objects."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from qenvs.envs.base import Environment, EnvState

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:  # pragma: no cover
    msg = "GymnasiumEnv requires gymnasium: pip install 'qenvs[gym]'"
    raise ImportError(msg) from e


class GymnasiumEnv(gym.Env):
    """Wrap a functional environment as a ``gymnasium.Env`` (single instance, numpy I/O).

    ``step`` and ``reset`` are jit-compiled once; subsequent calls are cheap.
    For large-batch training prefer ``jax.vmap`` over the functional API directly.
    """

    metadata = {"render_modes": []}

    def __init__(self, env: Environment, *, seed: int = 0) -> None:
        super().__init__()
        self.env = env
        self._state: EnvState | None = None
        self._key = jax.random.PRNGKey(seed)
        b = float(env.action_bound)
        self.action_space = spaces.Box(-b, b, shape=(env.action_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(env.observation_dim,), dtype=np.float32
        )
        self._reset = eqx.filter_jit(env.reset)
        self._step = eqx.filter_jit(env.step)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._key = jax.random.PRNGKey(seed)
        self._key, sub = jax.random.split(self._key)
        self._state, ts = self._reset(sub)
        return np.asarray(ts.observation, dtype=np.float32), _to_numpy(ts.info)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._state is None:
            msg = "call reset() before step()"
            raise RuntimeError(msg)
        self._state, ts = self._step(self._state, jnp.asarray(action))
        obs = np.asarray(ts.observation, dtype=np.float32)
        return obs, float(ts.reward), bool(ts.done), False, _to_numpy(ts.info)


def _to_numpy(info: dict[str, Any]) -> dict[str, Any]:
    return {k: np.asarray(v) for k, v in info.items()}
