"""Batched Gymnasium ``VectorEnv`` backed by ``jax.vmap`` of a functional environment."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from qenvs import keys
from qenvs.envs.base import Environment

try:
    import gymnasium as gym
    from gymnasium import spaces
    from gymnasium.vector import VectorEnv
except ImportError as e:  # pragma: no cover
    msg = "GymnasiumVectorEnv requires gymnasium: pip install 'qenvs[gym]'"
    raise ImportError(msg) from e

try:  # gymnasium >= 1.1 declares the autoreset mode in metadata
    from gymnasium.vector import AutoresetMode

    _AUTORESET_METADATA: dict[str, Any] = {"autoreset_mode": AutoresetMode.NEXT_STEP}
except ImportError:  # pragma: no cover
    # gymnasium 1.0 has no enum: NEXT_STEP is the only (implicit) behaviour there.
    _AUTORESET_METADATA = {}


def _bmask(flag: jax.Array, like: jax.Array) -> jax.Array:
    return flag.reshape((-1,) + (1,) * (like.ndim - 1))


class GymnasiumVectorEnv(VectorEnv):
    """``num_envs`` copies of a functional environment stepped by one jitted ``vmap``.

    Implements the gymnasium>=1.0 ``VectorEnv`` API with batched numpy I/O.

    Autoreset follows ``AutoresetMode.NEXT_STEP``: the step on which an episode ends
    returns its true final observation with ``terminated=True``; the *following*
    ``step`` call ignores the action of that sub-environment, returns the fresh
    initial observation, reward 0, ``terminated=False`` and ``truncated=False``.
    All returned arrays (observations, rewards, flags and every info value) are
    writable numpy copies. ``step`` expects actions of shape ``(num_envs, action_dim)``;
    a 1-D array of shape ``(num_envs,)`` raises ``ValueError``.

    Episodes in QEnvS always end by termination at the horizon, so ``truncated``
    is always False (see the RL libraries guide).

    Episode keys follow the single-run rule of :class:`GymnasiumEnv`: the run key is
    ``jax.random.PRNGKey(seed)`` and the episode with index ``i`` uses
    ``keys.episode_key(run_key, i) = fold_in(run_key, i)``. The ``j``-th episode of
    sub-environment ``b`` has index ``i = j * num_envs + b``, so the vector env replays, in
    an interleaved order, the episodes of ``GymnasiumEnv(seed=seed)``, and any episode can be
    regenerated from the seed and its index (stochastic resets included).
    The ``info`` of a wrapped environment may hold a nested dictionary (``"diagnostics"``);
    it is returned as a nested dictionary of batched arrays with its own ``_key`` masks.

    Args:
        env: functional environment.
        num_envs: number of parallel sub-environments.
        seed: base seed (used until ``reset(seed=...)`` is called).
        normalize_actions: if True, ``action_space`` is ``Box(-1, 1)`` and actions
            are multiplied by ``env.action_bound`` before stepping.
    """

    metadata = {"render_modes": [], **_AUTORESET_METADATA}

    def __init__(
        self,
        env: Environment,
        num_envs: int,
        *,
        seed: int = 0,
        normalize_actions: bool = False,
    ) -> None:
        if num_envs < 1:
            msg = f"num_envs must be >= 1, got {num_envs}"
            raise ValueError(msg)
        self.env = env
        self.num_envs = int(num_envs)
        self._action_dim = int(env.action_dim)  # cached: may build operators
        self.normalize_actions = normalize_actions
        self._seed = seed
        self._action_scale = float(env.action_bound) if normalize_actions else 1.0
        b = 1.0 if normalize_actions else float(env.action_bound)
        self.single_action_space = spaces.Box(-b, b, shape=(self._action_dim,), dtype=np.float32)
        self.single_observation_space = spaces.Box(
            -np.inf, np.inf, shape=(env.observation_dim,), dtype=np.float32
        )
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)
        self.observation_space = gym.vector.utils.batch_space(
            self.single_observation_space, self.num_envs
        )
        self._state: Any = None
        self._run_key = jax.random.PRNGKey(seed)
        self._episodes = jnp.zeros(self.num_envs, dtype=jnp.uint32)  # episodes started per env
        self._autoreset = jnp.zeros(self.num_envs, dtype=bool)
        scale = self._action_scale
        n_envs = self.num_envs
        slot = jnp.arange(n_envs, dtype=jnp.uint32)

        def _episode_keys(run_key: jax.Array, started: jax.Array) -> jax.Array:
            # sub-env b, j-th episode: i = j * num_envs + b
            return jax.vmap(keys.episode_key, in_axes=(None, 0))(run_key, started * n_envs + slot)

        def _reset(run_key: jax.Array, started: jax.Array):
            state, ts = jax.vmap(env.reset)(_episode_keys(run_key, started))
            return state, ts, started + 1

        def _step(state, actions, autoreset, run_key, started):
            stepped, ts = jax.vmap(env.step)(state, actions * scale)
            rstate, rts = jax.vmap(env.reset)(_episode_keys(run_key, started))

            def pick(r, s):
                return jnp.where(_bmask(autoreset, s), r, s)

            new_state = jax.tree.map(pick, rstate, stepped)
            obs = pick(rts.observation, ts.observation)
            reward = jnp.where(autoreset, 0.0, ts.reward)
            done = jnp.logical_and(ts.done, jnp.logical_not(autoreset))
            info = jax.tree.map(pick, rts.info, ts.info)
            new_started = jnp.where(autoreset, started + 1, started)
            return new_state, obs, reward, done, info, new_started

        self._reset_fn = jax.jit(_reset)
        self._step_fn = jax.jit(_step)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,  # noqa: ARG002 (gymnasium signature)
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._run_key = jax.random.PRNGKey(seed)
            self._episodes = jnp.zeros(self.num_envs, dtype=jnp.uint32)
        self._state, ts, self._episodes = self._reset_fn(self._run_key, self._episodes)
        self._autoreset = jnp.zeros(self.num_envs, dtype=bool)
        return np.array(ts.observation, dtype=np.float32), self._infos(ts.info)

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        if self._state is None:
            msg = "call reset() before step()"
            raise RuntimeError(msg)
        # A 1-D action of shape (num_envs,) is rejected on purpose: no implicit reshape.
        a = jnp.asarray(actions, dtype=jnp.float32)
        expected = (self.num_envs, self._action_dim)
        if a.shape != expected:
            msg = f"actions must have shape {expected}, got {a.shape}"
            raise ValueError(msg)
        self._state, obs, reward, done, info, self._episodes = self._step_fn(
            self._state, a, self._autoreset, self._run_key, self._episodes
        )
        self._autoreset = done
        # np.array copies: np.asarray of a jax array is a read-only view, which breaks
        # in-place edits and gymnasium wrappers such as ClipReward.
        return (
            np.array(obs, dtype=np.float32),
            np.array(reward, dtype=np.float32),
            np.array(done, dtype=np.bool_),
            np.zeros(self.num_envs, dtype=np.bool_),
            self._infos(info),
        )

    def _infos(self, info: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in info.items():
            out[k] = self._infos(v) if isinstance(v, dict) else np.array(v)  # writable copy
            out[f"_{k}"] = np.ones(self.num_envs, dtype=np.bool_)
        return out
