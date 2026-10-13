"""Gymnasium adapter passes the official env checker."""

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")
from gymnasium.utils.env_checker import check_env  # noqa: E402

import qenvs  # noqa: E402
from qenvs.wrappers import GymnasiumEnv  # noqa: E402


def test_check_env_passes():
    env = GymnasiumEnv(qenvs.make("qubit/StateTransfer-v0", horizon=5, dt=0.2))
    check_env(env, skip_render_check=True)


def test_episode_terminates_at_horizon_and_returns_fidelity():
    horizon, dt = 8, 0.25
    env = GymnasiumEnv(qenvs.make("qubit/StateTransfer-v0", horizon=horizon, dt=dt))
    obs, info = env.reset(seed=3)
    assert obs.shape == env.observation_space.shape
    total = 0.0
    u = np.pi / (horizon * dt)
    for i in range(horizon):
        obs, r, terminated, _truncated, info = env.step(np.array([u], dtype=np.float32))
        total += r
        assert terminated == (i == horizon - 1)
    assert total == pytest.approx(1.0, abs=1e-5)
    assert info["fidelity"] == pytest.approx(1.0, abs=1e-5)
