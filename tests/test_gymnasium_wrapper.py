"""Gymnasium adapter passes the official env checker."""

import jax
import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")
from gymnasium.utils.env_checker import check_env  # noqa: E402

import qenvs  # noqa: E402
from qenvs.wrappers import GymnasiumEnv  # noqa: E402


@pytest.fixture
def x32():
    """Run with jax x64 off (library default); conftest enables x64, which hides read-only views."""
    if hasattr(jax, "enable_x64"):
        ctx = jax.enable_x64(False)
    else:  # pragma: no cover (older jax)
        from jax.experimental import disable_x64

        ctx = disable_x64()
    with ctx:
        assert not jax.config.jax_enable_x64
        yield


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


def test_normalize_actions_space_and_checker_warning_free():
    import warnings

    env = GymnasiumEnv(
        qenvs.make("qubit/StateTransfer-v0", horizon=5, dt=0.2), normalize_actions=True
    )
    assert env.action_space.low[0] == -1.0
    assert env.action_space.high[0] == 1.0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_env(env, skip_render_check=True)
    # the unbounded observation space warning is expected (observation space is unchanged)
    assert not [w for w in caught if "action space" in str(w.message)]


def test_normalize_actions_scaling_exact():
    raw = qenvs.make("qubit/StateTransfer-v0", horizon=4, dt=0.2)
    b = float(raw.action_bound)
    e1 = GymnasiumEnv(raw, normalize_actions=True)
    e2 = GymnasiumEnv(raw)
    e1.reset(seed=1)
    e2.reset(seed=1)
    for u in (0.3, -0.7, 1.0, 0.0):
        o1, r1, *_ = e1.step(np.array([u], dtype=np.float32))
        o2, r2, *_ = e2.step(np.array([u * b], dtype=np.float32))
        assert r1 == r2
        np.testing.assert_array_equal(o1, o2)


def test_truncated_is_never_true():
    env = GymnasiumEnv(qenvs.make("qubit/StateTransfer-v0", horizon=3, dt=0.2))
    env.reset(seed=0)
    for _ in range(3):
        *_, terminated, truncated, _ = env.step(np.array([0.5], dtype=np.float32))
        assert truncated is False
    assert terminated is True


def test_returned_arrays_are_writable(x32):
    env = GymnasiumEnv(qenvs.make("qubit/StateTransfer-v0", horizon=3, dt=0.2))
    obs, info = env.reset(seed=0)
    assert obs.flags.writeable and all(v.flags.writeable for v in info.values())
    obs, _, _, _, info = env.step(np.array([0.5], dtype=np.float32))
    assert obs.flags.writeable and all(v.flags.writeable for v in info.values())
    obs[0] = 0.0  # in-place edit must not raise
    info["fidelity"][...] = 0.0
