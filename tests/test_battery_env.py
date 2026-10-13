"""Generic BatteryCharging env on the TwoLevel model."""

import jax
import jax.numpy as jnp
import pytest

from qenvs.envs import BatteryCharging
from qenvs.models import TwoLevel


@pytest.mark.parametrize("reward", ["energy", "ergotropy"])
def test_pi_pulse_charges_fully(reward):
    env = BatteryCharging(model=TwoLevel(omega0=1.5), horizon=10, dt=0.2, reward=reward)
    u = jnp.pi / (10 * 0.2)
    state, rewards = env.rollout(jnp.full((10, 1), u))
    assert float(rewards.sum()) == pytest.approx(1.5, abs=1e-6)
    _, ts = env.step(state, jnp.zeros(1))
    assert float(ts.info["energy"]) == pytest.approx(1.5, abs=1e-6)
    assert float(ts.info["ergotropy"]) == pytest.approx(1.5, abs=1e-6)


def test_bad_reward_rejected():
    with pytest.raises(ValueError, match="reward"):
        BatteryCharging(model=TwoLevel(), reward="power")


def test_open_battery_info_fields_and_vmap():
    env = BatteryCharging(model=TwoLevel(gamma1=0.2), horizon=3, dt=0.3)
    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    states, ts = jax.vmap(env.reset)(keys)
    states, ts = jax.jit(jax.vmap(env.step))(states, jnp.ones((4, 1)))
    for k in ("energy", "ergotropy", "power", "purity", "p1"):
        assert ts.info[k].shape == (4,)
    assert jnp.all(ts.info["ergotropy"] <= ts.info["energy"] + 1e-6)
    assert jnp.all(ts.info["purity"] <= 1.0 + 1e-6)
