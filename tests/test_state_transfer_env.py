"""Functional environment: pi-pulse action sequence reaches the target."""

import jax
import jax.numpy as jnp
import pytest

import qenvs


def test_registry_lists_task():
    assert "qubit/StateTransfer-v0" in qenvs.registry
    with pytest.raises(KeyError):
        qenvs.make("Nope-v0")


def test_rollout_step_count():
    env = qenvs.make("qubit/StateTransfer-v0", horizon=7, dt=0.2)
    state, rewards = env.rollout(jnp.zeros((7, env.action_dim)))
    assert int(state.step) == 7
    assert rewards.shape == (7,)


def test_rollout_is_differentiable():
    env = qenvs.make("qubit/StateTransfer-v0", horizon=5, dt=0.2)

    def loss(actions):
        _, r = env.rollout(actions)
        return -r.sum()

    a = jnp.full((5, env.action_dim), 1.0)
    g = jax.grad(loss)(a)
    assert g.shape == a.shape
    assert jnp.all(jnp.isfinite(g))


def test_open_system_uses_density_matrix_and_decays():
    env = qenvs.make("qubit/StateTransfer-v0", horizon=4, dt=0.5, gamma1=0.5)
    assert env.model.is_open
    state, ts = env.reset(jax.random.PRNGKey(0))
    assert ts.observation.shape == (env.observation_dim,)
    # zero drive: stays in |0>, fidelity to |1> is 0 throughout
    for _ in range(4):
        state, ts = env.step(state, jnp.zeros(env.action_dim))
    assert float(ts.info["fidelity"]) == pytest.approx(0.0, abs=1e-8)
    assert bool(ts.done)


def test_step_is_jit_and_vmap_compatible():
    env = qenvs.make("qubit/StateTransfer-v0", horizon=3, dt=0.3)
    keys = jax.random.split(jax.random.PRNGKey(1), 8)
    states, _ = jax.vmap(env.reset)(keys)
    actions = jnp.linspace(-1.0, 1.0, 8)[:, None]
    step = jax.jit(jax.vmap(env.step))
    states, ts = step(states, actions)
    assert ts.observation.shape == (8, env.observation_dim)
    assert ts.reward.shape == (8,)
