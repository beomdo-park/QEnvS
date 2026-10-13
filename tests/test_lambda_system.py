"""LambdaSystem model, STIRAP physics checks and env API (transfer/LambdaSTIRAP-v0)."""

import importlib

import dynamiqs as dq
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import qenvs
from qenvs.envs.state_transfer import StateTransfer
from qenvs.models.lambda_system import LambdaSystem, stirap_pulses
from qenvs.validation import lambda_system as validation


def _env(gamma=0.0, horizon=10, dt=0.2, **model_kw):
    model = LambdaSystem(gamma=gamma, **model_kw)
    target = dq.basis(3, 2)
    target = target.todm() if model.is_open else target
    return StateTransfer(model=model, target=target, horizon=horizon, dt=dt, action_bound=1.0)


# ------------------------------------------------------------- physics checks
@pytest.mark.parametrize("check", validation.CHECKS, ids=lambda c: c.__name__)
def test_physics_check_passes(check):
    r = check()
    assert r.model == "LambdaSystem"
    assert r.passed, f"{r.name}: error={r.error:.3e} > tol={r.tolerance:.0e} ({r.detail})"


# ----------------------------------------------------------------- model API
def test_closed_vs_open_state_types():
    assert LambdaSystem().initial_state().isket
    assert LambdaSystem(gamma=0.1).initial_state().isdm
    assert not LambdaSystem().is_open
    assert LambdaSystem(gamma=0.1).is_open
    assert LambdaSystem().dim == 3
    assert LambdaSystem().n_controls == 2


def test_initial_state_selection():
    for name, idx in (("1", 0), ("2", 1), ("3", 2)):
        psi = LambdaSystem(initial=name).initial_state()
        assert float(dq.fidelity(psi, dq.basis(3, idx))) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="initial"):
        LambdaSystem(initial="0")


def test_jump_ops_decay_from_excited_with_equal_branching():
    g = 0.8
    ops = LambdaSystem(gamma=g).jump_ops()
    assert len(ops) == 2
    excited = dq.basis(3, 1)
    for op, idx in zip(ops, (0, 2), strict=True):
        out = op @ excited
        assert float(out.norm()) ** 2 == pytest.approx(g / 2)
        assert float(dq.fidelity(out.unit(), dq.basis(3, idx))) == pytest.approx(1.0)
        assert float((op @ dq.basis(3, 0)).norm()) == 0.0
        assert float((op @ dq.basis(3, 2)).norm()) == 0.0


def test_operators_follow_basis_convention():
    pump, stokes = LambdaSystem().controls()
    expected_pump = np.zeros((3, 3))
    expected_pump[0, 1] = expected_pump[1, 0] = 0.5
    expected_stokes = np.zeros((3, 3))
    expected_stokes[2, 1] = expected_stokes[1, 2] = 0.5
    np.testing.assert_allclose(np.asarray(pump.to_jax()), expected_pump)
    np.testing.assert_allclose(np.asarray(stokes.to_jax()), expected_stokes)
    drift = np.asarray(LambdaSystem(delta_one_photon=0.3, delta_two_photon=-0.2).drift().to_jax())
    np.testing.assert_allclose(np.diag(drift).real, [0.0, 0.3, -0.2])
    np.testing.assert_allclose(drift - np.diag(np.diag(drift)), 0.0)
    assert set(LambdaSystem().observables()) == {"p1", "p2", "p3"}
    with pytest.raises(NotImplementedError):
        LambdaSystem().battery_hamiltonian()


def test_stirap_pulses_shape_and_order():
    horizon, dt = 50, 0.4
    a = stirap_pulses(horizon, dt, 1.0, 3.0, 4.0)
    assert a.shape == (horizon, 2)
    t = (jnp.arange(horizon) + 0.5) * dt
    # counter-intuitive: Stokes (column 1) peaks before the pump (column 0)
    assert float(t[jnp.argmax(a[:, 1])]) < float(t[jnp.argmax(a[:, 0])])
    assert float(t[jnp.argmax(a[:, 1])]) == pytest.approx(0.5 * horizon * dt - 3.0, abs=dt)
    assert 0.0 < float(a.max()) <= 1.0
    b = stirap_pulses(horizon, dt, 1.0, 3.0, 4.0, order="intuitive")
    np.testing.assert_allclose(np.asarray(b), np.asarray(a)[:, ::-1])
    with pytest.raises(ValueError, match="order"):
        stirap_pulses(horizon, dt, 1.0, 3.0, 4.0, order="simultaneous")


# ------------------------------------------------------------------- env API
def test_task_factory_and_defaults():
    importlib.import_module("qenvs.tasks.lambda_system")  # registers the task id
    assert "transfer/LambdaSTIRAP-v0" in qenvs.registry
    env = qenvs.make("transfer/LambdaSTIRAP-v0")
    assert isinstance(env, StateTransfer)
    assert (env.horizon, env.dt, env.action_bound) == (100, 0.2, 1.0)
    assert env.action_dim == 2
    assert env.observation_dim == 7
    assert not env.model.is_open
    assert env.target.isket
    env_open = qenvs.make(
        "transfer/LambdaSTIRAP-v0", gamma=0.2, horizon=5, dt=0.5, delta_one_photon=0.1
    )
    assert env_open.model.is_open
    assert env_open.target.isdm
    assert env_open.observation_dim == 19
    assert (env_open.horizon, env_open.dt) == (5, 0.5)
    assert env_open.model.delta_one_photon == 0.1


def test_observation_info_and_step_shapes():
    env = _env(horizon=3)
    state, ts = env.reset(jax.random.PRNGKey(0))
    assert ts.observation.shape == (env.observation_dim,)
    assert set(ts.info) == {"p1", "p2", "p3", "fidelity"}
    assert float(ts.info["p1"]) == pytest.approx(1.0)
    assert float(ts.info["fidelity"]) == pytest.approx(0.0)
    state, ts = env.step(state, jnp.array([0.5, -0.3]))
    assert ts.observation.shape == (env.observation_dim,)
    assert ts.reward.shape == ()
    assert float(ts.info["p1"] + ts.info["p2"] + ts.info["p3"]) == pytest.approx(1.0, abs=1e-6)
    assert float(ts.info["p2"]) > 0.0
    assert not bool(ts.done)


def test_step_is_jit_and_vmap_compatible():
    env = _env(horizon=3, dt=0.3)
    keys = jax.random.split(jax.random.PRNGKey(1), 8)
    states, _ = jax.vmap(env.reset)(keys)
    actions = jnp.stack([jnp.linspace(-1.0, 1.0, 8), jnp.linspace(1.0, -1.0, 8)], axis=1)
    step = jax.jit(jax.vmap(env.step))
    states, ts = step(states, actions)
    assert ts.observation.shape == (8, env.observation_dim)
    assert ts.reward.shape == (8,)
    assert ts.info["p3"].shape == (8,)


def test_rollout_is_differentiable():
    env = _env(horizon=5, dt=0.2)

    def loss(actions):
        _, r = env.rollout(actions)
        return -r.sum()

    a = stirap_pulses(5, 0.2, 1.0, 0.3, 0.4)
    g = jax.grad(loss)(a)
    assert g.shape == a.shape
    assert bool(jnp.all(jnp.isfinite(g)))
    assert float(jnp.abs(g).max()) > 0.0


def test_open_rollout_keeps_density_matrix_physical():
    env = _env(gamma=0.4, horizon=12, dt=0.3, delta_one_photon=0.2)
    actions = jax.random.uniform(jax.random.PRNGKey(2), (12, 2), minval=-1.0, maxval=1.0)
    state, rewards = env.rollout(actions)
    rho = np.asarray(state.qstate.to_jax())
    assert rho.shape == (3, 3)
    assert rewards.shape == (12,)
    assert int(state.step) == 12
    np.testing.assert_allclose(rho, rho.conj().T, atol=1e-8)
    assert np.trace(rho).real == pytest.approx(1.0, abs=1e-6)
    assert float(np.min(np.linalg.eigvalsh(rho))) >= -1e-8


def test_stirap_pulses_transfer_in_default_task_regime():
    # default task (T = 20, Omega_max = 1): Gaussian STIRAP pulses reach P3 > 0.99
    env = _env(horizon=100, dt=0.2)
    state, rewards = env.rollout(stirap_pulses(100, 0.2, 1.0, 3.0, 6.0))
    assert float(rewards.sum()) > 0.99
    assert float(dq.expect(env.model.observables()["p3"], state.qstate).real) > 0.99


def test_task_factory_forwards_detunings_and_gamma():
    import numpy as np

    import qenvs

    env = qenvs.make(
        "transfer/LambdaSTIRAP-v0",
        delta_one_photon=0.4,
        delta_two_photon=-0.15,
        gamma=0.2,
        horizon=3,
    )
    m = env.model
    assert m.delta_one_photon == 0.4 and m.delta_two_photon == -0.15 and m.gamma == 0.2
    np.testing.assert_allclose(np.diag(np.asarray(m.drift().to_jax())).real, [0.0, 0.4, -0.15])
