"""QubitBattery model, the battery/TwoLevel-v0 task factory and its physics checks."""

import dynamiqs as dq
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import qenvs
from qenvs.envs import BatteryCharging
from qenvs.models.qubit_battery import QubitBattery
from qenvs.tasks.qubit_battery import two_level_battery
from qenvs.validation import qubit_battery as validation


# ------------------------------------------------------------------ physics checks
@pytest.mark.parametrize("check", validation.CHECKS, ids=lambda c: c.__name__)
def test_physics_check_passes(check):
    r = check()
    assert r.model == "QubitBattery"
    assert r.passed, f"{r.name}: error={r.error:.3e} > tol={r.tolerance:.0e} ({r.detail})"


# ------------------------------------------------------------------ model API
def test_state_types_follow_dissipation():
    assert QubitBattery().initial_state().isket()
    assert QubitBattery(initial="thermal").initial_state().isket()  # n_th = 0 -> |0>
    assert QubitBattery(gamma1=0.1).initial_state().isdm()
    assert QubitBattery(gamma_phi=0.1, initial="thermal").initial_state().isdm()
    assert not QubitBattery().is_open
    assert QubitBattery(gamma1=0.1, n_th=0.5).is_open


def test_thermal_state_populations():
    n_th = 0.75
    m = QubitBattery(gamma1=0.2, n_th=n_th, initial="thermal")
    rho = np.asarray(m.initial_state().to_jax())
    p1 = n_th / (2 * n_th + 1)
    assert m.thermal_population == pytest.approx(p1)
    assert rho[1, 1].real == pytest.approx(p1)
    assert rho[0, 0].real == pytest.approx(1 - p1)
    assert abs(rho[0, 1]) == 0.0
    # Gibbs ratio p1/p0 = n_th/(n_th + 1)
    assert rho[1, 1].real / rho[0, 0].real == pytest.approx(n_th / (n_th + 1))


def test_jump_operators_and_rates():
    g, n = 0.5, 0.3
    emission, absorption, dephasing = QubitBattery(gamma1=g, n_th=n, gamma_phi=0.4).jump_ops()
    e1 = dq.basis(2, 1)
    e0 = dq.basis(2, 0)
    # emission |1> -> |0| at rate gamma1 (n_th + 1); annihilates |0>
    assert float((emission @ e1).norm()) ** 2 == pytest.approx(g * (n + 1))
    assert float(dq.fidelity((emission @ e1).unit(), e0)) == pytest.approx(1.0)
    assert float((emission @ e0).norm()) == 0.0
    # absorption |0> -> |1> at rate gamma1 n_th; annihilates |1>
    assert float((absorption @ e0).norm()) ** 2 == pytest.approx(g * n)
    assert float(dq.fidelity((absorption @ e0).unit(), e1)) == pytest.approx(1.0)
    assert float((absorption @ e1).norm()) == 0.0
    # dephasing sqrt(gamma_phi/2) sigma_z
    assert np.allclose(np.asarray(dephasing.to_jax()), np.sqrt(0.2) * np.diag([1.0, -1.0]))
    # zero temperature: absorption channel absent, identical to TwoLevel
    assert len(QubitBattery(gamma1=g).jump_ops()) == 1
    assert len(QubitBattery(gamma1=0.0, n_th=n).jump_ops()) == 0


def test_hamiltonian_and_observables():
    m = QubitBattery(omega0=2.0, detuning=0.6)
    assert m.dim == 2
    assert m.n_controls == 1
    assert set(m.observables()) == {"energy", "p1"}
    assert float(dq.expect(m.battery_hamiltonian(), dq.basis(2, 1)).real) == 2.0
    assert float(dq.expect(m.battery_hamiltonian(), dq.basis(2, 0)).real) == 0.0
    h = np.asarray(m.hamiltonian(jnp.array([1.5])).to_jax())
    assert np.allclose(h, np.array([[0.3, 0.75], [0.75, -0.3]]))


@pytest.mark.parametrize(
    "kw",
    [
        {"initial": "excited"},
        {"n_th": -0.1},
        {"gamma1": -1.0},
        {"gamma_phi": -1.0},
        {"initial": "thermal", "n_th": 0.5},  # mixed state but closed dynamics
    ],
)
def test_invalid_parameters_rejected(kw):
    with pytest.raises(ValueError):
        QubitBattery(**kw)


# ------------------------------------------------------------------ task factory
def test_task_factory_routes_kwargs():
    env = two_level_battery(
        horizon=5,
        dt=0.2,
        action_bound=1.5,
        reward="ergotropy",
        detuning=0.3,
        gamma1=0.1,
        n_th=0.2,
        gamma_phi=0.05,
        initial="thermal",
    )
    assert isinstance(env, BatteryCharging)
    assert isinstance(env.model, QubitBattery)
    assert (env.horizon, env.dt, env.action_bound, env.reward) == (5, 0.2, 1.5, "ergotropy")
    m = env.model
    assert (m.detuning, m.gamma1, m.n_th, m.gamma_phi, m.initial) == (
        0.3,
        0.1,
        0.2,
        0.05,
        "thermal",
    )
    assert env.action_dim == 1
    assert env.observation_dim == 2 * 4 + 1
    closed = two_level_battery()
    assert not closed.model.is_open
    assert closed.observation_dim == 2 * 2 + 1


def test_task_registered_on_import():
    assert "battery/TwoLevel-v0" in qenvs.registry
    spec = qenvs.registry["battery/TwoLevel-v0"]
    assert spec.references and set(spec.kwargs) >= {"reward", "n_th", "initial"}
    env = qenvs.make("battery/TwoLevel-v0", horizon=3, gamma1=0.2, n_th=0.1)
    assert env.model.is_open


# ------------------------------------------------------------------ environment API
def test_step_is_jit_and_vmap_compatible():
    env = BatteryCharging(model=QubitBattery(gamma1=0.2, n_th=0.3), horizon=3, dt=0.3)
    keys = jax.random.split(jax.random.PRNGKey(1), 8)
    states, ts = jax.vmap(env.reset)(keys)
    assert ts.observation.shape == (8, env.observation_dim)
    actions = jnp.linspace(-1.0, 1.0, 8)[:, None]
    step = jax.jit(jax.vmap(env.step))
    states, ts = step(states, actions)
    assert ts.observation.shape == (8, env.observation_dim)
    assert ts.reward.shape == (8,)
    assert ts.done.shape == (8,)
    for k in ("energy", "p1", "ergotropy", "power", "purity"):
        assert ts.info[k].shape == (8,)
    assert jnp.all(ts.info["ergotropy"] <= ts.info["energy"] + 1e-6)
    assert jnp.all(ts.info["purity"] <= 1.0 + 1e-6)


@pytest.mark.parametrize("reward", ["energy", "ergotropy"])
def test_rollout_is_differentiable(reward):
    env = BatteryCharging(
        model=QubitBattery(gamma1=0.1, n_th=0.2, gamma_phi=0.05), horizon=5, dt=0.2, reward=reward
    )

    def loss(actions):
        _, r = env.rollout(actions)
        return -r.sum()

    a = jnp.full((5, env.action_dim), 0.7)
    g = jax.grad(loss)(a)
    assert g.shape == a.shape
    assert jnp.all(jnp.isfinite(g))
    assert float(jnp.abs(g).max()) > 0.0


def test_open_rollout_keeps_density_matrix_physical():
    env = BatteryCharging(
        model=QubitBattery(detuning=0.4, gamma1=0.3, n_th=0.4, gamma_phi=0.1), horizon=8, dt=0.25
    )
    actions = jax.random.uniform(jax.random.PRNGKey(2), (8, 1), minval=-2.0, maxval=2.0)
    state, rewards = env.rollout(actions)
    assert int(state.step) == 8
    assert rewards.shape == (8,)
    rho = np.asarray(state.qstate.to_jax())
    assert rho.shape == (2, 2)
    assert np.allclose(rho, rho.conj().T, atol=1e-8)
    assert np.trace(rho).real == pytest.approx(1.0, abs=1e-8)
    assert np.min(np.linalg.eigvalsh(rho)) >= -1e-8


def test_zero_drive_from_ground_relaxes_to_thermal_population():
    n_th = 0.5
    env = BatteryCharging(model=QubitBattery(gamma1=2.0, n_th=n_th), horizon=10, dt=1.0)
    state, ts = env.reset(jax.random.PRNGKey(0))
    assert float(ts.info["energy"]) == 0.0
    for _ in range(env.horizon):
        state, ts = env.step(state, jnp.zeros(env.action_dim))
    assert bool(ts.done)
    assert float(ts.info["p1"]) == pytest.approx(n_th / (2 * n_th + 1), abs=1e-6)
    assert float(ts.info["ergotropy"]) == pytest.approx(0.0, abs=1e-6)  # thermal state is passive
