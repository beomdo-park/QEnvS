"""Dicke / Tavis-Cummings battery: physics checks (from qenvs.validation.dicke) and API tests."""

import dynamiqs as dq
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import qenvs
import qenvs.tasks.dicke  # registers battery/Dicke-v0
from qenvs.envs import BatteryCharging
from qenvs.models.dicke import DickeBattery, boson_matrices, collective_spin_matrices
from qenvs.validation import dicke as validation


@pytest.mark.parametrize("check", validation.CHECKS, ids=lambda c: c.__name__)
def test_physics_check_passes(check):
    r = check()
    assert r.model == "DickeBattery"
    assert r.passed, f"{r.name}: error={r.error:.3e} > tol={r.tolerance:.0e} ({r.detail})"


# ------------------------------------------------------------------ model API
def test_defaults_and_dimensions():
    m = DickeBattery()
    assert (m.n_cells, m.photons, m.cutoff) == (3, 3, 3)
    assert (m.dim_battery, m.dim_cavity, m.dim) == (4, 4, 16)
    assert m.n_controls == 1
    assert DickeBattery(rwa=False).cutoff == 7
    assert DickeBattery(n_cells=2, n_photons=5).cutoff == 5
    assert DickeBattery(n_cells=2, n_photons=1, n_max=4, rwa=False).dim == 3 * 5


def test_invalid_configurations_rejected():
    with pytest.raises(ValueError, match="n_cells"):
        DickeBattery(n_cells=0)
    with pytest.raises(ValueError, match="n_max"):
        DickeBattery(n_cells=2, n_photons=3, n_max=2)
    with pytest.raises(ValueError, match="kappa"):
        DickeBattery(kappa=-0.1)


def test_closed_vs_open_state_types():
    assert DickeBattery().initial_state().isket
    assert not DickeBattery().is_open
    for kw in ({"kappa": 0.1}, {"gamma": 0.1}, {"gamma_phi": 0.1}):
        m = DickeBattery(**kw)
        assert m.is_open
        assert m.initial_state().isdm
        assert len(m.jump_ops()) == 1


def test_basis_ordering_and_initial_state():
    m = DickeBattery(n_cells=2, n_photons=2)
    psi = m.initial_state()
    # all cells ground (index 0 of the spin factor) x Fock |2>
    assert float(dq.fidelity(psi, dq.tensor(dq.basis(3, 0), dq.basis(3, 2)))) == pytest.approx(1.0)
    assert float(dq.expect(m.battery_hamiltonian(), psi).real) == pytest.approx(0.0)
    assert float(dq.expect(m.observables()["n_photons"], psi).real) == pytest.approx(2.0)
    assert float(dq.expect(m.observables()["excitation"], psi).real) == pytest.approx(2.0)
    # J- annihilates the collective ground state, J+ raises it by one excitation
    assert float((m.jminus() @ psi).norm()) == pytest.approx(0.0)
    up = (m.jplus() @ psi).unit()
    assert float(dq.expect(m.battery_hamiltonian(), up).real) == pytest.approx(1.0)
    # fully charged state has energy N omega0
    top = dq.tensor(dq.basis(3, 2), dq.basis(3, 0))
    assert float(dq.expect(m.battery_hamiltonian(), top).real) == pytest.approx(2.0)


def test_single_cell_reduces_to_pauli_operators():
    jp, jm, jz = collective_spin_matrices(1)
    np.testing.assert_allclose(jp, [[0.0, 0.0], [1.0, 0.0]])  # |1><0|, index 1 = excited
    np.testing.assert_allclose(jm, [[0.0, 1.0], [0.0, 0.0]])
    np.testing.assert_allclose(jz, np.diag([-0.5, 0.5]))
    a, adag = boson_matrices(2)
    np.testing.assert_allclose(a @ np.array([0.0, 0.0, 1.0]), [0.0, np.sqrt(2.0), 0.0])
    np.testing.assert_allclose(adag, a.T)


def test_hamiltonians_are_hermitian_and_differ_by_counter_rotating_terms():
    tc = DickeBattery(n_cells=2, rwa=True, n_max=4)
    dicke = DickeBattery(n_cells=2, rwa=False, n_max=4)
    for m in (tc, dicke):
        h = m.hamiltonian(jnp.array([0.3])).to_jax()
        np.testing.assert_allclose(h, h.conj().T, atol=1e-12)
    diff = (dicke.controls()[0] - tc.controls()[0]).to_jax()
    expected = (tc.a() @ tc.jminus() + tc.adag() @ tc.jplus()).to_jax()
    np.testing.assert_allclose(diff, expected, atol=1e-12)


def test_observables_keys():
    assert set(DickeBattery().observables()) == {"energy", "n_photons", "excitation"}


# ------------------------------------------------------------------ env API
def test_registered_task_and_factory_kwargs():
    env = qenvs.make("battery/Dicke-v0", n_cells=2, rwa=False, kappa=0.05, horizon=4, dt=0.3)
    assert isinstance(env, BatteryCharging)
    assert env.model.n_cells == 2 and not env.model.rwa and env.model.is_open
    assert (env.horizon, env.dt, env.action_bound) == (4, 0.3, 0.5)
    assert env.action_dim == 1
    assert env.observation_dim == 2 * env.model.dim**2 + 1
    _, ts = env.reset(jax.random.PRNGKey(0))
    assert ts.observation.shape == (env.observation_dim,)
    for k in ("energy", "ergotropy", "power", "purity", "n_photons", "excitation"):
        assert ts.info[k].shape == ()


def test_step_is_jit_and_vmap_compatible():
    env = BatteryCharging(model=DickeBattery(n_cells=2), horizon=3, dt=0.2, action_bound=0.5)
    keys = jax.random.split(jax.random.PRNGKey(1), 6)
    states, _ = jax.vmap(env.reset)(keys)
    actions = jnp.linspace(-0.5, 0.5, 6)[:, None]
    states, ts = jax.jit(jax.vmap(env.step))(states, actions)
    assert ts.observation.shape == (6, env.observation_dim)
    assert ts.reward.shape == (6,)
    assert ts.info["energy"].shape == (6,)
    # g and -g are related by a phase convention in TC: identical energies
    e = np.asarray(ts.info["energy"])
    np.testing.assert_allclose(e, e[::-1], atol=1e-6)


def test_rollout_is_differentiable():
    env = BatteryCharging(
        model=DickeBattery(n_cells=2, rwa=False, kappa=0.05), horizon=4, dt=0.2, action_bound=0.5
    )

    def loss(actions):
        _, r = env.rollout(actions)
        return -r.sum()

    a = jnp.full((4, env.action_dim), 0.3)
    g = jax.grad(loss)(a)
    assert g.shape == a.shape
    assert jnp.all(jnp.isfinite(g))
    assert jnp.all(g < 0)  # more coupling charges faster early in the episode


def test_return_equals_energy_gain_and_zero_coupling_holds_charge():
    # N = 1, n = 1: P_e = sin^2(g t); g = 0.5 for t = pi charges fully, then g = 0 holds it
    dt = float(np.pi / 20)
    env = BatteryCharging(model=DickeBattery(n_cells=1), horizon=40, dt=dt, action_bound=0.5)
    actions = jnp.concatenate([jnp.full((20, 1), 0.5), jnp.zeros((20, 1))])
    state, rewards = env.rollout(actions)
    _, ts = env.step(state, jnp.zeros(1))
    assert float(rewards.sum()) == pytest.approx(1.0, abs=1e-5)
    assert float(ts.info["energy"]) == pytest.approx(1.0, abs=1e-5)
    assert float(ts.info["ergotropy"]) == pytest.approx(1.0, abs=1e-5)
    assert float(ts.info["n_photons"]) == pytest.approx(0.0, abs=1e-5)


def test_open_rollout_keeps_density_matrix_physical():
    m = DickeBattery(n_cells=2, rwa=False, kappa=0.2, gamma=0.1, gamma_phi=0.05)
    env = BatteryCharging(model=m, horizon=6, dt=0.3, action_bound=0.5)
    actions = jax.random.uniform(jax.random.PRNGKey(3), (6, 1), minval=-0.5, maxval=0.5)
    state, _ = env.rollout(actions)
    rho = np.asarray(state.qstate.to_jax())
    assert rho.shape == (m.dim, m.dim)
    np.testing.assert_allclose(rho, rho.conj().T, atol=1e-8)
    assert np.trace(rho).real == pytest.approx(1.0, abs=1e-6)
    assert np.min(np.linalg.eigvalsh(rho)) > -1e-6
    _, ts = env.step(state, jnp.zeros(1))
    assert float(ts.info["purity"]) < 1.0
    assert float(ts.info["ergotropy"]) <= float(ts.info["energy"]) + 1e-6
