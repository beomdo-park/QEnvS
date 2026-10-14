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
from qenvs.registry import registry as task_registry
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
    keys = {"energy", "n_photons", "excitation", "top_fock"}
    assert set(DickeBattery().observables()) == keys


def test_top_fock_observable_reads_cutoff_population():
    # n_max = n_photons: the initial Fock state sits at the cutoff; with headroom it does not
    tight = DickeBattery(n_cells=2, n_photons=2)
    wide = DickeBattery(n_cells=2, n_photons=2, n_max=5)
    assert float(dq.expect(tight.top_fock(), tight.initial_state()).real) == pytest.approx(1.0)
    assert float(dq.expect(wide.top_fock(), wide.initial_state()).real) == pytest.approx(0.0)
    top = np.asarray(wide.top_fock().to_jax())
    np.testing.assert_allclose(top, np.kron(np.eye(3), np.diag([0, 0, 0, 0, 0, 1.0])))


def test_battery_reduced_traces_out_the_cavity():
    m = DickeBattery(n_cells=2, n_photons=2)
    psi = m.initial_state()
    rho_b, hb = m.battery_reduced(psi)
    assert rho_b.shape == (3, 3) and hb.shape == (3, 3)
    np.testing.assert_allclose(np.asarray(rho_b.to_jax()), np.diag([1.0, 0.0, 0.0]), atol=1e-12)
    np.testing.assert_allclose(np.asarray(hb.to_jax()), np.diag([0.0, 1.0, 2.0]), atol=1e-12)
    # density matrices are accepted too, and omega0 is the energy unit of the local H_B
    rho_b2, hb2 = DickeBattery(n_cells=2, n_photons=2, omega0=1.5).battery_reduced(psi.todm())
    np.testing.assert_allclose(np.asarray(rho_b2.to_jax()), np.asarray(rho_b.to_jax()), atol=1e-12)
    np.testing.assert_allclose(np.asarray(hb2.to_jax()), np.diag([0.0, 1.5, 3.0]), atol=1e-12)


def test_reduced_ergotropy_is_below_energy_when_entangled():
    # N = 2, g = 0.5, t = 1 (closed TC): the cells are entangled with the cavity, so the
    # ergotropy of the reduced cell state is strictly below the stored energy
    env = BatteryCharging(model=DickeBattery(n_cells=2), horizon=1, dt=1.0, action_bound=0.5)
    state, _ = env.reset(jax.random.PRNGKey(0))
    _, ts = env.step(state, jnp.array([0.5]))
    assert float(ts.info["energy"]) == pytest.approx(0.784, abs=2e-3)
    assert float(ts.info["ergotropy"]) == pytest.approx(0.277, abs=2e-3)
    assert float(ts.info["purity"]) == pytest.approx(1.0, abs=1e-5)  # global state is pure


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
    for k in ("energy", "ergotropy", "power", "purity", "n_photons", "excitation", "top_fock"):
        assert ts.info[k].shape == ()


def test_task_factory_forwards_every_model_kwarg():
    kw = {
        "n_cells": 2,
        "n_photons": 2,
        "n_max": 5,
        "omega0": 1.1,
        "omega_c": 1.2,
        "rwa": False,
        "kappa": 0.03,
        "gamma": 0.1,
        "gamma_phi": 0.05,
    }
    env = qenvs.make("battery/Dicke-v0", **kw)
    m = env.model
    assert (m.n_cells, m.n_photons, m.n_max) == (2, 2, 5)
    assert (m.omega0, m.omega_c) == (1.1, 1.2)
    assert m.rwa is False
    assert (m.kappa, m.gamma, m.gamma_phi) == (0.03, 0.1, 0.05)
    assert (m.photons, m.cutoff, m.dim) == (2, 5, 18)
    # omega0 and omega_c reach the operators, not just the fields
    h0 = np.asarray(m.drift().to_jax()).real
    expected = np.kron(np.diag([0.0, 1.1, 2.2]), np.eye(6)) + np.kron(
        np.eye(3), np.diag(1.2 * np.arange(6.0))
    )
    np.testing.assert_allclose(h0, expected, atol=1e-12)
    assert "omega0" in task_registry["battery/Dicke-v0"].kwargs


def test_coupling_sign_reversal_within_protocol_undoes_charging():
    # N = 1, n = 1: g dt = pi/4 per step. [+g, +g] charges fully (sin^2(pi/2) = 1) while
    # [+g, -g] undoes the first step (the sign change inside the protocol is physical);
    # g -> -g for the whole protocol is a gauge symmetry, so [-g, -g] also charges fully.
    g = 0.5
    env = BatteryCharging(
        model=DickeBattery(n_cells=1), horizon=2, dt=float(np.pi / (4 * g)), action_bound=g
    )
    for actions, expected in (([g, g], 1.0), ([g, -g], 0.0), ([-g, -g], 1.0), ([-g, g], 0.0)):
        state, rewards = env.rollout(jnp.asarray(actions)[:, None])
        _, ts = env.step(state, jnp.zeros(1))
        assert float(rewards.sum()) == pytest.approx(expected, abs=1e-5), actions
        assert float(ts.info["energy"]) == pytest.approx(expected, abs=1e-5), actions


def test_step_is_jit_and_vmap_compatible():
    env = BatteryCharging(model=DickeBattery(n_cells=2), horizon=3, dt=0.2, action_bound=0.5)
    keys = jax.random.split(jax.random.PRNGKey(1), 6)
    states, _ = jax.vmap(env.reset)(keys)
    actions = jnp.linspace(-0.5, 0.5, 6)[:, None]
    states, ts = jax.jit(jax.vmap(env.step))(states, actions)
    assert ts.observation.shape == (6, env.observation_dim)
    assert ts.reward.shape == (6,)
    assert ts.info["energy"].shape == (6,)
    # a single step is a whole protocol: g -> -g is a gauge symmetry, identical energies
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
