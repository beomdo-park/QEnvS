"""Analytic validation of the TwoLevel model."""

import dynamiqs as dq
import jax.numpy as jnp
import numpy as np
import pytest

from qenvs.models import TwoLevel
from qenvs.observables import ergotropy


@pytest.mark.parametrize("u", [0.5, 1.0, 2.0])
def test_resonant_rabi_matches_sin_squared(u):
    m = TwoLevel()
    h = m.hamiltonian(jnp.array([u]))
    ts = jnp.linspace(0.0, 2 * jnp.pi / u, 9)
    res = dq.sesolve(h, m.initial_state(), ts)
    p1 = dq.expect(m.observables()["p1"], res.states).real
    np.testing.assert_allclose(p1, np.sin(u * np.asarray(ts) / 2) ** 2, atol=1e-6)


def test_pi_pulse_full_inversion():
    m = TwoLevel()
    psi = m.evolve(m.initial_state(), m.hamiltonian(jnp.array([1.0])), jnp.pi)
    assert float(dq.fidelity(psi, dq.basis(2, 1))) == pytest.approx(1.0, abs=1e-6)


def test_free_decay_is_exponential():
    gamma1 = 0.3
    m = TwoLevel(gamma1=gamma1, initial="excited")
    assert m.is_open
    ts = jnp.linspace(0.0, 5.0, 6)
    res = dq.mesolve(m.drift(), m.jump_ops(), m.initial_state(), ts)
    p1 = dq.expect(m.observables()["p1"], res.states).real
    np.testing.assert_allclose(p1, np.exp(-gamma1 * np.asarray(ts)), atol=1e-6)


def test_pure_dephasing_kills_coherence_at_gamma_phi():
    gamma_phi = 0.4
    m = TwoLevel(gamma_phi=gamma_phi)
    plus = (dq.basis(2, 0) + dq.basis(2, 1)).unit().todm()
    ts = jnp.linspace(0.0, 3.0, 4)
    res = dq.mesolve(m.drift(), m.jump_ops(), plus, ts)
    sx = dq.expect(dq.sigmax(), res.states).real
    np.testing.assert_allclose(sx, np.exp(-gamma_phi * np.asarray(ts)), atol=1e-6)


def test_ergotropy_of_excited_and_mixed_states():
    m = TwoLevel(omega0=1.7)
    hb = m.battery_hamiltonian()
    assert float(ergotropy(dq.basis(2, 1), hb)) == pytest.approx(1.7, abs=1e-9)
    assert float(ergotropy(dq.basis(2, 0), hb)) == pytest.approx(0.0, abs=1e-9)
    mixed = 0.5 * dq.eye(2)
    assert float(ergotropy(mixed, hb)) == pytest.approx(0.0, abs=1e-9)
    # |+> has energy omega0/2 but full ergotropy omega0/2 (pure state => passive state is |0>)
    plus = (dq.basis(2, 0) + dq.basis(2, 1)).unit()
    assert float(ergotropy(plus, hb)) == pytest.approx(0.85, abs=1e-9)
