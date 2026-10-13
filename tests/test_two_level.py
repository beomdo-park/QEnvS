"""TwoLevel model API (physics numbers live in test_validation.py)."""

import dynamiqs as dq

from qenvs.models import TwoLevel


def test_closed_vs_open_state_types():
    assert TwoLevel().initial_state().isket
    assert TwoLevel(gamma1=0.1).initial_state().isdm
    assert not TwoLevel().is_open
    assert TwoLevel(gamma_phi=0.1).is_open


def test_relaxation_operator_lowers_excited_state():
    (lower,) = TwoLevel(gamma1=1.0).jump_ops()
    out = lower @ dq.basis(2, 1)
    assert float(dq.fidelity(out.unit(), dq.basis(2, 0))) == 1.0
    assert float((lower @ dq.basis(2, 0)).norm()) == 0.0


def test_observables_and_controls():
    m = TwoLevel(omega0=2.0)
    assert set(m.observables()) == {"energy", "p1"}
    assert m.n_controls == 1
    assert float(dq.expect(m.battery_hamiltonian(), dq.basis(2, 1)).real) == 2.0
