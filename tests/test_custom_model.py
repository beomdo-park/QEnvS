"""HamiltonianModel (bring your own Hamiltonian) and the generic model checks.

Three groups: construction guarantees (what is rejected), the generic checks on valid models
(including every built-in model: no false positives), and deliberately broken models that
construct fine but must fail a named check (the checks are discriminating).
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys
import textwrap
import warnings

import dynamiqs as dq
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import qenvs
from qenvs import cli, toolkit
from qenvs.envs import BatteryCharging, StateTransfer
from qenvs.models import DickeBattery, LambdaSystem, QubitBattery, TwoLevel
from qenvs.models.custom import HamiltonianModel
from qenvs.registry import TaskSpec, registry
from qenvs.validation import generic
from qenvs.validation.generic import (
    TOL_CONSERVATION,
    TOL_SOLVER,
    _Context,
    default_dt,
    model_checks,
)

SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]])
SZ = np.diag([1.0, -1.0]).astype(complex)
ID2 = np.eye(2, dtype=complex)
LOWER = np.array([[0, 1], [0, 0]], dtype=complex)  # |0><1|, index 1 is excited
XY = 0.5 * (np.kron(SX, SX) + np.kron(SY, SY))  # iSWAP-type XY coupling
HB2 = np.diag([0.0, 1.0, 1.0, 2.0])  # two independent qubits, one quantum each


def two_qubit(**overrides):
    """Two detuned qubits, a tunable XY coupler and two local x drives, battery H_B."""
    kw = {
        "drift": 0.1 * np.kron(SZ, ID2) + 0.05 * np.kron(ID2, SZ),
        "controls": [XY, np.kron(SX, ID2) / 2, np.kron(ID2, SX) / 2],
        "initial_state": np.array([0, 1, 0, 0]),  # |01>
        "battery_hamiltonian": HB2,
        "dims": (2, 2),
    }
    kw.update(overrides)
    return HamiltonianModel.from_operators(**kw)


def open_two_qubit(**overrides):
    jumps = [0.3 * np.kron(LOWER, ID2), 0.2 * np.kron(ID2, LOWER)]
    return two_qubit(**{"jump_ops": jumps, **overrides})


def qutrit(**overrides):
    """Driven anharmonic qutrit (rotating frame): detuning, anharmonicity, I and Q drives."""
    a = np.diag(np.sqrt([1.0, 2.0]), 1).astype(complex)
    ad = a.conj().T
    kw = {
        "drift": 0.1 * ad @ a + 0.5 * (-0.3) * ad @ ad @ a @ a,
        "controls": [(a + ad) / 2, 1j * (ad - a) / 2],
        "observables": {"p2": np.diag([0.0, 0.0, 1.0])},
    }
    kw.update(overrides)
    return HamiltonianModel.from_operators(**kw)


def failed(results):
    return {r.name for r in results if not r.passed and not r.skipped}


def by_name(results):
    return {r.name: r for r in results}


# ------------------------------------------------------------ construction
def test_basic_properties_and_state_types():
    m = two_qubit()
    assert m.dim == 4 and m.dims == (2, 2) and m.n_controls == 3
    assert not m.is_open and m.initial_state().isket()
    assert m.initial_state().dims == (2, 2)
    assert dq.ptrace(m.initial_state(), 0).shape == (2, 2)
    # ket + jump operators becomes a density matrix, as in the built-in open models
    mo = open_two_qubit()
    assert mo.is_open and mo.initial_state().isdm() and len(mo.jump_ops()) == 2
    assert mo.hamiltonian(jnp.ones(3)).shape == (4, 4)
    # default initial state is |0>
    m0 = HamiltonianModel(np.diag([0.0, 1.0]), [SX])
    assert np.allclose(np.asarray(m0.initial_state().to_jax()), [[1], [0]])
    with pytest.raises(NotImplementedError, match="battery_hamiltonian"):
        m0.battery_hamiltonian()


def test_pure_density_matrix_without_jumps_becomes_ket_and_mixed_stays_dm():
    pure = HamiltonianModel(np.diag([0.0, 1.0]), [SX], initial_state=np.diag([0.0, 1.0]))
    assert not pure.is_open and pure.initial_state().isket()
    mixed = HamiltonianModel(np.diag([0.0, 1.0]), [SX], initial_state=0.5 * np.eye(2))
    assert mixed.is_open and mixed.initial_state().isdm()
    # the master equation with an empty dissipator integrates a mixed state unitarily
    env = StateTransfer(model=mixed, target=dq.basis(2, 1), horizon=4, dt=0.2)
    state, _ = env.rollout(jnp.ones((4, 1)))
    assert float(dq.trace(state.qstate).real) == pytest.approx(1.0, abs=1e-6)
    assert all(r.passed for r in model_checks(mixed) if not r.skipped)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {"controls": [SX, SX + 0.2j * SZ]},
            r"controls\[1\] is not Hermitian",
        ),
        ({"drift": np.array([[0, 1.0], [0.0, 0.0]])}, "drift is not Hermitian"),
        ({"observables": {"bad": np.array([[0, 1.0], [0.0, 1.0]])}}, "observables.*bad.*Hermitian"),
        ({"battery_hamiltonian": np.array([[0, 1.0], [0.0, 1.0]])}, "battery_hamiltonian.*Hermi"),
        ({"controls": [np.eye(3)]}, r"controls\[0\] has dimension 3"),
        ({"controls": [np.ones((2, 3))]}, "square"),
        ({"controls": []}, "at least one control"),
        ({"jump_ops": [np.eye(3)]}, r"jump_ops\[0\] has dimension 3"),
        ({"initial_state": np.array([1.0, 1.0])}, "not normalised"),
        ({"initial_state": np.array([0.0, 0.0])}, "not normalised"),
        ({"initial_state": np.diag([1.0, 1.0])}, "trace 2"),
        ({"initial_state": np.diag([1.5, -0.5])}, "not positive semidefinite"),
        ({"initial_state": np.array([[0.5, 0.3], [0.0, 0.5]])}, "not Hermitian"),
        ({"initial_state": np.ones((3, 3)) / 3}, "expected a ket"),
        ({"initial_state": np.array([[1.0, 0.0]])}, "bra"),
        ({"drift": np.array([[0, np.nan], [np.nan, 0]])}, "NaN or inf"),
        ({"dims": (3, 2)}, "do not multiply"),
    ],
)
def test_invalid_inputs_are_rejected_with_a_named_error(kwargs, match):
    base = {"drift": np.diag([0.0, 1.0]), "controls": [SX]}
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        HamiltonianModel.from_operators(**base)


def test_hermiticity_tolerance_is_relative_1e_8():
    ok = SX * 1e6 + 3e-3 * np.array([[0, 1], [-1, 0]])  # asymmetry 6e-9 relative
    HamiltonianModel(np.diag([0.0, 1.0]), [ok])
    bad = SX * 1e6 + 1e-1 * np.array([[0, 1], [-1, 0]])  # asymmetry 2e-7 relative
    with pytest.raises(ValueError, match="not Hermitian"):
        HamiltonianModel(np.diag([0.0, 1.0]), [bad])
    # accepted operators are symmetrised exactly
    m = HamiltonianModel(np.diag([0.0, 1.0]), [ok])
    c = np.asarray(m.control_matrices[0])
    assert np.array_equal(c, c.conj().T)


def test_dims_are_inferred_and_conflicts_rejected():
    a = dq.asqarray(np.eye(12), dims=(2, 6))
    b = dq.asqarray(np.eye(12), dims=(3, 4))
    flat = dq.asqarray(np.eye(12), dims=(12,))  # carries no composite information
    assert HamiltonianModel(a, [flat]).dims == (2, 6)
    assert HamiltonianModel(flat, [a]).dims == (2, 6)
    with pytest.raises(ValueError, match="conflicting dims"):
        HamiltonianModel(a, [b])
    assert HamiltonianModel(a, [b], dims=(4, 3)).dims == (4, 3)  # explicit dims win


# ------------------------------------------------------------ input types
def _qutip_two_qubit():
    qt = pytest.importorskip("qutip")
    sz, sx, sy, qe = qt.sigmaz(), qt.sigmax(), qt.sigmay(), qt.qeye(2)
    xy = 0.5 * (qt.tensor(sx, sx) + qt.tensor(sy, sy))
    return HamiltonianModel.from_operators(
        0.1 * qt.tensor(sz, qe) + 0.05 * qt.tensor(qe, sz),
        [xy, qt.tensor(sx, qe) / 2, qt.tensor(qe, sx) / 2],
        initial_state=qt.tensor(qt.basis(2, 0), qt.basis(2, 1)),
        battery_hamiltonian=qt.Qobj(HB2, dims=[[2, 2], [2, 2]]),
    )


def _qarray_two_qubit():
    sx, sy, sz, ie = dq.sigmax(), dq.sigmay(), dq.sigmaz(), dq.eye(2)
    xy = 0.5 * (dq.tensor(sx, sx) + dq.tensor(sy, sy))
    return HamiltonianModel.from_operators(
        0.1 * dq.tensor(sz, ie) + 0.05 * dq.tensor(ie, sz),
        [xy, dq.tensor(sx, ie) / 2, dq.tensor(ie, sx) / 2],
        initial_state=dq.tensor(dq.basis(2, 0), dq.basis(2, 1)),
        battery_hamiltonian=dq.asqarray(HB2, dims=(2, 2)),
    )


def _jax_two_qubit():
    j = jnp.asarray
    return two_qubit(
        drift=j(0.1 * np.kron(SZ, ID2) + 0.05 * np.kron(ID2, SZ)),
        controls=j(np.stack([XY, np.kron(SX, ID2) / 2, np.kron(ID2, SX) / 2])),  # (k, n, n) stack
        dims=None,
    )


@pytest.mark.parametrize("builder", [_qutip_two_qubit, _qarray_two_qubit, _jax_two_qubit])
def test_all_input_types_give_the_same_model_and_pass_every_check(builder):
    ref = two_qubit()
    m = builder()
    for name in ("drift_matrix", "control_matrices", "initial_matrix", "battery_matrix"):
        assert np.allclose(getattr(m, name), getattr(ref, name), atol=1e-12), name
    if builder is not _jax_two_qubit:
        assert m.dims == (2, 2)  # inferred from the QArray / Qobj
    results = model_checks(m)
    assert not failed(results), [(r.name, r.error) for r in results if not r.passed]
    assert not any(r.skipped for r in results)


# ------------------------------------------------------------ valid models pass
EXPECTED_CLOSED = {
    "structure",
    "hermiticity",
    "initial_state_valid",
    "norm_conservation",
    "solver_convergence",
    "qutip_solver_cross_check",
}


def test_valid_closed_battery_model_passes_all_generic_checks():
    results = model_checks(two_qubit())
    assert {r.name for r in results} == EXPECTED_CLOSED | {
        "battery_consistency",
        "ergotropy_bounds",
    }
    assert not failed(results) and not any(r.skipped for r in results)
    assert all(r.model == "HamiltonianModel" for r in results)


def test_valid_open_model_passes_and_reports_density_matrix_checks():
    results = model_checks(open_two_qubit())
    names = {r.name for r in results}
    assert "density_matrix_valid" in names and "norm_conservation" not in names
    assert not failed(results)


def test_valid_qutrit_passes_and_default_dt_rule():
    m = qutrit()
    results = model_checks(m)
    assert {r.name for r in results} == EXPECTED_CLOSED
    assert not failed(results)
    # dt = 1 / (||H0|| + b sum ||Hk|| + sum ||L||^2), documented rule
    w = np.linalg.norm(np.asarray(m.drift_matrix), 2)
    w += 2.0 * sum(np.linalg.norm(np.asarray(c), 2) for c in m.control_matrices)
    assert default_dt(m, 2.0) == pytest.approx(1.0 / w, rel=1e-6)
    mo = open_two_qubit()
    wj = sum(np.linalg.norm(np.asarray(j), 2) ** 2 for j in mo.jump_matrices)
    w0 = np.linalg.norm(np.asarray(mo.drift_matrix), 2) + sum(
        np.linalg.norm(np.asarray(c), 2) for c in mo.control_matrices
    )
    assert default_dt(mo) == pytest.approx(1.0 / (w0 + wj), rel=1e-6)
    with pytest.warns(UserWarning, match="all zero"):
        zero = HamiltonianModel(np.zeros((2, 2)), [np.zeros((2, 2))])
    assert default_dt(zero) == 1.0


@pytest.mark.parametrize(
    "model",
    [
        TwoLevel(),
        TwoLevel(gamma1=0.1, gamma_phi=0.05, detuning=0.2),
        QubitBattery(detuning=0.3, gamma1=0.2, n_th=0.4, gamma_phi=0.1, initial="thermal"),
        DickeBattery(n_cells=2),
        DickeBattery(n_cells=2, kappa=0.1),
        LambdaSystem(delta_one_photon=0.2),
        LambdaSystem(gamma=0.3),
    ],
    ids=lambda m: f"{type(m).__name__}-{'open' if m.is_open else 'closed'}",
)
def test_every_builtin_model_passes_the_generic_checks(model):
    results = model_checks(model)
    assert not failed(results), [(r.name, f"{r.error:.2e}") for r in results if not r.passed]
    assert not any(r.skipped for r in results)
    has_battery = type(model).__name__ in {"TwoLevel", "QubitBattery", "DickeBattery"}
    assert ("ergotropy_bounds" in {r.name for r in results}) == has_battery


def test_qutip_missing_is_skipped_not_raised(monkeypatch):
    monkeypatch.setitem(sys.modules, "qutip", None)  # makes `import qutip` raise ImportError
    results = by_name(model_checks(qutrit()))
    q = results["qutip_solver_cross_check"]
    assert q.skipped and not q.passed and "qutip is not installed" in q.detail
    assert not failed(results.values())
    # with a reference the skipped check carries the reference name
    ref = by_name(model_checks(qutrit(), horizon=2, reference=qutrit()))["qutip_cross_check"]
    assert ref.skipped and "qutip is not installed" in ref.detail


# ------------------------------------------------------------ what it cannot detect
def test_swapped_controls_and_wrong_signs_are_valid_physics_and_pass():
    """Documented limitation: the checks verify consistency, not that the physics is yours."""
    ctrls = [XY, np.kron(SX, ID2) / 2, np.kron(ID2, SX) / 2]
    swapped = two_qubit(controls=[ctrls[1], ctrls[0], ctrls[2]])
    flipped = two_qubit(controls=[-ctrls[0], ctrls[1], ctrls[2]])
    wrong_decay = open_two_qubit(jump_ops=[0.3 * np.kron(LOWER.T, ID2)])  # raises instead of lowers
    for m in (swapped, flipped, wrong_decay):
        assert not failed(model_checks(m))


# ------------------------------------------------------------ discriminating checks
def test_nonhermitian_control_that_bypasses_construction_fails_norm_and_hermiticity():
    m = two_qubit()
    bad = m.control_matrices.at[1].add(0.1j * np.kron(SX, ID2))  # tree_at skips validation
    broken = eqx.tree_at(lambda q: q.control_matrices, m, bad)
    res = by_name(model_checks(broken))
    assert {"hermiticity", "norm_conservation", "qutip_solver_cross_check"} <= failed(res.values())
    assert res["hermiticity"].error > 0.1 and "controls[1]" in res["hermiticity"].detail
    assert res["norm_conservation"].error > 1e-3


def test_too_loose_solver_fails_via_method_argument():
    loose = dq.method.Euler(dt=0.5)
    for m in (two_qubit(), open_two_qubit()):
        assert not failed(model_checks(m))  # same model, default solver
        res = by_name(model_checks(m, method=loose))
        assert "solver_convergence" in failed(res.values())
        assert res["solver_convergence"].error > 1e-2
        assert failed(res.values()) & {"norm_conservation", "density_matrix_valid"}


def test_wrong_jump_operator_is_caught_by_qutip_check_with_an_independent_reference():
    intended = open_two_qubit()
    wrong = two_qubit(jump_ops=[0.3 * np.kron(LOWER, ID2), 0.2 * np.kron(ID2, LOWER.T)])
    slightly_off = two_qubit(jump_ops=[0.3 * np.kron(LOWER, ID2), 0.21 * np.kron(ID2, LOWER)])
    assert not failed(model_checks(intended, reference=intended))
    for broken, min_err in ((wrong, 5e-2), (slightly_off, 1e-3)):
        res = by_name(model_checks(broken, reference=intended))
        assert failed(res.values()) == {"qutip_cross_check"}
        assert res["qutip_cross_check"].error > min_err
        assert "independent reference" in res["qutip_cross_check"].detail
        # without the reference the self-consistent cross-check cannot see the mistake
        assert not failed(model_checks(broken))


def test_reference_must_match_dimension():
    # a bad reference is reported as a failed cross-check, not raised
    res = by_name(model_checks(two_qubit(), horizon=2, reference=qutrit()))
    assert "qutip_cross_check" in failed(res.values())
    assert "reference has dimension" in res["qutip_cross_check"].detail


class WrongSubsystemBattery(HamiltonianModel):
    """Inconsistent battery: traces out the battery itself instead of the ancilla."""

    def battery_reduced(self, qstate):
        return dq.ptrace(qstate, 1), self.battery_hamiltonian()


def test_inconsistent_battery_hamiltonian_fails_battery_checks():
    kw = {
        "drift": np.kron(SZ, np.diag([0.0, 0.1, 0.25])),
        "controls": [np.kron(SX, np.eye(3))],
        "battery_hamiltonian": np.kron(np.diag([0.0, 1.0]), np.eye(3)),  # on the full space
        "dims": (2, 3),
    }
    good = HamiltonianModel.from_operators(**kw)
    assert not failed(model_checks(good))
    bad = WrongSubsystemBattery.from_operators(**kw)  # dims (2,3): reduced state is 3x3, H_B 6x6
    res = by_name(model_checks(bad))
    assert {"battery_consistency", "ergotropy_bounds"} <= failed(res.values())
    assert "do not conform" in res["battery_consistency"].detail

    # a non-Hermitian H_B that bypassed construction
    m = two_qubit()
    hb = m.battery_matrix.at[0, 1].set(0.3)
    res = by_name(model_checks(eqx.tree_at(lambda q: q.battery_matrix, m, hb)))
    assert {"hermiticity", "ergotropy_bounds"} <= failed(res.values())


def test_corrupted_initial_state_fails_initial_state_check():
    m = two_qubit()
    res = by_name(model_checks(eqx.tree_at(lambda q: q.initial_matrix, m, 2.0 * m.initial_matrix)))
    assert "initial_state_valid" in failed(res.values())
    assert res["initial_state_valid"].error == pytest.approx(3.0)


class OpenWithKetState(HamiltonianModel):
    def initial_state(self):
        return dq.basis(4, 1)  # open model must carry a density matrix


def test_open_model_with_ket_state_is_flagged():
    res = by_name(model_checks(OpenWithKetState.from_operators(**_open_kwargs()), horizon=2))
    assert "initial_state_valid" in failed(res.values())
    assert "density matrix" in res["initial_state_valid"].detail


def _open_kwargs():
    return {
        "drift": 0.1 * np.kron(SZ, ID2),
        "controls": [np.kron(SX, ID2)],
        "jump_ops": [0.2 * np.kron(LOWER, ID2)],
        "initial_state": np.array([0, 1, 0, 0]),
        "dims": (2, 2),
    }


# ------------------------------------------------------------ near-threshold (mutation) tests
def qubit(**overrides):
    """Driven qubit with a battery Hamiltonian: the cheapest model that exercises every check."""
    kw = {"drift": 0.3 * SZ, "controls": [SX / 2], "battery_hamiltonian": np.diag([0.0, 1.0])}
    kw.update(overrides)
    return HamiltonianModel.from_operators(**kw)


def open_qubit(**overrides):
    return qubit(**{"jump_ops": [0.2 * LOWER], "initial_state": [0, 1], **overrides})


def replaced(model, field, value):
    """Model with one leaf replaced past construction (``tree_at`` skips the validation)."""
    return eqx.tree_at(lambda q: getattr(q, field), model, jnp.asarray(value))


def ctx_of(model, horizon=3, dt=0.5):
    return _Context(model, horizon, dt, 1.0, 0, generic.DEFAULT_METHOD)


def static_check(model, name):
    """One check that needs no rollout (cheap: no compilation), called on a fresh context."""
    return getattr(generic, f"_check_{name}")(ctx_of(model))


def battery_check(model, name):
    """A battery check on a constant trajectory of the model's initial state (no ODE solve)."""
    psi = np.asarray(model.initial_state().to_jax(), dtype=np.complex128)
    return getattr(generic, f"_check_{name}")(ctx_of(model), np.stack([psi] * 3))


def n_level_open(n):
    a = np.diag(np.sqrt(np.arange(1.0, n)), 1).astype(complex)
    return HamiltonianModel(
        np.diag(np.linspace(0.0, 1.0, n)).astype(complex),
        [(a + a.conj().T) / 2],
        jump_ops=[0.1 * a],
        initial_state=np.eye(n)[1],
    )


@pytest.mark.parametrize("builder", [qubit, open_qubit], ids=["closed", "open"])
@pytest.mark.parametrize(
    ("method", "horizon", "dt_mult"),
    [
        (dq.method.Euler(dt=0.005), 4, 1.0),
        (dq.method.Tsit5(rtol=1e-2, atol=1e-2), 4, 3.0),  # loose adaptive solver, coarse dt
    ],
    ids=["euler", "loose-tsit5"],
)
def test_solver_error_between_tolerance_and_1e3_tolerance_fails(builder, method, horizon, dt_mult):
    m = builder()
    dt = dt_mult * default_dt(m)
    assert not failed(model_checks(m, horizon=horizon, dt=dt))  # the default solver is fine
    r = by_name(model_checks(m, horizon=horizon, dt=dt, method=method))["solver_convergence"]
    assert r.tolerance == pytest.approx(TOL_SOLVER)  # scale 1: horizon*dt below the reference span
    assert TOL_SOLVER < r.error < 1e3 * TOL_SOLVER  # a x1e3 loosened tolerance would pass
    assert not r.passed


@pytest.mark.parametrize("builder", [qubit, open_qubit], ids=["closed", "open"])
def test_solver_check_is_sensitive_to_each_reference_separately(monkeypatch, builder):
    m = builder()
    base = by_name(model_checks(m, horizon=3))["solver_convergence"]
    assert base.passed and "tight Tsit5" in base.detail and "vs Expm" in base.detail
    exact = _Context.exact_rollout
    with monkeypatch.context() as mp:  # a wrong Expm reference alone must fail the check
        mp.setattr(_Context, "exact_rollout", lambda self: exact(self) + 1e-3)
        r = by_name(model_checks(m, horizon=3))["solver_convergence"]
    assert not r.passed and 5e-4 < r.error < 2e-3
    with monkeypatch.context() as mp:  # a wrong tight reference alone must fail the check
        mp.setattr(generic, "_tight_method", lambda: dq.method.Euler(dt=0.005))
        r = by_name(model_checks(m, horizon=3))["solver_convergence"]
    assert not r.passed and r.error > 1e-4


def test_expm_reference_is_used_up_to_the_open_dimension_limit(monkeypatch):
    """Boundary at EXPM_MAX_DIM_OPEN (16 in the library; 3 here keeps the models tiny)."""
    assert generic.EXPM_MAX_DIM_OPEN == 16
    monkeypatch.setattr(generic, "EXPM_MAX_DIM_OPEN", 3)
    at = by_name(model_checks(n_level_open(3), horizon=1))["solver_convergence"]
    above = by_name(model_checks(n_level_open(4), horizon=1))["solver_convergence"]
    assert at.passed and "vs Expm" in at.detail
    assert above.passed and "Expm skipped" in above.detail and "vs Expm" not in above.detail


@pytest.mark.parametrize("where", ["drift", "control"])
def test_small_norm_loss_fails_norm_conservation(where):
    """A -0.01i |1><1| anti-Hermitian part loses norm: the check counts loss, not only gain."""
    p1 = np.diag([0.0, 1.0])
    if where == "drift":
        m = replaced(qubit(initial_state=[0, 1]), "drift_matrix", 0.3 * SZ - 0.01j * p1)
    else:
        m = replaced(qubit(initial_state=[0, 1]), "control_matrices", (SX / 2 - 0.01j * p1)[None])
    r = by_name(model_checks(m))["norm_conservation"]
    assert not r.passed and r.error > 1e-3
    if where == "drift":  # the drift cannot gain norm: every step has norm <= 1
        env = StateTransfer(model=m, target=dq.basis(2, 0), horizon=6, dt=default_dt(m))
        state, _ = env.rollout(jnp.zeros((6, 1)))
        assert float(jnp.linalg.norm(state.qstate.to_jax())) < 1.0 - 1e-3


def test_control_asymmetry_1e_4_fails_hermiticity_and_1e_7_passes():
    def with_asymmetry(eps):
        return replaced(qubit(), "control_matrices", (SX / 2 + eps * LOWER / 2)[None])

    ok = model_checks(with_asymmetry(1e-7))
    assert not failed(ok) and by_name(ok)["hermiticity"].error == pytest.approx(1e-7, rel=0.05)
    bad = static_check(with_asymmetry(1e-4), "hermiticity")
    assert not bad.passed and bad.error == pytest.approx(1e-4, rel=0.05)
    assert "controls[0]" in bad.detail


@pytest.mark.parametrize(
    ("rho", "key"),
    [
        (np.diag([1.001, -0.001]), "negativity"),  # trace 1, hermitian: the PSD term fires
        (np.diag([1.001, 0.0]), "tr-1"),  # PSD and hermitian: only the trace term fires
        (np.array([[0.5, 0.1 + 5e-4], [0.1 - 5e-4, 0.5]]), "hermiticity"),
    ],
    ids=["not-psd", "trace-1.001", "not-hermitian"],
)
def test_bypassed_open_initial_state_fails_each_term_near_threshold(rho, key):
    r = static_check(replaced(open_qubit(), "initial_matrix", rho), "initial_state")
    # 1e-3 is far above TOL_STATE = 1e-5 and below the 1e-2 a x1e3 loosening would accept
    assert not r.passed and 1e-5 < r.error < 1e-2, (r.error, r.detail)
    assert r.error == pytest.approx(1e-3, rel=0.01) and key in r.detail


def test_initial_state_tolerance_near_threshold_open_and_closed():
    ok = replaced(open_qubit(), "initial_matrix", np.diag([1.0 + 1e-6, 0.0]))
    assert static_check(ok, "initial_state").passed
    closed = qubit(initial_state=[0, 1])
    bad = replaced(closed, "initial_matrix", np.array([[0.0], [np.sqrt(1.001)]]))
    r = static_check(bad, "initial_state")
    assert not r.passed and r.error == pytest.approx(1e-3, rel=0.01)
    ok = replaced(closed, "initial_matrix", np.array([[0.0], [np.sqrt(1.000001)]]))
    assert static_check(ok, "initial_state").passed


class OneFlatControl(HamiltonianModel):
    """First control declared with flat dims (4,) while the drift is a (2, 2) system."""

    def controls(self):
        ctrls = super().controls()
        return [dq.asqarray(ctrls[0].to_jax(), dims=(4,)), *ctrls[1:]]


class CompositeControls(HamiltonianModel):
    """Flat drift (4,): controls with composite dims contradict nothing."""

    def controls(self):
        return [dq.asqarray(c.to_jax(), dims=(2, 2)) for c in super().controls()]


class SwappedDims(HamiltonianModel):
    def controls(self):
        return [dq.asqarray(c.to_jax(), dims=(3, 4)) for c in super().controls()]


class WrongShape(HamiltonianModel):
    def controls(self):
        return [dq.asqarray(np.eye(3))]


class NoControls(HamiltonianModel):
    def controls(self):
        return []


def structure_of(model):
    return static_check(model, "structure")


def test_structure_check_flags_inconsistent_dims_shapes_nan_and_missing_controls():
    r = structure_of(OneFlatControl.from_operators(**_kw2()))
    assert not r.passed and r.error == 1.0 and "controls[0] has dims (4,)" in r.detail
    ok = structure_of(CompositeControls.from_operators(**{**_kw2(), "dims": None}))
    assert ok.passed and ok.error == 0.0
    flat_dims = {"drift": np.diag(np.arange(12.0)), "controls": [np.eye(12)], "dims": (2, 6)}
    r = structure_of(SwappedDims.from_operators(**flat_dims))
    assert r.error == 1.0 and "dims (3, 4), inconsistent with the drift dims (2, 6)" in r.detail
    r = structure_of(WrongShape.from_operators(**_kw2()))
    assert not r.passed and "controls[0] has shape (3, 3), expected (4, 4)" in r.detail
    r = structure_of(NoControls.from_operators(**_kw2()))
    assert not r.passed and "no control operators" in r.detail
    nan_model = replaced(qubit(), "drift_matrix", np.array([[0.0, np.nan], [np.nan, 0.0]]))
    r = structure_of(nan_model)
    assert not r.passed and "drift contains NaN or inf" in r.detail


def _kw2():
    return {"drift": 0.1 * np.kron(SZ, ID2), "controls": [np.kron(SX, ID2)], "dims": (2, 2)}


def crafted_battery(rho, hb, scale=1.0):
    """Valid model whose ``battery_reduced`` returns a fixed (state, H_B) pair."""

    class Crafted(HamiltonianModel):
        def battery_reduced(self, qstate):
            return dq.asqarray(np.asarray(rho, dtype=complex)), dq.asqarray(scale * np.asarray(hb))

    return Crafted.from_operators(
        0.3 * SZ, [SX / 2], initial_state=[0, 1], battery_hamiltonian=np.diag([0.0, 1.0])
    )


HB = np.diag([0.0, 1.0])


def test_crafted_battery_pair_that_is_valid_passes():
    assert not failed(model_checks(crafted_battery(0.5 * np.eye(2), HB), horizon=3))


def _spread(hb):
    lam = jnp.linalg.eigvalsh(hb.to_jax())  # traced inside jax.vmap: no NumPy here
    return lam[-1] - lam[0]


@pytest.mark.parametrize("scale", [1.0, 1e3])
def test_ergotropy_lower_bound_violation_is_detected_at_any_scale_of_h_b(monkeypatch, scale):
    """A (buggy) ergotropy of -0.25 H_B widths violates 0 <= W while W <= E - E_min holds."""
    monkeypatch.setattr(generic, "ergotropy", lambda rho, hb: -0.25 * _spread(hb))
    m = crafted_battery(0.5 * np.eye(2), HB, scale)
    r = battery_check(m, "ergotropy")
    assert not r.passed and r.error == pytest.approx(0.25, rel=1e-6)  # relative to the H_B width


def test_ergotropy_above_energy_minus_ground_energy_is_detected_at_any_scale(monkeypatch):
    monkeypatch.setattr(generic, "ergotropy", lambda rho, hb: 5.0 * _spread(hb))
    r = battery_check(crafted_battery(0.5 * np.eye(2), HB, 1e3), "ergotropy")
    assert not r.passed and r.error > 1.0


def test_ergotropy_upper_bound_violation_is_detected():
    """Trace-one state with a negative eigenvalue: ergotropy 0 > E - E_min = -0.5."""
    r = battery_check(crafted_battery(np.diag([1.5, -0.5]), HB), "ergotropy")
    assert r.error == pytest.approx(0.5, rel=1e-6) and not r.passed


@pytest.mark.parametrize(
    ("rho", "expect"),
    [
        (np.diag([0.5005, 0.5005]), 1e-3),  # PSD, hermitian, trace 1.001: only the trace term
        (np.diag([1.001, -0.001]), 1e-3),  # trace 1, hermitian: only the PSD term
        (np.array([[0.5, 0.1 + 5e-4], [0.1 - 5e-4, 0.5]]), 1e-3),  # only the hermiticity term
    ],
    ids=["trace", "psd", "hermiticity"],
)
def test_battery_consistency_flags_each_invalid_reduced_state(rho, expect):
    r = battery_check(crafted_battery(rho, HB), "battery_consistency")
    assert not r.passed and r.error == pytest.approx(expect, rel=0.01)  # 20 x the tolerance


def test_ergotropy_tolerance_near_threshold():
    """Violation 1e-3 of the H_B width is above TOL_THERMO = 1e-5 and below a x1e3 loosening."""
    e = battery_check(crafted_battery(np.diag([1.001, -0.001]), HB), "ergotropy")
    assert not e.passed and 1e-5 < e.error < 1e-2
    ok = crafted_battery(np.diag([1.0 + 1e-7, -1e-7]), HB)
    assert battery_check(ok, "ergotropy").passed and battery_check(ok, "battery_consistency").passed


def test_ergotropy_check_exercises_mixed_states_not_only_the_rollout(monkeypatch):
    """A closed model only visits pure states; the random density matrices cover mixed ones."""
    real = generic.ergotropy

    def exact_on_pure_states_only(rho, hb):
        r = rho.todm().to_jax()
        return real(rho, hb) + 5.0 * (1.0 - jnp.real(jnp.trace(r @ r)))

    m = qubit(initial_state=[0, 1])
    assert battery_check(m, "ergotropy").passed
    monkeypatch.setattr(generic, "ergotropy", exact_on_pure_states_only)
    r = battery_check(m, "ergotropy")  # constant pure trajectory: only random states can fail
    assert not r.passed and r.error > 0.1 and "random density matrices" in r.detail


@pytest.mark.parametrize(
    ("make_kwargs", "bad", "good"),
    [
        (lambda v: {"initial_state": np.array([np.sqrt(v), 0.0])}, 1.0 + 1e-4, 1.0 + 1e-8),
        (lambda v: {"initial_state": np.diag([v, 0.0])}, 1.0 + 1e-4, 1.0 + 1e-8),
        (lambda v: {"initial_state": np.diag([1.0 + v, -v])}, 1e-4, 1e-8),  # negative eigenvalue
        (
            lambda v: {"initial_state": np.array([[0.5, 0.1 + v], [0.1 - v, 0.5]])},  # asymmetry 2v
            1e-4,
            1e-8,
        ),
    ],
    ids=["ket-norm", "dm-trace", "dm-negativity", "dm-hermiticity"],
)
def test_construction_state_tolerance_rejects_1e_4_and_accepts_1e_8(make_kwargs, bad, good):
    base = {"drift": np.diag([0.0, 1.0]), "controls": [SX]}
    with pytest.raises(ValueError, match=r"initial_state.*tolerance.*1e-06"):
        HamiltonianModel(**base, **make_kwargs(bad))
    m = HamiltonianModel(**base, **make_kwargs(good))
    a = np.asarray(m.initial_matrix)
    if a.shape[1] == 1:
        assert float(np.vdot(a, a).real) == pytest.approx(1.0, abs=1e-12)
    else:
        assert float(np.trace(a).real) == pytest.approx(1.0, abs=1e-12)


# ------------------------------------------------------------ long and coarse rollouts
def test_valid_models_pass_long_and_coarse_rollouts_with_scaled_tolerances():
    """Regression: TwoLevel() failed battery_consistency at horizon=400 (error 1.6e-5 vs 1e-5)."""
    two_level = TwoLevel()
    res = model_checks(two_level, horizon=400)
    assert not failed(res), [(r.name, r.error, r.tolerance) for r in res if not r.passed]
    scale = 400 / 12  # horizon * dt / (12 * default_dt) with dt = default_dt
    by = by_name(res)
    for name in ("norm_conservation", "solver_convergence", "qutip_solver_cross_check"):
        assert by[name].tolerance == pytest.approx(5e-5 * scale)
    assert by["battery_consistency"].tolerance == pytest.approx(TOL_CONSERVATION * scale)
    assert "tolerance 5e-05 x 33.3" in by["battery_consistency"].detail
    assert by["structure"].tolerance == 0.0 and by["hermiticity"].tolerance == 1e-6  # not scaled
    res = model_checks(two_qubit(), horizon=400)
    assert not failed(res), [(r.name, r.error, r.tolerance) for r in res if not r.passed]
    for model in (two_level, two_qubit(), open_two_qubit()):
        res = model_checks(model, horizon=60, dt=3 * default_dt(model))
        assert not failed(res), [(r.name, r.error, r.tolerance) for r in res if not r.passed]
        assert by_name(res)["solver_convergence"].tolerance == pytest.approx(5e-5 * 15)


def test_tolerances_do_not_scale_up_for_short_or_fine_rollouts():
    for kw in ({"horizon": 3}, {"horizon": 12, "dt": 0.1}):
        by = by_name(model_checks(qubit(), **kw))
        assert by["solver_convergence"].tolerance == pytest.approx(TOL_SOLVER)
        assert "integrated time" not in by["solver_convergence"].detail


# ------------------------------------------------------------ qutip check naming and the reference
def test_qutip_check_is_named_for_what_it_verifies():
    own = by_name(model_checks(qubit(), horizon=3))
    assert "qutip_cross_check" not in own
    d = own["qutip_solver_cross_check"].detail
    assert "does NOT verify" in d and "own matrices" in d and "reference=" in d
    ref = by_name(model_checks(qubit(), horizon=3, reference=qubit()))
    assert "qutip_solver_cross_check" not in ref and ref["qutip_cross_check"].passed
    assert "independent reference" in ref["qutip_cross_check"].detail


def test_reference_sensitivity_floor():
    """Caught only if rate * T * relative_error is above about the tolerance (see the guide)."""
    rate, horizon = 0.2, 12

    def decay(g):
        return qubit(jump_ops=[np.sqrt(g) * LOWER], initial_state=[0, 1])

    ref = decay(rate)
    span = horizon * default_dt(ref)  # episode length T
    for rel, caught in (
        (1e-2, True),
        (1e-5, False),
    ):  # rate * T * rel = 2.4e-2 and 2.4e-5 (x model)
        r = by_name(model_checks(decay(rate * (1 + rel)), horizon=horizon, reference=ref))
        r = r["qutip_cross_check"]
        assert (not r.passed) == caught, (rel, r.error, r.tolerance)
        assert r.error < rate * span * rel  # the state moves by at most gamma * T * relative error


# ------------------------------------------------------------ ergonomics
def test_construction_errors_format_norm_and_trace_with_3e():
    base = {"drift": np.diag([0.0, 1.0]), "controls": [SX]}
    with pytest.raises(ValueError, match="not normalised") as e:
        HamiltonianModel(**base, initial_state=np.array([np.sqrt(1 + 3e-6), 0.0]))
    assert "|<psi|psi> - 1| = 3.000e-06" in str(e.value) and "1.000003" in str(e.value)
    with pytest.raises(ValueError, match=r"trace 1\.000003") as e:
        HamiltonianModel(**base, initial_state=np.diag([1 + 3e-6, 0.0]))
    assert "|tr - 1| = 3.000e-06" in str(e.value)
    with pytest.raises(ValueError, match="not Hermitian") as e:
        HamiltonianModel(**base, initial_state=np.array([[0.5, 0.1 + 5e-6], [0.1 - 5e-6, 0.5]]))
    assert "1.000e-05" in str(e.value) and "tolerance 1e-06" in str(e.value)


@pytest.mark.parametrize("name", ["controls", "jump_ops"])
def test_nested_list_passed_as_a_single_operator_gets_a_hint(name):
    base = {"drift": np.diag([0.0, 1.0]), "controls": [SX]}
    for nested in ([[0, 1], [1, 0]], ((0.0, 1.0), (1.0, 0.0))):
        with pytest.raises(ValueError, match="wrap a single operator in a list"):
            HamiltonianModel(**{**base, name: nested})
    # a list holding the matrix, a list of nested lists and a 3-d array are all fine
    HamiltonianModel(**{**base, name: [[[0, 1], [1, 0]]]})
    HamiltonianModel(**{**base, name: [np.array([[0, 1], [1, 0]])]})
    HamiltonianModel(**{**base, name: np.array([[[0, 1], [1, 0]]])})


def test_all_zero_control_warns_and_is_reported_by_the_structure_check():
    base = {"drift": np.diag([0.0, 1.0])}
    with pytest.warns(UserWarning, match=r"controls\[1\] is all zero"):
        m = HamiltonianModel(**base, controls=[SX, np.zeros((2, 2))])
    assert "controls[1] is all zero" in structure_of(m).detail and structure_of(m).passed
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = HamiltonianModel(**base, controls=[SX, SZ])
    assert "all zero" not in structure_of(m).detail
    zeroed = replaced(m, "control_matrices", np.zeros((2, 2, 2)))  # bypassed
    assert "controls[0], controls[1] is all zero" in structure_of(zeroed).detail


def test_load_model_puts_cwd_first_and_restores_sys_path(tmp_path, monkeypatch):
    """A user file named like a stdlib module is found instead of the stdlib one."""
    name = next(
        n for n in ("colorsys", "sched", "mailcap", "cgitb", "pipes") if n not in sys.modules
    )
    assert importlib.util.find_spec(name) is not None  # a real stdlib module (or removed one)
    (tmp_path / f"{name}.py").write_text(MODULE)
    monkeypatch.chdir(tmp_path)
    before = list(sys.path)
    try:
        model = toolkit.load_model(f"{name}:MODEL")
        assert str(tmp_path) in model.__class__.__module__ or sys.modules[name].__file__.startswith(
            str(tmp_path)
        )
        assert isinstance(model, HamiltonianModel)
    finally:
        sys.modules.pop(name, None)
    assert sys.path == before
    # cwd already on sys.path (at the end): still loaded first, path unchanged afterwards
    monkeypatch.setattr(sys, "path", [*before, str(tmp_path)])
    snapshot = list(sys.path)
    try:
        assert isinstance(toolkit.load_model(f"{name}:build"), HamiltonianModel)
    finally:
        sys.modules.pop(name, None)
    assert sys.path == snapshot
    # a failing import still restores the path
    with pytest.raises(toolkit.InvalidModelError):
        toolkit.load_model("no_such_module_xyz:MODEL")
    assert sys.path == snapshot


# ------------------------------------------------------------ use with the environments
def test_iswap_state_transfer_with_exact_analytic_result():
    """H = g (XX + YY)/2 couples |01> and |10> with strength g: P(10) = sin^2(g t)."""
    model = HamiltonianModel(np.zeros((4, 4)), [XY], initial_state=np.array([0, 1, 0, 0]))
    target = dq.basis(4, 2)  # |10>
    horizon, dt = 10, 0.2
    env = StateTransfer(model=model, target=target, horizon=horizon, dt=dt, action_bound=2.0)
    total = horizon * dt
    # constant amplitude g over the episode: P(10) = sin^2(g T)
    for angle in (0.6, np.pi / 2):  # pi/2 is the iSWAP
        _, rewards = env.rollout(jnp.full((horizon, 1), angle / total))
        assert float(rewards.sum()) == pytest.approx(np.sin(angle) ** 2, abs=1e-5)


def test_driven_spin1_qutrit_matches_closed_form():
    """Spin-1 with H = u Jx from m=-1: P(m=+1) = sin^4(u t / 2) (Wigner d^1_{1,-1})."""
    s = 1 / np.sqrt(2)
    jx = np.array([[0, s, 0], [s, 0, s], [0, s, 0]], dtype=complex)
    model = HamiltonianModel(np.zeros((3, 3)), [jx], observables={"p2": np.diag([0.0, 0.0, 1.0])})
    env = StateTransfer(
        model=model,
        target=dq.basis(3, 2),
        horizon=6,
        dt=0.25,
        method=dq.method.Tsit5(rtol=1e-9, atol=1e-9),
    )
    u = 1.3
    state, _ = env.rollout(jnp.full((6, 1), u))
    p = float(dq.expect(model.observables()["p2"], state.qstate).real)
    assert p == pytest.approx(np.sin(u * 1.5 / 2) ** 4, abs=1e-6)


def test_battery_charging_on_custom_model_matches_closed_form():
    """Two independent resonant qubits driven together charge as 2 sin^2(u t / 2)."""
    drive = (np.kron(SX, ID2) + np.kron(ID2, SX)) / 2
    model = HamiltonianModel(np.zeros((4, 4)), [drive], battery_hamiltonian=HB2, dims=(2, 2))
    env = BatteryCharging(
        model=model, horizon=8, dt=0.25, action_bound=2.0, method=dq.method.Expm()
    )
    u = 1.1
    _, rewards = env.rollout(jnp.full((8, 1), u))
    assert float(rewards.sum()) == pytest.approx(2 * np.sin(u * 2.0 / 2) ** 2, abs=1e-6)
    _, ts = env.step(env.reset(jax.random.PRNGKey(0))[0], jnp.zeros(1))
    assert {"energy", "ergotropy", "power", "purity"} <= set(ts.info)
    with pytest.raises(NotImplementedError):
        BatteryCharging(model=qutrit()).reset(jax.random.PRNGKey(0))


def test_jit_vmap_and_grad_through_the_environment():
    model = open_two_qubit()
    env = BatteryCharging(model=model, horizon=4, dt=0.2)
    actions = 0.3 * jnp.ones((4, 3))
    ret = jax.jit(lambda a: env.rollout(a)[1].sum())(actions)
    grad = jax.grad(lambda a: env.rollout(a)[1].sum())(actions)
    assert jnp.isfinite(ret) and grad.shape == (4, 3) and bool(jnp.all(jnp.isfinite(grad)))
    assert float(jnp.abs(grad).max()) > 0
    # vmap over a batch of initial states / actions
    keys = jax.random.split(jax.random.PRNGKey(0), 3)
    states, _ = jax.vmap(env.reset)(keys)
    out, ts = jax.vmap(env.step)(states, jnp.stack([actions[0]] * 3))
    assert ts.reward.shape == (3,) and out.qstate.shape == (3, 4, 4)

    # gradient with respect to a model parameter (the drift matrix is a pytree leaf)
    def loss(h0):
        m = eqx.tree_at(lambda q: q.drift_matrix, model, h0)
        return BatteryCharging(model=m, horizon=3, dt=0.2).rollout(actions[:3])[1].sum().real

    g0 = jax.grad(loss)(model.drift_matrix)
    assert g0.shape == (4, 4) and bool(jnp.all(jnp.isfinite(g0)))


def test_gymnasium_wrapper_runs_a_custom_model():
    pytest.importorskip("gymnasium")
    from qenvs.wrappers import GymnasiumEnv

    env = GymnasiumEnv(BatteryCharging(model=open_two_qubit(), horizon=3, dt=0.2))
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape == (2 * 16 + 1,)
    for i in range(3):
        obs, _r, terminated, _t, info = env.step(env.action_space.sample())
        assert terminated == (i == 2)
    assert 0.0 <= float(info["ergotropy"]) <= float(info["energy"]) + 1e-6


def test_run_grape_works_on_a_task_built_from_a_custom_model(monkeypatch):
    pytest.importorskip("optax")
    import pathlib

    examples = str(pathlib.Path(__file__).resolve().parents[1] / "examples")
    monkeypatch.syspath_prepend(examples)
    import grape_jax

    model = HamiltonianModel(np.zeros((4, 4)), [XY], initial_state=np.array([0, 1, 0, 0]))

    def factory(**kw):
        return StateTransfer(model=model, target=dq.basis(4, 2), action_bound=2.0, **kw)

    monkeypatch.setitem(
        registry, "test/CustomISwap-v0", TaskSpec("test/CustomISwap-v0", factory, "iSWAP", "custom")
    )
    out = grape_jax.run_grape("test/CustomISwap-v0", steps=60, horizon=8, dt=0.3, lr=0.1)
    assert out["final_return"] > 0.95 and out["action_dim"] == 1
    assert qenvs.make("test/CustomISwap-v0", horizon=2, dt=0.1).horizon == 2


# ------------------------------------------------------------ toolkit + CLI
MODULE = textwrap.dedent(
    """
    import numpy as np
    from qenvs.models.custom import HamiltonianModel

    SX = np.array([[0, 1], [1, 0]], dtype=complex)
    LOWER = np.array([[0, 1], [0, 0]], dtype=complex)

    def build():
        return HamiltonianModel.from_operators(
            0.2 * np.diag([1.0, -1.0]), [SX / 2], jump_ops=[0.1 * LOWER],
            battery_hamiltonian=np.diag([0.0, 1.0]),
        )

    MODEL = build()
    WRONG = HamiltonianModel.from_operators(  # decay rate 50 percent too large
        0.2 * np.diag([1.0, -1.0]), [SX / 2], jump_ops=[0.15 * LOWER],
        battery_hamiltonian=np.diag([0.0, 1.0]),
    )
    NOT_A_MODEL = 42
    def make_bad():
        return HamiltonianModel.from_operators(np.diag([0.0, 1.0]), [np.array([[0, 1j], [1j, 0]])])
    """
)


@pytest.fixture
def user_module(tmp_path, monkeypatch):
    (tmp_path / "my_qenvs_model.py").write_text(MODULE)
    monkeypatch.syspath_prepend(str(tmp_path))
    yield "my_qenvs_model"
    sys.modules.pop("my_qenvs_model", None)


def _run(capsys, *argv):
    code = cli.main(list(argv))
    return code, capsys.readouterr().out


@pytest.mark.parametrize("attr", ["MODEL", "build"])
def test_cli_validate_model_round_trip(capsys, user_module, attr):
    code, out = _run(capsys, "validate", "--model", f"{user_module}:{attr}", "--json")
    assert code == 0
    res = json.loads(out)
    assert set(res) == {"all_passed", "n_skipped", "checks", "environment"}
    assert res["all_passed"]
    names = {c["name"] for c in res["checks"]}
    assert {
        "hermiticity",
        "density_matrix_valid",
        "solver_convergence",
        "ergotropy_bounds",
    } <= names
    assert all(c["model"] == "HamiltonianModel" for c in res["checks"])
    assert {"qenvs", "dynamiqs", "jax"} <= set(res["environment"])


def test_cli_validate_model_text_and_markdown(capsys, user_module):
    code, out = _run(capsys, "validate", "--model", f"{user_module}:MODEL")
    assert code == 0 and "PASS" in out and "FAIL" not in out
    assert "HamiltonianModel structure" in out  # model column is wide enough, not truncated
    assert "verifies the solver stack, not that the matrices are the physics" in out
    code, out = _run(capsys, "validate", "--model", f"{user_module}:MODEL", "--markdown")
    assert code == 0 and "## HamiltonianModel" in out and "| `hermiticity` |" in out
    assert "`qutip_solver_cross_check` integrates the model's own matrices" in out


def test_cli_validate_reference_option(capsys, user_module):
    spec = f"{user_module}:WRONG"
    code, out = _run(capsys, "validate", "--model", spec, "--json")  # self-consistent: passes
    assert code == 0 and "qutip_solver_cross_check" in out
    code, out = _run(capsys, "validate", "--model", spec, "--reference", f"{user_module}:MODEL")
    assert code == 1 and "FAIL" in out and "qutip_cross_check " in out
    assert "qutip_solver_cross_check" not in out and "solver stack, not" not in out  # no footer
    code, out = _run(
        capsys, "validate", "--model", spec, "--reference", f"{user_module}:MODEL", "--json"
    )
    res = json.loads(out)
    bad = {c["name"] for c in res["checks"] if not c["passed"]}
    assert code == 1 and bad == {"qutip_cross_check"}
    code, out = _run(  # the correct model against itself passes
        capsys, "validate", "--model", f"{user_module}:build", "--reference", f"{user_module}:MODEL"
    )
    assert code == 0 and "FAIL" not in out


def test_cli_validate_reference_errors(capsys, user_module):
    code, out = _run(capsys, "validate", "--reference", f"{user_module}:MODEL", "--json")
    assert code == 1 and json.loads(out)["error"]["code"] == "invalid_argument"
    code, out = _run(
        capsys,
        "validate",
        "--model",
        f"{user_module}:MODEL",
        "--reference",
        f"{user_module}:NOT_A_MODEL",
        "--json",
    )
    assert code == 1 and json.loads(out)["error"]["code"] == "invalid_model"
    with pytest.raises(toolkit.InvalidModelError, match="reference must be a qenvs Model"):
        toolkit.validate_model(qubit(), reference=object())


@pytest.mark.parametrize(
    "spec",
    [
        "no_such_module_xyz:MODEL",
        "{mod}:missing_attr",
        "{mod}:NOT_A_MODEL",
        "{mod}:make_bad",  # factory raises ValueError (non-Hermitian control)
        "just_a_module_name",
        "{mod}:",
    ],
)
def test_cli_validate_model_errors_are_invalid_model(capsys, user_module, spec):
    code, out = _run(capsys, "validate", "--model", spec.format(mod=user_module), "--json")
    assert code == 1
    err = json.loads(out)["error"]
    assert err["code"] == "invalid_model" and err["message"]


def test_cli_validate_model_error_text_mode(capsys, user_module):
    code = cli.main(["validate", "--model", f"{user_module}:NOT_A_MODEL"])
    assert code == 1
    err = capsys.readouterr().err
    assert "error [invalid_model]" in err and "not a qenvs Model" in err


def test_cli_validate_model_failing_model_exits_1(capsys, tmp_path, monkeypatch):
    (tmp_path / "broken_qenvs_model.py").write_text(
        MODULE
        + textwrap.dedent(
            """
            import equinox as eqx
            BROKEN = eqx.tree_at(
                lambda m: m.control_matrices, MODEL, MODEL.control_matrices + 0.2j * np.eye(2)
            )
            """
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    code, out = _run(capsys, "validate", "--model", "broken_qenvs_model:BROKEN", "--json")
    sys.modules.pop("broken_qenvs_model", None)
    assert code == 1
    res = json.loads(out)
    assert not res["all_passed"]
    assert "hermiticity" in {c["name"] for c in res["checks"] if not c["passed"]}


def test_load_model_uses_cwd_and_validate_model_kwargs(tmp_path, monkeypatch):
    (tmp_path / "cwd_qenvs_model.py").write_text(MODULE)
    monkeypatch.chdir(tmp_path)
    model = toolkit.load_model("cwd_qenvs_model:build")
    sys.modules.pop("cwd_qenvs_model", None)
    assert str(tmp_path) not in sys.path  # cwd is only importable while loading
    out = toolkit.validate_model(model, horizon=4, dt=0.3, seed=1)
    assert out["all_passed"]
    with pytest.raises(toolkit.InvalidModelError, match="expected a qenvs Model"):
        toolkit.validate_model(object())


# ------------------------------------------------------------ float32
def test_checks_hold_in_float32(tmp_path):
    """Run the generic checks in a fresh interpreter without x64 (library default)."""
    script = textwrap.dedent(
        """
        import json
        import numpy as np
        import jax
        assert not jax.config.jax_enable_x64
        from qenvs.models import DickeBattery, QubitBattery
        from qenvs.models.custom import HamiltonianModel
        from qenvs.validation.generic import model_checks

        sx = np.array([[0, 1], [1, 0]], dtype=complex)
        sy = np.array([[0, -1j], [1j, 0]])
        sz = np.diag([1.0, -1.0]).astype(complex)
        i2 = np.eye(2)
        lower = np.array([[0, 1], [0, 0]], dtype=complex)
        xy = 0.5 * (np.kron(sx, sx) + np.kron(sy, sy))
        m = HamiltonianModel(
            0.1 * np.kron(sz, i2), [xy, np.kron(sx, i2) / 2],
            jump_ops=[0.3 * np.kron(lower, i2)], initial_state=np.array([0, 1, 0, 0]),
            battery_hamiltonian=np.diag([0.0, 1.0, 1.0, 2.0]), dims=(2, 2),
        )
        closed = HamiltonianModel(0.1 * np.kron(sz, i2), [xy], initial_state=np.array([0, 1, 0, 0]))
        from qenvs.models import TwoLevel
        from qenvs.validation.generic import default_dt

        rng = np.random.default_rng(7)

        def random_model(n):
            def herm():
                a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
                return 0.5 * (a + a.conj().T)
            return HamiltonianModel(
                herm(), [herm(), herm()], battery_hamiltonian=herm(),
                initial_state=np.eye(n)[0],
            )

        two_level = TwoLevel()
        cases = {
            "custom_open": (m, {}), "custom_closed": (closed, {}),
            "QubitBattery": (QubitBattery(gamma1=0.2, n_th=0.3), {}),
            "Dicke": (DickeBattery(n_cells=2), {}),
            # regression: long and coarse rollouts of valid models (tolerances scale with time)
            "TwoLevel_h400": (two_level, {"horizon": 400}),
            "closed_h400": (closed, {"horizon": 400}),
            "TwoLevel_dt3": (two_level, {"horizon": 60, "dt": 3 * default_dt(two_level)}),
        }
        for k in range(1):
            rm = random_model(3)
            cases[f"random{k}_dt3"] = (rm, {"horizon": 60, "dt": 3 * default_dt(rm)})
        out = {}
        for name, (model, kw) in cases.items():
            out[name] = [(r.name, r.passed or r.skipped, r.error, r.tolerance)
                         for r in model_checks(model, **kw)]
        print(json.dumps(out))
        """
    )
    path = tmp_path / "f32.py"
    # `-I` ignores PYTHONPATH: make the child import the same qenvs as this process
    # (matters when testing a source checkout with PYTHONPATH=src instead of an install).
    import qenvs

    src_dir = pathlib.Path(qenvs.__file__).resolve().parents[1]
    path.write_text(f"import sys\nsys.path.insert(0, {str(src_dir)!r})\n" + script)
    env = {"JAX_PLATFORMS": "cpu", "PATH": "/usr/bin:/bin", "HOME": str(tmp_path)}
    proc = subprocess.run(
        [sys.executable, "-I", str(path)],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    for name, rows in out.items():
        bad = [r for r in rows if not r[1]]
        assert not bad, (name, bad)
