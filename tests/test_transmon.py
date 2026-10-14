"""Transmon model (hardware units: ns, rad/ns, GHz, us), pulse helpers and its physics checks."""

import math
import os
import subprocess
import sys

import dynamiqs as dq
import jax.numpy as jnp
import numpy as np
import pytest

from qenvs.models.transmon import (
    Transmon,
    drag_beta,
    gaussian_amplitude,
    gaussian_drag,
    ghz_to_rad_per_ns,
)
from qenvs.validation import transmon as validation


# ------------------------------------------------------------------ physics checks
@pytest.mark.parametrize("check", validation.CHECKS, ids=lambda c: c.__name__)
def test_physics_check_passes(check):
    r = check()
    assert r.model == "Transmon"
    assert not r.skipped, r.detail
    assert r.passed, f"{r.name}: error={r.error:.3e} > tol={r.tolerance:.0e} ({r.detail})"


def test_new_checks_reject_their_mutants(monkeypatch):
    """The SX and info-orientation checks fail on an adjoint SX and on a transposed map."""
    import qenvs.gates as gates
    from qenvs.envs.gate import GateControl

    assert validation.check_sx_orientation().passed
    assert validation.check_info_observables_orientation().passed
    sx = gates.TARGETS["SX"]
    adj = tuple(tuple(np.conj(sx[j][i]) for j in range(2)) for i in range(2))
    monkeypatch.setitem(gates.TARGETS, "SX", adj)
    assert not validation.check_sx_orientation().passed
    monkeypatch.setitem(gates.TARGETS, "SX", sx)

    def transposed(self, qstate):
        m = qstate.to_jax()
        psi0 = self.model.initial_state()
        if self.model.is_open:
            return dq.asqarray(gates.apply_superop(m.T, psi0.todm().to_jax()))
        return dq.asqarray(m.T @ psi0.to_jax())

    monkeypatch.setattr(GateControl, "evolved_state", transposed)
    assert not validation.check_info_observables_orientation().passed


def test_physics_checks_pass_in_float32():
    """The conftest enables x64; the same tolerances must hold in single precision."""
    code = (
        "import jax; jax.config.update('jax_enable_x64', False)\n"
        "from qenvs.validation import transmon as v\n"
        "bad = [r for r in (c() for c in v.CHECKS) if not (r.passed or r.skipped)]\n"
        "assert not bad, [(r.name, r.error, r.tolerance) for r in bad]\n"
        "import jax.numpy as jnp; assert jnp.asarray(1.0).dtype == jnp.float32\n"
    )
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "JAX_ENABLE_X64": "0"}
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120
    )
    assert out.returncode == 0, out.stderr[-2000:]


# ------------------------------------------------------------------ model API
def test_units_and_rates():
    m = Transmon(anharmonicity_ghz=-0.3, detuning_ghz=0.01, t1_us=100.0, t2_us=80.0)
    assert m.alpha == pytest.approx(2 * math.pi * -0.3)
    assert m.delta == pytest.approx(2 * math.pi * 0.01)
    assert m.gamma1 == pytest.approx(1 / 100_000.0)  # 1/ns
    assert m.gamma_phi == pytest.approx(1 / 80_000.0 - 0.5 / 100_000.0)
    assert ghz_to_rad_per_ns(0.05) == pytest.approx(0.1 * math.pi)


def test_hamiltonian_matrices():
    m = Transmon(levels=3, anharmonicity_ghz=-0.25, detuning_ghz=0.02)
    h0 = np.asarray(m.drift().to_jax())
    a = np.diag(np.sqrt([1.0, 2.0]), 1)
    n = np.diag([0.0, 1.0, 2.0])
    expected = 2 * np.pi * 0.02 * n + 0.5 * 2 * np.pi * -0.25 * n @ (n - np.eye(3))
    assert np.allclose(h0, expected)
    assert h0[2, 2].real == pytest.approx(2 * 2 * np.pi * 0.02 + 2 * np.pi * -0.25)
    h_i, h_q = (np.asarray(c.to_jax()) for c in m.controls())
    assert np.allclose(h_i, 0.5 * (a + a.T))
    assert np.allclose(h_q, 0.5j * (a.T - a))
    assert np.allclose(h_i, h_i.conj().T) and np.allclose(h_q, h_q.conj().T)
    assert m.n_controls == 2 and m.dim == 3


def test_jump_operators():
    a = np.diag(np.sqrt([1.0, 2.0]), 1)
    n = np.diag([0.0, 1.0, 2.0])
    m = Transmon(t1_us=50.0, t2_us=40.0)
    relax, deph = (np.asarray(op.to_jax()) for op in m.jump_ops())
    assert np.allclose(relax, math.sqrt(1 / 50_000.0) * a)
    g_phi = 1 / 40_000.0 - 0.5 / 50_000.0
    assert np.allclose(deph, math.sqrt(2 * g_phi) * n)
    # T1 only: relaxation only (T2 = 2 T1); T2 = 2 T1: no pure dephasing
    assert len(Transmon(t1_us=50.0).jump_ops()) == 1
    assert len(Transmon(t1_us=50.0, t2_us=100.0).jump_ops()) == 1
    # T2 only: pure dephasing at 1/T_phi = 1/T2
    (only,) = Transmon(t2_us=20.0).jump_ops()
    assert np.allclose(np.asarray(only.to_jax()), math.sqrt(2 / 20_000.0) * n)
    assert not Transmon().is_open and Transmon(t2_us=20.0).is_open


def test_states_and_observables():
    m = Transmon(levels=4)
    assert m.initial_state().isket()
    assert Transmon(t1_us=10.0).initial_state().isdm()
    obs = m.observables()
    assert set(obs) == {"p0", "p1", "p2", "p3", "leakage"}
    psi = (dq.basis(4, 1) + dq.basis(4, 2) + dq.basis(4, 3)).unit()
    assert float(dq.expect(obs["leakage"], psi).real) == pytest.approx(2 / 3)
    assert float(dq.expect(obs["p1"], psi).real) == pytest.approx(1 / 3)
    m1 = Transmon(initial="1")
    assert float(dq.expect(m1.observables()["p1"], m1.initial_state()).real) == 1.0


@pytest.mark.parametrize(
    "kw",
    [
        {"levels": 1},
        {"t1_us": 0.0},
        {"t2_us": -1.0},
        {"t1_us": 10.0, "t2_us": 20.5},  # T2 > 2 T1
        {"initial": "3"},
        {"initial": "excited"},
    ],
)
def test_invalid_parameters_rejected(kw):
    with pytest.raises(ValueError):
        Transmon(**kw)


# ------------------------------------------------------------------ pulse helpers
def test_gaussian_drag_shape_area_and_derivative():
    horizon, dt, sigma, beta = 200, 0.1, 5.0, 0.3
    amp = gaussian_amplitude(math.pi, horizon, dt, sigma)
    p = np.asarray(gaussian_drag(horizon, dt, amp, sigma, beta))
    assert p.shape == (horizon, 2)
    assert np.sum(p[:, 0]) * dt == pytest.approx(math.pi, rel=1e-9)
    assert p[:, 0].max() == pytest.approx(amp, rel=1e-3)  # unit-peak lifted Gaussian
    assert p[0, 0] < 1e-2 * amp and p[-1, 0] < 1e-2 * amp  # vanishes at the edges
    assert np.allclose(p[:, 0], p[::-1, 0])  # symmetric I
    assert np.allclose(p[:, 1], -p[::-1, 1])  # antisymmetric Q
    # Q = beta dI/dt: compare with central differences of I
    fd = (p[2:, 0] - p[:-2, 0]) / (2 * dt)
    assert np.max(np.abs(p[1:-1, 1] - beta * fd)) < 1e-3 * np.max(np.abs(beta * fd))
    assert np.allclose(np.asarray(gaussian_drag(10, 1.0, 0.1, 2.0))[:, 1], 0.0)


def test_drag_beta_sign_and_values():
    alpha = 2 * math.pi * -0.34
    assert drag_beta(-0.34) == pytest.approx(-1 / alpha)
    assert drag_beta(-0.34, "phase") == pytest.approx(-0.5 / alpha)
    assert drag_beta(-0.34) > 0  # transmon: alpha < 0
    with pytest.raises(ValueError):
        drag_beta(-0.34, "both")


def test_drag_sign_matters():
    """Flipping the sign of beta must increase leakage over beta = 0 (sign convention guard)."""
    from qenvs.envs.gate import GateControl
    from qenvs.gates import gate

    env = GateControl(model=Transmon(), target=gate("X"), horizon=48, dt=0.25, action_bound=10.0)
    amp = gaussian_amplitude(math.pi, 48, 0.25, 3.0)
    leak = {}
    for b in (0.0, drag_beta(-0.34), -drag_beta(-0.34)):
        s, _ = env.rollout(gaussian_drag(48, 0.25, amp, 3.0, b))
        leak[b] = float(env.leakage(s.qstate))
    b = drag_beta(-0.34)
    assert leak[b] < leak[0.0] / 100 < leak[-b] / 100


def test_pi_pulse_population_with_jnp_inputs():
    m = Transmon(levels=2)
    w = float(ghz_to_rad_per_ns(0.05))
    psi = m.evolve(m.initial_state(), m.hamiltonian(jnp.array([w, 0.0])), 10.0)
    assert float(dq.expect(m.observables()["p1"], psi).real) == pytest.approx(1.0, abs=1e-6)
