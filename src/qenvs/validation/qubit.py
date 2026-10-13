"""Validation checks for the driven two-level system (closed and open)."""

from __future__ import annotations

from collections.abc import Callable

import dynamiqs as dq
import jax
import jax.numpy as jnp
import numpy as np

from qenvs.models.two_level import TwoLevel
from qenvs.observables import ergotropy
from qenvs.validation.core import CheckResult
from qenvs.validation.core import result as _result


# --------------------------------------------------------------------- checks
def check_rabi_oscillation() -> CheckResult:
    """Closed qubit, constant drive u: P1(t) = sin^2(u t / 2)."""
    u = 1.3
    m = TwoLevel()
    ts = jnp.linspace(0.0, 2 * jnp.pi / u, 17)
    res = dq.sesolve(m.hamiltonian(jnp.array([u])), m.initial_state(), ts, progress_meter=False)
    p1 = np.asarray(dq.expect(m.observables()["p1"], res.states).real)
    err = np.max(np.abs(p1 - np.sin(u * np.asarray(ts) / 2) ** 2))
    return _result(
        "rabi_oscillation", "analytic", err, 1e-6, "max |P1 - sin^2(ut/2)| over one period"
    )


def check_pi_pulse_inversion() -> CheckResult:
    """Closed qubit: pi-pulse maps |0> to |1> with fidelity 1."""
    m = TwoLevel()
    psi = m.evolve(m.initial_state(), m.hamiltonian(jnp.array([1.0])), jnp.pi)
    err = 1.0 - float(dq.fidelity(psi, dq.basis(2, 1)))
    return _result("pi_pulse_inversion", "analytic", err, 1e-6, "1 - F(|1>) after u*t = pi")


def check_t1_decay() -> CheckResult:
    """Open qubit, no drive: P1(t) = exp(-gamma1 t)."""
    g = 0.3
    m = TwoLevel(gamma1=g, initial="excited")
    ts = jnp.linspace(0.0, 5.0, 11)
    res = dq.mesolve(m.drift(), m.jump_ops(), m.initial_state(), ts, progress_meter=False)
    p1 = np.asarray(dq.expect(m.observables()["p1"], res.states).real)
    err = np.max(np.abs(p1 - np.exp(-g * np.asarray(ts))))
    return _result("t1_exponential_decay", "open_system", err, 1e-6, "max |P1 - exp(-gamma1 t)|")


def check_tphi_dephasing() -> CheckResult:
    """Open qubit from |+>, pure dephasing: <sigma_x>(t) = exp(-gamma_phi t)."""
    g = 0.4
    m = TwoLevel(gamma_phi=g)
    plus = (dq.basis(2, 0) + dq.basis(2, 1)).unit().todm()
    ts = jnp.linspace(0.0, 3.0, 7)
    res = dq.mesolve(m.drift(), m.jump_ops(), plus, ts, progress_meter=False)
    sx = np.asarray(dq.expect(dq.sigmax(), res.states).real)
    err = np.max(np.abs(sx - np.exp(-g * np.asarray(ts))))
    return _result(
        "tphi_coherence_decay", "open_system", err, 1e-6, "max |<sx> - exp(-gamma_phi t)|"
    )


def check_ergotropy() -> CheckResult:
    """Qubit ergotropy: |1> -> omega0, |0> and I/2 -> 0, |+> -> omega0/2."""
    w = 1.7
    hb = TwoLevel(omega0=w).battery_hamiltonian()
    plus = (dq.basis(2, 0) + dq.basis(2, 1)).unit()
    vals = [
        float(ergotropy(dq.basis(2, 1), hb)) - w,
        float(ergotropy(dq.basis(2, 0), hb)),
        float(ergotropy(0.5 * dq.eye(2), hb)),
        float(ergotropy(plus, hb)) - w / 2,
    ]
    err = max(abs(v) for v in vals)
    return _result(
        "ergotropy_qubit", "thermodynamics", err, 1e-6, "max deviation over 4 reference states"
    )


def check_energy_conservation_closed() -> CheckResult:
    """Closed qubit, static H (detuned, no drive): <H> is conserved."""
    m = TwoLevel(detuning=0.8)
    plus = (dq.basis(2, 0) + dq.basis(2, 1)).unit()
    ts = jnp.linspace(0.0, 20.0, 41)
    res = dq.sesolve(m.drift(), plus, ts, progress_meter=False)
    e = np.asarray(dq.expect(m.drift(), res.states).real)
    err = np.max(np.abs(e - e[0]))
    return _result(
        "energy_conservation_closed", "conservation", err, 1e-6, "max |<H>(t) - <H>(0)|, static H"
    )


def check_density_matrix_constraints() -> CheckResult:
    """Open qubit under random piecewise-constant drive: rho stays Hermitian, trace-1, PSD."""
    from qenvs.registry import make

    env = make("qubit/StateTransfer-v0", horizon=15, dt=0.3, gamma1=0.2, gamma_phi=0.1)
    key = jax.random.PRNGKey(7)
    actions = jax.random.uniform(key, (15, env.action_dim), minval=-2.0, maxval=2.0)
    state, _ = env.rollout(actions)
    rho = np.asarray(state.qstate.to_jax())
    herm = np.max(np.abs(rho - rho.conj().T))
    tr = abs(np.trace(rho).real - 1.0)
    neg = max(0.0, -float(np.min(np.linalg.eigvalsh(rho))))
    err = max(herm, tr, neg)
    return _result(
        "density_matrix_constraints",
        "constraints",
        err,
        1e-6,
        f"hermiticity={herm:.2e}, |tr-1|={tr:.2e}, negativity={neg:.2e}",
    )


def check_solver_convergence() -> CheckResult:
    """Adaptive Tsit5 vs exact matrix exponential on the open pi-pulse rollout."""
    from qenvs.registry import make

    kw = {"horizon": 10, "dt": 0.2, "gamma1": 0.1}
    u = float(jnp.pi / (10 * 0.2))
    actions = jnp.full((10, 1), u)
    env_a = make("qubit/StateTransfer-v0", method=dq.method.Tsit5(), **kw)
    env_b = make("qubit/StateTransfer-v0", method=dq.method.Expm(), **kw)
    sa, _ = env_a.rollout(actions)
    sb, _ = env_b.rollout(actions)
    err = np.max(np.abs(np.asarray(sa.qstate.to_jax()) - np.asarray(sb.qstate.to_jax())))
    return _result(
        "solver_convergence", "numerics", err, 1e-6, "max |rho_Tsit5 - rho_Expm| at episode end"
    )


def check_env_pi_pulse_return() -> CheckResult:
    """Functional env: constant pi-pulse action sequence returns F_final - F_initial = 1."""
    from qenvs.registry import make

    env = make("qubit/StateTransfer-v0", horizon=10, dt=0.2)
    actions = jnp.full((10, env.action_dim), jnp.pi / (10 * 0.2))
    _, rewards = env.rollout(actions)
    err = abs(float(rewards.sum()) - 1.0)
    return _result(
        "env_pi_pulse_return", "control", err, 1e-6, "|sum(rewards) - 1| for the pi-pulse"
    )


CHECKS: tuple[Callable[[], CheckResult], ...] = (
    check_rabi_oscillation,
    check_pi_pulse_inversion,
    check_t1_decay,
    check_tphi_dephasing,
    check_ergotropy,
    check_energy_conservation_closed,
    check_density_matrix_constraints,
    check_solver_convergence,
    check_env_pi_pulse_return,
)
