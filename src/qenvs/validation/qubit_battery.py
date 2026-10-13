"""Validation checks for the open two-level quantum battery (:class:`QubitBattery`).

The QuTiP cross-check needs ``qutip`` (``qenvs[dev]`` extra); without it that single
check reports ``error = nan`` (FAIL, detail says why) instead of raising.
"""

from __future__ import annotations

from collections.abc import Callable

import dynamiqs as dq
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from qenvs.envs.base import EnvState
from qenvs.envs.battery import BatteryCharging
from qenvs.models.qubit_battery import QubitBattery
from qenvs.models.two_level import TwoLevel
from qenvs.observables import energy, ergotropy
from qenvs.validation.core import CheckResult, skipped
from qenvs.validation.core import result as _result

MODEL = "QubitBattery"
# Closed-system analytic checks integrate with a tight tolerance so the reported error
# reflects the model, not the integrator; ``check_solver_convergence`` validates the
# default Tsit5 settings separately.
TIGHT = dq.method.Tsit5(rtol=1e-8, atol=1e-8)


def _res(name: str, category: str, error: float, tol: float, detail: str) -> CheckResult:
    return _result(name, category, error, tol, detail, model=MODEL)


def _random_actions(key: int, horizon: int, bound: float) -> Array:
    return jax.random.uniform(jax.random.PRNGKey(key), (horizon, 1), minval=-bound, maxval=bound)


def _scan_info(env: BatteryCharging, actions: Array) -> dict[str, np.ndarray]:
    """Rollout ``actions`` and return ``info`` time series including the initial step."""
    state0, ts0 = env.reset(jax.random.PRNGKey(0))

    def body(state: EnvState, a: Array) -> tuple[EnvState, dict[str, Array]]:
        state, ts = env.step(state, a)
        return state, ts.info

    _, infos = jax.lax.scan(body, state0, actions)
    return {
        k: np.concatenate([np.atleast_1d(np.asarray(ts0.info[k])), np.asarray(v)])
        for k, v in infos.items()
    }


# --------------------------------------------------------------------- checks
def check_thermal_steady_state() -> CheckResult:
    """Zero drive from |0>, thermal bath: P1(t -> inf) = n_th / (2 n_th + 1)."""
    n_th = 0.6
    m = QubitBattery(gamma1=0.8, n_th=n_th, gamma_phi=0.3)
    # relaxation rate gamma1 (2 n_th + 1) = 1.76 -> exp(-52.8) transient at t = 30
    rho = m.evolve(m.initial_state(), m.hamiltonian(jnp.zeros(1)), 30.0)
    p1 = float(dq.expect(m.observables()["p1"], rho).real)
    err = p1 - n_th / (2 * n_th + 1)
    return _res(
        "thermal_steady_state", "open_system", err, 1e-5, "|P1(30) - n_th/(2n_th+1)|, zero drive"
    )


def check_thermal_state_stationary() -> CheckResult:
    """The Gibbs initial state is a fixed point of the thermal master equation (zero drive)."""
    m = QubitBattery(gamma1=0.7, n_th=0.4, gamma_phi=0.2, initial="thermal")
    ts = jnp.linspace(0.0, 10.0, 21)
    res = dq.mesolve(m.drift(), m.jump_ops(), m.initial_state(), ts, progress_meter=False)
    rho = np.asarray(res.states.to_jax())
    err = np.max(np.abs(rho - rho[0]))
    return _res(
        "thermal_state_stationary", "open_system", err, 1e-6, "max |rho(t) - rho(0)|, zero drive"
    )


def check_pi_pulse_charging() -> CheckResult:
    """Closed resonant pi-pulse from |0>: stored energy and ergotropy both equal omega0."""
    w, u = 1.3, 1.1
    m = QubitBattery(omega0=w)
    psi = m.evolve(m.initial_state(), m.hamiltonian(jnp.array([u])), jnp.pi / u, method=TIGHT)
    hb = m.battery_hamiltonian()
    e_err = float(energy(psi, hb)) - w
    w_err = float(ergotropy(psi, hb)) - w
    err = max(abs(e_err), abs(w_err))
    return _res(
        "pi_pulse_charging",
        "analytic",
        err,
        1e-6,
        f"|E - omega0|={abs(e_err):.2e}, |ergotropy - omega0|={abs(w_err):.2e} after u*t = pi",
    )


def check_ergotropy_le_energy() -> CheckResult:
    """Random open rollout: 0 <= ergotropy <= energy at every step."""
    m = QubitBattery(detuning=0.4, gamma1=0.3, n_th=0.5, gamma_phi=0.1)
    env = BatteryCharging(model=m, horizon=20, dt=0.25, action_bound=2.0)
    info = _scan_info(env, _random_actions(3, 20, 2.0))
    e, w = info["energy"], info["ergotropy"]
    err = max(np.max(w - e), np.max(-w), np.max(-e), 0.0)
    return _res(
        "ergotropy_le_energy",
        "thermodynamics",
        err,
        1e-7,
        f"min(energy - ergotropy)={np.min((e - w)[1:]):.2e}, min(ergotropy)={np.min(w[1:]):.2e} "
        "over 20 random open steps",
    )


def check_charging_power_bound() -> CheckResult:
    """Closed resonant drive u: |dE/dt| <= omega0 u / 2, saturated at u t = pi / 2.

    ``E(t) = omega0 sin^2(u t / 2)`` so ``dE/dt = (omega0 u / 2) sin(u t)``. The
    derivative is estimated with 4th-order central differences on a fine grid; the
    reported error is ``|max_t |dE/dt| - omega0 u / 2|`` (bound violated or not attained).
    """
    w, u = 0.9, 1.4
    m = QubitBattery(omega0=w)
    n = 201
    ts = jnp.linspace(0.0, 2 * jnp.pi / u, n)
    h = float(ts[1] - ts[0])
    res = dq.sesolve(
        m.hamiltonian(jnp.array([u])), m.initial_state(), ts, method=TIGHT, progress_meter=False
    )
    e = np.asarray(dq.expect(m.battery_hamiltonian(), res.states).real)
    de = (-e[4:] + 8 * e[3:-1] - 8 * e[1:-3] + e[:-4]) / (12 * h)
    bound = w * u / 2
    peak = float(np.max(np.abs(de)))
    err = peak - bound
    return _res(
        "charging_power_bound",
        "thermodynamics",
        err,
        1e-4,
        f"max|dE/dt|={peak:.6f} vs omega0*u/2={bound:.6f} (FD on {n} points)",
    )


def check_env_return_equals_gain() -> CheckResult:
    """BatteryCharging return equals Q(final) - Q(initial) for Q = energy and ergotropy."""
    m = QubitBattery(detuning=0.2, gamma1=0.25, n_th=0.4, gamma_phi=0.1)
    actions = _random_actions(5, 12, 2.0)
    errs = {}
    for reward, q in (("energy", energy), ("ergotropy", ergotropy)):
        env = BatteryCharging(model=m, horizon=12, dt=0.2, action_bound=2.0, reward=reward)
        state, rewards = env.rollout(actions)
        hb = m.battery_hamiltonian()
        gain = float(q(state.qstate, hb)) - float(q(m.initial_state(), hb))
        errs[reward] = abs(float(rewards.sum()) - gain)
    return _res(
        "env_return_equals_gain",
        "control",
        max(errs.values()),
        1e-6,
        f"|sum(rewards) - (Q_final - Q_initial)|: energy={errs['energy']:.2e}, "
        f"ergotropy={errs['ergotropy']:.2e}",
    )


def check_qutip_cross_check() -> CheckResult:
    """Random piecewise-constant drive with thermal bath + dephasing vs QuTiP 5 ``mesolve``."""
    try:
        import qutip as qt
    except ImportError:
        return skipped(
            "qutip_cross_check",
            "cross_check",
            1e-6,
            "not run: qutip is not installed (pip install 'qenvs[dev]')",
            model=MODEL,
        )
    delta, g1, n_th, gphi = 0.3, 0.4, 0.6, 0.15
    horizon, dt, bound = 12, 0.25, 2.0
    m = QubitBattery(detuning=delta, gamma1=g1, n_th=n_th, gamma_phi=gphi)
    env = BatteryCharging(model=m, horizon=horizon, dt=dt, action_bound=bound)
    actions = _random_actions(11, horizon, bound)
    state, _ = env.rollout(actions)
    rho_dq = np.asarray(state.qstate.to_jax())
    # the env casts actions to float32; feed QuTiP the identical amplitudes
    amps = np.asarray(jnp.asarray(actions[:, 0], dtype=jnp.float32), dtype=np.float64)
    amps = np.clip(amps, -bound, bound)
    # explicit matrices, index 1 = excited: sigma_z = diag(1, -1), |0><1| = [[0, 1], [0, 0]]
    sz = qt.Qobj(np.diag([1.0, -1.0]))
    sx = qt.Qobj(np.array([[0.0, 1.0], [1.0, 0.0]]))
    lower = qt.Qobj(np.array([[0.0, 1.0], [0.0, 0.0]]))
    c_ops = [
        np.sqrt(g1 * (n_th + 1)) * lower,
        np.sqrt(g1 * n_th) * lower.dag(),
        np.sqrt(gphi / 2) * sz,
    ]
    h0, hc = 0.5 * delta * sz, 0.5 * sx
    rho = qt.Qobj(np.diag([1.0, 0.0]))
    opts = {"atol": 1e-12, "rtol": 1e-10, "nsteps": 100_000, "progress_bar": ""}
    for a in amps:
        rho = qt.mesolve(h0 + float(a) * hc, rho, [0.0, dt], c_ops=c_ops, options=opts).states[-1]
    err = np.max(np.abs(rho_dq - rho.full()))
    return _res(
        "qutip_cross_check",
        "cross_check",
        err,
        1e-6,
        f"max |rho_dynamiqs - rho_qutip| after {horizon} random steps (qutip {qt.__version__})",
    )


def check_density_matrix_constraints() -> CheckResult:
    """Random open rollout: rho stays Hermitian, trace-1 and positive semidefinite."""
    m = QubitBattery(detuning=0.5, gamma1=0.2, n_th=0.3, gamma_phi=0.1)
    env = BatteryCharging(model=m, horizon=15, dt=0.3, action_bound=2.0)
    state, _ = env.rollout(_random_actions(7, 15, 2.0))
    rho = np.asarray(state.qstate.to_jax())
    herm = np.max(np.abs(rho - rho.conj().T))
    tr = abs(np.trace(rho).real - 1.0)
    neg = max(0.0, -float(np.min(np.linalg.eigvalsh(rho))))
    err = max(herm, tr, neg)
    return _res(
        "density_matrix_constraints",
        "constraints",
        err,
        1e-6,
        f"hermiticity={herm:.2e}, |tr-1|={tr:.2e}, negativity={neg:.2e}",
    )


def check_t1_decay_matches_two_level() -> CheckResult:
    """n_th = 0, no drive, from |1>: P1(t) = exp(-gamma1 t), identical to TwoLevel."""
    g = 0.3
    qb = QubitBattery(gamma1=g)
    tl = TwoLevel(gamma1=g, initial="excited")
    rho0 = dq.basis(2, 1).todm()
    ts = jnp.linspace(0.0, 5.0, 11)
    p1_op = qb.observables()["p1"]
    res_qb = dq.mesolve(qb.drift(), qb.jump_ops(), rho0, ts, progress_meter=False)
    res_tl = dq.mesolve(tl.drift(), tl.jump_ops(), tl.initial_state(), ts, progress_meter=False)
    p1_qb = np.asarray(dq.expect(p1_op, res_qb.states).real)
    p1_tl = np.asarray(dq.expect(p1_op, res_tl.states).real)
    exact = np.exp(-g * np.asarray(ts))
    err = max(np.max(np.abs(p1_qb - exact)), np.max(np.abs(p1_qb - p1_tl)))
    return _res(
        "t1_decay_matches_two_level",
        "open_system",
        err,
        1e-6,
        "max |P1 - exp(-gamma1 t)| and max |P1 - P1_TwoLevel|",
    )


def check_detuned_rabi() -> CheckResult:
    """Closed detuned drive: P1(t) = (u/Omega)^2 sin^2(Omega t / 2), Omega = sqrt(u^2 + Delta^2)."""
    u, delta = 1.2, 0.9
    omega = float(np.hypot(u, delta))
    m = QubitBattery(detuning=delta)
    ts = jnp.linspace(0.0, 2 * jnp.pi / omega, 17)
    res = dq.sesolve(
        m.hamiltonian(jnp.array([u])), m.initial_state(), ts, method=TIGHT, progress_meter=False
    )
    p1 = np.asarray(dq.expect(m.observables()["p1"], res.states).real)
    exact = (u / omega) ** 2 * np.sin(omega * np.asarray(ts) / 2) ** 2
    err = np.max(np.abs(p1 - exact))
    return _res("detuned_rabi", "analytic", err, 1e-6, "max |P1 - (u/Omega)^2 sin^2(Omega t/2)|")


def check_thermal_state_passive() -> CheckResult:
    """Thermal initial state: trace 1, energy omega0 n_th/(2 n_th + 1), zero ergotropy."""
    w, n_th = 1.4, 0.8
    m = QubitBattery(omega0=w, gamma1=0.5, n_th=n_th, initial="thermal")
    rho = m.initial_state()
    hb = m.battery_hamiltonian()
    p1 = n_th / (2 * n_th + 1)
    tr = abs(float(dq.trace(rho).real) - 1.0)
    e_err = abs(float(energy(rho, hb)) - w * p1)
    w_err = abs(float(ergotropy(rho, hb)))
    err = max(tr, e_err, w_err)
    return _res(
        "thermal_state_passive",
        "thermodynamics",
        err,
        1e-6,
        f"|tr-1|={tr:.2e}, |E - omega0 p1|={e_err:.2e}, |ergotropy|={w_err:.2e}",
    )


def check_solver_convergence() -> CheckResult:
    """Adaptive Tsit5 vs exact matrix exponential on an open constant-drive rollout."""
    m = QubitBattery(detuning=0.3, gamma1=0.1, n_th=0.3, gamma_phi=0.05)
    u = float(jnp.pi / (10 * 0.2))
    actions = jnp.full((10, 1), u)
    env_a = BatteryCharging(model=m, horizon=10, dt=0.2, method=dq.method.Tsit5())
    env_b = BatteryCharging(model=m, horizon=10, dt=0.2, method=dq.method.Expm())
    sa, _ = env_a.rollout(actions)
    sb, _ = env_b.rollout(actions)
    err = np.max(np.abs(np.asarray(sa.qstate.to_jax()) - np.asarray(sb.qstate.to_jax())))
    return _res(
        "solver_convergence", "numerics", err, 1e-6, "max |rho_Tsit5 - rho_Expm| at episode end"
    )


CHECKS: tuple[Callable[[], CheckResult], ...] = (
    check_thermal_steady_state,
    check_thermal_state_stationary,
    check_pi_pulse_charging,
    check_ergotropy_le_energy,
    check_charging_power_bound,
    check_env_return_equals_gain,
    check_qutip_cross_check,
    check_density_matrix_constraints,
    check_t1_decay_matches_two_level,
    check_detuned_rabi,
    check_thermal_state_passive,
    check_solver_convergence,
)
