"""Validation checks for the collective Dicke / Tavis-Cummings battery (:class:`DickeBattery`).

Analytic and cross-check comparisons integrate with a tight adaptive Tsit5
(``rtol=1e-8, atol=1e-10``) so that the reported error measures the physics, not
the default solver tolerance; the default solver is validated separately by
``solver_convergence``. The two QuTiP cross-checks need ``qutip`` (``qenvs[dev]``
extra); without it they report ``error = nan`` (FAIL, detail says why) instead of
raising.
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
from qenvs.models.dicke import DickeBattery, boson_matrices, collective_spin_matrices
from qenvs.validation.core import CheckResult, skipped
from qenvs.validation.core import result as _result

MODEL = "DickeBattery"
TIGHT = dq.method.Tsit5(rtol=1e-8, atol=1e-10)


def _res(name: str, category: str, error: float, tol: float, detail: str) -> CheckResult:
    return _result(name, category, error, tol, detail, model=MODEL)


def _energy_curve(model: DickeBattery, g: float, tsave: Array, method=TIGHT) -> np.ndarray:
    """Stored energy ``<H_B>(t)`` of the closed model under constant coupling ``g``."""
    res = dq.sesolve(
        model.hamiltonian(jnp.array([g])),
        model.initial_state(),
        tsave,
        method=method,
        progress_meter=False,
    )
    return np.asarray(dq.expect(model.battery_hamiltonian(), res.states).real, dtype=np.float64)


def _random_actions(seed: int, horizon: int, bound: float) -> Array:
    key = jax.random.PRNGKey(seed)
    return jax.random.uniform(key, (horizon, 1), minval=-bound, maxval=bound)


def _rollout_energies(env: BatteryCharging, actions: Array) -> np.ndarray:
    """Energy after every control step of an open-loop rollout (``lax.scan``)."""
    state0, _ = env.reset(jax.random.PRNGKey(0))

    def body(state: EnvState, a: Array) -> tuple[EnvState, Array]:
        state, ts = env.step(state, a)
        return state, ts.info["energy"]

    _, energies = jax.lax.scan(body, state0, actions)
    return np.asarray(energies, dtype=np.float64)


def _first_peak_time(ts: np.ndarray, e: np.ndarray) -> float:
    """Time of the first maximum of ``e`` on the grid, refined by a parabolic fit."""
    i = int(np.argmax(e))
    if 0 < i < len(e) - 1:
        y0, y1, y2 = e[i - 1], e[i], e[i + 1]
        denom = y0 - 2 * y1 + y2
        shift = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
    else:
        shift = 0.0
    return float(ts[i] + shift * (ts[1] - ts[0]))


# --------------------------------------------------------------------- checks
def check_jaynes_cummings_limit() -> CheckResult:
    """N = 1 cell, cavity in |n>, resonant TC, constant g: P_e(t) = sin^2(g sqrt(n) t)."""
    g = 0.5  # the formula is exact for TC at any g; a short period limits float32 roundoff
    errs = {}
    for n in (1, 3):
        m = DickeBattery(n_cells=1, n_photons=n)
        ts = np.linspace(0.0, np.pi / (g * np.sqrt(n)), 25)
        pe = _energy_curve(m, g, jnp.asarray(ts)) / m.omega0
        errs[n] = float(np.max(np.abs(pe - np.sin(g * np.sqrt(n) * ts) ** 2)))
    detail = ", ".join(f"n={n}: {e:.2e}" for n, e in errs.items())
    return _res(
        "jaynes_cummings_limit",
        "analytic",
        max(errs.values()),
        1e-6,
        f"max |P_e - sin^2(g sqrt(n) t)| over one period ({detail})",
    )


def check_excitation_conservation() -> CheckResult:
    """Closed TC, random piecewise-constant g(t): <a^dag a + Jz + N/2> is conserved.

    The Fock cutoff is raised above ``n_photons`` so that a spurious counter-rotating
    or mis-signed term could populate higher excitation sectors. The expectation value
    is taken on the renormalised state; the solver's norm drift is reported separately.
    """
    horizon, dt, bound = 12, 0.4, 0.5
    m = DickeBattery(n_cells=3, n_photons=3, n_max=6)
    amps = _random_actions(2, horizon, bound)
    times = jnp.arange(horizon + 1) * dt
    tsave = jnp.linspace(0.0, horizon * dt, 4 * horizon + 1)
    res = dq.sesolve(
        m.pwc_hamiltonian(times, amps.T),
        m.initial_state(),
        tsave,
        method=TIGHT,
        progress_meter=False,
    )
    psi = np.asarray(res.states.to_jax(), dtype=np.complex128)[..., 0]
    nop = np.asarray(m.excitation_number().to_jax(), dtype=np.complex128)
    norm = np.sum(np.abs(psi) ** 2, axis=1)
    exc = np.real(np.einsum("ti,ij,tj->t", psi.conj(), nop, psi)) / norm
    err = float(np.max(np.abs(exc - exc[0])))
    drift = float(np.max(np.abs(norm - 1.0)))
    return _res(
        "excitation_conservation",
        "conservation",
        err,
        1e-6,
        f"max |<N_exc>(t) - {exc[0]:.1f}| along random pwc g(t), N=3, n=3, n_max=6 "
        f"(renormalised state; solver norm drift {drift:.1e})",
    )


def check_energy_bounds() -> CheckResult:
    """Random rollouts (closed TC and open Dicke): 0 <= <H_B> <= N omega0 at every step."""
    horizon, bound = 15, 0.5
    envs = {
        "closed_tc": BatteryCharging(
            model=DickeBattery(n_cells=3), horizon=horizon, dt=0.4, action_bound=bound
        ),
        "open_dicke": BatteryCharging(
            model=DickeBattery(n_cells=3, rwa=False, kappa=0.1, gamma=0.05, gamma_phi=0.05),
            horizon=horizon,
            dt=0.3,
            action_bound=bound,
            method=TIGHT,
        ),
    }
    viol = {}
    for name, env in envs.items():
        e = _rollout_energies(env, _random_actions(4, horizon, bound))
        emax = env.model.n_cells * env.model.omega0
        viol[name] = max(0.0, float(np.max(-e)), float(np.max(e - emax)))
    detail = ", ".join(f"{k}: {v:.2e}" for k, v in viol.items())
    return _res(
        "energy_bounds",
        "constraints",
        max(viol.values()),
        1e-7,
        f"max violation of 0 <= E <= N omega0 along {horizon}-step random rollouts ({detail})",
    )


def check_collective_advantage() -> CheckResult:
    """Ferraro 2018: charging time t*(N) decreases with N, log-log slope near -1/2.

    Closed resonant TC with n_photons = n_cells, constant g = 0.05 omega0; t* is the
    first maximum of the stored energy on a dense time grid (parabolic refinement).
    """
    g = 0.05
    ns = (1, 2, 4, 8)
    tstars, emax = [], []
    for n in ns:
        m = DickeBattery(n_cells=n)
        ts = np.linspace(0.0, 1.3 * np.pi / (2 * g), 3001)
        e = _energy_curve(m, g, jnp.asarray(ts), method=dq.method.Tsit5())
        tstars.append(_first_peak_time(ts, e))
        emax.append(float(e.max()) / (n * m.omega0))
    slope = float(np.polyfit(np.log(ns), np.log(tstars), 1)[0])
    decreasing = bool(np.all(np.diff(tstars) < 0))
    err = abs(slope + 0.5) if decreasing else float("inf")
    detail = (
        f"fitted exponent {slope:.3f} (window [-0.75, -0.25]); t*(N) = "
        + ", ".join(f"{t:.2f}" for t in tstars)
        + "; E_max/(N omega0) = "
        + ", ".join(f"{x:.3f}" for x in emax)
        + ("" if decreasing else "; t*(N) NOT strictly decreasing")
    )
    return _res("collective_advantage", "scaling", err, 0.25, detail)


def _rwa_difference(n_cells: int, g: float) -> float:
    tc = DickeBattery(n_cells=n_cells)
    dicke = DickeBattery(n_cells=n_cells, rwa=False, n_max=n_cells + 6)
    ts = jnp.linspace(0.0, np.pi / (g * np.sqrt(n_cells)), 201)
    return float(np.max(np.abs(_energy_curve(tc, g, ts) - _energy_curve(dicke, g, ts))))


def check_rwa_weak_coupling() -> CheckResult:
    """Weak coupling g = 0.01 omega0: Dicke (n_max = n + 6) and TC energies agree."""
    err = _rwa_difference(2, 0.01)
    return _res(
        "rwa_weak_coupling",
        "analytic",
        err,
        2e-3,
        "max |E_Dicke - E_TC| over one collective Rabi period, N=2, g=0.01 omega0",
    )


def check_counter_rotating_terms() -> CheckResult:
    """Strong coupling g = 0.5 omega0: Dicke and TC energies differ by more than 0.05."""
    diff = _rwa_difference(2, 0.5)
    return _res(
        "counter_rotating_terms",
        "analytic",
        max(0.0, 0.05 - diff),
        0.0,
        f"max |E_Dicke - E_TC| = {diff:.3f} over one collective period, N=2, g=0.5 omega0 "
        "(error = shortfall below 0.05)",
    )


def _qutip_cross_check(name: str, model: DickeBattery) -> CheckResult:
    """Open random piecewise-constant g(t): dynamiqs env rollout vs QuTiP 5 ``mesolve``."""
    try:
        import qutip as qt
    except ImportError:
        return skipped(
            name, "cross_check", 1e-5, "not run: qutip is not installed (qenvs[dev])", model=MODEL
        )
    horizon, dt, bound = 8, 0.3, 0.4
    env = BatteryCharging(model=model, horizon=horizon, dt=dt, action_bound=bound, method=TIGHT)
    actions = _random_actions(3, horizon, bound)
    state, _ = env.rollout(actions)
    rho_dq = np.asarray(state.qstate.to_jax())
    # the env casts actions to float32; feed QuTiP the identical amplitudes
    amps = np.asarray(jnp.asarray(actions[:, 0], dtype=jnp.float32), dtype=np.float64)
    amps = np.clip(amps, -bound, bound)
    n, j, nc = model.n_cells, model.j, model.dim_cavity
    eye_c = qt.qeye(nc)
    jp, jm, jz = (qt.tensor(qt.jmat(j, s), eye_c) for s in ("+", "-", "z"))
    a = qt.tensor(qt.qeye(n + 1), qt.destroy(nc))
    h0 = model.omega0 * (jz + j) + model.omega_c * a.dag() * a
    hc = a.dag() * jm + a * jp if model.rwa else (a + a.dag()) * (jp + jm)
    c_ops = []
    for rate, op in ((model.kappa, a), (model.gamma, jm), (model.gamma_phi, jz)):
        if rate > 0:
            c_ops.append(np.sqrt(rate) * op)
    # qutip.jmat orders m = +j, ..., -j: all cells ground is index N
    rho = qt.tensor(qt.basis(n + 1, n), qt.basis(nc, model.photons)).proj()
    opts = {"atol": 1e-12, "rtol": 1e-10, "nsteps": 100_000, "progress_bar": ""}
    for amp in amps:
        rho = qt.mesolve(h0 + float(amp) * hc, rho, [0.0, dt], c_ops=c_ops, options=opts).states[-1]
    # map QuTiP's m-descending spin basis onto this library's m-ascending one
    perm = np.kron(np.eye(n + 1)[::-1], np.eye(nc))
    rho_qt = perm @ rho.full() @ perm.T
    err = float(np.max(np.abs(rho_dq - rho_qt)))
    kind = "Tavis-Cummings" if model.rwa else "Dicke"
    return _res(
        name,
        "cross_check",
        err,
        1e-5,
        f"max |rho_dynamiqs - rho_qutip| after {horizon} random steps, open {kind} "
        f"N={n}, n={model.photons}, n_max={model.cutoff} (qutip {qt.__version__})",
    )


def check_qutip_tavis_cummings() -> CheckResult:
    """Open TC (kappa, gamma, gamma_phi), random g(t), N = n = 3: match QuTiP 5 to 1e-5."""
    m = DickeBattery(n_cells=3, n_photons=3, kappa=0.1, gamma=0.05, gamma_phi=0.03)
    return _qutip_cross_check("qutip_tavis_cummings", m)


def check_qutip_dicke() -> CheckResult:
    """Open Dicke (counter-rotating terms, kappa), random g(t), N = n = 3: match QuTiP 5."""
    m = DickeBattery(n_cells=3, n_photons=3, rwa=False, kappa=0.1)
    return _qutip_cross_check("qutip_dicke", m)


def check_density_matrix_constraints() -> CheckResult:
    """Open Dicke under random driving: rho stays Hermitian, trace-1 and positive semidefinite."""
    horizon, bound = 12, 0.5
    m = DickeBattery(n_cells=3, rwa=False, kappa=0.1, gamma=0.05, gamma_phi=0.05)
    env = BatteryCharging(model=m, horizon=horizon, dt=0.3, action_bound=bound)
    state, _ = env.rollout(_random_actions(6, horizon, bound))
    rho = np.asarray(state.qstate.to_jax())
    herm = float(np.max(np.abs(rho - rho.conj().T)))
    tr = abs(float(np.trace(rho).real) - 1.0)
    neg = max(0.0, -float(np.min(np.linalg.eigvalsh(rho))))
    return _res(
        "density_matrix_constraints",
        "constraints",
        max(herm, tr, neg),
        1e-6,
        f"hermiticity={herm:.2e}, |tr-1|={tr:.2e}, negativity={neg:.2e} (dim {m.dim})",
    )


def check_solver_convergence() -> CheckResult:
    """Default adaptive Tsit5 (env rollout) vs exact Expm propagator, closed Dicke, random g(t)."""
    horizon, dt, bound = 8, 0.3, 0.4
    m = DickeBattery(n_cells=3, rwa=False)
    env = BatteryCharging(model=m, horizon=horizon, dt=dt, action_bound=bound)
    actions = jnp.asarray(_random_actions(5, horizon, bound), dtype=jnp.float32)
    state, _ = env.rollout(actions)
    times = jnp.arange(horizon + 1) * dt
    res = dq.sesolve(
        m.pwc_hamiltonian(times, actions.T),
        m.initial_state(),
        jnp.asarray([0.0, horizon * dt]),
        method=dq.method.Expm(),
        progress_meter=False,
    )
    psi_tsit5 = np.asarray(state.qstate.to_jax())
    psi_expm = np.asarray(res.states[-1].to_jax())
    err = float(np.max(np.abs(psi_tsit5 - psi_expm)))
    return _res(
        "solver_convergence",
        "numerics",
        err,
        2e-5,
        f"max |psi_Tsit5 - psi_Expm| after {horizon} random steps, closed Dicke dim {m.dim}",
    )


def check_operator_algebra() -> CheckResult:
    """[J+, J-] = 2 Jz (N = 1..6) and [a, a^dag] = 1 below the Fock cutoff (float64)."""
    errs = []
    for n in range(1, 7):
        jp, jm, jz = collective_spin_matrices(n)
        errs.append(np.max(np.abs(jp @ jm - jm @ jp - 2 * jz)))
        errs.append(np.max(np.abs(jp @ jp.T - jm.T @ jm)))  # J- = J+^dag
        errs.append(abs(jz[-1, -1] - n / 2) + abs(jz[0, 0] + n / 2))
    n_max = 7
    a, adag = boson_matrices(n_max)
    comm = a @ adag - adag @ a
    errs.append(np.max(np.abs(comm[:n_max, :n_max] - np.eye(n_max))))
    errs.append(np.max(np.abs(adag - a.T)))
    return _res(
        "operator_algebra",
        "algebra",
        float(max(errs)),
        1e-12,
        f"max |[J+, J-] - 2 Jz| for N <= 6 and |[a, a^dag] - 1| below cutoff n_max={n_max}",
    )


CHECKS: tuple[Callable[[], CheckResult], ...] = (
    check_operator_algebra,
    check_jaynes_cummings_limit,
    check_excitation_conservation,
    check_energy_bounds,
    check_collective_advantage,
    check_rwa_weak_coupling,
    check_counter_rotating_terms,
    check_qutip_tavis_cummings,
    check_qutip_dicke,
    check_density_matrix_constraints,
    check_solver_convergence,
)
