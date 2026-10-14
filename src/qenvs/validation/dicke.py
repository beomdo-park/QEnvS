"""Validation checks for the collective Dicke / Tavis-Cummings battery (:class:`DickeBattery`).

Analytic and cross-check comparisons integrate with a tight adaptive Tsit5
(``rtol=1e-8, atol=1e-10``) so that the reported error measures the physics, not
the default solver tolerance; the default solver is validated separately by
``solver_convergence`` (closed) and ``open_solver_convergence`` (Lindblad). The
QuTiP cross-checks need ``qutip`` (``qenvs[dev]`` extra); without it they are
reported as skipped (status SKIP, detail says why) instead of raising.
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
from qenvs.observables import energy, ergotropy
from qenvs.validation.core import CheckResult, skipped
from qenvs.validation.core import result as _result

MODEL = "DickeBattery"
TIGHT = dq.method.Tsit5(rtol=1e-8, atol=1e-10)


def _res(name: str, category: str, error: float, tol: float, detail: str) -> CheckResult:
    return _result(name, category, error, tol, detail, model=MODEL)


def _closed_states(model: DickeBattery, g: float, tsave: Array, method=TIGHT) -> dq.QArray:
    """States of the closed model at ``tsave`` under constant coupling ``g``."""
    res = dq.sesolve(
        model.hamiltonian(jnp.array([g])),
        model.initial_state(),
        tsave,
        method=method,
        progress_meter=False,
    )
    return res.states


def _energy_curve(model: DickeBattery, g: float, tsave: Array, method=TIGHT) -> np.ndarray:
    """Stored energy ``<H_B>(t)`` of the closed model under constant coupling ``g``."""
    states = _closed_states(model, g, tsave, method)
    return np.asarray(dq.expect(model.battery_hamiltonian(), states).real, dtype=np.float64)


def _random_actions(seed: int, horizon: int, bound: float) -> Array:
    key = jax.random.PRNGKey(seed)
    return jax.random.uniform(key, (horizon, 1), minval=-bound, maxval=bound)


def _scan_info(env: BatteryCharging, actions: Array) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Open-loop rollout via ``lax.scan``.

    Returns the ``info`` time series (initial step included) and ``tr rho`` after every
    control step.
    """
    state0, ts0 = env.reset(jax.random.PRNGKey(0))

    def body(state: EnvState, a: Array) -> tuple[EnvState, tuple[dict[str, Array], Array]]:
        state, ts = env.step(state, a)
        trace = jnp.trace(state.qstate.todm().to_jax()).real
        return state, (ts.info, trace)

    _, (infos, traces) = jax.lax.scan(body, state0, actions)
    info = {
        k: np.concatenate([np.atleast_1d(np.asarray(ts0.info[k])), np.asarray(v)]).astype(
            np.float64
        )
        for k, v in infos.items()
    }
    return info, np.asarray(traces, dtype=np.float64)


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
        3e-6,
        f"max |P_e - sin^2(g sqrt(n) t)| over one period ({detail})",
    )


def check_detuned_jaynes_cummings() -> CheckResult:
    """N = 1, |n> photons, constant g, detuning d = omega_c - omega0 (Jaynes-Cummings).

    ``P_e(t) = (4 g^2 n / Om^2) sin^2(Om t / 2)`` with ``Om^2 = d^2 + 4 g^2 n`` and, by
    excitation conservation, ``<a^dag a>(t) = n - P_e(t)``. ``P_e`` is read off as
    ``<H_B> / omega0``, so the cases with ``omega0 != 1`` also test the energy unit.
    ``P_e`` is even in ``d``. The sign of the detuning is pinned by the coupling
    expectation ``<V> = <a^dag J- + a J+>``: energy conservation inside the doublet
    ``{|g, n>, |e, n-1>}`` gives ``g <V> = d P_e``, which is odd in ``d``.
    """
    g = 0.5
    cases = ((1.0, 1.7, 1), (1.0, 0.3, 2), (1.3, 1.0, 1))
    errs = {}
    for omega0, omega_c, n in cases:
        m = DickeBattery(n_cells=1, n_photons=n, omega0=omega0, omega_c=omega_c)
        d = omega_c - omega0
        om = float(np.sqrt(d**2 + 4 * g**2 * n))
        ts = np.linspace(0.0, 2 * np.pi / om, 25)
        states = _closed_states(m, g, jnp.asarray(ts))
        pe = np.asarray(dq.expect(m.battery_hamiltonian(), states).real, dtype=np.float64)
        pe = pe / omega0
        nph = np.asarray(dq.expect(m.number(), states).real, dtype=np.float64)
        coupling = np.asarray(dq.expect(m.controls()[0], states).real, dtype=np.float64)
        exact = (4 * g**2 * n / om**2) * np.sin(om * ts / 2) ** 2
        errs[(omega0, omega_c, n)] = float(
            max(
                np.max(np.abs(pe - exact)),
                np.max(np.abs(nph - (n - pe))),
                np.max(np.abs(coupling - (d / g) * exact)),
            )
        )
    detail = ", ".join(
        f"(omega0={w0}, omega_c={wc}, n={n}): {e:.2e}" for (w0, wc, n), e in errs.items()
    )
    return _res(
        "detuned_jaynes_cummings",
        "analytic",
        max(errs.values()),
        3e-6,
        f"max of |P_e - (4g^2n/Om^2) sin^2(Om t/2)|, |<a^dag a> - (n - P_e)| and "
        f"|<V> - (d/g) P_e| (odd in d) over one period, g=0.5 ({detail})",
    )


def check_two_cells_closed_form() -> CheckResult:
    """N = 2, n = 2, resonant TC, constant g: first energy peak at t* = pi/(sqrt(6) g).

    In the excitation-2 manifold the coupling matrix has eigenfrequencies ``0`` and
    ``+-sqrt(6) g``, so ``E(t)/omega0 = (2/3) sin^2(w t) + (4/9)(1 - cos w t)^2`` with
    ``w = sqrt(6) g``, first maximum ``E_max = (8/9) N omega0 = 16/9 omega0`` at ``t*``.
    The error is ``max(|t_peak - t*|, 100 |E(t*) - E_max|)`` so that the single
    tolerance 1e-4 enforces 1e-4 on ``t*`` and 1e-6 on ``E_max``.
    """
    g = 0.5
    m = DickeBattery(n_cells=2, n_photons=2)
    w = np.sqrt(6.0) * g
    tstar = np.pi / w
    ts = np.linspace(0.0, 1.3 * tstar, 601)
    e = _energy_curve(m, g, jnp.asarray(ts))
    exact = m.omega0 * ((2 / 3) * np.sin(w * ts) ** 2 + (4 / 9) * (1 - np.cos(w * ts)) ** 2)
    curve_err = float(np.max(np.abs(e - exact)))
    t_err = abs(_first_peak_time(ts, e) - tstar)
    e_max = (8 / 9) * m.n_cells * m.omega0
    e_err = abs(float(_energy_curve(m, g, jnp.asarray([0.0, tstar]))[-1]) - e_max)
    return _res(
        "two_cells_closed_form",
        "analytic",
        max(t_err, 100 * e_err),
        1e-4,
        f"|t_peak - pi/(sqrt(6) g)|={t_err:.2e} (grid of 601 points, parabolic refinement), "
        f"|E(t*) - 16/9 omega0|={e_err:.2e}, max |E(t) - closed form|={curve_err:.2e}, g=0.5",
    )


def check_reduced_ergotropy() -> CheckResult:
    """N = 1, n = 1, resonant TC: ergotropy of the reduced cell state, closed form.

    ``psi(t) = cos(g t) |g, 1> - i sin(g t) |e, 0>`` so the reduced cell state is
    ``diag(cos^2 g t, sin^2 g t)``. At ``g t = pi/4`` it is ``diag(1/2, 1/2)``: energy
    ``omega0 / 2`` but ergotropy ``0`` (maximally mixed, passive). At ``g t = pi/2`` the
    cell is purely excited: ergotropy = energy = ``omega0``. Ergotropy evaluated on the
    full cells+cavity state would equal the energy at both times.
    """
    g = 0.5
    m = DickeBattery(n_cells=1, n_photons=1)
    h = m.hamiltonian(jnp.array([g]))
    errs = {}
    for label, gt, e_ref, w_ref in (("pi/4", np.pi / 4, 0.5, 0.0), ("pi/2", np.pi / 2, 1.0, 1.0)):
        psi = m.evolve(m.initial_state(), h, gt / g, method=TIGHT)
        rho_b, hb = m.battery_reduced(psi)
        e = float(energy(rho_b, hb))
        wk = float(ergotropy(rho_b, hb))
        errs[label] = max(abs(e - e_ref * m.omega0), abs(wk - w_ref * m.omega0))
    detail = ", ".join(f"g t = {k}: {v:.2e}" for k, v in errs.items())
    return _res(
        "reduced_ergotropy",
        "thermodynamics",
        max(errs.values()),
        1e-6,
        f"max(|E - E_ref|, |ergotropy - W_ref|) of the reduced cell state, N=1, n=1 ({detail}; "
        "references E=omega0/2, W=0 at pi/4 and E=W=omega0 at pi/2)",
    )


def check_omega0_scaling() -> CheckResult:
    """Resonant TC (omega_c = omega0), same random g(t): energy and ergotropy scale with omega0.

    On resonance ``H = omega0 N_exc + g(t) V`` with ``[N_exc, V] = 0``, so the populations
    do not depend on ``omega0`` and both ``<H_B>`` and the ergotropy must scale exactly by
    ``omega0`` (catches an ``H_B`` that ignores ``omega0``). Env rollouts, N = 2.
    """
    horizon, dt, bound = 10, 0.3, 0.5
    actions = _random_actions(2, horizon, bound)
    series = {}
    for w in (1.0, 1.5):
        env = BatteryCharging(
            model=DickeBattery(n_cells=2, omega0=w, omega_c=w),
            horizon=horizon,
            dt=dt,
            action_bound=bound,
            method=TIGHT,
        )
        series[w], _ = _scan_info(env, actions)
    e_err = float(np.max(np.abs(series[1.5]["energy"] - 1.5 * series[1.0]["energy"])))
    w_err = float(np.max(np.abs(series[1.5]["ergotropy"] - 1.5 * series[1.0]["ergotropy"])))
    return _res(
        "omega0_scaling",
        "analytic",
        max(e_err, w_err),
        1e-6,
        f"max |Q(omega0=1.5) - 1.5 Q(omega0=1)| over {horizon} random steps: energy={e_err:.2e}, "
        f"ergotropy={w_err:.2e} (N=2, resonant TC)",
    )


def check_excitation_conservation() -> CheckResult:
    """Closed TC, random piecewise-constant g(t): <a^dag a + Jz + N/2> = n_photons throughout.

    The Fock cutoff is raised above ``n_photons`` so that a spurious counter-rotating
    or mis-signed term could populate higher excitation sectors. The initial value must
    equal ``n_photons`` exactly (catches a missing ``+N/2`` offset). The expectation
    value is taken on the renormalised state; the solver's norm drift is reported
    separately.
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
    offset = abs(float(exc[0]) - m.photons)
    err = max(float(np.max(np.abs(exc - m.photons))), offset)
    drift = float(np.max(np.abs(norm - 1.0)))
    return _res(
        "excitation_conservation",
        "conservation",
        err,
        1e-6,
        f"max |<N_exc>(t) - n_photons| along random pwc g(t), N=3, n=3, n_max=6 "
        f"(|<N_exc>(0) - 3|={offset:.1e}; renormalised state; solver norm drift {drift:.1e})",
    )


def check_ergotropy_bounds() -> CheckResult:
    """Random rollouts: 0 <= ergotropy <= energy <= N omega0 for the reduced cell state.

    Closed TC and open Dicke (kappa, gamma, gamma_phi) environments, 15 random steps
    each with the tight solver. For the open rollout ``|tr rho - 1|`` after every step
    is part of the error (trace preservation of the Lindblad generator); the ket norm
    drift of the closed rollout is pure solver roundoff and is only reported. The
    minimum gap ``energy - ergotropy`` is reported: it is positive whenever the cells
    are entangled with the cavity or mixed by dissipation.
    """
    horizon, bound = 15, 0.5
    envs = {
        "closed_tc": BatteryCharging(
            model=DickeBattery(n_cells=3), horizon=horizon, dt=0.4, action_bound=bound, method=TIGHT
        ),
        "open_dicke": BatteryCharging(
            model=DickeBattery(n_cells=3, rwa=False, kappa=0.1, gamma=0.05, gamma_phi=0.05),
            horizon=horizon,
            dt=0.3,
            action_bound=bound,
            method=TIGHT,
        ),
    }
    viol, gap, drift = {}, {}, {}
    for name, env in envs.items():
        info, traces = _scan_info(env, _random_actions(4, horizon, bound))
        e, w = info["energy"], info["ergotropy"]
        emax = env.model.n_cells * env.model.omega0
        drift[name] = float(np.max(np.abs(traces - 1.0)))
        viol[name] = max(
            0.0,
            float(np.max(w - e)),
            float(np.max(-w)),
            float(np.max(-e)),
            float(np.max(e - emax)),
            drift[name] if env.model.is_open else 0.0,
        )
        gap[name] = float(np.min((e - w)[1:]))
    detail = ", ".join(
        f"{k}: violation {viol[k]:.1e}, min(energy - ergotropy) {gap[k]:.2e}, "
        f"|tr rho - 1| {drift[k]:.1e}"
        for k in envs
    )
    return _res(
        "ergotropy_bounds",
        "thermodynamics",
        max(viol.values()),
        1e-6,
        f"max violation of 0 <= ergotropy <= E <= N omega0 (both rollouts) and open trace "
        f"drift along {horizon}-step random rollouts ({detail})",
    )


def check_collective_advantage() -> CheckResult:
    """Ferraro 2018: charging time t*(N) decreases with N, log-log slope near -1/2.

    Closed resonant TC with n_photons = n_cells, constant g = 0.05 omega0; t* is the
    first maximum of the stored energy on a dense time grid (parabolic refinement).
    The fitted exponent in this protocol is about -0.40 (also for N up to 48, measured
    offline); the asymptotic N^(-1/2) law is accepted within a window of +-0.25.
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
        "max |E_Dicke - E_TC| for t in [0, pi/(g sqrt(N))], N=2, g=0.01 omega0",
    )


def check_counter_rotating_terms() -> CheckResult:
    """Strong coupling g = 0.5 omega0: Dicke and TC energies differ by more than 0.05."""
    diff = _rwa_difference(2, 0.5)
    return _res(
        "counter_rotating_terms",
        "analytic",
        max(0.0, 0.05 - diff),
        0.0,
        f"max |E_Dicke - E_TC| = {diff:.3f} for t in [0, pi/(g sqrt(N))], N=2, g=0.5 omega0 "
        "(error = shortfall below 0.05)",
    )


def check_cutoff_convergence() -> CheckResult:
    """Closed Dicke (rwa=False), N = 2, constant g = 0.1 omega0: default n_max vs n_max + 6.

    ``<H_B>(t)`` along a 20-step rollout (dt = 0.2, the task default) with the default
    cutoff ``n_max = n + 4 = 6`` is compared with ``n_max = 12``. The population of the
    top Fock level (observable ``top_fock``) of the default-cutoff run is reported. At
    the task default ``g = 0.5`` the same comparison is not converged (see the docs).
    """
    horizon, dt, g = 20, 0.2, 0.1
    base = DickeBattery(n_cells=2, rwa=False)
    wide = DickeBattery(n_cells=2, rwa=False, n_max=base.cutoff + 6)
    actions = jnp.full((horizon, 1), g)
    curves, top = [], 0.0
    for m in (base, wide):
        env = BatteryCharging(model=m, horizon=horizon, dt=dt, action_bound=0.5, method=TIGHT)
        info, _ = _scan_info(env, actions)
        curves.append(info["energy"])
        if m is base:
            top = float(np.max(info["top_fock"]))
    err = float(np.max(np.abs(curves[0] - curves[1])))
    return _res(
        "cutoff_convergence",
        "numerics",
        err,
        1e-3,
        f"max |E(n_max={base.cutoff}) - E(n_max={wide.cutoff})| over {horizon} steps of "
        f"g = 0.1 omega0, N=2 (max top_fock population at n_max={base.cutoff}: {top:.1e})",
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
        f"N={n}, n={model.photons}, n_max={model.cutoff}, omega0={model.omega0}, "
        f"omega_c={model.omega_c} (qutip {qt.__version__})",
    )


def check_qutip_tavis_cummings() -> CheckResult:
    """Open TC (kappa, gamma, gamma_phi), random g(t), N = n = 3: match QuTiP 5 to 1e-5."""
    m = DickeBattery(n_cells=3, n_photons=3, kappa=0.1, gamma=0.05, gamma_phi=0.03)
    return _qutip_cross_check("qutip_tavis_cummings", m)


def check_qutip_dicke() -> CheckResult:
    """Open Dicke (counter-rotating terms, kappa), random g(t), N = n = 3: match QuTiP 5."""
    m = DickeBattery(n_cells=3, n_photons=3, rwa=False, kappa=0.1)
    return _qutip_cross_check("qutip_dicke", m)


def check_qutip_detuned() -> CheckResult:
    """Open detuned TC, omega0 = 0.8, omega_c = 1.3, N = n = 2, random g(t): match QuTiP 5.

    QuTiP builds ``H`` independently with explicit ``omega0`` and ``omega_c``. Populations
    are even in the detuning, but the coherences of ``rho`` are not, so a drift that uses
    ``omega0`` for the cavity, a flipped detuning sign or an ``H_B`` without ``omega0``
    changes the final density matrix at the 1e-2 level (tolerance 1e-5).
    """
    m = DickeBattery(n_cells=2, n_photons=2, omega0=0.8, omega_c=1.3, kappa=0.05, gamma=0.02)
    return _qutip_cross_check("qutip_detuned", m)


def check_collective_dissipators() -> CheckResult:
    """Closed-form Lindblad dynamics of the three dissipators, N = 1, g = 0 (no QuTiP).

    * cavity loss ``sqrt(kappa) a`` from ``|g> (x) |1>``: ``<a^dag a>(t) = exp(-kappa t)``;
    * collective emission ``sqrt(gamma) J-`` from ``|e> (x) |0>``:
      ``<H_B>(t) / omega0 = exp(-gamma t)``;
    * collective dephasing ``sqrt(gamma_phi) Jz`` (``Jz = sigma_z / 2``) from
      ``(|g> + |e>)/sqrt(2) (x) |0>``: ``2 |<J->(t)| = exp(-gamma_phi t / 2)`` (the
      modulus removes the free precession at ``omega0`` of the lab-frame drift).

    A reversed jump operator or a wrong rate prefactor fails one of the three, so the jump
    operators are pinned even when the QuTiP cross-checks are skipped.
    """
    ts = np.linspace(0.0, 4.0, 17)
    tsave = jnp.asarray(ts)

    def evolve(m: DickeBattery, rho0: dq.QArray, op: dq.QArray) -> np.ndarray:
        res = dq.mesolve(m.drift(), m.jump_ops(), rho0, tsave, method=TIGHT, progress_meter=False)
        return np.asarray(dq.expect(op, res.states).real, dtype=np.float64)

    kappa, gamma, gamma_phi = 0.3, 0.4, 0.5
    m_k = DickeBattery(n_cells=1, n_photons=1, kappa=kappa)
    e_k = np.max(np.abs(evolve(m_k, m_k.initial_state(), m_k.number()) - np.exp(-kappa * ts)))

    m_g = DickeBattery(n_cells=1, n_photons=0, n_max=1, gamma=gamma)
    excited = dq.tensor(dq.basis(2, 1), dq.basis(2, 0)).todm()
    e_g = np.max(
        np.abs(evolve(m_g, excited, m_g.battery_hamiltonian()) / m_g.omega0 - np.exp(-gamma * ts))
    )

    m_p = DickeBattery(n_cells=1, n_photons=0, n_max=1, gamma_phi=gamma_phi)
    plus = dq.tensor((dq.basis(2, 0) + dq.basis(2, 1)).unit(), dq.basis(2, 0)).todm()
    res_p = dq.mesolve(m_p.drift(), m_p.jump_ops(), plus, tsave, method=TIGHT, progress_meter=False)
    coherence = 2 * np.abs(np.asarray(dq.expect(m_p.jminus(), res_p.states), dtype=np.complex128))
    e_p = np.max(np.abs(coherence - np.exp(-gamma_phi * ts / 2)))

    return _res(
        "collective_dissipators",
        "open_system",
        max(e_k, e_g, e_p),
        2e-6,
        f"N=1, g=0: |<a^dag a> - exp(-kappa t)|={e_k:.2e}, |<H_B>/omega0 - exp(-gamma t)|="
        f"{e_g:.2e}, |2|<J->| - exp(-gamma_phi t/2)|={e_p:.2e} (t <= 4)",
    )


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


def check_open_solver_convergence() -> CheckResult:
    """Open Dicke (kappa = 0.1, gamma = 0.05), random g(t): default Tsit5 vs tight Tsit5.

    The default env solver (``rtol=1e-6, atol=1e-6``) is compared with
    ``Tsit5(rtol=1e-10, atol=1e-12)`` on the final density matrix.
    """
    horizon, dt, bound = 8, 0.3, 0.4
    m = DickeBattery(n_cells=2, rwa=False, kappa=0.1, gamma=0.05)
    actions = _random_actions(5, horizon, bound)
    env = BatteryCharging(model=m, horizon=horizon, dt=dt, action_bound=bound)
    tight = BatteryCharging(
        model=m,
        horizon=horizon,
        dt=dt,
        action_bound=bound,
        method=dq.method.Tsit5(rtol=1e-10, atol=1e-12),
    )
    sa, _ = env.rollout(actions)
    sb, _ = tight.rollout(actions)
    err = float(np.max(np.abs(np.asarray(sa.qstate.to_jax()) - np.asarray(sb.qstate.to_jax()))))
    return _res(
        "open_solver_convergence",
        "numerics",
        err,
        2e-5,
        f"max |rho_Tsit5(default) - rho_Tsit5(rtol=1e-10, atol=1e-12)| after {horizon} random "
        f"steps, open Dicke dim {m.dim}, kappa=0.1, gamma=0.05",
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
    check_detuned_jaynes_cummings,
    check_two_cells_closed_form,
    check_reduced_ergotropy,
    check_omega0_scaling,
    check_excitation_conservation,
    check_ergotropy_bounds,
    check_collective_advantage,
    check_rwa_weak_coupling,
    check_counter_rotating_terms,
    check_cutoff_convergence,
    check_qutip_tavis_cummings,
    check_qutip_dicke,
    check_qutip_detuned,
    check_collective_dissipators,
    check_density_matrix_constraints,
    check_solver_convergence,
    check_open_solver_convergence,
)
