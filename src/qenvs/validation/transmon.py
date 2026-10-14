"""Validation checks for the transmon model, the gate-fidelity tools and :class:`GateControl`.

UNITS: time in ns, angular frequency in rad/ns, model parameters in GHz and us.
The QuTiP cross-check needs ``qutip`` (``qenvs[dev]`` extra); without it that check is
reported as skipped instead of raising.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import dynamiqs as dq
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from qenvs.envs.gate import GateControl
from qenvs.gates import (
    average_gate_fidelity,
    gate,
    leakage,
    rz,
    superop_average_gate_fidelity,
    superop_leakage,
    superop_process_fidelity,
    unitary_to_superop,
)
from qenvs.models.transmon import (
    Transmon,
    drag_beta,
    gaussian_amplitude,
    gaussian_drag,
    ghz_to_rad_per_ns,
)
from qenvs.validation.core import CheckResult, skipped
from qenvs.validation.core import result as _result

MODEL = "Transmon"
TWO_PI = 2.0 * math.pi


def _res(name: str, category: str, error: float, tol: float, detail: str) -> CheckResult:
    return _result(name, category, error, tol, detail, model=MODEL)


def _arith_tol() -> float:
    """Tolerance for pure arithmetic: 1e-9 in float64, 10 machine epsilons in float32."""
    eps = float(jnp.finfo(jnp.asarray(1.0).dtype).eps)
    return max(1e-9, 10.0 * eps)


def _random_actions(key: int, horizon: int, bound: float) -> Array:
    return jax.random.uniform(jax.random.PRNGKey(key), (horizon, 2), minval=-bound, maxval=bound)


def _map(env: GateControl, actions: Array) -> np.ndarray:
    state, _ = env.rollout(actions)
    return np.asarray(state.qstate.to_jax())


def _random_dm(rng: np.random.Generator, d: int) -> np.ndarray:
    x = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    rho = x @ x.conj().T
    return rho / np.trace(rho)


def _np_expm(a: np.ndarray) -> np.ndarray:
    """Matrix exponential by scaling and squaring with a 24-term Taylor series (numpy only)."""
    s = max(0, math.ceil(math.log2(max(float(np.linalg.norm(a, 1)), 1e-300))) + 1)
    b = a / 2.0**s
    term, out = np.eye(a.shape[0], dtype=complex), np.eye(a.shape[0], dtype=complex)
    for k in range(1, 25):
        term = term @ b / k
        out = out + term
    for _ in range(s):
        out = out @ out
    return out


def _np_hamiltonian(levels: int, anh: float, det: float, wi: float, wq: float) -> np.ndarray:
    """Rotating-frame transmon Hamiltonian built directly from GHz parameters (numpy only)."""
    a = np.diag(np.sqrt(np.arange(1.0, levels)), 1).astype(complex)
    n = np.diag(np.arange(float(levels))).astype(complex)
    drift = TWO_PI * det * n + 0.5 * TWO_PI * anh * n @ (n - np.eye(levels))
    return drift + wi * 0.5 * (a + a.conj().T) + wq * 0.5j * (a.conj().T - a)


def _np_map(
    levels: int,
    anh: float,
    det: float,
    actions: np.ndarray,
    dt: float,
    t1_us: float | None = None,
    t2_us: float | None = None,
) -> np.ndarray:
    """Accumulated propagator ``U`` (closed) or column-stacking ``S`` (``t1_us`` / ``t2_us``).

    Independent of dynamiqs and of the model classes: the Hamiltonian, the jump operators
    and the matrix exponentials are built in numpy.
    """
    a = np.diag(np.sqrt(np.arange(1.0, levels)), 1).astype(complex)
    n = np.diag(np.arange(float(levels))).astype(complex)
    eye = np.eye(levels)
    open_system = t1_us is not None or t2_us is not None
    collapse: list[np.ndarray] = []
    if open_system:
        g1 = 0.0 if t1_us is None else 1.0 / (1000.0 * t1_us)
        gphi = 0.0 if t2_us is None else 1.0 / (1000.0 * t2_us) - 0.5 * g1
        if g1 > 0:
            collapse.append(math.sqrt(g1) * a)
        if gphi > 1e-15:
            collapse.append(math.sqrt(2.0 * gphi) * n)
    diss = np.zeros((levels**2, levels**2), dtype=complex)
    for c in collapse:
        cdc = c.conj().T @ c
        diss += np.kron(c.conj(), c) - 0.5 * np.kron(eye, cdc) - 0.5 * np.kron(cdc.T, eye)
    total = np.eye(levels**2 if open_system else levels, dtype=complex)
    for wi, wq in np.asarray(actions, dtype=float):
        h = _np_hamiltonian(levels, anh, det, float(wi), float(wq))
        # column stacking: vec(H rho) = (1 kron H) vec(rho), vec(rho H) = (H^T kron 1) vec(rho)
        liouv = -1j * (np.kron(eye, h) - np.kron(h.T, eye)) + diss
        gen = liouv if open_system else -1j * h
        total = _np_expm(gen * dt) @ total
    return total


# --------------------------------------------------------------------- checks
def check_qubit_limit_gates() -> CheckResult:
    """Qubit limit: controls are sigma_x/2, sigma_y/2; Omega_I T = pi is X, Omega_Q T = pi/2 Ry."""
    m = Transmon(levels=2)
    h_i, h_q = (np.asarray(c.to_jax()) for c in m.controls())
    sx = np.array([[0, 1], [1, 0]]) / 2
    sy = np.array([[0, -1j], [1j, 0]]) / 2
    op_err = max(np.max(np.abs(h_i - sx)), np.max(np.abs(h_q - sy)))
    horizon, dt = 4, 2.5
    w = math.pi / (horizon * dt)
    env_x = GateControl(model=m, target=gate("X"), horizon=horizon, dt=dt, action_bound=1.0)
    u_x = _map(env_x, jnp.tile(jnp.array([w, 0.0]), (horizon, 1)))
    f_x = float(average_gate_fidelity(u_x, gate("X")))
    c = s = 1 / math.sqrt(2)
    ry = jnp.array([[c, -s], [s, c]], dtype=complex)  # exp(-i (pi/4) sigma_y)
    u_y = _map(env_x, jnp.tile(jnp.array([0.0, w / 2]), (horizon, 1)))
    f_y = float(average_gate_fidelity(u_y, ry))
    err = max(op_err, abs(1 - f_x), abs(1 - f_y))
    return _res(
        "qubit_limit_gates",
        "analytic",
        err,
        2e-6,
        f"levels 2: |H_I - sx/2|, |H_Q - sy/2| = {op_err:.1e}; "
        f"1 - F(X) = {1 - f_x:.1e} after Omega_I T = pi; "
        f"1 - F(Ry(pi/2)) = {1 - f_y:.1e} after Omega_Q T = pi/2",
    )


def check_rabi_units() -> CheckResult:
    """Rabi frequency f (GHz) flips |0> to |1> at t = 1/(2 f) ns; detuned: P1 = W^2/(W^2+D^2)."""
    f = 0.025  # 25 MHz
    m = Transmon(levels=2)
    env = GateControl(model=m, target=gate("X"), horizon=1, dt=1 / (2 * f), action_bound=1.0)
    w = float(ghz_to_rad_per_ns(f))
    state, _ = env.rollout(jnp.array([[w, 0.0]]))
    p1 = float(env._info(state.qstate)["p1"])
    det = 0.02  # GHz
    md = Transmon(levels=2, detuning_ghz=det)
    big_w = math.hypot(w, TWO_PI * det)
    env_d = GateControl(model=md, target=gate("X"), horizon=1, dt=math.pi / big_w, action_bound=1.0)
    state_d, _ = env_d.rollout(jnp.array([[w, 0.0]]))
    p1_d = float(env_d._info(state_d.qstate)["p1"])
    exact_d = w**2 / big_w**2
    err = max(abs(p1 - 1.0), abs(p1_d - exact_d))
    return _res(
        "rabi_units",
        "analytic",
        err,
        1e-6,
        f"P1(1/(2f)) = {p1:.8f} for f = 25 MHz; detuned 20 MHz: P1 = {p1_d:.8f} vs {exact_d:.8f}",
    )


def check_drift_spectrum() -> CheckResult:
    """Idle propagator = diag(exp(-i (Delta n + alpha n (n-1) / 2) t)) with GHz -> rad/ns."""
    det, anh, t = 0.013, -0.31, 1.5
    m = Transmon(levels=4, detuning_ghz=det, anharmonicity_ghz=anh)
    env = GateControl(model=m, target=gate("I"), horizon=3, dt=t / 3)
    u = _map(env, jnp.zeros((3, 2)))
    n = np.arange(4)
    energies = TWO_PI * det * n + 0.5 * TWO_PI * anh * n * (n - 1)
    err = np.max(np.abs(u - np.diag(np.exp(-1j * energies * t))))
    return _res(
        "drift_spectrum",
        "analytic",
        err,
        1e-5,
        "max |U_idle - diag(exp(-i E_n t))|, E_n = 2 pi (0.013 n - 0.155 n (n-1)) rad/ns, "
        "t = 1.5 ns, 4 levels",
    )


def check_fidelity_closed_forms() -> CheckResult:
    """Rz(theta) vs I: (2 + 4 cos^2(theta/2))/6; |1> -> |2> swap: 1/3 and 1/6; I vs X: 1/3."""
    errs: list[float] = []
    eye2 = gate("I")
    for theta in (0.3, 1.1, 2.5, math.pi):
        f = float(average_gate_fidelity(rz(theta), eye2))
        errs.append(abs(f - (2 + 4 * math.cos(theta / 2) ** 2) / 6))
        fs = float(superop_average_gate_fidelity(unitary_to_superop(rz(theta)), eye2))
        errs.append(abs(fs - (2 + 4 * math.cos(theta / 2) ** 2) / 6))
    perm = jnp.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=complex)  # |1> <-> |2>
    # M = P U P = diag(1, 0): vs I, Tr(MM^dag) = |Tr M|^2 = 1 -> 2/6; vs X, M = [[0,0],[1,0]] -> 1/6
    cases = (
        (float(average_gate_fidelity(perm, eye2)), 1 / 3),
        (float(average_gate_fidelity(perm, gate("X"))), 1 / 6),
        (float(leakage(perm, 2)), 1 / 2),
        (float(superop_average_gate_fidelity(unitary_to_superop(perm), eye2)), 1 / 3),
        (float(superop_average_gate_fidelity(unitary_to_superop(perm), gate("X"))), 1 / 6),
        (float(superop_leakage(unitary_to_superop(perm), 2)), 1 / 2),
        (float(average_gate_fidelity(jnp.eye(3), gate("X"))), 1 / 3),
        (float(average_gate_fidelity(jnp.eye(2), gate("SX"))), 2 / 3),
    )
    errs += [abs(a - b) for a, b in cases]
    tol = _arith_tol()
    return _res(
        "fidelity_closed_forms",
        "analytic",
        max(errs),
        tol,
        "Rz(theta) vs I (unitary and superoperator), leaky |1><->|2> permutation vs I / X and its "
        "leakage 1/2, F(I, X) = 1/3, F(I, SX) = 2/3 (tol 1e-9 in x64, 10 eps in float32)",
    )


def check_closed_open_agree() -> CheckResult:
    """Random I/Q pulses: closed U and open S (T1 = T2 = 1e6 us) give the same F and leakage."""
    closed = GateControl(model=Transmon(), target=gate("SX"))
    opened = GateControl(model=Transmon(t1_us=1e6, t2_us=1e6), target=gate("SX"))
    actions = _random_actions(4, closed.horizon, closed.action_bound)
    u = _map(closed, actions)
    s = _map(opened, actions)
    s_err = np.max(np.abs(s - np.asarray(unitary_to_superop(u))))
    # complex targets (SX, H, Rz) catch conjugation / transposition slips in either formula
    f_err = max(
        abs(float(average_gate_fidelity(u, t)) - float(superop_average_gate_fidelity(s, t)))
        for t in (gate("SX"), gate("H"), rz(0.7), gate("Y"))
    )
    l_err = abs(float(closed.leakage(u)) - float(opened.leakage(s)))
    f_env = float(closed.fidelity(u))
    err = max(s_err, f_err, l_err)
    return _res(
        "closed_open_agree",
        "cross_check",
        err,
        1e-5,
        f"max |S - conj(U) kron U| = {s_err:.1e}, max |dF| over SX, H, Rz(0.7), Y = {f_err:.1e}, "
        f"|dL| = {l_err:.1e} (3 levels, 16 random steps, F_SX = {f_env:.3f}, T1 = T2 = 1e6 us)",
    )


def check_info_observables_orientation() -> CheckResult:
    """``info`` populations equal ``|U[:, i0]|^2`` (closed) and ``diag S vec(|i0><i0|)`` (open).

    The reference map is built in numpy (independent of dynamiqs). Random I/Q pulse, 3 levels,
    detuning 10 MHz and initial level |1>: a transposed map would give ``|U[i0, :]|^2``.
    """
    levels, anh, det, horizon, dt = 3, -0.3, 0.01, 16, 2.0
    actions = np.asarray(_random_actions(5, horizon, 0.3))
    errs: list[float] = []
    gaps: list[float] = []
    for t1, t2 in ((None, None), (0.05, 0.04)):
        model = Transmon(
            levels=levels,
            anharmonicity_ghz=anh,
            detuning_ghz=det,
            t1_us=t1,
            t2_us=t2,
            initial="1",
        )
        env = GateControl(model=model, target=gate("X"), horizon=horizon, dt=dt, action_bound=1.0)
        state, _ = env.rollout(jnp.asarray(actions))
        info = env._info(state.qstate)
        ref = _np_map(levels, anh, det, actions, dt, t1, t2)
        if t1 is None:
            pops, wrong = np.abs(ref[:, 1]) ** 2, np.abs(ref[1, :]) ** 2
        else:
            rho0 = np.zeros((levels, levels), dtype=complex)
            rho0[1, 1] = 1.0
            rho = (ref @ rho0.reshape(-1, order="F")).reshape(levels, levels, order="F")
            pops = rho.diagonal().real
            wrong = (ref.T @ rho0.reshape(-1, order="F")).reshape(levels, levels, order="F")
            wrong = wrong.diagonal().real
        got = np.array([float(info[f"p{k}"]) for k in range(levels)])
        errs.append(float(np.max(np.abs(got - pops))))
        errs.append(abs(float(info["state_leakage"]) - pops[2]))
        gaps.append(float(np.max(np.abs(pops - wrong))))
    err = max(errs)
    tol = max(1e-9, 1e2 * float(jnp.finfo(jnp.asarray(1.0).dtype).eps))
    return _res(
        "info_observables_orientation",
        "cross_check",
        err if min(gaps) > 1e-3 else 1.0,
        tol,
        f"max |info p_k - numpy reference| = {err:.1e} (closed |U[:, 1]|^2 and open "
        f"diag(S vec(|1><1|)); 3 levels, 16 random I/Q steps, 10 MHz detuning; the transposed "
        f"map would differ by {min(gaps):.2f})",
    )


def check_sx_orientation() -> CheckResult:
    """A pi/2 rotation about x is SX, not its adjoint: ``SX = e^{i pi/4} Rx(pi/2)``."""
    horizon, dt = 4, 2.5
    w = (math.pi / 2) / (horizon * dt)
    env = GateControl(
        model=Transmon(levels=2), target=gate("SX"), horizon=horizon, dt=dt, action_bound=1.0
    )
    u = _map(env, jnp.tile(jnp.array([w, 0.0]), (horizon, 1)))
    sx = np.asarray(gate("SX"))
    # exp(-i (pi/4) sigma_x) = [[1, -i], [-i, 1]] / sqrt 2 and SX = e^{i pi/4} times it
    err = float(np.max(np.abs(np.exp(0.25j * math.pi) * u - sx)))
    adj = float(np.max(np.abs(np.exp(0.25j * math.pi) * u - sx.conj().T)))
    return _res(
        "sx_orientation",
        "analytic",
        err if adj > 0.5 else 1.0,
        2e-6,
        f"levels 2, Omega_I T = pi/2: max |e^(i pi/4) U - SX| = {err:.1e} (elementwise; "
        f"against SX^dag it is {adj:.2f})",
    )


def check_idle_coherence_fidelity() -> CheckResult:
    """Idle qubit for t: F_avg(I) = (2 F_pro + 1)/3, F_pro = (1 + e^{-t/T1} + 2 e^{-t/T2})/4."""
    t1, t2, horizon, dt = 0.04, 0.05, 16, 2.0  # us, us, steps, ns
    t = horizon * dt / 1000.0  # us
    f_pro = (1 + math.exp(-t / t1) + 2 * math.exp(-t / t2)) / 4
    exact = (2 * f_pro + 1) / 3
    errs = []
    for levels in (2, 3):
        env = GateControl(
            model=Transmon(levels=levels, t1_us=t1, t2_us=t2),
            target=gate("I"),
            horizon=horizon,
            dt=dt,
        )
        s = _map(env, jnp.zeros((horizon, 2)))
        errs.append(abs(float(env.fidelity(s)) - exact))
        errs.append(abs(float(superop_process_fidelity(s, gate("I"))) - f_pro))
    return _res(
        "idle_coherence_fidelity",
        "open_system",
        max(errs),
        1e-6,
        f"T1 = 40 ns, T2 = 50 ns, t = 32 ns: F_avg exact {exact:.6f}, F_pro exact {f_pro:.6f}; "
        "levels 2 and 3",
    )


def check_harmonic_t1_ladder() -> CheckResult:
    """From |2>, no drive: P2 = e^{-2t/T1}, P1 = 2 (e^{-t/T1} - e^{-2t/T1}) (rate n / T1)."""
    t1 = 0.05
    env = GateControl(model=Transmon(t1_us=t1, initial="2"), target=gate("I"), horizon=10, dt=3.0)
    s = _map(env, jnp.zeros((10, 2)))
    info = env._info(jnp.asarray(s))
    x = 30.0 / (1000.0 * t1)
    e2 = abs(float(info["p2"]) - math.exp(-2 * x))
    e1 = abs(float(info["p1"]) - 2 * (math.exp(-x) - math.exp(-2 * x)))
    return _res(
        "harmonic_t1_ladder",
        "open_system",
        max(e1, e2),
        1e-6,
        f"|P2 - e^(-2t/T1)| = {e2:.1e}, |P1 - 2(e^(-t/T1) - e^(-2t/T1))| = {e1:.1e}, "
        "T1 = 50 ns, t = 30 ns",
    )


def check_superop_trace_preserving_cp() -> CheckResult:
    """Open random pulse: S preserves trace and Hermiticity, and its Choi matrix is PSD."""
    env = GateControl(model=Transmon(t1_us=0.05, t2_us=0.04), target=gate("X"))
    s = _map(env, _random_actions(9, env.horizon, env.action_bound))
    d = env.model.dim
    rng = np.random.default_rng(0)
    errs = []
    for _ in range(5):
        rho = _random_dm(rng, d)
        out = (s @ rho.reshape(-1, order="F")).reshape(d, d, order="F")
        errs.append(abs(np.trace(out) - 1.0))
        errs.append(np.max(np.abs(out - out.conj().T)))
    # Choi matrix sum_ce |c><e| (x) E(|c><e|) from the column-stacking S
    e = s.reshape(d, d, d, d).transpose(1, 0, 3, 2)  # e[a, b, c, f] = E(|c><f|)[a, b]
    choi = e.transpose(2, 0, 3, 1).reshape(d * d, d * d)
    neg = max(0.0, -float(np.min(np.linalg.eigvalsh(0.5 * (choi + choi.conj().T)))))
    err = max(max(errs), neg)
    return _res(
        "superop_trace_preserving_cp",
        "constraints",
        err,
        1e-6,
        f"max |tr S(rho) - 1| and hermiticity over 5 random rho = {max(errs):.1e}; "
        f"Choi negativity = {neg:.1e} (3 levels, T1 = 50 ns, T2 = 40 ns)",
    )


def _drag_sweep(env: GateControl, amp: float, sigma: float, betas: Array) -> tuple[Array, Array]:
    def run(b: Array) -> tuple[Array, Array]:
        s, _ = env.rollout(gaussian_drag(env.horizon, env.dt, amp, sigma, b))
        return env.fidelity(s.qstate), env.leakage(s.qstate)

    return jax.jit(jax.vmap(run))(betas)


def _argmin_refined(
    env: GateControl, amp: float, sigma: float, theory: float, which: int
) -> tuple[float, float, float]:
    """Coarse grid on [0, 2 theory] then a fine grid around the minimum (resolution 0.25 %)."""

    def objective(betas: Array) -> np.ndarray:
        f, lk = _drag_sweep(env, amp, sigma, betas)
        return np.asarray(lk if which == 1 else 1.0 - f)

    coarse = jnp.linspace(0.0, 2.0 * theory, 41)
    y = objective(coarse)
    k = int(np.argmin(y))
    h = float(coarse[1] - coarse[0])
    fine = jnp.linspace(float(coarse[k]) - h, float(coarse[k]) + h, 41)
    yf = objective(fine)
    j = int(np.argmin(yf))
    return float(fine[j]), float(yf[j]), float(y[0])


def check_drag_coefficient() -> CheckResult:
    """12 ns Gaussian X, 3 levels: beta minima match -1/alpha (leakage), -1/(2 alpha) (fidelity)."""
    anh, gate_ns, dt = -0.34, 12.0, 0.25
    horizon, sigma = int(gate_ns / dt), gate_ns / 4
    env = GateControl(
        model=Transmon(levels=3, anharmonicity_ghz=anh),
        target=gate("X"),
        horizon=horizon,
        dt=dt,
        action_bound=10.0,
    )
    amp = gaussian_amplitude(math.pi, horizon, dt, sigma)
    b_l, b_p = drag_beta(anh, "leakage"), drag_beta(anh, "phase")
    bl_num, l_min, l_0 = _argmin_refined(env, amp, sigma, b_l, which=1)
    bp_num, inf_min, inf_0 = _argmin_refined(env, amp, sigma, b_p, which=0)
    rel = max(abs(bl_num / b_l - 1), abs(bp_num / b_p - 1))
    # require large improvements too: leakage / 100 and infidelity / 50 versus beta = 0
    gains_ok = l_min < l_0 / 100 and inf_min < inf_0 / 50
    err = rel if gains_ok else 1.0
    return _res(
        "drag_coefficient",
        "analytic",
        err,
        0.05,
        f"leakage-optimal beta / (-1/alpha) = {bl_num / b_l:.4f} (L {l_0:.2e} -> {l_min:.2e}); "
        f"fidelity-optimal beta / (-1/(2 alpha)) = {bp_num / b_p:.4f} "
        f"(1 - F {inf_0:.2e} -> {inf_min:.2e}); 12 ns, sigma 3 ns, dt 0.25 ns",
    )


def check_qutip_cross_check() -> CheckResult:
    """3-level transmon, random I/Q pulses vs QuTiP 5: open mesolve (rho) and closed propagator."""
    try:
        import qutip as qt
    except ImportError:
        return skipped(
            "qutip_cross_check",
            "cross_check",
            1e-5,
            "not run: qutip is not installed (pip install 'qenvs[dev]')",
            model=MODEL,
        )
    d, anh, det, t1, t2 = 3, -0.3, 0.004, 0.03, 0.025
    horizon, dt, bound = 12, 2.0, 0.4
    actions = _random_actions(21, horizon, bound)
    amps = np.clip(np.asarray(actions, dtype=np.float64), -bound, bound)
    m_open = Transmon(levels=d, anharmonicity_ghz=anh, detuning_ghz=det, t1_us=t1, t2_us=t2)
    env_o = GateControl(model=m_open, target=gate("X"), horizon=horizon, dt=dt, action_bound=bound)
    s = _map(env_o, actions)
    # independent QuTiP construction from the GHz / us parameters
    a = qt.destroy(d)
    n = a.dag() * a
    h0 = TWO_PI * det * n + 0.5 * TWO_PI * anh * n * (n - 1)
    hi, hq = 0.5 * (a + a.dag()), 0.5j * (a.dag() - a)
    g1 = 1 / (1000 * t1)
    gphi = 1 / (1000 * t2) - g1 / 2
    c_ops = [math.sqrt(g1) * a, math.sqrt(2 * gphi) * n]
    rng = np.random.default_rng(3)
    rho0 = _random_dm(rng, d)
    rho = qt.Qobj(rho0)
    opts = {"atol": 1e-12, "rtol": 1e-10, "nsteps": 100_000, "progress_bar": ""}
    u_qt = qt.qeye(d)
    for wi, wq in amps:
        h = h0 + float(wi) * hi + float(wq) * hq
        rho = qt.mesolve(h, rho, [0.0, dt], c_ops=c_ops, options=opts).states[-1]
        u_qt = (-1j * h * dt).expm() * u_qt
    rho_dq = (s @ rho0.reshape(-1, order="F")).reshape(d, d, order="F")
    e_open = np.max(np.abs(rho_dq - rho.full()))
    m_closed = Transmon(levels=d, anharmonicity_ghz=anh, detuning_ghz=det)
    env_c = GateControl(
        model=m_closed, target=gate("X"), horizon=horizon, dt=dt, action_bound=bound
    )
    u = _map(env_c, actions)
    e_closed = np.max(np.abs(u - u_qt.full()))
    err = max(e_open, e_closed)
    return _res(
        "qutip_cross_check",
        "cross_check",
        err,
        1e-5,
        f"open: max |rho_dq - rho_qutip| = {e_open:.1e} (random initial rho, T1 = 30 ns, "
        f"T2 = 25 ns); closed: max |U_dq - U_qutip| = {e_closed:.1e}; 12 random I/Q steps "
        f"(qutip {qt.__version__})",
    )


def check_solver_convergence() -> CheckResult:
    """Default Expm step propagators vs tight adaptive Tsit5, closed and open.

    Tolerance 1e-4: in float32 the Pade matrix exponential is off by about 6e-7 per
    2 ns step and the error accumulates coherently over the 16 steps (about 1e-5);
    in float64 the measured difference is about 2e-9.
    """
    tight = dq.method.Tsit5(rtol=1e-10, atol=1e-10)
    errs = []
    for model in (Transmon(), Transmon(t1_us=0.05, t2_us=0.04)):
        env_a = GateControl(model=model, target=gate("X"))
        env_b = GateControl(model=model, target=gate("X"), method=tight)
        actions = _random_actions(13, env_a.horizon, env_a.action_bound)
        errs.append(np.max(np.abs(_map(env_a, actions) - _map(env_b, actions))))
    return _res(
        "solver_convergence",
        "numerics",
        max(errs),
        1e-4,
        f"max |map_Expm - map_Tsit5(rtol=atol=1e-10)| at episode end: closed {errs[0]:.1e}, "
        f"open {errs[1]:.1e}",
    )


def check_env_return_equals_gain() -> CheckResult:
    """GateControl return equals F_final - F_initial (closed and open; F_initial(X) = 1/3)."""
    errs = []
    for model in (Transmon(), Transmon(t1_us=0.1, t2_us=0.08)):
        env = GateControl(model=model, target=gate("X"))
        actions = _random_actions(17, env.horizon, env.action_bound)
        state, rewards = env.rollout(actions)
        f_final = float(env.fidelity(state.qstate))
        errs.append(abs(float(rewards.sum()) - (f_final - 1 / 3)))
    return _res(
        "env_return_equals_gain",
        "control",
        max(errs),
        1e-6,
        f"|sum(rewards) - (F_final - 1/3)|: closed {errs[0]:.1e}, open {errs[1]:.1e}",
    )


CHECKS: tuple[Callable[[], CheckResult], ...] = (
    check_qubit_limit_gates,
    check_rabi_units,
    check_drift_spectrum,
    check_fidelity_closed_forms,
    check_closed_open_agree,
    check_info_observables_orientation,
    check_sx_orientation,
    check_idle_coherence_fidelity,
    check_harmonic_t1_ladder,
    check_superop_trace_preserving_cp,
    check_drag_coefficient,
    check_qutip_cross_check,
    check_solver_convergence,
    check_env_return_equals_gain,
)
