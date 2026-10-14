"""Validation checks for two coupled transmons (:class:`TwoTransmon`) and ``transmon/CRGate-v0``.

UNITS: time in ns, angular frequency in rad/ns, model parameters in GHz and us. Tensor order
control (x) target; computational states at indices ``(0, 1, d, d + 1)``. The QuTiP
cross-check needs ``qutip`` (``qenvs[dev]`` extra); without it that check is reported as
skipped instead of raising.
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
    leakage,
    superop_average_gate_fidelity,
    superop_leakage,
    unitary_to_superop,
)
from qenvs.models.transmon import Transmon, gaussian_amplitude, gaussian_drag
from qenvs.models.two_transmon import (
    TwoTransmon,
    static_zz_second_order,
    to_qubit_frame,
    zx_rate_first_order,
)
from qenvs.tasks.cross_resonance import cr_target, make_cr_task, zx_gate
from qenvs.validation.core import CheckResult, skipped
from qenvs.validation.core import result as _result

MODEL = "TwoTransmon"
TWO_PI = 2.0 * math.pi
#: Unequal anharmonicities so that swapping control and target parameters is detectable.
ANH_C, ANH_T, DET = -0.34, -0.30, -0.1
#: Checks that compare full ``9 x 9`` maps elementwise use 0.5 ns steps, so that
#: ``||H dt||_1 < 3.93`` and ``jax.scipy.linalg.expm`` needs no scaling and squaring. For
#: ``||H dt||_1`` between ``3.93 * 2^k`` and ``7.85 * 2^k`` the float32 expm of jax 0.10 is
#: off by up to about 1e-3 per step (it rounds the number of squarings down), mostly in the
#: phases of doubly excited states; see docs/models/cross_resonance.md.
_SMALL_DT = 0.5


def _res(name: str, category: str, error: float, tol: float, detail: str) -> CheckResult:
    return _result(name, category, error, tol, detail, model=MODEL)


def _eps() -> float:
    return float(jnp.finfo(jnp.asarray(1.0).dtype).eps)


def _arith_tol() -> float:
    """Tolerance for pure arithmetic: 1e-9 in float64, 10 machine epsilons in float32."""
    return max(1e-9, 10.0 * _eps())


def _env(model: TwoTransmon, horizon: int, dt: float, bound: float = 1.0) -> GateControl:
    """GateControl towards ZX(pi/2) (pulled back to the common frame) on the 4 qubit states."""
    return GateControl(
        model=model,
        target=cr_target(model, horizon * dt),
        horizon=horizon,
        dt=dt,
        action_bound=bound,
        subspace=model.computational_indices,
    )


def _map(env: GateControl, actions: Array) -> np.ndarray:
    state, _ = env.rollout(jnp.asarray(actions))
    return np.asarray(state.qstate.to_jax())


def _random_actions(key: int, horizon: int, bound: float, n: int = 4) -> Array:
    return jax.random.uniform(jax.random.PRNGKey(key), (horizon, n), minval=-bound, maxval=bound)


def _random_dm(rng: np.random.Generator, d: int) -> np.ndarray:
    x = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    rho = x @ x.conj().T
    return rho / np.trace(rho)


def _sub(u: np.ndarray, idx: tuple[int, ...]) -> np.ndarray:
    return u[np.ix_(idx, idx)]


# --------------------------------------------------------------------- checks
def check_frame_change_identity() -> CheckResult:
    """J = 0, no drive: the qubit-frame map ``exp(+i Delta n_c T) U_r`` is 1 on the 4 qubit states.

    ``T = 37 ns`` so that ``Delta T`` is not a multiple of pi: the opposite sign
    ``exp(-i Delta n_c T)`` leaves ``|e^{2 i Delta T} - 1|`` on the ``|1x>`` diagonal.
    """
    model = TwoTransmon(
        detuning_ghz=DET, anharmonicity_c_ghz=ANH_C, anharmonicity_t_ghz=ANH_T, coupling_ghz=0.0
    )
    horizon, dt = 4, 9.25
    duration = horizon * dt
    env = _env(model, horizon, dt)
    u = _map(env, jnp.zeros((horizon, 4)))
    idx = model.computational_indices
    uq = _sub(np.asarray(to_qubit_frame(model, u, duration)), idx)
    err = float(np.max(np.abs(uq - np.eye(4))))
    wrong = np.asarray(model.frame_unitary(-duration)) @ u
    gap = float(np.max(np.abs(_sub(wrong, idx) - np.eye(4))))
    f_env = float(env.fidelity(u))
    f_id = float(average_gate_fidelity(uq, zx_gate(), subspace=None))
    err_f = abs(f_env - f_id)
    return _res(
        "frame_change_identity",
        "analytic",
        max(err, err_f) if gap > 0.5 else 1.0,
        1e-5,
        f"J = 0, idle 37 ns, Delta/2pi = -100 MHz: max |U_q - 1| on |00>,|01>,|10>,|11> = "
        f"{err:.1e}; with exp(-i Delta n_c T) instead it is {gap:.2f}; env fidelity (pulled-back "
        f"target) equals F(U_q, ZX) to {err_f:.1e}",
    )


def check_j0_decoupling() -> CheckResult:
    """J = 0: the map factorises into two single :class:`Transmon` maps; DRAG on the target is X.

    (a) Random pulses on all four channels: ``U = U_c (x) U_t`` with ``U_c`` from
    ``Transmon(detuning = Delta)`` driven by the CR channels and ``U_t`` from
    ``Transmon(detuning = 0)`` driven by the target channels (no entanglement).
    (b) The Gaussian DRAG X pulse of ``transmon/XGate-v0`` on the target alone: the
    qubit-frame map on the 4 qubit states equals ``1 (x) V`` with ``V`` the single-transmon
    map on ``|0>, |1>``, and its fidelity to ``1 (x) X`` is the single-transmon value.
    """
    model = TwoTransmon(
        detuning_ghz=DET, anharmonicity_c_ghz=ANH_C, anharmonicity_t_ghz=ANH_T, coupling_ghz=0.0
    )
    horizon, dt = 16, 0.5  # ||H dt||_1 < 3.9: see _SMALL_DT
    actions = _random_actions(31, horizon, 0.3)
    u = _map(_env(model, horizon, dt), actions)
    tc = GateControl(
        model=Transmon(levels=3, anharmonicity_ghz=ANH_C, detuning_ghz=DET),
        target=jnp.eye(2),
        horizon=horizon,
        dt=dt,
        action_bound=1.0,
    )
    tt = GateControl(
        model=Transmon(levels=3, anharmonicity_ghz=ANH_T, detuning_ghz=0.0),
        target=jnp.eye(2),
        horizon=horizon,
        dt=dt,
        action_bound=1.0,
    )
    prod = np.kron(_map(tc, actions[:, :2]), _map(tt, actions[:, 2:]))
    e_fact = float(np.max(np.abs(u - prod)))
    # coupled control: the same pulse does not factorise
    coupled = _map(_env(_replace(model, coupling_ghz=0.003), horizon, dt), actions)
    gap = float(np.max(np.abs(coupled - prod)))
    # (b) DRAG X on the target alone
    h2, dt2 = 16, 2.0
    sigma = h2 * dt2 / 4
    amp = gaussian_amplitude(math.pi, h2, dt2, sigma)
    drag = gaussian_drag(h2, dt2, amp, sigma, -1.0 / (2.0 * TWO_PI * ANH_T))
    acts = jnp.concatenate([jnp.zeros((h2, 2)), drag], axis=1)
    u2 = _map(_env(model, h2, dt2), acts)
    uq = _sub(np.asarray(to_qubit_frame(model, u2, h2 * dt2)), model.computational_indices)
    single = GateControl(
        model=Transmon(levels=3, anharmonicity_ghz=ANH_T),
        target=jnp.asarray([[0, 1], [1, 0]], dtype=complex),
        horizon=h2,
        dt=dt2,
        action_bound=1.0,
    )
    v = _map(single, drag)
    e_x = float(np.max(np.abs(uq - np.kron(np.eye(2), v[:2, :2]))))
    ix = np.kron(np.eye(2), np.array([[0, 1], [1, 0]]))
    f_two = float(average_gate_fidelity(uq, ix))
    f_one = float(single.fidelity(v))
    m = np.array([[0, 1], [1, 0]]) @ v[:2, :2]
    a, b = float(np.sum(np.abs(m) ** 2)), float(abs(np.trace(m)) ** 2)
    e_f = abs(f_two - (2 * a + 4 * b) / 20)
    err = max(e_fact, e_x, e_f)
    return _res(
        "j0_decoupling",
        "analytic",
        err if gap > 1e-3 else 1.0,
        1e-5,
        f"(a) 16 random 4-channel steps of 0.5 ns: max |U - U_c (x) U_t| = {e_fact:.1e} "
        f"(with J/2pi = 3 MHz it is {gap:.1e}); (b) DRAG X on the target: "
        f"max |U_q - 1 (x) V| = {e_x:.1e}, "
        f"1 - F(1 (x) X) = {1 - f_two:.1e} (single transmon 1 - F(X) = {1 - f_one:.1e}), "
        f"F matches (2 Tr(mm^dag) + 4 |Tr m|^2)/20 to {e_f:.1e}",
    )


def _replace(model: TwoTransmon, **changes: object) -> TwoTransmon:
    """A copy of ``model`` with some fields changed."""
    fields = {
        name: getattr(model, name)
        for name in (
            "levels",
            "detuning_ghz",
            "anharmonicity_c_ghz",
            "anharmonicity_t_ghz",
            "coupling_ghz",
            "t1_c_us",
            "t2_c_us",
            "t1_t_us",
            "t2_t_us",
            "initial",
        )
    }
    fields.update(changes)
    return TwoTransmon(**fields)  # type: ignore[arg-type]


def check_static_zz() -> CheckResult:
    """Static ZZ, exact diagonalisation vs ``2 J^2 [1/(Delta - alpha_t) - 1/(Delta + alpha_c)]``.

    Relative error at J/2pi = 3 MHz (J / |Delta| = 0.03); the fourth-order remainder is about
    4e-4 of zeta in float64. Unequal anharmonicities so that alpha_c <-> alpha_t is caught.
    """
    errs, lines = [], []
    for j in (0.003, 0.005):
        m = TwoTransmon(
            detuning_ghz=DET, anharmonicity_c_ghz=ANH_C, anharmonicity_t_ghz=ANH_T, coupling_ghz=j
        )
        exact = float(m.static_zz()) / TWO_PI
        pert = static_zz_second_order(DET, ANH_C, ANH_T, j)
        rel = exact / pert - 1.0
        errs.append(abs(rel) if j == 0.003 else 0.0)
        lines.append(
            f"J = {j * 1e3:.0f} MHz: exact {exact * 1e6:.3f} kHz, 2nd order "
            f"{pert * 1e6:.3f} kHz, rel {rel:.1e}"
        )
    return _res(
        "static_zz",
        "analytic",
        max(errs),
        5e-3,
        "; ".join(lines) + " (Delta/2pi = -100 MHz, alpha_c/2pi = -340 MHz, alpha_t/2pi = "
        "-300 MHz; zeta = E11 - E10 - E01 + E00; tol on the 3 MHz case)",
    )


def _block_rotation(b: np.ndarray) -> np.ndarray:
    """Rotation vector ``theta n`` of the unitary part of a 2 x 2 block (``exp(-i theta n.s/2)``).

    Polar part via SVD, then the SU(2) representative with non-negative real trace.
    """
    w, _, vh = np.linalg.svd(b)
    u = w @ vh
    u = u / np.sqrt(np.linalg.det(u))
    if np.real(np.trace(u)) < 0:
        u = -u
    c = min(1.0, max(-1.0, float(np.real(np.trace(u))) / 2))
    half = math.acos(c)  # theta / 2
    paulis = (
        np.array([[0, 1], [1, 0]]),
        np.array([[0, -1j], [1j, 0]]),
        np.array([[1, 0], [0, -1]]),
    )
    s = math.sin(half)
    if s < 1e-12:
        return np.zeros(3)
    # u = cos(theta/2) 1 - i sin(theta/2) n.sigma  ->  n_k = i tr(u sigma_k) / (2 sin(theta/2))
    n = np.array([np.real(1j * np.trace(u @ p)) / (2 * s) for p in paulis])
    return 2 * half * n


def check_zx_rate_weak_drive() -> CheckResult:
    """Weak constant CR drive: ZX and IX rates from the propagator vs first order (Magesan 2020).

    J/2pi = 2 MHz, Omega/2pi = 5 MHz, 2 us of constant drive on the control I channel. The
    target rotation for the control in ``|c>`` is read from the ``|c0>, |c1>`` block of the
    qubit-frame map; ``nu_ZX = (w_0 - w_1)/2``, ``nu_IX = (w_0 + w_1)/2`` (x components).
    """
    j, rabi = 0.002, 0.005
    m = TwoTransmon(
        detuning_ghz=DET, anharmonicity_c_ghz=ANH_C, anharmonicity_t_ghz=ANH_T, coupling_ghz=j
    )
    horizon, dt = 100, 20.0
    duration = horizon * dt
    actions = jnp.zeros((horizon, 4)).at[:, 0].set(TWO_PI * rabi)
    u = _map(_env(m, horizon, dt), actions)
    uq = np.asarray(to_qubit_frame(m, u, duration))
    i00, i01, i10, i11 = m.computational_indices
    w0 = _block_rotation(uq[np.ix_((i00, i01), (i00, i01))])[0] / duration
    w1 = _block_rotation(uq[np.ix_((i10, i11), (i10, i11))])[0] / duration
    zx_num, ix_num = 0.5 * (w0 - w1) / TWO_PI, 0.5 * (w0 + w1) / TWO_PI
    zx_th, ix_th = zx_rate_first_order(DET, ANH_C, j, rabi)
    e_zx, e_ix = zx_num / zx_th - 1.0, ix_num / ix_th - 1.0
    return _res(
        "zx_rate_weak_drive",
        "analytic",
        max(abs(e_zx), abs(e_ix)),
        0.01,
        f"nu_ZX = {zx_num * 1e6:.3f} kHz vs -(J Omega/Delta) alpha_c/(alpha_c + Delta) = "
        f"{zx_th * 1e6:.3f} kHz (rel {e_zx:.1e}); nu_IX = {ix_num * 1e6:.3f} kHz vs "
        f"-J Omega/(Delta + alpha_c) = {ix_th * 1e6:.3f} kHz (rel {e_ix:.1e}); J/2pi = 2 MHz, "
        "Omega/2pi = 5 MHz, 2 us",
    )


def check_subspace_fidelity_closed_forms() -> CheckResult:
    """Closed forms on the two-qutrit subspace ``(0, 1, 3, 4)`` (unitary and superoperator).

    F(1, ZX(pi/2)) = 3/5; F(CZ, 1) = 2/5; embedded ZX(pi/2) is perfect (and differs on the
    default first-4-levels subspace); the |11> <-> |20> swap has L = 1/4 and F(., 1) = 3/5.
    """
    idx = (0, 1, 3, 4)
    eye9 = np.eye(9, dtype=complex)
    zx = np.asarray(zx_gate())
    emb = eye9.copy()
    emb[np.ix_(idx, idx)] = zx
    emb[8, 8] = np.exp(0.3j)  # a phase outside the subspace does not matter
    cz = np.diag([1, 1, 1, -1]).astype(complex)
    emb_cz = eye9.copy()
    emb_cz[np.ix_(idx, idx)] = cz
    swap = eye9[:, [0, 1, 2, 3, 6, 5, 4, 7, 8]]  # |11> (index 4) <-> |20> (index 6)
    eye4 = np.eye(4, dtype=complex)
    cases = []
    for fid, lk in (
        (average_gate_fidelity, leakage),
        (
            lambda u, t, subspace: superop_average_gate_fidelity(
                unitary_to_superop(u), t, subspace=subspace
            ),
            lambda u, k=2, subspace=None: superop_leakage(
                unitary_to_superop(u), k, subspace=subspace
            ),
        ),
    ):
        cases += [
            (float(fid(eye9, zx, subspace=idx)), 3 / 5),
            (float(fid(emb, zx, subspace=idx)), 1.0),
            (float(fid(emb_cz, eye4, subspace=idx)), 2 / 5),
            (float(fid(swap, eye4, subspace=idx)), 3 / 5),
            (float(lk(swap, 4, subspace=idx)), 1 / 4),
            (float(lk(emb, 4, subspace=idx)), 0.0),
        ]
    errs = [abs(a - b) for a, b in cases]
    # the default subspace (first 4 levels: |00>, |01>, |02>, |10>) gives a different answer
    f_default = float(average_gate_fidelity(emb, zx))
    gap = abs(f_default - 1.0)
    return _res(
        "subspace_fidelity_closed_forms",
        "analytic",
        max(errs) if gap > 0.1 else 1.0,
        _arith_tol(),
        f"F(1, ZX) = 3/5, F(ZX embedded, ZX) = 1, F(CZ, 1) = 2/5, |11> <-> |20>: F = 3/5 and "
        f"L = 1/4, unitary and superoperator forms; max error {max(errs):.1e}; on the default "
        f"first-4-levels subspace the embedded ZX gives F = {f_default:.3f}",
    )


def check_qutip_cross_check() -> CheckResult:
    """Two qutrits, random 4-channel pulse vs QuTiP 5: open mesolve (rho) and closed propagator."""
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
    d, j = 3, 0.003
    t1c, t2c, t1t, t2t = 0.03, 0.025, 0.04, 0.02
    horizon, dt, bound = 16, 0.5, 0.3  # ||H dt||_1 < 3.9: see _SMALL_DT
    actions = _random_actions(41, horizon, bound)
    amps = np.clip(np.asarray(actions, dtype=np.float64), -bound, bound)
    kw = {"detuning_ghz": DET, "anharmonicity_c_ghz": ANH_C, "anharmonicity_t_ghz": ANH_T}
    m_open = TwoTransmon(coupling_ghz=j, t1_c_us=t1c, t2_c_us=t2c, t1_t_us=t1t, t2_t_us=t2t, **kw)
    s = _map(_env(m_open, horizon, dt, bound), actions)
    a = qt.tensor(qt.destroy(d), qt.qeye(d))
    b = qt.tensor(qt.qeye(d), qt.destroy(d))
    nc, nt = a.dag() * a, b.dag() * b
    h0 = (
        TWO_PI * DET * nc
        + 0.5 * TWO_PI * ANH_C * nc * (nc - 1)
        + 0.5 * TWO_PI * ANH_T * nt * (nt - 1)
        + TWO_PI * j * (a.dag() * b + a * b.dag())
    )
    ctrl = [0.5 * (a + a.dag()), 0.5j * (a.dag() - a), 0.5 * (b + b.dag()), 0.5j * (b.dag() - b)]
    c_ops = []
    for low, num, t1, t2 in ((a, nc, t1c, t2c), (b, nt, t1t, t2t)):
        g1 = 1 / (1000 * t1)
        gphi = 1 / (1000 * t2) - g1 / 2
        c_ops += [math.sqrt(g1) * low, math.sqrt(2 * gphi) * num]
    rng = np.random.default_rng(5)
    rho0 = _random_dm(rng, d * d)
    dims = [[d, d], [d, d]]
    rho = qt.Qobj(rho0, dims=dims)
    opts = {"atol": 1e-12, "rtol": 1e-10, "nsteps": 100_000, "progress_bar": ""}
    u_qt = qt.tensor(qt.qeye(d), qt.qeye(d))
    for row in amps:
        h = h0 + sum(float(x) * c for x, c in zip(row, ctrl, strict=True))
        rho = qt.mesolve(h, rho, [0.0, dt], c_ops=c_ops, options=opts).states[-1]
        u_qt = (-1j * h * dt).expm() * u_qt
    rho_dq = (s @ rho0.reshape(-1, order="F")).reshape(d * d, d * d, order="F")
    e_open = float(np.max(np.abs(rho_dq - rho.full())))
    u = _map(_env(TwoTransmon(coupling_ghz=j, **kw), horizon, dt, bound), actions)
    e_closed = float(np.max(np.abs(u - u_qt.full())))
    return _res(
        "qutip_cross_check",
        "cross_check",
        max(e_open, e_closed),
        1e-5,
        f"open: max |rho_dq - rho_qutip| = {e_open:.1e} (random rho0, T1/T2 = 30/25 ns control, "
        f"40/20 ns target); closed: max |U_dq - U_qutip| = {e_closed:.1e}; 16 random 4-channel "
        f"steps of 0.5 ns, J/2pi = 3 MHz (qutip {qt.__version__})",
    )


def check_superop_cptp() -> CheckResult:
    """Open random pulse: S preserves trace and Hermiticity, and its Choi matrix is PSD."""
    m = TwoTransmon(t1_c_us=0.05, t2_c_us=0.04, t1_t_us=0.06, t2_t_us=0.03)
    env = _env(m, 6, 5.0, 0.3)
    s = _map(env, _random_actions(43, 6, 0.3))
    dim = m.dim
    rng = np.random.default_rng(1)
    errs = []
    for _ in range(4):
        rho = _random_dm(rng, dim)
        out = (s @ rho.reshape(-1, order="F")).reshape(dim, dim, order="F")
        errs.append(abs(np.trace(out) - 1.0))
        errs.append(float(np.max(np.abs(out - out.conj().T))))
    e = s.reshape(dim, dim, dim, dim).transpose(1, 0, 3, 2)  # e[a, b, c, f] = E(|c><f|)[a, b]
    choi = e.transpose(2, 0, 3, 1).reshape(dim * dim, dim * dim)
    neg = max(0.0, -float(np.min(np.linalg.eigvalsh(0.5 * (choi + choi.conj().T)))))
    err = max(max(errs), neg)
    return _res(
        "superop_cptp",
        "constraints",
        err,
        1e-5,
        f"max |tr S(rho) - 1| and hermiticity over 4 random rho = {max(errs):.1e}; Choi "
        f"negativity = {neg:.1e} (two qutrits, 6 random steps of 5 ns, T1/T2 50/40 and 60/30 ns)",
    )


def check_solver_convergence() -> CheckResult:
    """Default Expm step propagators vs tight adaptive Tsit5, closed and open (16 x 0.5 ns)."""
    tight = dq.method.Tsit5(rtol=1e-10, atol=1e-10)
    errs = []
    for model in (TwoTransmon(), TwoTransmon(t1_c_us=0.05, t2_c_us=0.04, t1_t_us=0.06)):
        env_a = _env(model, 16, 0.5, 0.4)
        env_b = GateControl(
            model=model,
            target=env_a.target,
            horizon=16,
            dt=0.5,
            action_bound=0.4,
            subspace=env_a.subspace,
            method=tight,
        )
        actions = _random_actions(47, 16, 0.4)
        errs.append(float(np.max(np.abs(_map(env_a, actions) - _map(env_b, actions)))))
    return _res(
        "solver_convergence",
        "numerics",
        max(errs),
        1e-4,
        f"max |map_Expm - map_Tsit5(rtol=atol=1e-10)| after 16 random steps of 0.5 ns: "
        f"closed {errs[0]:.1e}, open {errs[1]:.1e}",
    )


def check_env_return_equals_gain() -> CheckResult:
    """Task env: return = F_final - F_initial, and F_final = F(qubit-frame map, ZX(pi/2)).

    ``horizon = 12, dt = 4.3 ns`` so that the frame change is not trivial
    (``Delta T / 2 pi = -5.16``); closed and open.
    """
    errs = []
    for kw in ({}, {"t1_c_us": 0.1, "t2_c_us": 0.08, "t1_t_us": 0.12, "t2_t_us": 0.09}):
        env = make_cr_task(horizon=12, dt=4.3, **kw)
        assert isinstance(env, GateControl)
        actions = _random_actions(53, 12, float(env.action_bound))
        state, rewards = env.rollout(actions)
        f_final = float(env.fidelity(state.qstate))
        f0 = float(env.fidelity(jnp.eye(env.map_dim, dtype=complex)))
        mq = to_qubit_frame(env.model, state.qstate.to_jax(), 12 * 4.3)  # type: ignore[arg-type]
        idx = env.subspace
        f_q = (
            superop_average_gate_fidelity(mq, zx_gate(), subspace=idx)
            if env.model.is_open
            else average_gate_fidelity(mq, zx_gate(), subspace=idx)
        )
        errs.append(abs(float(rewards.sum()) - (f_final - f0)))
        errs.append(abs(f_final - float(f_q)))
    return _res(
        "env_return_equals_gain",
        "control",
        max(errs),
        1e-5,
        f"|sum(rewards) - (F_T - F_0)| and |F_T - F(exp(i Delta n_c T) U, ZX)|: closed "
        f"{max(errs[:2]):.1e}, open {max(errs[2:]):.1e}",
    )


CHECKS: tuple[Callable[[], CheckResult], ...] = (
    check_frame_change_identity,
    check_j0_decoupling,
    check_static_zz,
    check_zx_rate_weak_drive,
    check_subspace_fidelity_closed_forms,
    check_qutip_cross_check,
    check_superop_cptp,
    check_solver_convergence,
    check_env_return_equals_gain,
)
