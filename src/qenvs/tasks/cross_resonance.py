"""Cross-resonance (CR) two-qubit gate task on two coupled transmons, hardware units (ns, GHz, us).

UNITS: time in ns, angular frequency (Hamiltonian, actions, ``action_bound``) in rad/ns,
user-facing frequencies in GHz (converted with ``2 pi``), T1 / T2 in microseconds.

The target is ``ZX(pi/2) = exp(-i (pi/4) Z (x) X)`` in control (x) target order, defined in
the bare qubit frames at the end of the gate. The environment simulates in the common
frame rotating at the target frequency; since the frame change ``exp(+i Delta n_c T)`` is
diagonal and maps the computational subspace to itself, the fidelity of the qubit-frame
map against ``ZX(pi/2)`` equals the fidelity of the common-frame map against
``D^dag ZX(pi/2)`` with ``D = diag(1, 1, e^{i Delta T}, e^{i Delta T})``
(:func:`cr_target`). That pulled-back target is what ``env.target`` holds.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from qenvs.envs.base import Environment
from qenvs.envs.gate import GateControl
from qenvs.models.transmon import ghz_to_rad_per_ns
from qenvs.models.two_transmon import TwoTransmon, to_qubit_frame, zx_rate_first_order
from qenvs.registry import register

TASK_ID = "transmon/CRGate-v0"
DEFAULT_HORIZON = 64
DEFAULT_DT_NS = 5.0
DEFAULT_MAX_RABI_GHZ = 0.06
DEFAULT_ACTION_BOUND = 2.0 * math.pi * DEFAULT_MAX_RABI_GHZ  # rad/ns

#: Units of the CR task (also shown by ``qenvs info``).
UNITS: dict[str, str] = {
    "time": "ns (dt, horizon * dt); hbar = 1",
    "frequency": "angular frequency in rad/ns inside the Hamiltonian; kwargs *_ghz are ordinary "
    "frequencies in GHz, converted with 2 pi",
    "control": "actions = [Omega_cI, Omega_cQ, Omega_tI, Omega_tQ] in rad/ns (CR drive on the "
    "control, drive on the target, both at the target frequency), clipped to +-action_bound; "
    "Omega / (2 pi) is a Rabi frequency in GHz",
    "coherence": "t1_c_us, t2_c_us, t1_t_us, t2_t_us in microseconds",
}

MODEL_FLOAT_KWARGS = ("detuning_ghz", "anharmonicity_c_ghz", "anharmonicity_t_ghz", "coupling_ghz")
MODEL_COHERENCE_KWARGS = ("t1_c_us", "t2_c_us", "t1_t_us", "t2_t_us")


def zx_gate(theta: float = math.pi / 2) -> Array:
    """``exp(-i (theta/2) Z (x) X)`` in control (x) target order, basis ``|00>, |01>, |10>, |11>``.

    ``Z = diag(1, -1)`` (``|0>`` is the ground state), so the target is rotated by ``+theta``
    about x when the control is in ``|0>`` and by ``-theta`` when it is in ``|1>``.
    """
    z = jnp.diag(jnp.array([1.0, -1.0]))
    x = jnp.array([[0.0, 1.0], [1.0, 0.0]])
    zx = jnp.kron(z, x).astype(complex)
    return math.cos(theta / 2) * jnp.eye(4, dtype=complex) - 1j * math.sin(theta / 2) * zx


def cr_target(model: TwoTransmon, duration: float, theta: float = math.pi / 2) -> Array:
    """Common-frame equivalent ``D^dag ZX(theta)`` of the qubit-frame target after ``duration`` ns.

    ``D`` is the frame change ``exp(+i Delta n_c T)`` restricted to the computational states.
    """
    idx = np.asarray(model.computational_indices)
    d = jnp.diag(model.frame_unitary(duration))[idx]
    return jnp.conj(d)[:, None] * zx_gate(theta)


def _optional_float(v: object) -> float | None:
    return None if v is None else float(v)  # type: ignore[arg-type]


def make_cr_task(**kw: object) -> Environment:
    """Factory of ``transmon/CRGate-v0``: :class:`GateControl` on :class:`TwoTransmon`."""
    model_kw: dict[str, object] = {}
    if "levels" in kw:
        model_kw["levels"] = int(kw.pop("levels"))  # type: ignore[call-overload]
    for name in MODEL_FLOAT_KWARGS:
        if name in kw:
            model_kw[name] = float(kw.pop(name))  # type: ignore[arg-type]
    for name in MODEL_COHERENCE_KWARGS:
        if name in kw:
            model_kw[name] = _optional_float(kw.pop(name))
    if "max_rabi_ghz" in kw:
        if "action_bound" in kw:
            msg = "give either action_bound (rad/ns) or max_rabi_ghz (GHz), not both"
            raise ValueError(msg)
        kw["action_bound"] = float(ghz_to_rad_per_ns(float(kw.pop("max_rabi_ghz"))))  # type: ignore[arg-type]
    kw.setdefault("horizon", DEFAULT_HORIZON)
    kw.setdefault("dt", DEFAULT_DT_NS)
    kw.setdefault("action_bound", DEFAULT_ACTION_BOUND)
    model = TwoTransmon(**model_kw)  # type: ignore[arg-type]
    duration = int(kw["horizon"]) * float(kw["dt"])  # type: ignore[call-overload, arg-type]
    return GateControl(
        model=model,
        target=cr_target(model, duration),
        subspace=model.computational_indices,
        **kw,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------- reference pulse
def flat_top(horizon: int, dt: float, ramp: float, width: float, oversample: int = 16) -> Array:
    """Unit-height flat-top envelope with ``sin^2`` edges, averaged over each control step.

    The continuous envelope rises over ``ramp`` ns from ``t = 0``, stays at 1 for ``width``
    ns and falls over ``ramp`` ns; it is zero afterwards (``2 ramp + width <= horizon dt``).
    Step values are averages over ``oversample`` points per step, so ``width`` acts
    continuously. Returns shape ``(horizon,)``.
    """
    t = (jnp.arange(horizon * oversample) + 0.5) * (dt / oversample)
    total = 2.0 * ramp + width
    up = jnp.sin(0.5 * jnp.pi * jnp.clip(t / ramp, 0.0, 1.0)) ** 2
    down = jnp.sin(0.5 * jnp.pi * jnp.clip((total - t) / ramp, 0.0, 1.0)) ** 2
    g = jnp.where(t < total, jnp.minimum(up, down), 0.0)
    return g.reshape(horizon, oversample).mean(axis=1)


def conditional_angles(u_q: Array, model: TwoTransmon) -> Array:
    """Target rotation angles about x, ``(theta_0, theta_1)``, for the control in |0> and |1>.

    ``u_q`` is a qubit-frame propagator; ``theta_c = atan2(-<Y>, <Z>)`` of the target state
    reached from ``|c 0>`` (projected on the computational states of control level ``c``).
    For ``exp(-i (pi/4) ZX)``, ``theta_0 = pi/2`` and ``theta_1 = -pi/2``.
    """
    i00, i01, i10, i11 = model.computational_indices
    out = []
    for start, pair in ((i00, (i00, i01)), (i10, (i10, i11))):
        a0, a1 = u_q[..., pair[0], start], u_q[..., pair[1], start]
        z = jnp.abs(a0) ** 2 - jnp.abs(a1) ** 2
        y = 2.0 * jnp.imag(jnp.conj(a0) * a1)  # <Y> = 2 Im(conj(a0) a1)
        out.append(jnp.arctan2(-y, z))
    return jnp.stack(out, axis=-1)


def _closed(model: TwoTransmon) -> TwoTransmon:
    return eqx.tree_at(
        lambda m: (m.t1_c_us, m.t2_c_us, m.t1_t_us, m.t2_t_us),
        model,
        (None, None, None, None),
        is_leaf=lambda x: x is None,
    )


def cancellation_ratio(model: TwoTransmon) -> float:
    """Leading-order ratio ``c`` of the target tone to the CR tone that cancels ``nu_IX``.

    ``c = J / (Delta + alpha_c)`` for ``levels >= 3``: ``nu_IX = -J Omega / (Delta + alpha_c)``
    (Magesan and Gambetta 2020; only the control anharmonicity enters). With ``levels = 2``
    the control has no ``|2>`` level, which is the ``alpha_c -> -inf`` limit: dressing the
    control qubit by the drive gives ``H_eff = -(J Omega / 2 Delta) Z (x) X`` and no IX
    term (``nu_IX = 0``), so ``c = 0``. Using the three-level value there would add an IX
    term instead of cancelling one.
    """
    if model.levels < 3:
        return 0.0
    return model.coupling_ghz / (model.detuning_ghz + model.anharmonicity_c_ghz)


def cr_amplitude_sign(model: TwoTransmon) -> float:
    """Sign of the CR amplitude that rotates the target by ``+pi/2`` for the control in ``|0>``.

    It is the sign of the first-order ``nu_ZX`` (:func:`zx_rate_first_order`):
    ``-sign(Delta) sign(alpha_c / (alpha_c + Delta))`` for ``levels >= 3`` and, with the
    ``alpha_c -> -inf`` limit of that expression (no ``|2>`` level), ``-sign(Delta)`` for
    ``levels = 2``. With ``alpha_c < 0`` it is ``+1`` for ``Delta < 0`` and ``-1`` for
    ``Delta > 0``, unless ``Delta`` lies beyond ``-alpha_c``.
    """
    d, ac = model.detuning_ghz, model.anharmonicity_c_ghz
    if model.levels < 3:
        return 1.0 if d <= 0 else -1.0
    zx, _ = zx_rate_first_order(d, ac, model.coupling_ghz, 1.0)
    return 1.0 if zx >= 0 else -1.0


def reference_actions(
    env: GateControl, n_widths: int = 16, ramp_ns: float = 30.0, bisections: int = 24
) -> Array:
    """Calibrated flat-top CR pulse with a target cancellation tone, shape ``(horizon, 4)``.

    * CR drive on the control I channel: ``s A f(t)`` with ``f`` = :func:`flat_top`
      (``sin^2`` edges of ``ramp_ns``, capped at a quarter of the gate time), ``A >= 0``
      and ``s`` = :func:`cr_amplitude_sign`, the sign of the first-order ``nu_ZX``
      (``+1`` at the default ``Delta = -0.1`` GHz, ``-1`` for ``Delta = +0.1`` GHz), so the
      calibration reaches the ``+pi/2`` conditional rotation for either sign of ``Delta``.
    * Cancellation tone on the target I channel: ``c s A f(t)`` with the leading-order ratio
      ``c = `` :func:`cancellation_ratio` (``J / (Delta + alpha_c)`` for ``levels >= 3``,
      ``0`` for ``levels = 2``, where ``nu_IX = 0``), which cancels ``nu_IX`` (Magesan and
      Gambetta 2020).
    * For each flat-top width on a grid of ``n_widths`` values, ``A`` is calibrated by
      ``bisections`` bisection steps on ``[0, action_bound]`` so that the conditional
      rotation of the target is ``pi/2`` (``(theta_0 - theta_1) / 2 = pi/2``,
      :func:`conditional_angles`) in the closed (coherent) model; the grid is then refined
      once around the best width, and the width with the highest average gate fidelity
      is returned.
      The width matters because the control's Stark shift (``ZI``) grows with the integral
      of ``A^2`` and there are no virtual-Z frame updates to remove it.

    Calibration always uses the closed model, even when ``env`` has T1 / T2. Widths that
    need ``A > action_bound`` are skipped. Raises ``ValueError`` if no width works.
    """
    model = env.model
    if not isinstance(model, TwoTransmon):
        msg = f"reference_actions needs a TwoTransmon model, got {type(model).__name__}"
        raise ValueError(msg)
    closed = GateControl(
        model=_closed(model),
        target=env.target,
        horizon=env.horizon,
        dt=env.dt,
        action_bound=env.action_bound,
        subspace=env.subspace,
    )
    horizon, dt, bound = int(env.horizon), float(env.dt), float(env.action_bound)
    duration = horizon * dt
    ramp = min(ramp_ns, duration / 4.0)
    ratio = cancellation_ratio(model)
    sign = cr_amplitude_sign(model)

    def pulse(amp: Array, width: Array) -> Array:
        """CR amplitude ``sign * amp`` (``amp >= 0``) with the cancellation tone."""
        f = flat_top(horizon, dt, ramp, width)
        zero = jnp.zeros_like(f)
        return jnp.stack([sign * amp * f, zero, sign * ratio * amp * f, zero], axis=-1)

    def evaluate(amp: Array, width: Array) -> tuple[Array, Array]:
        state, _ = closed.rollout(pulse(amp, width))
        u = state.qstate.to_jax()
        th = conditional_angles(to_qubit_frame(model, u, duration), model)
        return 0.5 * (th[0] - th[1]) - 0.5 * jnp.pi, closed.fidelity(u)

    batch = jax.jit(jax.vmap(evaluate))

    def calibrate(widths: Array) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bisection on A for every width at once; returns amplitudes, fidelities, validity."""
        lo, hi = jnp.zeros_like(widths), jnp.full_like(widths, bound)
        feasible = np.asarray(batch(hi, widths)[0]) > 0
        for _ in range(bisections):
            mid = 0.5 * (lo + hi)
            below = batch(mid, widths)[0] < 0
            lo, hi = jnp.where(below, mid, lo), jnp.where(below, hi, mid)
        amps = 0.5 * (lo + hi)
        err, fid = (np.asarray(x) for x in batch(amps, widths))
        return np.asarray(amps), fid, feasible & (np.abs(err) < 1e-4)

    w_max = duration - 2 * ramp
    coarse = jnp.linspace(0.25 * w_max, w_max, n_widths)
    amps, fid, ok = calibrate(coarse)
    if not ok.any():
        msg = "no flat-top width reaches a pi/2 conditional rotation within action_bound"
        raise ValueError(msg)
    k = int(np.argmax(np.where(ok, fid, -np.inf)))
    step = float(coarse[1] - coarse[0])
    fine = jnp.linspace(float(coarse[k]) - step, float(coarse[k]) + step, n_widths)
    fine = jnp.clip(fine, 0.0, w_max)
    amps_f, fid_f, ok_f = calibrate(fine)
    widths = np.concatenate([np.asarray(coarse), np.asarray(fine)])
    amps, fid = np.concatenate([amps, amps_f]), np.concatenate([fid, fid_f])
    ok = np.concatenate([ok, ok_f])
    k = int(np.argmax(np.where(ok, fid, -np.inf)))
    return jnp.clip(pulse(jnp.asarray(amps[k]), jnp.asarray(widths[k])), -bound, bound)


# ---------------------------------------------------------------------- registration
REFERENCES = (
    "Magesan & Gambetta, Phys. Rev. A 101, 052308 (2020)",
    "Koch et al., Phys. Rev. A 76, 042319 (2007)",
    "Krantz et al., Appl. Phys. Rev. 6, 021318 (2019)",
    "Pedersen, Moller & Molmer, Phys. Lett. A 367, 47 (2007)",
    "Wood & Gambetta, Phys. Rev. A 97, 032306 (2018)",
)

KWARGS = {
    "horizon": f"int, control steps per episode (default {DEFAULT_HORIZON})",
    "dt": f"float, control period in ns (default {DEFAULT_DT_NS}; gate time = horizon * dt)",
    "action_bound": "float, bound on each of the 4 channels in rad/ns (primary; default "
    f"2 pi x {DEFAULT_MAX_RABI_GHZ} = {DEFAULT_ACTION_BOUND:.5f})",
    "max_rabi_ghz": "float, alternative to action_bound in GHz: action_bound = 2 pi max_rabi_ghz",
    "levels": "int, levels per transmon (default 3; 2 = ideal qubits, no leakage)",
    "detuning_ghz": "float, f_control - f_target in GHz (default -0.1); drive at f_target",
    "anharmonicity_c_ghz": "float, control alpha / 2 pi in GHz (default -0.34)",
    "anharmonicity_t_ghz": "float, target alpha / 2 pi in GHz (default -0.34)",
    "coupling_ghz": "float, exchange coupling J / 2 pi in GHz (default 0.003)",
    "t1_c_us": "float or None, control T1 in microseconds; any T1 / T2 switches to Lindblad "
    "dynamics (default None)",
    "t2_c_us": "float or None, control T2 in microseconds, T2 <= 2 T1 (default None)",
    "t1_t_us": "float or None, target T1 in microseconds (default None)",
    "t2_t_us": "float or None, target T2 in microseconds, T2 <= 2 T1 (default None)",
    "method": "dynamiqs Method for each step propagator (default Expm(), exact for "
    "piecewise-constant controls)",
}

DESCRIPTION = (
    "Synthesise the cross-resonance gate ZX(pi/2) = exp(-i (pi/4) Z(x)X) (control (x) target, "
    "bare qubit frames at the end of the gate) on two exchange-coupled transmons (3 levels each "
    "by default), H = Delta n_c + (alpha_c/2) n_c(n_c-1) + (alpha_t/2) n_t(n_t-1) "
    "+ J (a^dag b + a b^dag) + drives, in the frame rotating at the target frequency (RWA). "
    "Actions [Omega_cI, Omega_cQ, Omega_tI, Omega_tQ] in rad/ns: CR drive on the control and a "
    "drive on the target, both at the target frequency. UNITS: time in ns, frequencies in "
    "rad/ns (kwargs in GHz and us). State = 9 x 9 propagator (closed) or 81 x 81 superoperator "
    "(with T1/T2). Reward per step = gain of the average gate fidelity on the computational "
    "states |00>, |01>, |10>, |11> (leakage counts as error); return = F_final - F_initial. "
    "Default 64 x 5 ns = 320 ns, |Omega|/2pi <= 60 MHz, Delta/2pi = -100 MHz, J/2pi = 3 MHz. "
    "No virtual-Z frame updates: Stark and static phases count as errors."
)

register(
    TASK_ID,
    make_cr_task,
    description=DESCRIPTION,
    model="qenvs.models.two_transmon.TwoTransmon",
    references=REFERENCES,
    kwargs=KWARGS,
    units=UNITS,
    reference=reference_actions,
)
