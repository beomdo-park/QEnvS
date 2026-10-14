"""Single-qubit gate tasks on a multi-level transmon in hardware units (ns, GHz, us).

UNITS: time in ns, angular frequency (Hamiltonian, actions, ``action_bound``) in
rad/ns, user-facing frequencies in GHz (converted with ``2 pi``), T1 / T2 in
microseconds.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from jax import Array

from qenvs.envs.base import Environment
from qenvs.envs.gate import GateControl
from qenvs.gates import gate
from qenvs.models.transmon import (
    Transmon,
    drag_beta,
    gaussian_amplitude,
    gaussian_drag,
    ghz_to_rad_per_ns,
)
from qenvs.registry import register

DEFAULT_HORIZON = 16
DEFAULT_DT_NS = 2.0
DEFAULT_MAX_RABI_GHZ = 0.05
DEFAULT_ACTION_BOUND = 2.0 * math.pi * DEFAULT_MAX_RABI_GHZ  # rad/ns

#: Units of every transmon gate task (also shown by ``qenvs info``).
UNITS: dict[str, str] = {
    "time": "ns (dt, horizon * dt); hbar = 1",
    "frequency": "angular frequency in rad/ns inside the Hamiltonian; kwargs *_ghz are ordinary "
    "frequencies in GHz, converted with 2 pi",
    "control": "actions = [Omega_I, Omega_Q] in rad/ns, clipped to +-action_bound; "
    "Omega / (2 pi) is the Rabi frequency in GHz (constant Omega_I for pi / Omega ns = X gate)",
    "coherence": "t1_us, t2_us in microseconds",
}


#: Rotation angle about x of each task's target (used by :func:`reference_actions`).
ROTATION_ANGLE = {"transmon/XGate-v0": math.pi, "transmon/SXGate-v0": math.pi / 2}


def _optional_float(v: object) -> float | None:
    return None if v is None else float(v)  # type: ignore[arg-type]


def make_gate_task(target: Array) -> Callable[..., Environment]:
    """Factory for a :class:`GateControl` task on :class:`Transmon` with the given target."""

    def factory(**kw: object) -> Environment:
        model_kw: dict[str, object] = {}
        if "levels" in kw:
            model_kw["levels"] = int(kw.pop("levels"))  # type: ignore[call-overload]
        for name in ("anharmonicity_ghz", "detuning_ghz"):
            if name in kw:
                model_kw[name] = float(kw.pop(name))  # type: ignore[arg-type]
        for name in ("t1_us", "t2_us"):
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
        model = Transmon(**model_kw)  # type: ignore[arg-type]
        return GateControl(model=model, target=target, **kw)  # type: ignore[arg-type]

    return factory


REFERENCES = (
    "Koch et al., Phys. Rev. A 76, 042319 (2007)",
    "Krantz et al., Appl. Phys. Rev. 6, 021318 (2019)",
    "Motzoi, Gambetta, Rebentrost & Wilhelm, Phys. Rev. Lett. 103, 110501 (2009)",
    "Pedersen, Moller & Molmer, Phys. Lett. A 367, 47 (2007)",
    "Wood & Gambetta, Phys. Rev. A 97, 032306 (2018)",
)

KWARGS = {
    "horizon": f"int, control steps per episode (default {DEFAULT_HORIZON})",
    "dt": f"float, control period in ns (default {DEFAULT_DT_NS}; gate time = horizon * dt)",
    "action_bound": "float, |Omega_I|, |Omega_Q| <= action_bound in rad/ns (primary; default "
    f"2 pi x {DEFAULT_MAX_RABI_GHZ} = {DEFAULT_ACTION_BOUND:.5f})",
    "max_rabi_ghz": "float, alternative to action_bound in GHz: action_bound = 2 pi max_rabi_ghz",
    "levels": "int, transmon levels kept (default 3; 2 = ideal qubit, no leakage)",
    "anharmonicity_ghz": "float, alpha / 2 pi in GHz (default -0.34)",
    "detuning_ghz": "float, f_qubit - f_drive in GHz (default 0)",
    "t1_us": "float or None, T1 in microseconds; given T1 or T2 switches to Lindblad "
    "dynamics (default None)",
    "t2_us": "float or None, T2 in microseconds, T2 <= 2 T1 (default None)",
    "method": "dynamiqs Method for each step propagator (default Expm(), exact for "
    "piecewise-constant controls)",
}

_DESCRIPTION = (
    "Synthesise the {name} gate on the computational subspace {{|0>, |1>}} of a transmon "
    "(default 3 levels), H = Delta n + (alpha/2) n(n-1) + (Omega_I/2)(a + a^dag) "
    "+ (Omega_Q/2) i(a^dag - a) "
    "in the drive frame (RWA). UNITS: time in ns, frequencies and actions [Omega_I, Omega_Q] in "
    "rad/ns (kwargs in GHz and us). State = propagator U (closed) or superoperator (with "
    "t1_us/t2_us). Reward per step = gain of average gate fidelity (leakage counts as error); "
    "return = F_final - F_initial, F_initial = {f0}. Default 16 x 2 ns = 32 ns gate, "
    "|Omega|/2pi <= 50 MHz."
)

register(
    "transmon/XGate-v0",
    make_gate_task(gate("X")),
    description=_DESCRIPTION.format(name="X", f0="1/3"),
    model="qenvs.models.transmon.Transmon",
    references=REFERENCES,
    kwargs=KWARGS,
    units=UNITS,
    reference=lambda env: reference_actions("transmon/XGate-v0", env),
)

register(
    "transmon/SXGate-v0",
    make_gate_task(gate("SX")),
    description=_DESCRIPTION.format(
        name="SX = (1/2)[[1+i, 1-i], [1-i, 1+i]] (IBM convention, SX^2 = X)", f0="2/3"
    ),
    model="qenvs.models.transmon.Transmon",
    references=REFERENCES,
    kwargs=KWARGS,
    units=UNITS,
    reference=lambda env: reference_actions("transmon/SXGate-v0", env),
)


def reference_actions(task_id: str, env: GateControl) -> Array:
    """Analytic baseline pulse for a transmon gate task: Gaussian DRAG on ``env``'s time grid.

    Lifted Gaussian on I with ``sigma = T / 4`` (``T = horizon * dt``) and area equal to
    the target rotation angle (pi for X, pi/2 for SX), DRAG quadrature with the
    phase-error coefficient ``beta = -1/(2 alpha)`` (:func:`~qenvs.models.transmon.drag_beta`).
    Actions are in rad/ns, shape ``(horizon, 2)``, clipped to ``env.action_bound``.
    On the default X task (closed, 3 levels, 16 x 2 ns) this gives 1 - F of about 5e-6.
    """
    try:
        angle = ROTATION_ANGLE[task_id]
    except KeyError as e:
        msg = f"no reference pulse for {task_id!r}; known: {sorted(ROTATION_ANGLE)}"
        raise ValueError(msg) from e
    model = env.model
    horizon, dt = int(env.horizon), float(env.dt)
    sigma = horizon * dt / 4.0
    amp = gaussian_amplitude(angle, horizon, dt, sigma)
    beta = drag_beta(float(model.anharmonicity_ghz), "phase")  # type: ignore[attr-defined]
    actions = gaussian_drag(horizon, dt, amp, sigma, beta)
    return actions.clip(-env.action_bound, env.action_bound)
