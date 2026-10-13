"""Qubit tasks."""

from __future__ import annotations

import dynamiqs as dq

from qenvs.envs.base import Environment
from qenvs.envs.state_transfer import StateTransfer
from qenvs.models.two_level import TwoLevel
from qenvs.registry import register


def _qubit_state_transfer(**kw: object) -> Environment:
    gamma1 = float(kw.pop("gamma1", 0.0))
    gamma_phi = float(kw.pop("gamma_phi", 0.0))
    detuning = float(kw.pop("detuning", 0.0))
    model = TwoLevel(gamma1=gamma1, gamma_phi=gamma_phi, detuning=detuning)
    target = dq.basis(2, 1)  # pure target kept as a ket: exact, differentiable fidelity
    return StateTransfer(model=model, target=target, **kw)  # type: ignore[arg-type]


register(
    "qubit/StateTransfer-v0",
    _qubit_state_transfer,
    description=(
        "Drive a qubit from |0> to |1> with a piecewise-constant resonant field u(t) sigma_x/2. "
        "Reward per step = fidelity gain; return = F_final - F_initial. "
        "Optimum: any pulse with integral(u dt) = pi (pi-pulse), F = 1 for the closed system."
    ),
    model="qenvs.models.TwoLevel",
    references=(
        "Rabi, Phys. Rev. 51, 652 (1937)",
        "Breuer & Petruccione, The Theory of Open Quantum Systems (2002), Sec. 3.4",
    ),
    kwargs={
        "horizon": "int, control steps per episode (default 20)",
        "dt": "float, control period (default 0.1)",
        "action_bound": "float, |u| <= action_bound (default 2.0)",
        "detuning": "float, drive-qubit detuning (default 0)",
        "gamma1": "float, relaxation rate; >0 switches to Lindblad dynamics (default 0)",
        "gamma_phi": "float, pure dephasing rate (default 0)",
    },
)
