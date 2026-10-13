"""Task registry: ``qenvs.make("qubit/StateTransfer-v0")``.

Task ids follow the Gymnasium convention ``namespace/Name-vN``. Bump ``N``
whenever a change could alter learning results (reward, observation, defaults);
never silently change a released task.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import dynamiqs as dq

from qenvs.envs.base import Environment
from qenvs.envs.state_transfer import StateTransfer
from qenvs.models.two_level import TwoLevel


@dataclass(frozen=True)
class TaskSpec:
    """Metadata attached to a registered task."""

    id: str
    factory: Callable[..., Environment]
    description: str
    model: str
    references: tuple[str, ...] = field(default_factory=tuple)
    kwargs: dict[str, str] = field(default_factory=dict)


registry: dict[str, TaskSpec] = {}


def register(
    task_id: str,
    factory: Callable[..., Environment],
    *,
    description: str,
    model: str,
    references: tuple[str, ...] = (),
    kwargs: dict[str, str] | None = None,
) -> None:
    """Register a task factory under ``task_id``."""
    if task_id in registry:
        msg = f"task id already registered: {task_id!r}"
        raise ValueError(msg)
    registry[task_id] = TaskSpec(
        id=task_id,
        factory=factory,
        description=description,
        model=model,
        references=references,
        kwargs=kwargs or {},
    )


def make(task_id: str, **kwargs: object) -> Environment:
    """Instantiate a registered task, forwarding ``kwargs`` to its factory."""
    try:
        spec = registry[task_id]
    except KeyError as e:
        known = ", ".join(sorted(registry))
        msg = f"unknown task id {task_id!r}; known: {known}"
        raise KeyError(msg) from e
    return spec.factory(**kwargs)


def _qubit_state_transfer(**kw: object) -> Environment:
    gamma1 = float(kw.pop("gamma1", 0.0))
    gamma_phi = float(kw.pop("gamma_phi", 0.0))
    detuning = float(kw.pop("detuning", 0.0))
    model = TwoLevel(gamma1=gamma1, gamma_phi=gamma_phi, detuning=detuning)
    target = dq.basis(2, 1)
    target = target.todm() if model.is_open else target
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
