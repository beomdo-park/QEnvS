"""Task registry: ``qenvs.make("qubit/StateTransfer-v0")``.

Task ids follow the Gymnasium convention ``namespace/Name-vN``. Bump ``N``
whenever a change could alter learning results (reward, observation, defaults);
never silently change a released task.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from qenvs.envs.base import Environment


@dataclass(frozen=True)
class TaskSpec:
    """Metadata attached to a registered task."""

    id: str
    factory: Callable[..., Environment]
    description: str
    model: str
    references: tuple[str, ...] = field(default_factory=tuple)
    kwargs: dict[str, str] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)


#: Units of the dimensionless models (TwoLevel, QubitBattery, DickeBattery, LambdaSystem).
DIMENSIONLESS_UNITS: dict[str, str] = {
    "energy": "units of omega0 (the transition frequency of the model); omega0 = 1 by default",
    "time": "units of 1/omega0; hbar = 1",
    "rates": "decay, dephasing and loss rates in units of omega0",
}

registry: dict[str, TaskSpec] = {}


def register(
    task_id: str,
    factory: Callable[..., Environment],
    *,
    description: str,
    model: str,
    references: tuple[str, ...] = (),
    kwargs: dict[str, str] | None = None,
    units: dict[str, str] | None = None,
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
        units=dict(units) if units else dict(DIMENSIONLESS_UNITS),
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
