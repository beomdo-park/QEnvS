"""Minimal task registry: ``qenvs.make("TwoLevelStateTransfer-v0")``.

Task ids carry a version suffix; bump it whenever a change could alter learning
results (same convention as Gymnasium/Jumanji).
"""

from __future__ import annotations

from collections.abc import Callable

import dynamiqs as dq

from qenvs.envs.base import Environment
from qenvs.envs.state_transfer import StateTransfer
from qenvs.models.two_level import TwoLevel

registry: dict[str, Callable[..., Environment]] = {}


def register(task_id: str, factory: Callable[..., Environment]) -> None:
    """Register a task factory under ``task_id``."""
    if task_id in registry:
        msg = f"task id already registered: {task_id!r}"
        raise ValueError(msg)
    registry[task_id] = factory


def make(task_id: str, **kwargs: object) -> Environment:
    """Instantiate a registered task, forwarding ``kwargs`` to its factory."""
    try:
        factory = registry[task_id]
    except KeyError as e:
        known = ", ".join(sorted(registry))
        msg = f"unknown task id {task_id!r}; known: {known}"
        raise KeyError(msg) from e
    return factory(**kwargs)


def _two_level_state_transfer(**kw: object) -> Environment:
    gamma1 = float(kw.pop("gamma1", 0.0))
    gamma_phi = float(kw.pop("gamma_phi", 0.0))
    model = TwoLevel(gamma1=gamma1, gamma_phi=gamma_phi)
    target = dq.basis(2, 1)
    target = target.todm() if model.is_open else target
    return StateTransfer(model=model, target=target, **kw)  # type: ignore[arg-type]


register("TwoLevelStateTransfer-v0", _two_level_state_transfer)
