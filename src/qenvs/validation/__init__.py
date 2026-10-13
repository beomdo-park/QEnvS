"""Physics validation checks, shared by the test-suite and the ``qenvs validate`` CLI.

Each check compares a simulation against a closed-form result or a physical
constraint and returns the measured error together with the tolerance.
Tolerances are chosen to hold in float32 as well as float64 (the CLI enables
x64; library users may run either). Every model contributes its own module
(``qenvs.validation.<model>``) exposing a ``CHECKS`` tuple; register it in
``MODULES`` below.
"""

from __future__ import annotations

from collections.abc import Callable

from qenvs.validation import dicke, lambda_system, qubit, qubit_battery
from qenvs.validation.core import CheckResult, environment_info, result, skipped

MODULES = (qubit, qubit_battery, dicke, lambda_system)

CHECKS: tuple[Callable[[], CheckResult], ...] = tuple(c for m in MODULES for c in m.CHECKS)


def run_all() -> list[CheckResult]:
    """Run every registered check."""
    return [c() for c in CHECKS]


__all__ = ["CHECKS", "MODULES", "CheckResult", "environment_info", "result", "run_all", "skipped"]
