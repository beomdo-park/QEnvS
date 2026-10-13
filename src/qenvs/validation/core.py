"""Shared types for physics validation checks."""

from __future__ import annotations

import platform
from dataclasses import asdict, dataclass

import dynamiqs as dq
import jax


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one validation check."""

    name: str
    category: str
    error: float
    tolerance: float
    passed: bool
    detail: str
    model: str = ""
    skipped: bool = False

    def to_dict(self) -> dict[str, object]:
        """Plain-dict form for JSON output."""
        return asdict(self)


def result(
    name: str, category: str, error: float, tol: float, detail: str, model: str = ""
) -> CheckResult:
    """Build a :class:`CheckResult` from an absolute error and a tolerance."""
    error = abs(float(error))
    return CheckResult(name, category, error, tol, bool(error <= tol), detail, model)


def skipped(name: str, category: str, tol: float, detail: str, model: str = "") -> CheckResult:
    """A check that could not run (e.g. optional dependency missing); never counts as passed."""
    return CheckResult(name, category, 0.0, tol, False, detail, model, skipped=True)


def environment_info() -> dict[str, str]:
    """Versions and platform, reported next to validation numbers."""
    from qenvs import __version__

    return {
        "qenvs": __version__,
        "dynamiqs": dq.__version__,
        "jax": jax.__version__,
        "jax_backend": jax.default_backend(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "x64": str(bool(jax.config.jax_enable_x64)),
    }
