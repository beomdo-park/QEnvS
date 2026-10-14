"""Task modules register themselves on import; add new modules here."""

from qenvs.tasks import (
    cross_resonance,
    dicke,
    dicke_local,
    lambda_system,
    qubit,
    qubit_battery,
    transmon,
)

__all__ = [
    "cross_resonance",
    "dicke",
    "dicke_local",
    "lambda_system",
    "qubit",
    "qubit_battery",
    "transmon",
]
