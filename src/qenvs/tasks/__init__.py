"""Task modules register themselves on import; add new modules here."""

from qenvs.tasks import dicke, lambda_system, qubit, qubit_battery

__all__ = ["dicke", "lambda_system", "qubit", "qubit_battery"]
