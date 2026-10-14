"""QEnvS: physically validated quantum control environments for RL and optimal control."""

from qenvs import envs, gates, models, observables, tasks, validation
from qenvs.envs.base import Environment, EnvState, TimeStep
from qenvs.registry import make, register, registry

try:
    from qenvs._version import __version__
except ImportError:  # pragma: no cover - editable install without build hook
    __version__ = "0.0.0+unknown"

__all__ = [
    "EnvState",
    "Environment",
    "TimeStep",
    "__version__",
    "envs",
    "gates",
    "make",
    "models",
    "observables",
    "register",
    "registry",
    "tasks",
    "validation",
]
