"""Environment wrappers.

``Wrapper`` and the uncertainty wrappers need only the core dependencies. The adapters to
third-party RL interfaces (``GymnasiumEnv``, ``GymnasiumVectorEnv``) need the ``gym``
extra: ``pip install 'qenvs[gym]'``.
"""

from __future__ import annotations

from typing import Any

from qenvs.distributions import Normal, TruncatedNormal, Uniform
from qenvs.wrappers.control import ControlErrors
from qenvs.wrappers.core import WrappedState, Wrapper
from qenvs.wrappers.observation import ObservationModel
from qenvs.wrappers.randomization import ParameterRandomization

try:
    from qenvs.wrappers.gymnasium import GymnasiumEnv
    from qenvs.wrappers.vector import GymnasiumVectorEnv
except ImportError as _gym_error:  # gymnasium not installed
    _GYM_ERROR: ImportError | None = _gym_error
else:
    _GYM_ERROR = None

__all__ = [
    "ControlErrors",
    "GymnasiumEnv",
    "GymnasiumVectorEnv",
    "Normal",
    "ObservationModel",
    "ParameterRandomization",
    "TruncatedNormal",
    "Uniform",
    "WrappedState",
    "Wrapper",
]


def __getattr__(name: str) -> Any:
    if name in ("GymnasiumEnv", "GymnasiumVectorEnv") and _GYM_ERROR is not None:
        raise _GYM_ERROR
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
