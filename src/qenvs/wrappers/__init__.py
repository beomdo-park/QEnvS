"""Adapters to third-party RL interfaces (need the ``gym`` extra: ``pip install 'qenvs[gym]'``)."""

from qenvs.wrappers.gymnasium import GymnasiumEnv
from qenvs.wrappers.vector import GymnasiumVectorEnv

__all__ = ["GymnasiumEnv", "GymnasiumVectorEnv"]
