"""Functional environment API (``reset``/``step`` return ``(state, timestep)``).

The design follows Jumanji (Apache-2.0): the environment object is stateless,
all mutable data lives in :class:`EnvState`, so ``step`` can be ``jit``-ed and
``vmap``-ed over batches of states and actions.
"""

from __future__ import annotations

import abc

import equinox as eqx
from dynamiqs import QArray
from jax import Array


class EnvState(eqx.Module):
    """Everything needed to advance the environment."""

    qstate: QArray
    t: Array
    step: Array
    key: Array


class TimeStep(eqx.Module):
    """Output of ``reset``/``step``.

    Attributes:
        observation: real-valued observation vector.
        reward: scalar reward.
        done: whether the episode terminated at this step.
        info: dictionary of named diagnostics (observable expectation values, ...).
    """

    observation: Array
    reward: Array
    done: Array
    info: dict[str, Array]


class Environment(eqx.Module):
    """Abstract functional environment."""

    @property
    @abc.abstractmethod
    def action_dim(self) -> int:
        """Dimension of the continuous action vector."""

    @property
    @abc.abstractmethod
    def observation_dim(self) -> int:
        """Dimension of the observation vector."""

    @property
    @abc.abstractmethod
    def action_bound(self) -> float:
        """Actions are clipped to ``[-action_bound, action_bound]``."""

    @property
    @abc.abstractmethod
    def horizon(self) -> int:
        """Number of control steps per episode."""

    @abc.abstractmethod
    def reset(self, key: Array) -> tuple[EnvState, TimeStep]:
        """Start a new episode."""

    @abc.abstractmethod
    def step(self, state: EnvState, action: Array) -> tuple[EnvState, TimeStep]:
        """Advance one control interval."""
