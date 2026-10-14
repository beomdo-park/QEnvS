"""Gate-synthesis task: build a target unitary on the computational subspace of ``model``.

The environment state is the map accumulated so far: the propagator ``U`` (closed
models, start = identity) or the column-stacking superoperator ``S`` (open models,
start = identity). Each step multiplies by the propagator of one piecewise-constant
control interval, computed with ``dynamiqs.sepropagator`` / ``dynamiqs.mepropagator``.
"""

from __future__ import annotations

import math

import dynamiqs as dq
import equinox as eqx
import jax
import jax.numpy as jnp
from dynamiqs import QArray
from dynamiqs.method import Expm, Method
from jax import Array

from qenvs.envs.base import Environment, EnvState, TimeStep
from qenvs.gates import (
    apply_superop,
    average_gate_fidelity,
    leakage,
    superop_average_gate_fidelity,
    superop_leakage,
)
from qenvs.models.base import Model

RESERVED_INFO = ("fidelity", "leakage")


def _as_matrix(x: object) -> Array:
    to_jax = getattr(x, "to_jax", None)
    m = jnp.asarray(to_jax() if to_jax is not None else x, dtype=complex)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        msg = f"target must be a square (k, k) matrix, got shape {m.shape}"
        raise ValueError(msg)
    return m


class GateControl(Environment):
    """Piecewise-constant control towards a target gate (average gate fidelity reward).

    Each step applies the action as constant control amplitudes for ``dt`` and
    composes the step propagator with the accumulated map. The reward is the gain of
    the average gate fidelity on the computational subspace (the first ``k`` levels,
    ``k = target.shape[0]``), so the undiscounted return equals
    ``F_final - F_initial`` with ``F_initial = F(identity, target)``.

    The observation is the flattened real and imaginary parts of the accumulated map
    (``2 d^2`` numbers for ``U``, ``2 d^4`` for ``S``) plus the normalised time.
    ``info`` reports ``fidelity`` (average gate fidelity, Wood and Gambetta 2018),
    ``leakage`` (leakage rate out of the subspace), and the model observables
    evaluated on ``map(model.initial_state())``; a model observable whose name
    clashes with ``fidelity`` or ``leakage`` is reported as ``state_<name>``.

    Attributes:
        model: physical model (any :class:`~qenvs.models.Model`; units are the model's).
        target: ``(k, k)`` target unitary with ``k <= model.dim``.
        horizon: number of control steps.
        dt: duration of each step (control period).
        action_bound: actions are clipped to ``[-action_bound, action_bound]``.
        method: dynamiqs method for each step propagator. The default ``Expm()`` is
            exact for piecewise-constant controls; ODE methods (``Tsit5()``) integrate
            the propagator instead.
    """

    model: Model
    target: Array = eqx.field(converter=_as_matrix)
    horizon: int = 16
    dt: float = 2.0
    action_bound: float = 0.1 * math.pi
    method: Method = Expm()

    def __check_init__(self) -> None:
        k = self.target.shape[0]
        if k > self.model.dim:
            msg = f"target acts on {k} levels but the model has only {self.model.dim}"
            raise ValueError(msg)

    @property
    def action_dim(self) -> int:
        return self.model.n_controls

    @property
    def n_comp(self) -> int:
        """Dimension ``k`` of the computational subspace (size of the target)."""
        return int(self.target.shape[0])

    @property
    def map_dim(self) -> int:
        """Side of the accumulated map: ``d`` (unitary) or ``d^2`` (superoperator)."""
        d = self.model.dim
        return d * d if self.model.is_open else d

    @property
    def observation_dim(self) -> int:
        return 2 * self.map_dim**2 + 1

    @property
    def duration(self) -> float:
        """Total episode duration ``horizon * dt``."""
        return self.horizon * self.dt

    # ------------------------------------------------------------- quantities
    def fidelity(self, qstate: QArray | Array) -> Array:
        """Average gate fidelity of the accumulated map against ``target``."""
        if self.model.is_open:
            return superop_average_gate_fidelity(qstate, self.target)
        return average_gate_fidelity(qstate, self.target)

    def leakage(self, qstate: QArray | Array) -> Array:
        """Leakage rate of the accumulated map out of the first ``k`` levels."""
        if self.model.is_open:
            return superop_leakage(qstate, self.n_comp)
        return leakage(qstate, self.n_comp)

    def evolved_state(self, qstate: QArray | Array) -> QArray:
        """The accumulated map applied to ``model.initial_state()``."""
        m = qstate.to_jax() if isinstance(qstate, QArray) else qstate
        psi0 = self.model.initial_state()
        if self.model.is_open:
            return dq.asqarray(apply_superop(m, psi0.todm().to_jax()))
        return dq.asqarray(m @ psi0.to_jax())

    def _observe(self, qstate: QArray, t: Array) -> Array:
        x = qstate.to_jax().reshape(-1)
        return jnp.concatenate([x.real, x.imag, jnp.atleast_1d(t / self.duration)])

    def _info(self, qstate: QArray, t: Array | None = None) -> dict[str, Array]:  # noqa: ARG002
        psi = self.evolved_state(qstate)
        info: dict[str, Array] = {}
        for name, op in self.model.observables().items():
            key = f"state_{name}" if name in RESERVED_INFO else name
            info[key] = dq.expect(op, psi).real
        info["fidelity"] = self.fidelity(qstate)
        info["leakage"] = self.leakage(qstate)
        return info

    # ------------------------------------------------------------- dynamics
    def step_propagator(self, action: Array) -> Array:
        """Propagator (``U`` or ``S``) of one control interval with constant ``action``."""
        h = self.model.hamiltonian(action)
        tsave = jnp.asarray([0.0, self.dt])
        if self.model.is_open:
            res = dq.mepropagator(
                h, self.model.jump_ops(), tsave, method=self.method, progress_meter=False
            )
        else:
            res = dq.sepropagator(h, tsave, method=self.method, progress_meter=False)
        return res.final_propagator.to_jax()

    def reset(self, key: Array) -> tuple[EnvState, TimeStep]:
        qstate = dq.asqarray(jnp.eye(self.map_dim, dtype=complex))
        t = jnp.asarray(0.0)
        state = EnvState(qstate=qstate, t=t, step=jnp.asarray(0), key=key)
        ts = TimeStep(
            observation=self._observe(qstate, t),
            reward=jnp.asarray(0.0),
            done=jnp.asarray(False),
            info=self._info(qstate),
        )
        return state, ts

    def step(self, state: EnvState, action: Array) -> tuple[EnvState, TimeStep]:
        action = jnp.clip(jnp.asarray(action, dtype=float), -self.action_bound, self.action_bound)
        m = self.step_propagator(action) @ state.qstate.to_jax()
        qstate = dq.asqarray(m)
        f_prev = self.fidelity(state.qstate)
        f_new = self.fidelity(qstate)
        t = state.t + self.dt
        step = state.step + 1
        new_state = EnvState(qstate=qstate, t=t, step=step, key=state.key)
        ts = TimeStep(
            observation=self._observe(qstate, t),
            reward=f_new - f_prev,
            done=step >= self.horizon,
            info=self._info(qstate),
        )
        return new_state, ts

    def rollout(self, actions: Array, key: Array | None = None) -> tuple[EnvState, Array]:
        """Open-loop rollout of ``actions`` of shape ``(horizon, action_dim)`` via ``lax.scan``.

        Returns the final state (``state.qstate`` is the full map) and the per-step
        rewards; differentiable with respect to ``actions``.
        """
        if key is None:
            key = jax.random.PRNGKey(0)
        state0, _ = self.reset(key)

        def body(state: EnvState, a: Array) -> tuple[EnvState, Array]:
            state, ts = self.step(state, a)
            return state, ts.reward

        return jax.lax.scan(body, state0, actions)
