"""State-transfer task: drive ``model`` from its initial state to ``target``."""

from __future__ import annotations

import dynamiqs as dq
import jax
import jax.numpy as jnp
from dynamiqs import QArray
from dynamiqs.method import Method, Tsit5
from jax import Array

from qenvs.envs.base import Environment, EnvState, TimeStep
from qenvs.models.base import Model
from qenvs.observables import fidelity


class StateTransfer(Environment):
    """Piecewise-constant control towards a target state.

    Each step applies the action as constant control amplitudes for ``dt``. The
    observation is the flattened real/imaginary parts of the quantum state plus
    the normalised time. The reward is the fidelity *gain* of the step, so the
    undiscounted return equals ``F(final) - F(initial)``.

    Attributes:
        model: physical model.
        target: target state (ket or density matrix).
        horizon: number of control steps.
        dt: duration of each step.
        action_bound: maximum control amplitude.
        method: dynamiqs ODE method used inside each control interval. The
            integrator step is independent of ``dt`` (adaptive by default).
    """

    model: Model
    target: QArray
    horizon: int = 20
    dt: float = 0.1
    action_bound: float = 2.0
    method: Method = Tsit5()

    @property
    def action_dim(self) -> int:
        return self.model.n_controls

    @property
    def observation_dim(self) -> int:
        n = self.model.dim
        return 2 * (n * n if self.model.is_open else n) + 1

    @property
    def duration(self) -> float:
        """Total episode duration ``horizon * dt``."""
        return self.horizon * self.dt

    def _observe(self, qstate: QArray, t: Array) -> Array:
        x = qstate.to_jax().reshape(-1)
        return jnp.concatenate([x.real, x.imag, jnp.atleast_1d(t / self.duration)])

    def _info(self, qstate: QArray, t: Array | None = None) -> dict[str, Array]:  # noqa: ARG002
        info = {name: dq.expect(op, qstate).real for name, op in self.model.observables().items()}
        info["fidelity"] = fidelity(qstate, self.target)
        return info

    def reset(self, key: Array) -> tuple[EnvState, TimeStep]:
        qstate = self.model.initial_state()
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
        action = jnp.clip(
            jnp.asarray(action, dtype=jnp.float32), -self.action_bound, self.action_bound
        )
        h = self.model.hamiltonian(action)
        qstate = self.model.evolve(state.qstate, h, self.dt, method=self.method)
        f_prev = fidelity(state.qstate, self.target)
        f_new = fidelity(qstate, self.target)
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
        """Open-loop rollout of ``actions`` of shape ``(horizon, action_dim)``.

        Returns the final state and the per-step rewards. Implemented with
        ``lax.scan`` so it is differentiable with respect to ``actions``.
        """
        if key is None:
            key = jax.random.PRNGKey(0)
        state0, _ = self.reset(key)

        def body(state: EnvState, a: Array) -> tuple[EnvState, Array]:
            state, ts = self.step(state, a)
            return state, ts.reward

        return jax.lax.scan(body, state0, actions)
