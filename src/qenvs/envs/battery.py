"""Battery-charging task: maximise stored energy or ergotropy of ``model.battery_hamiltonian()``."""

from __future__ import annotations

import dynamiqs as dq
import jax
import jax.numpy as jnp
from dynamiqs import QArray
from dynamiqs.method import Method, Tsit5
from jax import Array

from qenvs.envs.base import Environment, EnvState, TimeStep
from qenvs.models.base import Model
from qenvs.observables import energy, ergotropy

REWARDS = ("energy", "ergotropy")


class BatteryCharging(Environment):
    """Piecewise-constant charging control of a quantum battery.

    Works for any :class:`~qenvs.models.Model` that implements
    ``battery_hamiltonian()``. The reward is the per-step *gain* of the chosen
    quantity, so the undiscounted return equals ``Q(final) - Q(initial)`` with
    ``Q`` the stored energy or the ergotropy. ``info`` reports energy,
    ergotropy, average charging power ``E(t) / t`` and purity every step.

    Attributes:
        model: physical model with a battery Hamiltonian.
        horizon: number of control steps.
        dt: duration of each step (control period; the integrator adapts inside it).
        action_bound: maximum control amplitude.
        reward: ``"energy"`` or ``"ergotropy"``.
        method: dynamiqs ODE method.
    """

    model: Model
    horizon: int = 20
    dt: float = 0.1
    action_bound: float = 2.0
    reward: str = "energy"
    method: Method = Tsit5()

    def __check_init__(self) -> None:
        if self.reward not in REWARDS:
            msg = f"reward must be one of {REWARDS}, got {self.reward!r}"
            raise ValueError(msg)

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

    def _quantity(self, qstate: QArray) -> Array:
        rho_b, hb = self.model.battery_reduced(qstate)
        return energy(rho_b, hb) if self.reward == "energy" else ergotropy(rho_b, hb)

    def _observe(self, qstate: QArray, t: Array) -> Array:
        x = qstate.to_jax().reshape(-1)
        return jnp.concatenate([x.real, x.imag, jnp.atleast_1d(t / self.duration)])

    def _info(self, qstate: QArray, t: Array) -> dict[str, Array]:
        rho_b, hb = self.model.battery_reduced(qstate)
        e = energy(rho_b, hb)
        rho = qstate.todm().to_jax()
        info = {name: dq.expect(op, qstate).real for name, op in self.model.observables().items()}
        info.update(
            {
                "energy": e,
                "ergotropy": ergotropy(rho_b, hb),
                "power": jnp.where(t > 0, e / jnp.maximum(t, 1e-12), 0.0),
                "purity": jnp.real(jnp.trace(rho @ rho)),
            }
        )
        return info

    def reset(self, key: Array) -> tuple[EnvState, TimeStep]:
        qstate = self.model.initial_state()
        t = jnp.asarray(0.0)
        state = EnvState(qstate=qstate, t=t, step=jnp.asarray(0), key=key)
        ts = TimeStep(
            observation=self._observe(qstate, t),
            reward=jnp.asarray(0.0),
            done=jnp.asarray(False),
            info=self._info(qstate, t),
        )
        return state, ts

    def step(self, state: EnvState, action: Array) -> tuple[EnvState, TimeStep]:
        action = jnp.clip(
            jnp.asarray(action, dtype=jnp.float32), -self.action_bound, self.action_bound
        )
        h = self.model.hamiltonian(action)
        qstate = self.model.evolve(state.qstate, h, self.dt, method=self.method)
        q_prev = self._quantity(state.qstate)
        q_new = self._quantity(qstate)
        t = state.t + self.dt
        step = state.step + 1
        new_state = EnvState(qstate=qstate, t=t, step=step, key=state.key)
        ts = TimeStep(
            observation=self._observe(qstate, t),
            reward=q_new - q_prev,
            done=step >= self.horizon,
            info=self._info(qstate, t),
        )
        return new_state, ts

    def rollout(self, actions: Array, key: Array | None = None) -> tuple[EnvState, Array]:
        """Open-loop rollout of ``actions`` of shape ``(horizon, action_dim)`` via ``lax.scan``."""
        if key is None:
            key = jax.random.PRNGKey(0)
        state0, _ = self.reset(key)

        def body(state: EnvState, a: Array) -> tuple[EnvState, Array]:
            state, ts = self.step(state, a)
            return state, ts.reward

        return jax.lax.scan(body, state0, actions)
