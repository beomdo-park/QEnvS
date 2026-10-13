"""Base class for physical models.

A :class:`Model` is a pure description of the physics: the drift Hamiltonian,
the control operators the agent can drive, optional Lindblad jump operators,
the initial state and a dictionary of named observables. It knows nothing about
rewards, horizons or action spaces; those live in :mod:`qenvs.envs`.

Every model ships with analytic reference results that the validation test-suite
checks (see ``tests/``). That is the contract behind "physically validated".
"""

from __future__ import annotations

import abc

import dynamiqs as dq
import equinox as eqx
import jax.numpy as jnp
from dynamiqs import QArray, TimeQArray
from dynamiqs.method import Method, Tsit5
from jax import Array

DEFAULT_METHOD: Method = Tsit5()


class Model(eqx.Module):
    """Abstract physical model.

    Subclasses are immutable equinox modules (JAX pytrees), so they can be
    jitted, vmapped and differentiated through.
    """

    @property
    @abc.abstractmethod
    def dim(self) -> int:
        """Hilbert-space dimension."""

    @abc.abstractmethod
    def drift(self) -> QArray:
        """Drift (uncontrolled) Hamiltonian ``H_0``."""

    @abc.abstractmethod
    def controls(self) -> list[QArray]:
        """Control operators ``H_k``; the total Hamiltonian is ``H_0 + sum_k u_k(t) H_k``."""

    def jump_ops(self) -> list[QArray]:
        """Lindblad jump operators. Empty list means closed (unitary) dynamics."""
        return []

    @abc.abstractmethod
    def initial_state(self) -> QArray:
        """Initial state (ket for closed systems, density matrix otherwise)."""

    def observables(self) -> dict[str, QArray]:
        """Named Hermitian observables reported by environments (may be empty)."""
        return {}

    # ------------------------------------------------------------------ helpers
    @property
    def n_controls(self) -> int:
        """Number of control channels."""
        return len(self.controls())

    @property
    def is_open(self) -> bool:
        """Whether the model has dissipation."""
        return len(self.jump_ops()) > 0

    def hamiltonian(self, amplitudes: Array) -> QArray:
        """Constant total Hamiltonian for fixed control amplitudes of shape ``(n_controls,)``."""
        h = self.drift()
        for k, hk in enumerate(self.controls()):
            h = h + amplitudes[k] * hk
        return h

    def pwc_hamiltonian(self, times: Array, amplitudes: Array) -> TimeQArray:
        """Piecewise-constant Hamiltonian.

        Args:
            times: interval boundaries, shape ``(N + 1,)``.
            amplitudes: control amplitudes, shape ``(n_controls, N)``.
        """
        h: TimeQArray | QArray = self.drift()
        for k, hk in enumerate(self.controls()):
            h = h + dq.pwc(times, amplitudes[k], hk)
        return h

    def evolve(
        self,
        state: QArray,
        hamiltonian: QArray | TimeQArray,
        duration: float | Array,
        *,
        method: Method | None = None,
    ) -> QArray:
        """Evolve ``state`` under ``hamiltonian`` for ``duration`` and return the final state.

        Uses the Schrodinger equation for closed models and the Lindblad master
        equation otherwise. Density matrices are always accepted.

        The ODE ``method`` (adaptive Tsit5 by default) takes as many internal
        integration steps as needed inside ``duration``: the agent's control
        period and the integrator's time step are independent. For piecewise
        constant Hamiltonians ``dynamiqs.method.Expm()`` is exact.
        """
        tsave = jnp.asarray([0.0, duration])
        method = DEFAULT_METHOD if method is None else method
        if self.is_open:
            # dynamiqs 0.3.6: mesolve(..., method=Expm(), save_states=False).final_state
            # returns the *vectorised* density matrix (n^2, 1); states[-1] is always (n, n).
            res = dq.mesolve(
                hamiltonian, self.jump_ops(), state, tsave, method=method, progress_meter=False
            )
            return res.states[-1]
        res = dq.sesolve(
            hamiltonian, state, tsave, method=method, save_states=False, progress_meter=False
        )
        return res.final_state
