"""Thermodynamic and information observables shared by models and environments."""

from __future__ import annotations

import dynamiqs as dq
import jax.numpy as jnp
from dynamiqs import QArray
from jax import Array


def energy(state: QArray, hamiltonian: QArray) -> Array:
    """Mean energy ``tr(rho H)`` (real)."""
    return dq.expect(hamiltonian, state).real


def ergotropy(state: QArray, hamiltonian: QArray) -> Array:
    """Maximum work extractable by a unitary (Allahverdyan et al., 2004).

    ``E = tr(rho H) - min_U tr(U rho U^dag H)``, where the minimum is attained by the
    passive state: eigenvalues of ``rho`` in decreasing order paired with eigenvalues
    of ``H`` in increasing order.
    """
    rho = state.todm().to_jax()
    h = hamiltonian.to_jax()
    r = jnp.sort(jnp.linalg.eigvalsh(rho))[::-1]
    e = jnp.sort(jnp.linalg.eigvalsh(h))
    passive = jnp.sum(r * e)
    return jnp.real(jnp.trace(rho @ h)) - passive


def fidelity(state: QArray, target: QArray) -> Array:
    """State fidelity in ``[0, 1]`` (dynamiqs convention, squared overlap for pure states)."""
    return dq.fidelity(state, target).real
