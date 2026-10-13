"""Driven two-level system (qubit), closed or open.

Rotating-frame Hamiltonian

    H(t) = (Delta / 2) sigma_z + u(t) sigma_x / 2

with optional relaxation (rate ``gamma1``) and pure dephasing (rate ``gamma_phi``).
Energies are measured in units of the qubit frequency ``omega0`` through the
battery Hamiltonian ``H_B = omega0 |1><1|``.

Analytic references (checked in ``tests/test_two_level.py``):

* Resonant Rabi drive of constant amplitude ``u``: ``P_1(t) = sin^2(u t / 2)``;
  a pi-pulse (``u t = pi``) gives full inversion.
* Free decay from |1> with ``gamma1``: ``P_1(t) = exp(-gamma1 t)``.
* Ergotropy of |1> equals ``omega0``; of the maximally mixed state equals 0.
"""

from __future__ import annotations

import dynamiqs as dq
import jax.numpy as jnp
from dynamiqs import QArray

from qenvs.models.base import Model


class TwoLevel(Model):
    """Driven qubit with optional T1/T_phi dissipation.

    Attributes:
        omega0: qubit (battery) frequency, sets the energy unit of ``H_B``.
        detuning: drive-qubit detuning ``Delta`` in the rotating frame.
        gamma1: relaxation rate (0 = none).
        gamma_phi: pure dephasing rate (0 = none).
        initial: ``"ground"`` or ``"excited"``.
    """

    omega0: float = 1.0
    detuning: float = 0.0
    gamma1: float = 0.0
    gamma_phi: float = 0.0
    initial: str = "ground"

    @property
    def dim(self) -> int:
        return 2

    def drift(self) -> QArray:
        return 0.5 * self.detuning * dq.sigmaz()

    def controls(self) -> list[QArray]:
        return [0.5 * dq.sigmax()]

    def jump_ops(self) -> list[QArray]:
        ops: list[QArray] = []
        if self.gamma1 > 0:
            # explicit |0><1| (index 1 = excited); dynamiqs' sigmam() follows the
            # QuTiP spin convention where index 0 is "up", which is the opposite.
            lower = dq.basis(2, 0) @ dq.basis(2, 1).dag()
            ops.append(jnp.sqrt(self.gamma1) * lower)
        if self.gamma_phi > 0:
            ops.append(jnp.sqrt(self.gamma_phi / 2.0) * dq.sigmaz())
        return ops

    def initial_state(self) -> QArray:
        ket = dq.basis(2, 1) if self.initial == "excited" else dq.basis(2, 0)
        return ket.todm() if self.is_open else ket

    def battery_hamiltonian(self) -> QArray:
        """``H_B = omega0 |1><1|`` (ground energy set to zero)."""
        return self.omega0 * dq.basis(2, 1).todm()

    def observables(self) -> dict[str, QArray]:
        return {"energy": self.battery_hamiltonian(), "p1": dq.basis(2, 1).todm()}
