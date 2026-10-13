"""Open two-level quantum battery coupled to a thermal bath.

Rotating-frame Hamiltonian

    H(t) = (Delta / 2) sigma_z + u(t) sigma_x / 2

with battery Hamiltonian ``H_B = omega0 |1><1|`` (ground energy 0). The bath is a
thermal reservoir with Bose occupation ``n_th`` coupled at rate ``gamma1``
(Breuer & Petruccione 2002, Sec. 3.4), giving the jump operators

    sqrt(gamma1 (n_th + 1)) |0><1|    (emission)
    sqrt(gamma1 n_th)       |1><0|    (absorption)

plus pure dephasing ``sqrt(gamma_phi / 2) sigma_z``. Index 1 is the excited
state; ``sigma_z = |0><0| - |1><1|`` so ``Delta > 0`` lowers the excited level in
the rotating frame, exactly as in :class:`~qenvs.models.TwoLevel`.

Analytic references (checked in ``qenvs.validation.qubit_battery``):

* Zero drive, ``gamma1 > 0``: ``P_1(t) -> n_th / (2 n_th + 1)`` at rate
  ``gamma1 (2 n_th + 1)``; for ``n_th = 0`` this is ``exp(-gamma1 t)`` decay.
* The Gibbs state ``diag(1 - p1, p1)`` with ``p1 = n_th / (2 n_th + 1)`` is
  stationary, has energy ``omega0 p1`` and zero ergotropy (passive state).
* Closed resonant drive ``u``: ``P_1(t) = sin^2(u t / 2)``, so a pi-pulse
  stores energy ``omega0`` with ergotropy ``omega0``, and the charging power
  obeys ``|dE/dt| <= omega0 u / 2``.
"""

from __future__ import annotations

import dynamiqs as dq
import jax.numpy as jnp
from dynamiqs import QArray

from qenvs.models.base import Model

INITIAL_STATES = ("ground", "thermal")


class QubitBattery(Model):
    """Driven qubit battery with a thermal bath and pure dephasing.

    Attributes:
        omega0: qubit (battery) frequency, sets the energy unit of ``H_B``.
        detuning: drive-qubit detuning ``Delta`` in the rotating frame.
        gamma1: bath coupling rate; the emission rate is ``gamma1 (n_th + 1)`` and the
            absorption rate ``gamma1 n_th`` (0 = no bath).
        n_th: Bose occupation of the bath at the qubit frequency (0 = zero temperature).
        gamma_phi: pure dephasing rate (0 = none).
        initial: ``"ground"`` (|0>) or ``"thermal"`` (Gibbs state of ``H_B`` at the bath
            occupation, ``p1 = n_th / (2 n_th + 1)``).
    """

    omega0: float = 1.0
    detuning: float = 0.0
    gamma1: float = 0.0
    n_th: float = 0.0
    gamma_phi: float = 0.0
    initial: str = "ground"

    def __check_init__(self) -> None:
        if self.initial not in INITIAL_STATES:
            msg = f"initial must be one of {INITIAL_STATES}, got {self.initial!r}"
            raise ValueError(msg)
        for name in ("gamma1", "n_th", "gamma_phi"):
            if getattr(self, name) < 0:
                msg = f"{name} must be non-negative, got {getattr(self, name)!r}"
                raise ValueError(msg)
        if self.initial == "thermal" and self.n_th > 0 and not self.is_open:
            msg = (
                "initial='thermal' with n_th > 0 is a mixed state and needs the master "
                "equation: set gamma1 > 0 (or gamma_phi > 0)"
            )
            raise ValueError(msg)

    @property
    def dim(self) -> int:
        return 2

    @property
    def thermal_population(self) -> float:
        """Excited-state population ``n_th / (2 n_th + 1)`` of the Gibbs state at the bath."""
        return self.n_th / (2.0 * self.n_th + 1.0)

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
            ops.append(jnp.sqrt(self.gamma1 * (self.n_th + 1.0)) * lower)
            if self.n_th > 0:
                ops.append(jnp.sqrt(self.gamma1 * self.n_th) * lower.dag())
        if self.gamma_phi > 0:
            ops.append(jnp.sqrt(self.gamma_phi / 2.0) * dq.sigmaz())
        return ops

    def initial_state(self) -> QArray:
        if self.initial == "thermal" and self.n_th > 0:
            p1 = self.thermal_population
            return (1.0 - p1) * dq.basis(2, 0).todm() + p1 * dq.basis(2, 1).todm()
        ket = dq.basis(2, 0)
        return ket.todm() if self.is_open else ket

    def battery_hamiltonian(self) -> QArray:
        """``H_B = omega0 |1><1|`` (ground energy set to zero)."""
        return self.omega0 * dq.basis(2, 1).todm()

    def observables(self) -> dict[str, QArray]:
        return {"energy": self.battery_hamiltonian(), "p1": dq.basis(2, 1).todm()}
