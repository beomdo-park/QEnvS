"""Driven two-level system (qubit), closed or open.

Rotating-frame Hamiltonian

    H(t) = (Delta / 2) sigma_z + u(t) sigma_x / 2

with optional relaxation (rate ``gamma1``) and pure dephasing (rate ``gamma_phi``).
Energies are measured in units of the qubit frequency ``omega0`` through the
battery Hamiltonian ``H_B = omega0 |1><1|``.

Analytic references (checked in ``qenvs.validation.qubit``):

* Resonant Rabi drive of constant amplitude ``u``: ``P_1(t) = sin^2(u t / 2)``;
  a pi-pulse (``u t = pi``) gives full inversion.
* Free decay from |1> with ``gamma1``: ``P_1(t) = exp(-gamma1 t)``.
* Ergotropy of |1> equals ``omega0``; of the maximally mixed state equals 0.

Randomisable parameter (:meth:`TwoLevel.parameter_specs`): the detuning ``detuning``
only. ``omega0`` is not declared: it enters ``H_B`` and the ``energy`` observable (the
energy unit of the readout) but not the Hamiltonian, so randomising it would rescale the
reward instead of changing the physics.
"""

from __future__ import annotations

from typing import ClassVar

import dynamiqs as dq
import jax.numpy as jnp
from dynamiqs import QArray

from qenvs.models.base import DISSIPATION, STRUCTURAL, Model, ParameterSpec


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

    non_parameters: ClassVar[dict[str, str]] = {
        "omega0": (
            "it only sets the energy unit of H_B and of the energy readout and does not enter "
            "the Hamiltonian, so randomising it would rescale the reward, not the physics"
        ),
        "gamma1": DISSIPATION,
        "gamma_phi": DISSIPATION,
        "initial": STRUCTURAL,
    }

    def parameter_specs(self) -> dict[str, ParameterSpec]:
        """``detuning``: ``(-inf, inf)``, reference width 0.05.

        The detuning has no sign constraint. The reference width 0.05 is about 3% of the
        Rabi rate ``pi / T = 1.57`` of the default pi pulse (``T = 2``): a visible but
        small error (about ``1 - F = 1e-3``), the scale of a calibration drift relative to
        the drive strength. Units are those of the model (``omega0 = 1``).
        """
        return {"detuning": ParameterSpec("omega0", (-float("inf"), float("inf")), 0.05)}

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
