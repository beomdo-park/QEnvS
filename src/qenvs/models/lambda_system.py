"""Three-level Lambda system for STIRAP-style state transfer, closed or open.

Basis (index = Hilbert-space index): ``0 = |1>`` initial ground state, ``1 = |2>``
excited (radiatively decaying) state, ``2 = |3>`` target ground state. In the
rotating frame of the two laser fields, under the rotating-wave approximation,

    H(t) = Delta |2><2| + delta |3><3|
           + (Omega_p(t) / 2) (|1><2| + |2><1|) + (Omega_s(t) / 2) (|3><2| + |2><3|)

with ``Delta`` the one-photon (pump) detuning and ``delta`` the two-photon
detuning. The two control channels are, in this order, the pump Rabi frequency
``Omega_p`` (couples |1> <-> |2>) and the Stokes Rabi frequency ``Omega_s``
(couples |3> <-> |2>). Spontaneous emission from |2> at total rate ``gamma`` is
modelled by two Lindblad jump operators with equal branching into the two
ground states, ``sqrt(gamma / 2) |1><2|`` and ``sqrt(gamma / 2) |3><2|``.

Analytic references (checked in ``qenvs.validation.lambda_system``):

* For ``delta = 0`` the dark state ``|D> ~ Omega_s |1> - Omega_p |3>`` is an exact
  zero-energy eigenstate of ``H`` for any ``Delta``.
* With ``Omega_s = 0`` and ``Delta = 0`` the |1>-|2> pair is a resonantly driven
  qubit: ``P_2(t) = sin^2(Omega_p t / 2)``.
* Counter-intuitive pulse order (Stokes before pump) with ``Omega_0 T >> 1``
  transfers |1> -> |3> adiabatically through the dark state with negligible
  population of the lossy state |2> (STIRAP; Bergmann et al. 1998, Vitanov et al. 2017).

Randomisable parameters (:meth:`LambdaSystem.parameter_specs`): the one-photon detuning
``delta_one_photon`` and the two-photon detuning ``delta_two_photon``. The decay rate
``gamma`` is dissipation and is not declared.
"""

from __future__ import annotations

from typing import ClassVar

import dynamiqs as dq
import jax.numpy as jnp
from dynamiqs import QArray
from jax import Array

from qenvs.models.base import DISSIPATION, STRUCTURAL, Model, ParameterSpec

PULSE_ORDERS = ("counterintuitive", "intuitive")
INITIAL_STATES = ("1", "2", "3")


class LambdaSystem(Model):
    """Lambda system |1> <-> |2> <-> |3> driven by pump and Stokes fields.

    Attributes:
        delta_one_photon: one-photon (pump) detuning ``Delta`` of |2>.
        delta_two_photon: two-photon detuning ``delta`` of |3>.
        gamma: total spontaneous-emission rate of |2> (0 = closed system). The
            decay branches equally into |1> and |3>; losses to states outside
            the Lambda manifold are not modelled.
        initial: ``"1"``, ``"2"`` or ``"3"``, the initially populated bare state.
    """

    delta_one_photon: float = 0.0
    delta_two_photon: float = 0.0
    gamma: float = 0.0
    initial: str = "1"

    non_parameters: ClassVar[dict[str, str]] = {"gamma": DISSIPATION, "initial": STRUCTURAL}

    def parameter_specs(self) -> dict[str, ParameterSpec]:
        """Both detunings: ``(-inf, inf)``; reference widths 0.05 and 0.01.

        Units are those of the Rabi frequencies (the default bound is 1, ``T = 20``).
        Adiabatic transfer tolerates a one-photon detuning of a few percent of the peak
        Rabi frequency, but is more sensitive to the two-photon detuning, which tilts the
        dark state; its reference width is therefore five times smaller. Measured on the
        reference STIRAP pulse of the default task: ``1 - F`` is 3.6e-3 at nominal, 4.2e-3
        for ``delta_one_photon = 0.05``, 3.8e-3 for ``delta_two_photon = 0.01`` and 9.7e-3
        for ``delta_two_photon = 0.05``.
        """
        inf = float("inf")
        return {
            "delta_one_photon": ParameterSpec("Rabi frequency", (-inf, inf), 0.05),
            "delta_two_photon": ParameterSpec("Rabi frequency", (-inf, inf), 0.01),
        }

    def __check_init__(self) -> None:
        if self.initial not in INITIAL_STATES:
            msg = f"initial must be one of {INITIAL_STATES}, got {self.initial!r}"
            raise ValueError(msg)

    @property
    def dim(self) -> int:
        return 3

    @staticmethod
    def _proj(i: int) -> QArray:
        return dq.basis(3, i).todm()

    @staticmethod
    def _coupling(i: int, j: int) -> QArray:
        """``(|i><j| + |j><i|) / 2`` for Hilbert-space indices ``i``, ``j``."""
        ket_i, ket_j = dq.basis(3, i), dq.basis(3, j)
        return 0.5 * (ket_i @ ket_j.dag() + ket_j @ ket_i.dag())

    def drift(self) -> QArray:
        """``Delta |2><2| + delta |3><3|``."""
        return self.delta_one_photon * self._proj(1) + self.delta_two_photon * self._proj(2)

    def controls(self) -> list[QArray]:
        """``[pump, Stokes]``: ``(|1><2| + h.c.) / 2`` and ``(|3><2| + h.c.) / 2``."""
        return [self._coupling(0, 1), self._coupling(2, 1)]

    def jump_ops(self) -> list[QArray]:
        """Spontaneous emission from |2> with equal branching into |1> and |3>."""
        if self.gamma <= 0:
            return []
        rate = jnp.sqrt(self.gamma / 2.0)
        excited = dq.basis(3, 1)
        return [
            rate * (dq.basis(3, 0) @ excited.dag()),
            rate * (dq.basis(3, 2) @ excited.dag()),
        ]

    def initial_state(self) -> QArray:
        ket = dq.basis(3, INITIAL_STATES.index(self.initial))
        return ket.todm() if self.is_open else ket

    def observables(self) -> dict[str, QArray]:
        """Bare-state populations ``p1``, ``p2``, ``p3``."""
        return {"p1": self._proj(0), "p2": self._proj(1), "p3": self._proj(2)}


def stirap_pulses(
    horizon: int,
    dt: float,
    omega0: float,
    tau: float,
    width: float,
    order: str = "counterintuitive",
) -> Array:
    """Gaussian pump/Stokes pulse pair sampled on the control grid of an environment.

    Both pulses have peak Rabi frequency ``omega0`` and ``1/e`` half-width ``width``,
    ``Omega(t) = omega0 exp(-((t - t_c) / width)^2)``, and are centred at
    ``T/2 -+ tau`` with ``T = horizon * dt``. In the counter-intuitive order the
    Stokes pulse comes first (centre ``T/2 - tau``) and the pump second
    (``T/2 + tau``); the intuitive order swaps them. Adiabatic transfer needs
    ``omega0 * width >> 1`` (``>= 10`` is comfortably adiabatic) and ``tau`` of the
    order of ``width`` (Vitanov et al., Rev. Mod. Phys. 89, 015006 (2017)).

    Args:
        horizon: number of control steps.
        dt: control period; each amplitude is the pulse sampled at the interval midpoint.
        omega0: peak Rabi frequency of both pulses.
        tau: half the delay between the two pulse centres.
        width: ``1/e`` half-width of each Gaussian pulse.
        order: ``"counterintuitive"`` (Stokes first) or ``"intuitive"`` (pump first).

    Returns:
        Array of shape ``(horizon, 2)`` with columns ``[Omega_p, Omega_s]``, ready for
        :meth:`qenvs.envs.StateTransfer.rollout`.
    """
    if order not in PULSE_ORDERS:
        msg = f"order must be one of {PULSE_ORDERS}, got {order!r}"
        raise ValueError(msg)
    t = (jnp.arange(horizon) + 0.5) * dt
    centre = 0.5 * horizon * dt
    first = omega0 * jnp.exp(-(((t - (centre - tau)) / width) ** 2))
    second = omega0 * jnp.exp(-(((t - (centre + tau)) / width) ** 2))
    pump, stokes = (second, first) if order == "counterintuitive" else (first, second)
    return jnp.stack([pump, stokes], axis=1)
