"""Driven multi-level transmon in the rotating frame of the drive (hardware units).

UNITS: time in nanoseconds (ns), angular frequency in rad/ns, hbar = 1. User-facing
frequencies are ordinary frequencies in GHz and are converted with ``2 pi``
(``omega = 2 pi f``); coherence times ``t1_us`` / ``t2_us`` are in microseconds.

Hamiltonian (rotating-wave approximation, frame rotating at the drive frequency)::

    H(t) = Delta n + (alpha / 2) n (n - 1)
           + (Omega_I(t) / 2) (a + a^dag) + (Omega_Q(t) / 2) i (a^dag - a)

with ``n = a^dag a`` truncated to ``levels`` states, ``Delta = 2 pi (f_q - f_d)`` and
``alpha = 2 pi * anharmonicity_ghz`` (negative for a transmon). The two controls are,
in this order, ``Omega_I`` and ``Omega_Q`` in rad/ns; ``Omega / (2 pi)`` is the Rabi
frequency in GHz. For ``levels = 2`` the control operators are exactly
``sigma_x / 2`` and ``sigma_y / 2`` in the computational basis (index 0 = ground,
index 1 = excited), so a constant ``Omega_I`` with ``Omega_I T = pi`` is an X gate.
They correspond to a lab-frame drive ``Omega_I(t) cos(w_d t) + Omega_Q(t) sin(w_d t)``
coupling to ``a + a^dag`` before the rotating-wave approximation.

Dissipation (Lindblad, enabled when ``t1_us`` and/or ``t2_us`` is given):

* relaxation ``sqrt(1/T1) a``: harmonic-ladder approximation, level ``n`` decays to
  ``n - 1`` at rate ``n / T1``;
* pure dephasing ``sqrt(2/T_phi) n`` with ``1/T_phi = 1/T2 - 1/(2 T1)``, so the
  0-1 coherence decays at ``1/T2`` and the ``m``-``n`` coherence at
  ``(n - m)^2 / T_phi + (m + n) / (2 T1)``.

Randomisable parameters (:meth:`Transmon.parameter_specs`): ``detuning_ghz`` and
``anharmonicity_ghz``, both in GHz. Relaxation and dephasing times, ``levels`` and
``initial`` are not declared. For ``levels = 2`` the anharmonicity term vanishes
(``n (n - 1) = 0``), so randomising it has no effect on the dynamics.

Analytic references are checked in :mod:`qenvs.validation.transmon`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import ClassVar

import dynamiqs as dq
import equinox as eqx
import jax.numpy as jnp
from dynamiqs import QArray
from jax import Array

from qenvs.models.base import DISSIPATION, STRUCTURAL, Model, ParameterSpec

TWO_PI = 2.0 * math.pi
NS_PER_US = 1000.0


def ghz_to_rad_per_ns(f_ghz: float | Array) -> float | Array:
    """Convert an ordinary frequency in GHz to an angular frequency in rad/ns (``2 pi f``)."""
    return TWO_PI * f_ghz


class Transmon(Model):
    """Multi-level transmon (Duffing oscillator) with I/Q drive and optional T1/T2.

    UNITS: time in ns, angular frequency in rad/ns; the fields below are in GHz
    (ordinary frequency) and microseconds.

    Attributes:
        levels: number of transmon levels kept (>= 2; 3 captures leakage to |2>).
        anharmonicity_ghz: ``alpha / (2 pi)`` in GHz (transmon: negative, about -0.3).
        detuning_ghz: ``(f_q - f_d)`` in GHz, qubit frequency minus drive frequency.
        t1_us: energy relaxation time T1 in microseconds (``None`` = no relaxation).
        t2_us: coherence time T2 in microseconds, ``T2 <= 2 T1`` (``None`` = no pure
            dephasing, i.e. ``T2 = 2 T1``).
        initial: initial level as a string, ``'0'`` (ground) by default; used by
            environments to report observables of ``map(initial state)``.
    """

    levels: int = eqx.field(default=3, static=True)
    anharmonicity_ghz: float = -0.34
    detuning_ghz: float = 0.0
    t1_us: float | None = None
    t2_us: float | None = None
    initial: str = "0"

    non_parameters: ClassVar[dict[str, str]] = {
        "levels": STRUCTURAL,
        "initial": STRUCTURAL,
        "t1_us": DISSIPATION,
        "t2_us": DISSIPATION,
    }

    def parameter_specs(self) -> dict[str, ParameterSpec]:
        """``detuning_ghz`` on the whole line and ``anharmonicity_ghz`` in ``(-inf, 0]`` (GHz).

        Reference widths: 2 MHz for the detuning (frequency drift and two-level-system
        shifts of a fixed-frequency transmon between calibrations are of order 0.1 to a few
        MHz, against the 50 MHz Rabi bound of the gate tasks) and 5 MHz for the
        anharmonicity (about 1.5% of -340 MHz, the device-to-device spread of a
        fabricated junction). A transmon anharmonicity is negative, so a Gaussian offset
        needs a ``TruncatedNormal``. At this width the anharmonicity barely changes the
        error of the reference X-gate pulse, so a robustness
        result under ``anharmonicity_ghz`` alone at ``w_ref`` is close to a null test; widen
        it deliberately if that axis is the point.
        """
        inf = float("inf")
        return {
            "detuning_ghz": ParameterSpec("GHz", (-inf, inf), 0.002),
            "anharmonicity_ghz": ParameterSpec("GHz", (-inf, 0.0), 0.005),
        }

    def __check_init__(self) -> None:
        if not isinstance(self.levels, int) or self.levels < 2:
            msg = f"levels must be an integer >= 2, got {self.levels!r}"
            raise ValueError(msg)
        if not (str(self.initial).isdigit() and int(self.initial) < self.levels):
            msg = f"initial must be a level index '0'..'{self.levels - 1}', got {self.initial!r}"
            raise ValueError(msg)
        for name in ("t1_us", "t2_us"):
            v = getattr(self, name)
            if v is not None and not v > 0:
                msg = f"{name} must be positive (microseconds) or None, got {v!r}"
                raise ValueError(msg)
        both = self.t1_us is not None and self.t2_us is not None
        if both and self.t2_us > 2.0 * self.t1_us * (1.0 + 1e-12):
            msg = (
                f"t2_us = {self.t2_us} exceeds 2 * t1_us = {2.0 * self.t1_us}; "
                "T2 <= 2 T1 is required (1/T2 = 1/(2 T1) + 1/T_phi)"
            )
            raise ValueError(msg)

    @classmethod
    def from_properties(
        cls, props: Mapping[str, object], *, levels: int = 3, detuning_ghz: float = 0.0
    ) -> Transmon:
        """Build a transmon from device properties (explicit units or SI values).

        ``props`` is a mapping with ``anharmonicity_ghz``, ``t1_us``, ``t2_us`` and
        ``frequency_ghz``, or with the SI values (seconds and hertz, each either a bare
        number or a ``(value, timestamp)`` tuple) that Qiskit's
        ``BackendProperties.qubit_property(q)`` returned: ``'T1'``, ``'T2'``,
        ``'frequency'``, ``'anharmonicity'``. Qiskit is not imported. The rotating-frame
        model does not use the qubit frequency (give the drive offset as ``detuning_ghz``);
        read it with :func:`qenvs.pulses.parse_properties` if you want it as pulse metadata.

        A missing anharmonicity keeps the model default and emits a ``UserWarning``; missing
        ``t1_us`` / ``t2_us`` mean no dissipation from that source. ``T2 > 2 T1`` raises
        ``ValueError``. See :func:`qenvs.pulses.parse_properties` for the accepted keys.
        """
        import warnings

        from qenvs.pulses import parse_properties

        p = parse_properties(props)
        extra: dict[str, float] = {}
        if p.anharmonicity_ghz is None:
            warnings.warn(
                "no anharmonicity in the device properties; keeping the Transmon default "
                f"anharmonicity_ghz = {cls.anharmonicity_ghz}",
                UserWarning,
                stacklevel=2,
            )
        else:
            if p.anharmonicity_ghz > 0:
                warnings.warn(
                    f"anharmonicity_ghz = {p.anharmonicity_ghz:g} is positive; a transmon has "
                    "a negative anharmonicity (the sign is kept as given)",
                    UserWarning,
                    stacklevel=2,
                )
            extra["anharmonicity_ghz"] = p.anharmonicity_ghz
        return cls(
            levels=levels,
            detuning_ghz=detuning_ghz,
            t1_us=p.t1_us,
            t2_us=p.t2_us,
            **extra,
        )

    # ----------------------------------------------------------------- rates
    @property
    def dim(self) -> int:
        return self.levels

    @property
    def alpha(self) -> float:
        """Anharmonicity in rad/ns."""
        return TWO_PI * self.anharmonicity_ghz

    @property
    def delta(self) -> float:
        """Detuning ``f_q - f_d`` in rad/ns."""
        return TWO_PI * self.detuning_ghz

    @property
    def gamma1(self) -> float:
        """Relaxation rate ``1/T1`` in 1/ns (0 if ``t1_us`` is None)."""
        return 0.0 if self.t1_us is None else 1.0 / (NS_PER_US * self.t1_us)

    @property
    def gamma_phi(self) -> float:
        """Pure dephasing rate ``1/T_phi = 1/T2 - 1/(2 T1)`` in 1/ns (0 if ``t2_us`` is None)."""
        if self.t2_us is None:
            return 0.0
        rate = 1.0 / (NS_PER_US * self.t2_us) - 0.5 * self.gamma1
        # T2 == 2 T1 up to rounding: no pure dephasing
        return max(rate, 0.0) if rate > 1e-12 * self.gamma1 else 0.0

    # ----------------------------------------------------------------- operators
    def _a(self) -> QArray:
        # dense ladder operator a|n> = sqrt(n)|n-1> (dq.destroy is sparse, which warns
        # and densifies when added to the dense drift)
        lower = jnp.diag(jnp.sqrt(jnp.arange(1, self.levels, dtype=float)), 1)
        return dq.asqarray(lower.astype(complex))

    def number(self) -> QArray:
        """Number operator ``n = a^dag a``."""
        a = self._a()
        return a.dag() @ a

    def drift(self) -> QArray:
        n = jnp.arange(self.levels, dtype=float)
        diag = self.delta * n + 0.5 * self.alpha * n * (n - 1.0)
        return dq.asqarray(jnp.diag(diag).astype(complex))

    def controls(self) -> list[QArray]:
        a = self._a()
        return [0.5 * (a + a.dag()), 0.5j * (a.dag() - a)]

    def jump_ops(self) -> list[QArray]:
        ops: list[QArray] = []
        if self.gamma1 > 0:
            ops.append(math.sqrt(self.gamma1) * self._a())
        if self.gamma_phi > 0:
            ops.append(math.sqrt(2.0 * self.gamma_phi) * self.number())
        return ops

    def initial_state(self) -> QArray:
        ket = dq.basis(self.levels, int(self.initial))
        return ket.todm() if self.is_open else ket

    def observables(self) -> dict[str, QArray]:
        obs = {f"p{k}": dq.basis(self.levels, k).todm() for k in range(self.levels)}
        leak = jnp.diag((jnp.arange(self.levels) >= 2).astype(complex))
        obs["leakage"] = dq.asqarray(leak)
        return obs


# ---------------------------------------------------------------------- pulses
def _gaussian_parts(horizon: int, dt: float, sigma: float) -> tuple[Array, Array]:
    """Lifted unit-peak Gaussian ``g`` and its analytic derivative ``dg/dt`` at step midpoints."""
    t = (jnp.arange(horizon) + 0.5) * dt
    center = 0.5 * horizon * dt
    gauss = jnp.exp(-((t - center) ** 2) / (2.0 * sigma**2))
    edge = math.exp(-(center**2) / (2.0 * sigma**2))
    g = (gauss - edge) / (1.0 - edge)
    dg = -(t - center) / sigma**2 * gauss / (1.0 - edge)
    return g, dg


def gaussian_drag(
    horizon: int, dt: float, amplitude: float, sigma: float, beta: float = 0.0
) -> Array:
    """Piecewise-constant Gaussian DRAG pulse, shape ``(horizon, 2)`` = ``[Omega_I, Omega_Q]``.

    UNITS: ``dt`` and ``sigma`` in ns, ``amplitude`` in rad/ns, ``beta`` in ns; the
    returned amplitudes are in rad/ns.

    ``Omega_I(t) = amplitude * (G(t) - G(0)) / (1 - G(0))`` with
    ``G(t) = exp(-(t - T/2)^2 / (2 sigma^2))`` and ``T = horizon * dt``: a Gaussian
    centred in the window and lifted so that it vanishes at the edges ``t = 0`` and
    ``t = T``. ``Omega_Q(t) = beta * dOmega_I/dt``, evaluated analytically. Both are
    sampled at the step midpoints ``t_k = (k + 1/2) dt`` and held for ``dt``.
    Use :func:`drag_beta` for the DRAG coefficient in this library's sign convention.
    """
    g, dg = _gaussian_parts(horizon, dt, sigma)
    return jnp.stack([amplitude * g, beta * amplitude * dg], axis=-1)


def gaussian_amplitude(angle: float, horizon: int, dt: float, sigma: float) -> float:
    """Peak amplitude (rad/ns) giving the sampled I envelope the area ``sum(Omega_I dt) = angle``.

    For a two-level system on resonance this gives a rotation by ``angle`` about x
    (``pi`` for X, ``pi / 2`` for SX).
    """
    g, _ = _gaussian_parts(horizon, dt, sigma)
    return float(angle / (jnp.sum(g) * dt))


def drag_beta(anharmonicity_ghz: float, objective: str = "leakage") -> float:
    """First-order DRAG coefficient ``beta`` (ns) for ``Omega_Q = beta dOmega_I/dt``.

    In this library's convention (``H_Q = (Omega_Q/2) i (a^dag - a)``, drive on
    resonance) the 1-2 transition is driven by the Fourier component of
    ``Omega_I + i Omega_Q`` at ``alpha = 2 pi * anharmonicity_ghz``; integrating by
    parts, ``Omega_I + i beta dOmega_I/dt`` has that component multiplied by
    ``(1 + beta alpha)``, so first-order leakage vanishes for ``beta = -1/alpha``.
    The relative phase error (drive-induced shift of the 0-1 transition) is
    minimised instead at half that value, ``beta = -1/(2 alpha)``. Both values are
    confirmed numerically by ``qenvs.validation.transmon.check_drag_coefficient``.
    Magnitudes agree with ``lambda / |alpha|`` for ``lambda = 1`` (leakage) and
    ``lambda = 0.5`` (phase) in Krantz et al., Appl. Phys. Rev. 6, 021318 (2019),
    Eq. (101); the sign differs because that reference uses a different sign for
    the Q quadrature.

    Args:
        anharmonicity_ghz: ``alpha / (2 pi)`` in GHz (negative for a transmon).
        objective: ``'leakage'`` (``-1/alpha``) or ``'phase'`` (``-1/(2 alpha)``).
    """
    if objective not in ("leakage", "phase"):
        msg = f"objective must be 'leakage' or 'phase', got {objective!r}"
        raise ValueError(msg)
    alpha = TWO_PI * anharmonicity_ghz
    lam = 1.0 if objective == "leakage" else 0.5
    return -lam / alpha
