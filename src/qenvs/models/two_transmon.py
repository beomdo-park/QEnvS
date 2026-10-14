"""Two exchange-coupled transmons driven at the target frequency (cross-resonance), hardware units.

UNITS: time in nanoseconds (ns), angular frequency in rad/ns, hbar = 1. User-facing
frequencies are ordinary frequencies in GHz, converted with ``2 pi``; coherence times are
in microseconds. Tensor order is control (x) target: basis index ``i_c * d + i_t`` for
``d = levels`` per transmon, so the computational states ``|00>, |01>, |10>, |11>`` sit at
indices ``0, 1, d, d + 1``.

Lab-frame Hamiltonian (Duffing transmons, exchange coupling in the rotating-wave
approximation)::

    H_lab = sum_i [w_i n_i + (alpha_i / 2) n_i (n_i - 1)] + J (a^dag b + a b^dag) + drives

with ``a`` (control) and ``b`` (target). Both transmons are written in one common frame
rotating at the drive frequency ``w_d = w_t`` (the bare target frequency), where the
Hamiltonian is time independent apart from the control envelopes::

    H = Delta n_c + (alpha_c / 2) n_c (n_c - 1) + (alpha_t / 2) n_t (n_t - 1)
        + J (a^dag b + a b^dag)
        + (Omega_cI / 2) (a + a^dag) + (Omega_cQ / 2) i (a^dag - a)
        + (Omega_tI / 2) (b + b^dag) + (Omega_tQ / 2) i (b^dag - b)

with ``Delta = 2 pi (f_c - f_t)`` (``detuning_ghz``) and ``J = 2 pi coupling_ghz``. The four
controls, in this order, are the cross-resonance (CR) drive on the control (I, Q) and a
drive on the target (I, Q) at the same frequency, e.g. for cancellation tones; each uses
the operator convention of :class:`~qenvs.models.transmon.Transmon`.

Qubit frames. Gates are defined in the frames rotating at each bare qubit frequency. A
state in the common frame is ``psi_r = exp(i w_d N t) psi_lab`` and in the qubit frames
``psi_q = exp(i (w_c n_c + w_t n_t) t) psi_lab``, so ``psi_q = exp(+i Delta n_c t) psi_r``
and the qubit-frame map after a duration ``T`` is ``exp(+i Delta n_c T) U_r(T)``
(:func:`to_qubit_frame`). With ``J = 0`` and no drive this is the identity on the
computational subspace (checked in ``qenvs.validation.cross_resonance``).

Dissipation (Lindblad, per transmon, enabled by ``t1_*_us`` / ``t2_*_us``): the same
harmonic-ladder relaxation ``sqrt(1/T1) a`` and dephasing ``sqrt(2/T_phi) n`` as
:class:`~qenvs.models.transmon.Transmon`, applied to the bare (uncoupled) operators.
"""

from __future__ import annotations

import math

import dynamiqs as dq
import equinox as eqx
import jax.numpy as jnp
import numpy as np
from dynamiqs import QArray
from jax import Array

from qenvs.gates import _arr, unitary_to_superop
from qenvs.models.base import Model

TWO_PI = 2.0 * math.pi
NS_PER_US = 1000.0


def _rates(t1_us: float | None, t2_us: float | None) -> tuple[float, float]:
    """``(1/T1, 1/T_phi)`` in 1/ns, as in :class:`~qenvs.models.transmon.Transmon`."""
    g1 = 0.0 if t1_us is None else 1.0 / (NS_PER_US * t1_us)
    if t2_us is None:
        return g1, 0.0
    gphi = 1.0 / (NS_PER_US * t2_us) - 0.5 * g1
    return g1, (max(gphi, 0.0) if gphi > 1e-12 * g1 else 0.0)


class TwoTransmon(Model):
    """Control and target transmon with exchange coupling, both driven at the target frequency.

    UNITS: time in ns, angular frequency in rad/ns; the fields below are in GHz (ordinary
    frequency) and microseconds. Tensor order control (x) target.

    Attributes:
        levels: levels kept per transmon (>= 2; 3 captures leakage to |2>).
        detuning_ghz: ``f_c - f_t`` in GHz (bare control minus bare target frequency); the
            drive frequency is ``f_t``.
        anharmonicity_c_ghz: control anharmonicity ``alpha_c / (2 pi)`` in GHz.
        anharmonicity_t_ghz: target anharmonicity ``alpha_t / (2 pi)`` in GHz.
        coupling_ghz: exchange coupling ``J / (2 pi)`` in GHz.
        t1_c_us, t2_c_us: control T1 / T2 in microseconds (``None``: no such channel).
        t1_t_us, t2_t_us: target T1 / T2 in microseconds.
        initial: initial computational state ``'00'``, ``'01'``, ``'10'`` or ``'11'``
            (control digit first); environments report observables of ``map(initial)``.
    """

    levels: int = eqx.field(default=3, static=True)
    detuning_ghz: float = -0.1
    anharmonicity_c_ghz: float = -0.34
    anharmonicity_t_ghz: float = -0.34
    coupling_ghz: float = 0.003
    t1_c_us: float | None = None
    t2_c_us: float | None = None
    t1_t_us: float | None = None
    t2_t_us: float | None = None
    initial: str = "00"

    def __check_init__(self) -> None:
        if not isinstance(self.levels, int) or self.levels < 2:
            msg = f"levels must be an integer >= 2, got {self.levels!r}"
            raise ValueError(msg)
        if self.initial not in ("00", "01", "10", "11"):
            msg = f"initial must be one of '00', '01', '10', '11', got {self.initial!r}"
            raise ValueError(msg)
        for q in ("c", "t"):
            t1, t2 = getattr(self, f"t1_{q}_us"), getattr(self, f"t2_{q}_us")
            for name, v in ((f"t1_{q}_us", t1), (f"t2_{q}_us", t2)):
                if v is not None and not v > 0:
                    msg = f"{name} must be positive (microseconds) or None, got {v!r}"
                    raise ValueError(msg)
            if t1 is not None and t2 is not None and t2 > 2.0 * t1 * (1.0 + 1e-12):
                msg = f"t2_{q}_us = {t2} exceeds 2 * t1_{q}_us = {2.0 * t1}; T2 <= 2 T1 is required"
                raise ValueError(msg)

    # ----------------------------------------------------------------- parameters
    @property
    def dim(self) -> int:
        return self.levels**2

    @property
    def delta(self) -> float:
        """Detuning ``2 pi (f_c - f_t)`` in rad/ns."""
        return TWO_PI * self.detuning_ghz

    @property
    def alpha_c(self) -> float:
        """Control anharmonicity in rad/ns."""
        return TWO_PI * self.anharmonicity_c_ghz

    @property
    def alpha_t(self) -> float:
        """Target anharmonicity in rad/ns."""
        return TWO_PI * self.anharmonicity_t_ghz

    @property
    def coupling(self) -> float:
        """Exchange coupling ``J`` in rad/ns."""
        return TWO_PI * self.coupling_ghz

    @property
    def computational_indices(self) -> tuple[int, int, int, int]:
        """Indices of ``|00>, |01>, |10>, |11>`` (control digit first) in the ``d^2`` basis."""
        d = self.levels
        return (0, 1, d, d + 1)

    # ----------------------------------------------------------------- operators
    def _lower(self) -> Array:
        return jnp.diag(jnp.sqrt(jnp.arange(1, self.levels, dtype=float)), 1).astype(complex)

    def ops(self) -> tuple[QArray, QArray]:
        """Dense ladder operators ``(a, b)`` of control and target on the ``d^2`` space."""
        low, eye = self._lower(), jnp.eye(self.levels, dtype=complex)
        return dq.asqarray(jnp.kron(low, eye)), dq.asqarray(jnp.kron(eye, low))

    def numbers(self) -> tuple[QArray, QArray]:
        """Number operators ``(n_c, n_t)``."""
        a, b = self.ops()
        return a.dag() @ a, b.dag() @ b

    def drift(self) -> QArray:
        n = jnp.arange(self.levels, dtype=float)
        nc = jnp.repeat(n, self.levels)
        nt = jnp.tile(n, self.levels)
        diag = (
            self.delta * nc
            + 0.5 * self.alpha_c * nc * (nc - 1.0)
            + 0.5 * self.alpha_t * nt * (nt - 1.0)
        )
        a, b = self.ops()
        exchange = a.dag() @ b + a @ b.dag()
        return dq.asqarray(jnp.diag(diag).astype(complex)) + self.coupling * exchange

    def controls(self) -> list[QArray]:
        a, b = self.ops()
        return [
            0.5 * (a + a.dag()),
            0.5j * (a.dag() - a),
            0.5 * (b + b.dag()),
            0.5j * (b.dag() - b),
        ]

    def jump_ops(self) -> list[QArray]:
        out: list[QArray] = []
        a, b = self.ops()
        nc, nt = self.numbers()
        for low, num, (t1, t2) in (
            (a, nc, (self.t1_c_us, self.t2_c_us)),
            (b, nt, (self.t1_t_us, self.t2_t_us)),
        ):
            g1, gphi = _rates(t1, t2)
            if g1 > 0:
                out.append(math.sqrt(g1) * low)
            if gphi > 0:
                out.append(math.sqrt(2.0 * gphi) * num)
        return out

    def initial_state(self) -> QArray:
        i = self.computational_indices[int(self.initial, 2)]
        ket = dq.basis(self.dim, i)
        return ket.todm() if self.is_open else ket

    def observables(self) -> dict[str, QArray]:
        idx = self.computational_indices
        names = ("00", "01", "10", "11")
        obs = {f"p{s}": dq.basis(self.dim, i).todm() for s, i in zip(names, idx, strict=True)}
        outside = np.ones(self.dim)
        outside[list(idx)] = 0.0
        obs["leakage"] = dq.asqarray(jnp.diag(jnp.asarray(outside, dtype=complex)))
        return obs

    # ----------------------------------------------------------------- frames, spectra
    def frame_unitary(self, duration: float | Array) -> Array:
        """``exp(+i Delta n_c T)``: common frame to bare qubit frames after ``T = duration`` ns."""
        n = jnp.arange(self.levels, dtype=float)
        nc = jnp.repeat(n, self.levels)
        return jnp.diag(jnp.exp(1j * self.delta * nc * duration))

    def dressed_energies(self) -> Array:
        """Drift eigenvalues assigned to ``|00>, |01>, |10>, |11>`` by maximal overlap (rad/ns).

        Energies are in the common frame; ``E11 - E10 - E01 + E00`` is frame independent.
        """
        evals, evecs = jnp.linalg.eigh(self.drift().to_jax())
        weights = jnp.abs(evecs) ** 2  # weights[i, k] = |<i|v_k>|^2
        idx = jnp.asarray(self.computational_indices)
        return evals[jnp.argmax(weights[idx, :], axis=1)]

    def static_zz(self) -> Array:
        """Static ZZ rate ``zeta = E11 - E10 - E01 + E00`` of the dressed states (rad/ns).

        Exact diagonalisation of the undriven Hamiltonian; with ``H_ZZ = (zeta / 4) Z (x) Z``
        this is the shift of the ``|11>`` energy relative to the uncoupled sum.
        """
        e = self.dressed_energies()
        return e[3] - e[2] - e[1] + e[0]


def static_zz_second_order(
    detuning_ghz: float, anharmonicity_c_ghz: float, anharmonicity_t_ghz: float, coupling_ghz: float
) -> float:
    """Second-order static ZZ ``zeta = 2 J^2 [1/(Delta - alpha_t) - 1/(Delta + alpha_c)]`` (GHz).

    ``Delta = f_c - f_t``. Derivation: only ``|01> <-> |10>`` (matrix element ``J``) and
    ``|11> <-> |20>, |02>`` (``sqrt 2 J``) couple; ``zeta`` collects their second-order
    shifts in ``E11 - E10 - E01 + E00``. Equals ``2 xi`` of Magesan and Gambetta, Phys. Rev.
    A 101, 052308 (2020), Eq. (4.5), whose Hamiltonian term is ``xi Z (x) Z / 2``.
    Works in GHz throughout (``zeta`` is quadratic in J and inverse in frequencies, so the
    ``2 pi`` factors cancel to one overall ``2 pi`` that is not applied here).
    """
    d, ac, at, j = detuning_ghz, anharmonicity_c_ghz, anharmonicity_t_ghz, coupling_ghz
    return 2.0 * j**2 * (1.0 / (d - at) - 1.0 / (d + ac))


def zx_rate_first_order(
    detuning_ghz: float, anharmonicity_c_ghz: float, coupling_ghz: float, rabi_ghz: float
) -> tuple[float, float]:
    """Leading-order CR rates ``(nu_ZX, nu_IX)`` in GHz for ``H_eff = (nu_ZX/2) ZX + (nu_IX/2) IX``.

    ``nu_ZX = -(J Omega / Delta) alpha_c / (alpha_c + Delta)`` and
    ``nu_IX = -J Omega / (Delta + alpha_c)``, with ``Omega`` the Rabi rate of the control's I
    channel (``Omega_cI / 2 pi = rabi_ghz``), ``Delta = f_c - f_t`` and ``Z |0> = +|0>``.
    These are Eq. (4.26) and the IX entry of Appendix C of Magesan and Gambetta,
    Phys. Rev. A 101, 052308 (2020), whose drive term ``Omega cos(w_d t)(b^dag + b)``
    becomes ``(Omega/2)(b^dag + b)`` in the rotating frame (our convention) and whose Pauli
    terms carry a factor 1/2. Our own first-order derivation (dress the control by the
    drive, then project the exchange term) gives the same expressions.
    """
    d, ac, j, w = detuning_ghz, anharmonicity_c_ghz, coupling_ghz, rabi_ghz
    return -(j * w / d) * ac / (ac + d), -j * w / (d + ac)


def to_qubit_frame(model: TwoTransmon, m: object, duration: float | Array) -> Array:
    """Convert a common-frame map (``d^2`` propagator or ``d^4`` superoperator) to qubit frames.

    ``U_q = exp(+i Delta n_c T) U_r`` (unitary) and ``S_q = S_F S_r`` with ``S_F`` the
    column-stacking superoperator of the frame unitary (open systems).
    """
    m = _arr(m)
    f = model.frame_unitary(duration)
    if m.shape[-1] == model.dim:
        return f @ m
    return unitary_to_superop(f) @ m
