"""Collective (Dicke / Tavis-Cummings) quantum battery: N two-level cells and one cavity mode.

The battery is a set of ``n_cells`` identical two-level systems (the cells) and the
charger is a single cavity mode, initially holding ``n_photons`` photons (Ferraro et
al. 2018). Working in the permutation-symmetric (Dicke) subspace with collective
spin ``j = N/2``, the Hamiltonians are (hbar = 1)

    Tavis-Cummings (rwa=True):
        H = omega0 (Jz + N/2) + omega_c a^dag a + g(t) (a^dag J- + a J+)
    Dicke (rwa=False):
        H = omega0 (Jz + N/2) + omega_c a^dag a + g(t) (a + a^dag)(J+ + J-)

with the single control ``g(t)``, the cell-cavity coupling. The two models differ by
the counter-rotating terms ``a J-`` and ``a^dag J+``; dropping them (rotating-wave
approximation) is justified for ``g << omega0`` near resonance. The Tavis-Cummings
Hamiltonian conserves the excitation number ``a^dag a + Jz + N/2``; the Dicke
Hamiltonian only conserves its parity.

Conventions:

* Dicke basis ``|j, m>`` with ``m = -j, ..., +j`` stored at indices ``0, ..., N``:
  index 0 is all cells in the ground state, index N is all cells excited (consistent
  with the library rule "higher index = more excited").
* ``J+`` is built from ``<j, m+1| J+ |j, m> = sqrt(j(j+1) - m(m+1))``, ``J- = J+^dag``
  and ``Jz = diag(-j, ..., j)``.
* Cavity Fock space ``|0>, ..., |n_max>`` with ``a |n> = sqrt(n) |n-1>``.
* Tensor order: battery (first) x cavity (second), ``dim = (N + 1)(n_max + 1)``.
* Battery Hamiltonian ``H_B = omega0 (Jz + N/2)``: 0 when all cells are in the ground
  state, ``N omega0`` when fully charged.
* Energy and ergotropy are properties of the cells alone: :meth:`battery_reduced`
  traces out the cavity and returns ``H_B`` on the ``(N + 1)``-dimensional Dicke space.
  For closed dynamics the reduced cell state is mixed whenever cells and cavity are
  entangled, so the ergotropy is then strictly below the stored energy.
* No diamagnetic ``A^2`` term: this is the bare Dicke Hamiltonian of Ferraro et al. 2018.
* ``g -> -g`` for the whole protocol is a gauge symmetry (``U = (-1)^(a^dag a)`` maps
  ``a -> -a``); a sign change *within* a protocol is physical and can reverse charging.

Dissipation (all collective, i.e. permutation invariant, so the symmetric subspace
stays exact): cavity loss ``sqrt(kappa) a``, collective emission ``sqrt(gamma) J-``
and collective dephasing ``sqrt(gamma_phi) Jz``. Local (per-cell) decay or dephasing
couples the symmetric subspace to lower-``j`` sectors and is out of scope for v0
(a PIQS-style permutation-invariant solver is planned).

Randomisable parameters (:meth:`DickeBattery.parameter_specs`): the cavity frequency
``omega_c`` only. ``omega0`` is refused in 0.3 (see ``non_parameters``): it is also the
energy unit of ``H_B``, so a sampled ``omega0`` would rescale the reward and the
evaluation metric of its episode. Randomise ``omega_c`` to vary the cell-cavity detuning
at a fixed energy scale.
"""

from __future__ import annotations

from typing import ClassVar

import dynamiqs as dq
import jax.numpy as jnp
import numpy as np
from dynamiqs import QArray

from qenvs.models.base import DISSIPATION, STRUCTURAL, Model, ParameterSpec

NON_RWA_EXTRA_PHOTONS = 4
"""Default Fock headroom above ``n_photons`` for the Dicke model (``rwa=False``)."""


def collective_spin_matrices(n_cells: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(J+, J-, Jz)`` in the Dicke basis ``|j, m>``, ``m = -j, ..., +j``.

    Matrices are float64 NumPy arrays of shape ``(N + 1, N + 1)`` with ``j = N/2``;
    index ``i`` holds ``m = -j + i`` (index 0 = all cells ground).

    Args:
        n_cells: number of two-level cells ``N >= 1``.
    """
    j = n_cells / 2.0
    m = -j + np.arange(n_cells + 1, dtype=np.float64)
    jplus = np.zeros((n_cells + 1, n_cells + 1), dtype=np.float64)
    for i in range(n_cells):
        jplus[i + 1, i] = np.sqrt(j * (j + 1) - m[i] * (m[i] + 1))
    return jplus, jplus.T.copy(), np.diag(m)


SECOND_MOMENT_TOL = 1e-12
"""Relative Hermiticity tolerance of :func:`spin_second_moments`."""


def spin_second_moments(jp: np.ndarray, jm: np.ndarray, jz: np.ndarray) -> dict[str, np.ndarray]:
    """The six symmetrised second moments of a collective spin, as dense complex matrices.

    With ``Jx = (J+ + J-)/2`` and ``Jy = (J+ - J-)/(2i)`` the returned entries are
    ``Jx2 = Jx^2``, ``Jy2``, ``Jz2``, ``JxJy = (Jx Jy + Jy Jx)/2``, ``JyJz`` and
    ``JzJx``. Every entry is Hermitian; this is checked (``ValueError`` beyond a relative
    ``1e-12``) and then enforced exactly. ``jp``, ``jm``, ``jz`` may act on any space (for
    example the Dicke space tensored with the cavity identity).
    """
    jx = 0.5 * (jp + jm)
    jy = (jp - jm) / 2j
    jz = jz.astype(complex)

    def sym(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return 0.5 * (a @ b + b @ a)

    out = {
        "Jx2": jx @ jx,
        "Jy2": jy @ jy,
        "Jz2": jz @ jz,
        "JxJy": sym(jx, jy),
        "JyJz": sym(jy, jz),
        "JzJx": sym(jz, jx),
    }
    for name, m in out.items():
        asym = float(np.max(np.abs(m - m.conj().T), initial=0.0))
        if asym > SECOND_MOMENT_TOL * max(float(np.max(np.abs(m), initial=0.0)), 1.0):
            msg = f"second moment {name} is not Hermitian: max|A - A^dagger| = {asym:.3e}"
            raise ValueError(msg)
        out[name] = 0.5 * (m + m.conj().T)
    return out


def boson_matrices(n_max: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(a, a^dag)`` on the truncated Fock space ``|0>, ..., |n_max>``.

    Args:
        n_max: highest Fock state kept (dimension ``n_max + 1``).
    """
    n = np.arange(1, n_max + 1, dtype=np.float64)
    a = np.diag(np.sqrt(n), k=1)
    return a, a.T.copy()


class DickeBattery(Model):
    """N two-level cells collectively coupled to one cavity mode (Dicke / Tavis-Cummings).

    Attributes:
        n_cells: number of cells ``N`` (collective spin ``j = N/2``).
        n_photons: initial cavity Fock number; ``None`` means ``n_cells`` (Ferraro
            protocol: as many photons as cells).
        n_max: Fock cutoff. ``None`` means ``n_photons`` for ``rwa=True`` (exact,
            since Tavis-Cummings conserves the excitation number) and
            ``n_photons + 4`` for ``rwa=False`` (counter-rotating terms create
            photons; the user must check convergence in the cutoff).
        omega0: cell transition frequency, energy unit of ``H_B``.
        omega_c: cavity frequency (detuning ``omega_c - omega0``).
        rwa: ``True`` for Tavis-Cummings, ``False`` for the full Dicke model.
        kappa: cavity photon loss rate (jump ``sqrt(kappa) a``).
        gamma: collective emission rate (jump ``sqrt(gamma) J-``).
        gamma_phi: collective dephasing rate (jump ``sqrt(gamma_phi) Jz``).
    """

    n_cells: int = 3
    n_photons: int | None = None
    n_max: int | None = None
    omega0: float = 1.0
    omega_c: float = 1.0
    rwa: bool = True
    kappa: float = 0.0
    gamma: float = 0.0
    gamma_phi: float = 0.0

    non_parameters: ClassVar[dict[str, str]] = {
        "omega0": (
            "not supported in this release: omega0 is also the energy unit of H_B, so a sampled "
            "omega0 rescales the reward and the evaluation metric of its episode (a change of "
            "units, roughly 7 to 10 times larger than its effect on the charging dynamics), "
            "not only the physics. Randomise omega_c for a cell-cavity detuning at a fixed "
            "energy scale. Support would need a scale-normalised objective (for example energy "
            "over N omega0) and the same contract on the other battery models"
        ),
        "n_cells": STRUCTURAL,
        "n_photons": STRUCTURAL,
        "n_max": STRUCTURAL,
        "rwa": STRUCTURAL,
        "kappa": DISSIPATION,
        "gamma": DISSIPATION,
        "gamma_phi": DISSIPATION,
    }

    def parameter_specs(self) -> dict[str, ParameterSpec]:
        """``omega_c`` in ``[0, inf)``, reference width 0.05.

        Units are those of the model (``omega0 = omega_c = 1`` by default). A frequency is
        non-negative. 0.05 is a 5% spread, the order of the frequency mismatch between an
        emitter ensemble and its cavity. ``omega0`` is not declared in this release (it is
        also the energy unit of ``H_B``; see ``non_parameters``). It is a modest
        perturbation of the charging protocol: with the constant coupling ``g = 0.25``
        over the default episode (20 steps of 0.2, ``N = 3``) the stored energy
        ``<H_B>`` changes from 2.582 to 2.568 for ``omega_c = 0.95`` or ``1.05``
        (0.5%) and to 2.527 for ``omega_c = 1.1``; a pulse tuned for the nominal
        detuning loses more.
        """
        return {"omega_c": ParameterSpec("omega0", (0.0, float("inf")), 0.05)}

    def __check_init__(self) -> None:
        if self.n_cells < 1:
            msg = f"n_cells must be >= 1, got {self.n_cells}"
            raise ValueError(msg)
        if self.photons < 0:
            msg = f"n_photons must be >= 0, got {self.photons}"
            raise ValueError(msg)
        if self.cutoff < self.photons:
            msg = f"n_max ({self.cutoff}) must be >= n_photons ({self.photons})"
            raise ValueError(msg)
        for name in ("kappa", "gamma", "gamma_phi"):
            if getattr(self, name) < 0:
                msg = f"{name} must be >= 0, got {getattr(self, name)}"
                raise ValueError(msg)

    # ------------------------------------------------------------ resolved sizes
    @property
    def photons(self) -> int:
        """Initial cavity photon number (``n_photons`` with its default resolved)."""
        return self.n_cells if self.n_photons is None else int(self.n_photons)

    @property
    def cutoff(self) -> int:
        """Fock cutoff ``n_max`` with its default resolved."""
        if self.n_max is not None:
            return int(self.n_max)
        return self.photons if self.rwa else self.photons + NON_RWA_EXTRA_PHOTONS

    @property
    def j(self) -> float:
        """Collective spin ``j = N/2``."""
        return self.n_cells / 2.0

    @property
    def dim_battery(self) -> int:
        """Dimension of the symmetric spin subspace, ``N + 1``."""
        return self.n_cells + 1

    @property
    def dim_cavity(self) -> int:
        """Dimension of the truncated Fock space, ``n_max + 1``."""
        return self.cutoff + 1

    @property
    def dim(self) -> int:
        return self.dim_battery * self.dim_cavity

    # ------------------------------------------------------------------ operators
    def _qarray(self, matrix: np.ndarray) -> QArray:
        """Wrap a dense ``(dim, dim)`` NumPy matrix as a QArray with composite dims."""
        return dq.asqarray(jnp.asarray(matrix), dims=(self.dim_battery, self.dim_cavity))

    def _spin(self, which: int) -> QArray:
        mats = collective_spin_matrices(self.n_cells)
        return self._qarray(np.kron(mats[which], np.eye(self.dim_cavity)))

    def jplus(self) -> QArray:
        """Collective raising operator ``J+ (x) 1``."""
        return self._spin(0)

    def jminus(self) -> QArray:
        """Collective lowering operator ``J- (x) 1``."""
        return self._spin(1)

    def jz(self) -> QArray:
        """Collective inversion ``Jz (x) 1`` with eigenvalues ``-N/2, ..., N/2``."""
        return self._spin(2)

    def a(self) -> QArray:
        """Cavity annihilation operator ``1 (x) a``."""
        a, _ = boson_matrices(self.cutoff)
        return self._qarray(np.kron(np.eye(self.dim_battery), a))

    def adag(self) -> QArray:
        """Cavity creation operator ``1 (x) a^dag``."""
        return self.a().dag()

    def number(self) -> QArray:
        """Photon number ``a^dag a``."""
        return self.adag() @ self.a()

    def excitation_number(self) -> QArray:
        """Total excitation number ``a^dag a + Jz + N/2`` (conserved by Tavis-Cummings)."""
        return self.number() + self.jz() + self.j * self._qarray(np.eye(self.dim))

    # --------------------------------------------------------------------- Model
    def drift(self) -> QArray:
        return self.battery_hamiltonian() + self.omega_c * self.number()

    def controls(self) -> list[QArray]:
        if self.rwa:
            return [self.adag() @ self.jminus() + self.a() @ self.jplus()]
        return [(self.a() + self.adag()) @ (self.jplus() + self.jminus())]

    def jump_ops(self) -> list[QArray]:
        ops: list[QArray] = []
        if self.kappa > 0:
            ops.append(jnp.sqrt(self.kappa) * self.a())
        if self.gamma > 0:
            ops.append(jnp.sqrt(self.gamma) * self.jminus())
        if self.gamma_phi > 0:
            ops.append(jnp.sqrt(self.gamma_phi) * self.jz())
        return ops

    def initial_state(self) -> QArray:
        """All cells in the ground state, cavity in Fock state ``|n_photons>``."""
        ket = dq.tensor(dq.basis(self.dim_battery, 0), dq.basis(self.dim_cavity, self.photons))
        return ket.todm() if self.is_open else ket

    def battery_hamiltonian(self) -> QArray:
        """``H_B = omega0 (Jz + N/2)`` on the full space (0 all ground, ``N omega0`` charged)."""
        return self.omega0 * (self.jz() + self.j * self._qarray(np.eye(self.dim)))

    def local_battery_hamiltonian(self) -> QArray:
        """``H_B = omega0 (Jz + N/2)`` on the ``(N + 1)``-dimensional Dicke space alone."""
        _, _, jz = collective_spin_matrices(self.n_cells)
        return self.omega0 * dq.asqarray(jnp.asarray(jz + self.j * np.eye(self.dim_battery)))

    def battery_reduced(self, qstate: QArray) -> tuple[QArray, QArray]:
        """Reduced cell state ``tr_cavity(rho)`` and ``H_B`` on the Dicke space.

        Energy and ergotropy are properties of the cells, so the cavity (subsystem 1)
        is traced out first. Evaluating the ergotropy on the full cells+cavity state
        with ``H_B (x) 1`` would give ``ergotropy == energy`` for every pure global
        state (closed dynamics); on the reduced state it is strictly smaller whenever
        the cells are entangled with the cavity.
        """
        return dq.ptrace(qstate, 0), self.local_battery_hamiltonian()

    def top_fock(self) -> QArray:
        """Projector ``1 (x) |n_max><n_max|`` on the highest kept Fock level.

        Its expectation value is the population at the Fock cutoff; a value that is
        not negligible means the truncation is affecting the dynamics (raise ``n_max``).
        """
        proj = np.zeros((self.dim_cavity, self.dim_cavity))
        proj[self.cutoff, self.cutoff] = 1.0
        return self._qarray(np.kron(np.eye(self.dim_battery), proj))

    def observables(self) -> dict[str, QArray]:
        return {
            "energy": self.battery_hamiltonian(),
            "n_photons": self.number(),
            "excitation": self.excitation_number(),
            "top_fock": self.top_fock(),
        }

    def correlation_observables(self) -> dict[str, QArray]:
        """Second moments of the collective spin on the full (cells x cavity) space.

        ``Jx2``, ``Jy2``, ``Jz2``, ``JxJy``, ``JyJz`` and ``JzJx``, where the mixed ones
        are symmetrised, ``JxJy = (Jx Jy + Jy Jx)/2``, so all six are Hermitian (checked).
        """
        eye = np.eye(self.dim_cavity)
        jp, jm, jz = (np.kron(m, eye) for m in collective_spin_matrices(self.n_cells))
        return {k: self._qarray(v) for k, v in spin_second_moments(jp, jm, jz).items()}
