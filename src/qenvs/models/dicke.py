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
"""

from __future__ import annotations

import dynamiqs as dq
import jax.numpy as jnp
import numpy as np
from dynamiqs import QArray

from qenvs.models.base import Model

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
