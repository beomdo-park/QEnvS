"""Dicke / Tavis-Cummings battery with local (per-cell) dissipation, permutation-invariant method.

Same cells, cavity and Hamiltonians as :class:`~qenvs.models.dicke.DickeBattery`
(``hbar = 1``)::

    Tavis-Cummings (rwa=True):  H = omega0 (Jz + N/2) + omega_c a^dag a + g(t) (a^dag J- + a J+)
    Dicke (rwa=False):          H = omega0 (Jz + N/2) + omega_c a^dag a + g(t) (a + a^dag)(J+ + J-)

plus Lindblad dissipation that acts on every cell separately (rates are per cell)::

    local emission    sqrt(gamma_local)     sigma_-^(i)        (sigma_- = |0><1|, index 1 excited)
    local dephasing   sqrt(gamma_phi_local) sigma_z^(i) / 2    (sigma_z = |1><1| - |0><0|)
    local pumping     sqrt(pumping_local)   sigma_+^(i)        (sigma_+ = |1><0|)

and the collective processes of ``battery/Dicke-v0``: cavity loss ``sqrt(kappa) a``,
collective emission ``sqrt(gamma) J-`` and collective dephasing ``sqrt(gamma_phi) Jz``.

Method (Shammah et al., Phys. Rev. A 98, 063815 (2018)). Local dissipators break the
symmetric (``j = N/2``) subspace but keep the joint cells+cavity state invariant under
permutations of the cells. By Schur-Weyl duality such a state is block diagonal in the
total-spin (Schur) basis ``|j, m, alpha> (x) |n>``::

    rho = sum_j  R_j (x) 1_{d_j},     j = N/2, N/2 - 1, ..., 0 or 1/2,

where ``R_j`` acts on spin ``j`` (x) cavity and ``1_{d_j}`` on the multiplicity space of
dimension ``d_j = C(N, N/2 - j) - C(N, N/2 - j - 1)`` (``sum_j d_j (2j + 1) = 2^N``).
The dynamics is closed on the blocks ``sigma_j = d_j R_j`` (unit total trace), whose
number of entries grows only polynomially, ``sum_j ((2j + 1) n_c)^2`` with
``n_c = n_max + 1``. The Liouvillian on the vectorised blocks (row-major,
``vec(A X B) = (A (x) B^T) vec(X)``) is

* ``-i [H, .]`` and the collective dissipators ``D[L] X = L X L^dag - {L^dag L, X}/2``
  sector by sector (collective operators act within each ``j``);
* the local dissipators through the closed-form coefficients that couple
  ``sigma_{j, m, m'}`` to ``sigma_{j', m +- 1, m' +- 1}`` and ``sigma_{j', m, m'}``,
  ``j' in {j - 1, j, j + 1}`` (Shammah et al. 2018; the formulas are transcribed in
  float64 from the QuTiP PIQS implementation, BSD-3-Clause), with the cavity index a
  spectator.

Each control step is exact for piecewise-constant ``g``: ``sigma(t + dt) =
expm(L(g) dt) sigma(t)`` (dense ``jax.scipy.linalg.expm``; differentiable, ``jit`` and
``vmap`` compatible).

State representation (what ``EnvState.qstate`` holds). The model stores the *full*
``2^N n_c``-dimensional density matrix of cells and cavity written in the Schur basis
(a fixed unitary change of the cells' basis), block diagonal with ``d_j`` identical copies
of ``R_j = sigma_j / d_j``. Every basis-independent quantity is then exact without special
cases: trace, purity ``tr rho^2``, expectation values of the collective observables, the
partial trace over the cavity, and the energy and ergotropy of the full ``2^N``-dimensional
cell state. The ergotropy is computed by :meth:`battery_reduced` (cavity traced out) on
that ``2^N`` matrix, i.e. the eigenvalues of each block ``sigma_j`` divided by ``d_j`` and
repeated ``d_j`` times, paired with ``H_B = omega0 (Jz + N/2)`` whose level ``k`` has
multiplicity ``C(N, k)``; :meth:`ergotropy_from_blocks` is the same quantity computed
from the blocks directly.

Ordering: sectors ``j = N/2, N/2 - 1, ...``; inside a sector the copies
``alpha = 0, ..., d_j - 1``, then ``m = -j, ..., +j`` (index 0 = lowest ``m``, library
rule "higher index = more excited"), then the Fock index ``n``. The qarrays carry
``dims = (2^N, n_max + 1)``, so ``dq.ptrace(state, 0)`` is the cell state. The initial
state (all cells ground, cavity in ``|n_photons>``) lies in the ``j = N/2`` sector.

Fock cutoff. For ``rwa=True`` without local pumping the excitation number
``a^dag a + Jz + N/2`` is conserved by the Hamiltonian and never increases under the
dissipators, so the default cutoff ``n_max = n_photons`` is exact. With
``pumping_local > 0`` that no longer holds: a pumped cell can emit into the cavity and be
pumped again, so the excitation number is unbounded and *no* finite cutoff is exact. The
constructor then requires an explicit ``n_max`` and the population of the top Fock level
(:meth:`DickeBatteryLocal.top_fock`, published as ``info['top_fock']``) is the convergence
diagnostic: raise ``n_max`` until it is negligible. For ``rwa=False`` the default
``n_photons + 4`` is not a convergence guarantee either.

Scope: dense Liouvillians. The construction raises ``ValueError`` when the Liouvillian
dimension exceeds :data:`MAX_LIOUVILLIAN_DIM` or the stored matrix dimension exceeds
:data:`MAX_HILBERT_DIM`. Rates are static model parameters (not differentiable).

Randomisable parameters (:meth:`DickeBatteryLocal.parameter_specs`): ``omega_c`` only,
as in :class:`~qenvs.models.dicke.DickeBattery` (``omega0`` is refused in 0.3: it is also
the energy unit of ``H_B``). The dissipator is cached by its (static) rates and does not
depend on the frequencies, so a randomised ``omega_c`` never changes it.
"""

from __future__ import annotations

import functools
import math
from typing import ClassVar, NamedTuple

import dynamiqs as dq
import jax
import jax.numpy as jnp
import numpy as np
from dynamiqs import QArray, TimeQArray
from dynamiqs.method import Method
from jax import Array

from qenvs.models.base import DISSIPATION, STRUCTURAL, Model, ParameterSpec
from qenvs.models.dicke import (
    NON_RWA_EXTRA_PHOTONS,
    boson_matrices,
    collective_spin_matrices,
    spin_second_moments,
)

MAX_LIOUVILLIAN_DIM = 2100
"""Largest allowed ``sum_j ((2j + 1)(n_max + 1))^2`` (dense ``expm`` per control step)."""

MAX_HILBERT_DIM = 512
"""Largest allowed ``2^N (n_max + 1)`` (dimension of the stored density matrix)."""


# --------------------------------------------------------------------- structure
def sector_spins(n_cells: int) -> tuple[int, ...]:
    """Twice the total spin of every sector, ``2j = N, N - 2, ..., N mod 2``."""
    return tuple(range(n_cells, -1, -2))


def degeneracy(n_cells: int, two_j: int) -> int:
    """Multiplicity ``d_j = C(N, N/2 - j) - C(N, N/2 - j - 1)`` of spin ``j = two_j / 2``."""
    k = (n_cells - two_j) // 2
    return math.comb(n_cells, k) - (math.comb(n_cells, k - 1) if k >= 1 else 0)


class Layout(NamedTuple):
    """Static index bookkeeping for ``n_cells`` cells and ``n_c`` Fock levels."""

    two_js: tuple[int, ...]
    degeneracies: tuple[int, ...]
    block_sizes: tuple[int, ...]  # (2j + 1) n_c
    full_offsets: tuple[int, ...]  # first row of each sector in the stored matrix
    vec_offsets: tuple[int, ...]  # first entry of each sector in the compact vector
    gather: np.ndarray  # compact entry -> flat index of its first copy in the stored matrix
    weights: np.ndarray  # d_j of every compact entry
    scatter: np.ndarray  # flat index of the stored matrix -> compact entry (or size: zero)
    dim: int  # 2^N n_c
    size: int  # Liouvillian dimension


@functools.lru_cache(maxsize=64)
def layout(n_cells: int, n_c: int) -> Layout:
    """Index maps between the stored ``2^N n_c`` matrix and the compact block vector."""
    two_js = sector_spins(n_cells)
    degs = tuple(degeneracy(n_cells, tj) for tj in two_js)
    sizes = tuple((tj + 1) * n_c for tj in two_js)
    dim = (2**n_cells) * n_c
    full_off, vec_off = [], []
    f = v = 0
    for d, b in zip(degs, sizes, strict=True):
        full_off.append(f)
        vec_off.append(v)
        f += d * b
        v += b * b
    size = v
    gather = np.empty(size, dtype=np.int64)
    weights = np.empty(size, dtype=np.float64)
    scatter = np.full(dim * dim, size, dtype=np.int64)
    for d, b, fo, vo in zip(degs, sizes, full_off, vec_off, strict=True):
        p, q = np.meshgrid(np.arange(b), np.arange(b), indexing="ij")
        compact = vo + (p * b + q).reshape(-1)
        gather[compact] = ((fo + p) * dim + (fo + q)).reshape(-1)
        weights[compact] = d
        for alpha in range(d):
            base = fo + alpha * b
            scatter[((base + p) * dim + (base + q)).reshape(-1)] = compact
    for arr in (gather, weights, scatter):
        arr.setflags(write=False)  # cached and shared
    return Layout(
        two_js, degs, sizes, tuple(full_off), tuple(vec_off), gather, weights, scatter, dim, size
    )


# ------------------------------------------------------------- local dissipator
def _sqrt(x: float) -> float:
    return math.sqrt(x) if x > 0 else 0.0


def local_cell_dissipator(
    n_cells: int, gamma_local: float, gamma_phi_local: float, pumping_local: float
) -> np.ndarray:
    """Local-process generator on the cells' compact Dicke blocks (float64, cavity excluded).

    Acts on the concatenation of the row-major vectorised blocks ``sigma_j`` (sectors
    ``j = N/2, N/2 - 1, ...``, ``m`` ascending inside a block) of a permutation-invariant
    cell state normalised as ``sum_j tr sigma_j = 1``. Jump operators per cell:
    ``sqrt(gamma_local) sigma_-``, ``sqrt(gamma_phi_local) sigma_z / 2`` and
    ``sqrt(pumping_local) sigma_+``. Coefficients after Shammah et al. (2018), transcribed
    from the QuTiP PIQS implementation (BSD-3-Clause) and evaluated in float64.
    """
    n = float(n_cells)
    half = n / 2.0
    ge, gd, gp = float(gamma_local), float(gamma_phi_local), float(pumping_local)
    two_js = sector_spins(n_cells)
    offsets, size = {}, 0
    for tj in two_js:
        offsets[tj] = size
        size += (tj + 1) ** 2

    def index(tj: int, m: float, m1: float) -> int | None:
        j = tj / 2.0
        if tj not in offsets or abs(m) > j + 1e-9 or abs(m1) > j + 1e-9:
            return None
        b = tj + 1
        return offsets[tj] + round(m + j) * b + round(m1 + j)

    out = np.zeros((size, size), dtype=np.float64)
    for tj in two_js:
        j = tj / 2.0
        ms = [-j + k for k in range(tj + 1)]
        for m in ms:
            for m1 in ms:
                row = index(tj, m, m1)
                dephase = (
                    gd * n / 4.0
                    if j <= 0
                    else gd / 2.0 * (half - m * m1 * (half + 1) / (j * (j + 1)))
                )
                out[row, row] -= ge / 2.0 * (n + m + m1) + gp / 2.0 * (n - m - m1) + dephase
                terms: list[tuple[int, float, float, float]] = []
                # emission: source (J, M, M1) = (j + {0, +1, -1}, m + 1, m1 + 1)
                big_j, big_m, big_m1 = j, m + 1, m1 + 1
                if ge and j > 0:
                    c = _sqrt((big_j + big_m) * (big_j - big_m + 1) * (big_j + big_m1))
                    c *= _sqrt(big_j - big_m1 + 1)
                    terms.append((tj, big_m, big_m1, ge / 2 * c * (half + 1) / (j * (j + 1))))
                big_j = j + 1
                if ge:
                    c = _sqrt((big_j + big_m) * (big_j + big_m - 1))
                    c *= _sqrt((big_j + big_m1) * (big_j + big_m1 - 1))
                    terms.append(
                        (
                            tj + 2,
                            big_m,
                            big_m1,
                            ge / 2 * c * (half + big_j + 1) / (big_j * (2 * big_j + 1)),
                        )
                    )
                big_j = j - 1
                if ge and big_j >= 0:
                    c = _sqrt((big_j - big_m + 1) * (big_j - big_m + 2))
                    c *= _sqrt((big_j - big_m1 + 1) * (big_j - big_m1 + 2))
                    coef = ge / 2 * c * (half - big_j) / ((big_j + 1) * (2 * big_j + 1))
                    terms.append((tj - 2, big_m, big_m1, coef))
                # dephasing: source (j + {+1, -1}, m, m1)
                big_j = j + 1
                if gd:
                    c = _sqrt((big_j**2 - m**2) * (big_j**2 - m1**2))
                    coef = gd / 2 * c * (half + big_j + 1) / (big_j * (2 * big_j + 1))
                    terms.append((tj + 2, m, m1, coef))
                big_j = j - 1
                if gd and big_j >= 0:
                    c = _sqrt(((big_j + 1) ** 2 - m**2) * ((big_j + 1) ** 2 - m1**2))
                    coef = gd / 2 * c * (half - big_j) / ((big_j + 1) * (2 * big_j + 1))
                    terms.append((tj - 2, m, m1, coef))
                # pumping: source (j + {+1, 0, -1}, m - 1, m1 - 1)
                big_m, big_m1 = m - 1, m1 - 1
                big_j = j + 1
                if gp:
                    c = _sqrt((big_j - big_m - 1) * (big_j - big_m))
                    c *= _sqrt((big_j - big_m1 - 1) * (big_j - big_m1))
                    coef = gp / 2 * c * (half + big_j + 1) / (big_j * (2 * big_j + 1))
                    terms.append((tj + 2, big_m, big_m1, coef))
                big_j = j
                if gp and j > 0:
                    c = _sqrt((big_j + big_m + 1) * (big_j - big_m) * (big_j + big_m1 + 1))
                    c *= _sqrt(big_j - big_m1)
                    terms.append((tj, big_m, big_m1, gp / 2 * c * (half + 1) / (j * (j + 1))))
                big_j = j - 1
                if gp and big_j >= 0:
                    c = _sqrt((big_j + big_m + 1) * (big_j + big_m + 2))
                    c *= _sqrt((big_j + big_m1 + 1) * (big_j + big_m1 + 2))
                    coef = gp / 2 * c * (half - big_j) / ((big_j + 1) * (2 * big_j + 1))
                    terms.append((tj - 2, big_m, big_m1, coef))
                for src_tj, sm, sm1, coef in terms:
                    col = index(src_tj, sm, sm1)
                    if col is not None and coef != 0.0:
                        out[row, col] += coef
    return out


def _lindblad_superop(op: np.ndarray) -> np.ndarray:
    """Row-major superoperator of ``D[L] X = L X L^dag - {L^dag L, X} / 2``."""
    eye = np.eye(op.shape[0])
    ldl = op.conj().T @ op
    return np.kron(op, op.conj()) - 0.5 * np.kron(ldl, eye) - 0.5 * np.kron(eye, ldl.T)


def _sector_operators(two_j: int, n_c: int) -> dict[str, np.ndarray]:
    """Spin-``j`` (x) cavity operators of one sector (spin first, Fock second)."""
    jp, jm, jz = collective_spin_matrices(two_j)
    a, adag = boson_matrices(n_c - 1)
    es, ec = np.eye(two_j + 1), np.eye(n_c)
    return {
        "jp": np.kron(jp, ec),
        "jm": np.kron(jm, ec),
        "jz": np.kron(jz, ec),
        "a": np.kron(es, a),
        "adag": np.kron(es, adag),
        "eye": np.eye((two_j + 1) * n_c),
    }


@functools.lru_cache(maxsize=16)
def _dissipator(
    n_cells: int,
    n_c: int,
    kappa: float,
    gamma: float,
    gamma_phi: float,
    gamma_local: float,
    gamma_phi_local: float,
    pumping_local: float,
) -> np.ndarray:
    """Full (collective + local) dissipator on the compact joint vector, float64."""
    lay = layout(n_cells, n_c)
    out = np.zeros((lay.size, lay.size), dtype=np.complex128)
    for tj, vo, b in zip(lay.two_js, lay.vec_offsets, lay.block_sizes, strict=True):
        ops = _sector_operators(tj, n_c)
        block = np.zeros((b * b, b * b), dtype=np.complex128)
        for rate, op in ((kappa, ops["a"]), (gamma, ops["jm"]), (gamma_phi, ops["jz"])):
            if rate > 0:
                block += rate * _lindblad_superop(op)
        out[vo : vo + b * b, vo : vo + b * b] = block
    if gamma_local > 0 or gamma_phi_local > 0 or pumping_local > 0:
        cells = local_cell_dissipator(n_cells, gamma_local, gamma_phi_local, pumping_local)
        # cells compact index -> (sector position, a, a1); joint index adds Fock n, n1
        cell_index = []
        for s, tj in enumerate(lay.two_js):
            for a in range(tj + 1):
                for a1 in range(tj + 1):
                    cell_index.append((s, a, a1))
        n, n1 = np.meshgrid(np.arange(n_c), np.arange(n_c), indexing="ij")
        n, n1 = n.reshape(-1), n1.reshape(-1)

        def joint(s: int, a: int, a1: int) -> np.ndarray:
            b = lay.block_sizes[s]
            return lay.vec_offsets[s] + (a * n_c + n) * b + (a1 * n_c + n1)

        rows, cols = np.nonzero(cells)
        for r, c in zip(rows, cols, strict=True):
            out[joint(*cell_index[r]), joint(*cell_index[c])] += cells[r, c]
    out.setflags(write=False)  # shared through lru_cache: callers must not mutate it
    return out


class DickeBatteryLocal(Model):
    """N cells collectively coupled to a cavity, with local and collective dissipation.

    Attributes:
        n_cells: number of cells ``N``.
        n_photons: initial cavity Fock number; ``None`` means ``n_cells``.
        n_max: Fock cutoff. ``None`` means ``n_photons`` for ``rwa=True`` (exact when
            ``pumping_local = 0``: the Tavis-Cummings excitation number cannot exceed its
            initial value) and ``n_photons + 4`` for ``rwa=False`` (not a convergence
            guarantee, see ``battery/Dicke-v0``). With ``pumping_local > 0`` the
            excitation number is unbounded (a pumped cell emits into the cavity and is
            pumped again), no finite cutoff is exact, and ``n_max`` must be given
            explicitly; check convergence with :meth:`top_fock` (``info['top_fock']``),
            e.g. by comparing ``n_max`` and ``n_max + 4``.
        omega0: cell transition frequency, energy unit of ``H_B``.
        omega_c: cavity frequency.
        rwa: ``True`` for Tavis-Cummings, ``False`` for the full Dicke model.
        kappa: cavity loss rate (jump ``sqrt(kappa) a``).
        gamma: collective emission rate (jump ``sqrt(gamma) J-``).
        gamma_phi: collective dephasing rate (jump ``sqrt(gamma_phi) Jz``).
        gamma_local: per-cell emission rate (jump ``sqrt(gamma_local) sigma_-`` on each cell).
        gamma_phi_local: per-cell dephasing rate (jump ``sqrt(gamma_phi_local) sigma_z / 2``;
            a single-cell coherence decays at ``gamma_phi_local / 2``).
        pumping_local: per-cell incoherent pumping rate (jump ``sqrt(pumping_local) sigma_+``).
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
    gamma_local: float = 0.0
    gamma_phi_local: float = 0.0
    pumping_local: float = 0.0

    def __check_init__(self) -> None:
        if self.n_cells < 1:
            msg = f"n_cells must be >= 1, got {self.n_cells}"
            raise ValueError(msg)
        if self.photons < 0:
            msg = f"n_photons must be >= 0, got {self.photons}"
            raise ValueError(msg)
        if self.n_max is None and self.pumping_local > 0:
            msg = (
                "n_max must be given explicitly when pumping_local > 0: a pumped cell can emit "
                "into the cavity and be pumped again, so the excitation number a^dag a + Jz + "
                "N/2 is unbounded and no finite Fock cutoff is exact (the default n_photons "
                "holds only without pumping). Choose n_max and raise it until the population of "
                "the top Fock level (model.top_fock(), info['top_fock']) is negligible, e.g. "
                "compare n_max and n_max + 4"
            )
            raise ValueError(msg)
        if self.cutoff < self.photons:
            msg = f"n_max ({self.cutoff}) must be >= n_photons ({self.photons})"
            raise ValueError(msg)
        for name in self.RATES:
            if getattr(self, name) < 0:
                msg = f"{name} must be >= 0, got {getattr(self, name)}"
                raise ValueError(msg)
        lay_dim = (2**self.n_cells) * self.dim_cavity
        size = self.dim_cavity**2 * sum((tj + 1) ** 2 for tj in sector_spins(self.n_cells))
        if size > MAX_LIOUVILLIAN_DIM or lay_dim > MAX_HILBERT_DIM:
            msg = (
                f"n_cells={self.n_cells}, n_max={self.cutoff}: Liouvillian dimension {size} "
                f"(limit {MAX_LIOUVILLIAN_DIM}) and stored state dimension {lay_dim} (limit "
                f"{MAX_HILBERT_DIM}); this dense permutation-invariant solver supports e.g. "
                "N <= 5 with n_max <= 5, N = 4 with n_max <= 6, N = 3 with n_max <= 9 "
                "(see docs/models/dicke_local.md)"
            )
            raise ValueError(msg)

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
        "gamma_local": DISSIPATION,
        "gamma_phi_local": DISSIPATION,
        "pumping_local": DISSIPATION,
    }

    def parameter_specs(self) -> dict[str, ParameterSpec]:
        """``omega_c`` in ``[0, inf)``, reference width 0.05.

        Same declaration and justification as :meth:`DickeBattery.parameter_specs`
        (``omega0`` is not declared in this release).
        """
        return {"omega_c": ParameterSpec("omega0", (0.0, float("inf")), 0.05)}

    RATES = (
        "kappa",
        "gamma",
        "gamma_phi",
        "gamma_local",
        "gamma_phi_local",
        "pumping_local",
    )

    # ------------------------------------------------------------ resolved sizes
    @property
    def photons(self) -> int:
        """Initial cavity photon number (``n_photons`` with its default resolved)."""
        return self.n_cells if self.n_photons is None else int(self.n_photons)

    @property
    def cutoff(self) -> int:
        """Fock cutoff ``n_max`` with its default resolved.

        ``n_photons`` for ``rwa=True`` (exact only without local pumping, which requires an
        explicit ``n_max``) and ``n_photons + 4`` for ``rwa=False`` (no convergence guarantee).
        """
        if self.n_max is not None:
            return int(self.n_max)
        if not self.rwa:
            return self.photons + NON_RWA_EXTRA_PHOTONS
        return self.photons

    @property
    def j(self) -> float:
        """Largest collective spin ``N/2`` (the symmetric sector)."""
        return self.n_cells / 2.0

    @property
    def dim_cavity(self) -> int:
        """Dimension of the truncated Fock space, ``n_max + 1``."""
        return self.cutoff + 1

    @property
    def dim_cells(self) -> int:
        """Dimension of the full cell space, ``2^N``."""
        return 2**self.n_cells

    @property
    def dim(self) -> int:
        """Dimension ``2^N (n_max + 1)`` of the stored (Schur-basis) density matrix."""
        return self.dim_cells * self.dim_cavity

    @property
    def layout(self) -> Layout:
        """Static index maps of the permutation-invariant representation."""
        return layout(self.n_cells, self.dim_cavity)

    @property
    def liouvillian_dim(self) -> int:
        """Number of independent entries ``sum_j ((2j + 1)(n_max + 1))^2``."""
        return self.layout.size

    @property
    def sectors(self) -> tuple[tuple[float, int], ...]:
        """``(j, d_j)`` of every sector in storage order (``j = N/2`` first)."""
        lay = self.layout
        return tuple((tj / 2.0, d) for tj, d in zip(lay.two_js, lay.degeneracies, strict=True))

    @property
    def is_open(self) -> bool:
        """Always ``True``: the state is a density matrix even with every rate at zero."""
        return True

    # ------------------------------------------------------------------ operators
    def _qarray(self, matrix: np.ndarray | Array) -> QArray:
        return dq.asqarray(jnp.asarray(matrix), dims=(self.dim_cells, self.dim_cavity))

    def _full(self, name: str) -> np.ndarray:
        """Block-diagonal stored-basis matrix of a collective sector operator."""
        lay = self.layout
        blocks = []
        for tj, d in zip(lay.two_js, lay.degeneracies, strict=True):
            op = _sector_operators(tj, self.dim_cavity)[name]
            blocks.extend([op] * d)
        out = np.zeros((lay.dim, lay.dim), dtype=np.float64)
        i = 0
        for blk in blocks:
            out[i : i + blk.shape[0], i : i + blk.shape[0]] = blk
            i += blk.shape[0]
        return out

    def jplus(self) -> QArray:
        """Collective raising operator ``J+ (x) 1`` in the stored basis."""
        return self._qarray(self._full("jp"))

    def jminus(self) -> QArray:
        """Collective lowering operator ``J- (x) 1``."""
        return self._qarray(self._full("jm"))

    def jz(self) -> QArray:
        """Collective inversion ``Jz (x) 1``."""
        return self._qarray(self._full("jz"))

    def a(self) -> QArray:
        """Cavity annihilation operator ``1 (x) a``."""
        return self._qarray(self._full("a"))

    def adag(self) -> QArray:
        """Cavity creation operator ``1 (x) a^dag``."""
        return self._qarray(self._full("adag"))

    def number(self) -> QArray:
        """Photon number ``a^dag a``."""
        a = self._full("a")
        return self._qarray(a.T @ a)

    def _hb_matrix(self) -> np.ndarray:
        return self.omega0 * (self._full("jz") + self.j * np.eye(self.dim))

    def excitation_number(self) -> QArray:
        """``a^dag a + Jz + N/2`` (conserved by closed Tavis-Cummings dynamics only)."""
        a = self._full("a")
        return self._qarray(a.T @ a + self._full("jz") + self.j * np.eye(self.dim))

    def top_fock(self) -> QArray:
        """Projector on the highest kept Fock level (Fock truncation monitor).

        Its expectation value is the convergence diagnostic of the cutoff (``info['top_fock']``,
        the same name as in ``battery/Dicke-v0``): a population that is not negligible means
        the truncation affects the dynamics, so raise ``n_max``. It is the only guard when
        ``pumping_local > 0`` or ``rwa=False``.
        """
        proj = np.zeros((self.dim_cavity, self.dim_cavity))
        proj[self.cutoff, self.cutoff] = 1.0
        return self._qarray(np.kron(np.eye(self.dim_cells), proj))

    def symmetric_projector(self) -> QArray:
        """Projector on the symmetric sector ``j = N/2`` (the subspace of ``battery/Dicke-v0``).

        Its trace is ``(N + 1)`` times the cavity dimension ``n_max + 1``.
        """
        proj = np.zeros((self.dim, self.dim))
        b = self.layout.block_sizes[0]
        proj[:b, :b] = np.eye(b)
        return self._qarray(proj)

    # --------------------------------------------------------------------- Model
    def drift(self) -> QArray:
        a = self._full("a")
        return self._qarray(self._hb_matrix() + self.omega_c * (a.T @ a))

    def controls(self) -> list[QArray]:
        a, ad = self._full("a"), self._full("adag")
        jp, jm = self._full("jp"), self._full("jm")
        if self.rwa:
            return [self._qarray(ad @ jm + a @ jp)]
        return [self._qarray((a + ad) @ (jp + jm))]

    def jump_ops(self) -> list[QArray]:
        """Not available: local jumps act across copies of the multiplicity space.

        The dissipation enters :meth:`evolve` through :meth:`liouvillian`. Raising
        here prevents a generic solver from silently dropping the local processes.
        """
        msg = (
            "DickeBatteryLocal has no jump-operator list in its permutation-invariant "
            "representation; use model.evolve or model.liouvillian"
        )
        raise NotImplementedError(msg)

    def initial_state(self) -> QArray:
        """All cells ground (``j = N/2, m = -N/2``), cavity in ``|n_photons>`` (density matrix)."""
        rho = np.zeros((self.dim, self.dim), dtype=np.complex128)
        rho[self.photons, self.photons] = 1.0
        return self._qarray(rho)

    def battery_hamiltonian(self) -> QArray:
        """``H_B = omega0 (Jz + N/2)`` on the full stored space."""
        return self._qarray(self._hb_matrix())

    def local_battery_hamiltonian(self) -> QArray:
        """``H_B`` on the ``2^N`` cell space (Schur basis).

        Level ``k omega0`` has multiplicity ``C(N, k)``.
        """
        lay = self.layout
        diag = []
        for tj, d in zip(lay.two_js, lay.degeneracies, strict=True):
            m = -tj / 2.0 + np.arange(tj + 1)
            diag.extend(list(m + self.j) * d)
        return dq.asqarray(jnp.asarray(self.omega0 * np.diag(np.asarray(diag))))

    def battery_reduced(self, qstate: QArray) -> tuple[QArray, QArray]:
        """Full ``2^N``-dimensional cell state (cavity traced out) and ``H_B`` on it.

        Both are written in the Schur basis; energy and ergotropy are basis independent,
        so they equal those of the cell state in the computational basis. The eigenvalues
        of the reduced state are those of every block ``sigma_j`` divided by ``d_j``, each
        repeated ``d_j`` times (see :meth:`ergotropy_from_blocks`).
        """
        return dq.ptrace(qstate, 0), self.local_battery_hamiltonian()

    def observables(self) -> dict[str, QArray]:
        return {
            "energy": self.battery_hamiltonian(),
            "n_photons": self.number(),
            "excitation": self.excitation_number(),
            "top_fock": self.top_fock(),
            "symmetric": self.symmetric_projector(),
        }

    def correlation_observables(self) -> dict[str, QArray]:
        """Second moments of the collective spin on the full ``2^N (n_max + 1)`` stored space.

        ``Jx2``, ``Jy2``, ``Jz2``, ``JxJy``, ``JyJz`` and ``JzJx`` (the mixed ones
        symmetrised, all Hermitian, checked), built from the collective operators in the
        Schur basis, so each is block diagonal with identical copies per sector.
        """
        mats = (self._full(n) for n in ("jp", "jm", "jz"))
        return {k: self._qarray(v) for k, v in spin_second_moments(*mats).items()}

    # ------------------------------------------------- permutation-invariant core
    def to_blocks(self, qstate: QArray | Array) -> Array:
        """Compact vector of the joint blocks ``sigma_j = d_j R_j`` (row-major, sum of traces 1)."""
        rho = qstate.todm().to_jax() if isinstance(qstate, QArray) else jnp.asarray(qstate)
        lay = self.layout
        flat = rho.reshape(*rho.shape[:-2], -1)
        return flat[..., lay.gather] * jnp.asarray(lay.weights, dtype=flat.real.dtype)

    def from_blocks(self, vec: Array) -> QArray:
        """Stored density matrix (``d_j`` copies of ``sigma_j / d_j``) from the compact vector."""
        lay = self.layout
        vec = jnp.asarray(vec)
        vec = vec.astype(jnp.result_type(vec.dtype, jnp.complex64))
        scaled = vec / jnp.asarray(lay.weights, dtype=vec.real.dtype)
        padded = jnp.concatenate([scaled, jnp.zeros((1,), dtype=scaled.dtype)])
        rho = padded[jnp.asarray(lay.scatter)].reshape(lay.dim, lay.dim)
        return self._qarray(rho)

    def dicke_density(self, qstate: QArray | Array) -> Array:
        """Cell state in the Dicke basis: block-diagonal ``(n_ds, n_ds)``, unit trace.

        ``n_ds = sum_j (2j + 1)``; blocks ``sigma_j`` (cavity traced out) for
        ``j = N/2, N/2 - 1, ...``, ``m`` ascending inside a block. This is the
        representation of Shammah et al. (2018) and QuTiP PIQS (which orders ``m``
        descending inside a block).
        """
        lay = self.layout
        vec = self.to_blocks(qstate)
        nc = self.dim_cavity
        blocks = []
        for tj, vo, b in zip(lay.two_js, lay.vec_offsets, lay.block_sizes, strict=True):
            blk = vec[vo : vo + b * b].reshape(tj + 1, nc, tj + 1, nc)
            blocks.append(jnp.einsum("anbn->ab", blk))
        return jax.scipy.linalg.block_diag(*blocks)

    def ergotropy_from_blocks(self, qstate: QArray | Array) -> Array:
        """Ergotropy of the full cell state from the Dicke blocks and their degeneracies.

        Eigenvalues of every cell block ``sigma_j`` divided by ``d_j`` and repeated
        ``d_j`` times give the ``2^N`` spectrum of the cell state; ``H_B`` has level
        ``k omega0`` with multiplicity ``C(N, k)``. Equals the ``ergotropy`` of
        :meth:`battery_reduced` (checked in the validation).
        """
        lay = self.layout
        rho_d = self.dicke_density(qstate)
        evals, i = [], 0
        for tj, d in zip(lay.two_js, lay.degeneracies, strict=True):
            blk = rho_d[i : i + tj + 1, i : i + tj + 1]
            evals.append(jnp.repeat(jnp.linalg.eigvalsh(blk) / d, d))
            i += tj + 1
        r = jnp.sort(jnp.concatenate(evals))[::-1]
        # level k (multiplicity C(N, k)) in ascending order, scaled by omega0 with jnp so that
        # a traced omega0 works
        steps = np.concatenate(
            [np.full(math.comb(self.n_cells, k), k) for k in range(self.n_cells + 1)]
        )
        levels = self.omega0 * jnp.asarray(steps, dtype=r.dtype)
        energy = jnp.real(jnp.trace(rho_d @ self._dicke_hb()))
        return energy - jnp.sum(r * levels)

    def _dicke_hb(self) -> Array:
        diag = []
        for tj in self.layout.two_js:
            diag.extend(list(-tj / 2.0 + np.arange(tj + 1) + self.j))
        return self.omega0 * jnp.asarray(np.diag(np.asarray(diag)))

    def dissipator(self) -> np.ndarray:
        """Constant (collective + local) part of the Liouvillian on the compact vector.

        The array is cached and shared between models with equal parameters, so it is
        read-only; copy it before modifying.
        """
        return _dissipator(
            self.n_cells,
            self.dim_cavity,
            *(float(getattr(self, name)) for name in self.RATES),
        )

    def liouvillian(self, hamiltonian: QArray | Array) -> Array:
        """Full Liouvillian ``-i [H, .] + D`` on the compact vector for a collective ``H``.

        ``H`` is given in the stored basis (e.g. ``model.hamiltonian(g)``); only its sector
        blocks are used (a collective Hamiltonian is block diagonal with identical copies).
        """
        h = hamiltonian.to_jax() if isinstance(hamiltonian, QArray) else jnp.asarray(hamiltonian)
        lay = self.layout
        flat = h.reshape(-1)
        supers = []
        for vo, b in zip(lay.vec_offsets, lay.block_sizes, strict=True):
            hb = flat[jnp.asarray(lay.gather[vo : vo + b * b])].reshape(b, b)
            eye = jnp.eye(b, dtype=hb.dtype)
            supers.append(-1j * (jnp.kron(hb, eye) - jnp.kron(eye, hb.T)))
        lh = jax.scipy.linalg.block_diag(*supers)
        return lh + jnp.asarray(self.dissipator(), dtype=lh.dtype)

    def evolve(
        self,
        state: QArray,
        hamiltonian: QArray | TimeQArray,
        duration: float | Array,
        *,
        method: Method | None = None,
    ) -> QArray:
        """Exact step ``vec sigma -> expm(L(H) duration) vec sigma`` for a constant ``H``.

        ``method`` is ignored (the matrix exponential is exact for piecewise-constant
        controls); time-dependent Hamiltonians are rejected.
        """
        del method
        if isinstance(hamiltonian, TimeQArray):
            msg = "DickeBatteryLocal.evolve needs a constant Hamiltonian (one control step)"
            raise TypeError(msg)
        lv = self.liouvillian(hamiltonian)
        vec = self.to_blocks(state).astype(lv.dtype)
        prop = jax.scipy.linalg.expm(lv * duration)
        return self.from_blocks(prop @ vec)
