"""Validation checks for the Dicke battery with local dissipation (:class:`DickeBatteryLocal`).

The reference for the permutation-invariant representation is an independent brute-force
simulation in the full ``2^N (x) cavity`` Hilbert space with explicit per-cell jump
operators, propagated exactly by dynamiqs (``mesolve`` with ``Expm``, which builds its own
Lindblad superoperator). The brute-force state is mapped to the Schur basis by a unitary
built here from scratch (highest-weight vectors of each ``j`` from an SVD, lowered with
``J-``), so the comparison covers the full joint state and not only expectation values.

Agreement tolerances depend on the precision: ``1e-8`` with ``jax_enable_x64`` (the
release gate) and ``2e-5`` in float32, where both simulations carry single-precision
roundoff (the analytic and constraint checks use ``1e-5`` instead of ``2e-6`` / ``1e-6`` in
float32). The QuTiP PIQS cross-check needs ``qutip`` (``qenvs[dev]``); without it the check
is reported as skipped.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import dynamiqs as dq
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from qenvs.envs.base import EnvState
from qenvs.envs.battery import BatteryCharging
from qenvs.models.dicke import DickeBattery
from qenvs.models.dicke_local import (
    DickeBatteryLocal,
    degeneracy,
    layout,
    local_cell_dissipator,
    sector_spins,
)
from qenvs.observables import energy, ergotropy
from qenvs.validation.core import CheckResult, skipped
from qenvs.validation.core import result as _result

MODEL = "DickeBatteryLocal"
EXPM = dq.method.Expm()


def _res(name: str, category: str, error: float, tol: float, detail: str) -> CheckResult:
    return _result(name, category, error, tol, detail, model=MODEL)


def _x64() -> bool:
    return bool(jax.config.jax_enable_x64)


def _agreement_tol() -> float:
    """1e-8 in float64 (release gate), 2e-5 in float32."""
    return 1e-8 if _x64() else 2e-5


def _precision() -> str:
    return "float64" if _x64() else "float32"


def _random_actions(seed: int, horizon: int, bound: float) -> Array:
    key = jax.random.PRNGKey(seed)
    return jax.random.uniform(key, (horizon, 1), minval=-bound, maxval=bound)


def _f32_actions(actions: Array) -> np.ndarray:
    """The env casts actions to float32; references must use the identical amplitudes."""
    return np.asarray(jnp.asarray(actions[:, 0], dtype=jnp.float32), dtype=np.float64)


# ------------------------------------------------------------ brute force (2^N)
def _site(op: np.ndarray, i: int, n_cells: int) -> np.ndarray:
    out = np.eye(1)
    for k in range(n_cells):
        out = np.kron(out, op if k == i else np.eye(2))
    return out


def _brute_force(m: DickeBatteryLocal) -> dict[str, object]:
    """Hamiltonian pieces and explicit jump operators on ``2^N (x) cavity`` (index 1 excited)."""
    n, nc = m.n_cells, m.dim_cavity
    sm = np.array([[0.0, 1.0], [0.0, 0.0]])  # |0><1|, built explicitly (library rule 4)
    sz = np.diag([-1.0, 1.0])
    jm = sum(_site(sm, i, n) for i in range(n))
    jp = jm.T
    jz = sum(_site(sz, i, n) for i in range(n)) / 2.0
    a = np.diag(np.sqrt(np.arange(1.0, nc)), 1)
    ec, eb = np.eye(nc), np.eye(2**n)
    hb = m.omega0 * (jz + n / 2.0 * eb)
    h0 = np.kron(hb, ec) + m.omega_c * np.kron(eb, a.T @ a)
    v = np.kron(jm, a.T) + np.kron(jp, a) if m.rwa else np.kron(jp + jm, a + a.T)
    ops = []
    for rate, op in (
        (m.kappa, np.kron(eb, a)),
        (m.gamma, np.kron(jm, ec)),
        (m.gamma_phi, np.kron(jz, ec)),
    ):
        if rate > 0:
            ops.append(np.sqrt(rate) * op)
    for i in range(n):
        for rate, op in (
            (m.gamma_local, sm),
            (m.gamma_phi_local, sz / 2.0),
            (m.pumping_local, sm.T),
        ):
            if rate > 0:
                ops.append(np.sqrt(rate) * np.kron(_site(op, i, n), ec))
    dims = (2**n, nc)
    return {
        "h0": h0,
        "v": v,
        "jumps": [dq.asqarray(jnp.asarray(o), dims=dims) for o in ops],
        "hb": hb,
        "number": np.kron(eb, a.T @ a),
        "dims": dims,
    }


def schur_transform(n_cells: int) -> np.ndarray:
    """Rows = Dicke states ``|j, m, alpha>`` in the computational basis (storage order).

    Independent of the model code: for each ``j`` the highest-weight states span the
    kernel of ``J+`` on the ``k = j + N/2`` excitation subspace (orthonormal, from an
    SVD); lower ``m`` follow from ``J- |j, m> = sqrt(j(j+1) - m(m-1)) |j, m-1>``.
    Order: ``j = N/2, N/2 - 1, ...``, then ``alpha``, then ``m`` ascending.
    """
    sm = np.array([[0.0, 1.0], [0.0, 0.0]])
    jm = sum(_site(sm, i, n_cells) for i in range(n_cells))
    jp = jm.T
    nexc = np.array([bin(i).count("1") for i in range(2**n_cells)])
    rows = []
    for tj in range(n_cells, -1, -2):
        j = tj / 2.0
        idx = np.where(nexc == round(j + n_cells / 2.0))[0]
        _, s, vh = np.linalg.svd(jp[:, idx])
        rank = int(np.sum(s > 1e-10))
        for vec in vh[rank:]:
            hw = np.zeros(2**n_cells)
            hw[idx] = vec
            ladder, mm = [hw], j
            for _ in range(tj):
                ladder.append(jm @ ladder[-1] / np.sqrt(j * (j + 1) - mm * (mm - 1)))
                mm -= 1
            rows.extend(ladder[::-1])
    return np.asarray(rows)


def _dicke_from_full_cells(rho_cells: np.ndarray, u: np.ndarray, n_cells: int) -> np.ndarray:
    """Project a computational-basis cell state on the Dicke blocks: sum over alpha."""
    r = u @ rho_cells @ u.conj().T
    blocks, i = [], 0
    for tj in sector_spins(n_cells):
        b, d = tj + 1, degeneracy(n_cells, tj)
        blk = np.zeros((b, b), dtype=complex)
        for _ in range(d):
            blk += r[i : i + b, i : i + b]
            i += b
        blocks.append(blk)
    out = np.zeros((sum(x.shape[0] for x in blocks),) * 2, dtype=complex)
    k = 0
    for blk in blocks:
        out[k : k + blk.shape[0], k : k + blk.shape[0]] = blk
        k += blk.shape[0]
    return out


def _schur_cells(cells: dq.QArray, u: np.ndarray) -> np.ndarray:
    """Brute-force cell state in the Schur basis (computational-basis input)."""
    return u @ np.asarray(cells.to_jax(), dtype=np.complex128) @ u.conj().T


def _brute_vs_model(m: DickeBatteryLocal, actions: np.ndarray, dt: float) -> dict[str, float]:
    """Step both simulations through ``actions``; max deviations over all steps."""
    bf = _brute_force(m)
    n, nc = m.n_cells, m.dim_cavity
    u = schur_transform(n)
    uu = np.kron(u, np.eye(nc))
    rho0 = np.zeros((2**n * nc,) * 2)
    rho0[m.photons, m.photons] = 1.0  # computational index 0 = all ground
    rho = dq.asqarray(jnp.asarray(rho0, dtype=complex), dims=bf["dims"])
    hb_c = dq.asqarray(jnp.asarray(bf["hb"]))
    state = m.initial_state()
    errs = {
        "state": 0.0,
        "energy": 0.0,
        "ergotropy": 0.0,
        "n_photons": 0.0,
        "dicke": 0.0,
        "symmetric": 0.0,
    }
    # the projector on j = N/2 has rank (N + 1) times the cavity dimension
    proj = np.asarray(m.symmetric_projector().to_jax())
    errs["projector"] = abs(float(np.trace(proj).real) - (n + 1) * nc)
    for g in actions:
        h = dq.asqarray(jnp.asarray(bf["h0"] + float(g) * bf["v"]), dims=bf["dims"])
        rho = dq.mesolve(
            h, bf["jumps"], rho, jnp.asarray([0.0, dt]), method=EXPM, progress_meter=False
        ).states[-1]
        rho = dq.asqarray(rho.to_jax(), dims=bf["dims"])
        state = m.evolve(state, m.hamiltonian(jnp.asarray([g])), dt)
        r = np.asarray(rho.to_jax(), dtype=np.complex128)
        s = np.asarray(state.to_jax(), dtype=np.complex128)
        cells_bf = dq.ptrace(rho, 0)
        rho_b, hb = m.battery_reduced(state)
        dd = np.asarray(m.dicke_density(state), dtype=np.complex128)
        dd_bf = _dicke_from_full_cells(np.asarray(cells_bf.to_jax(), dtype=np.complex128), u, n)
        new = {
            "state": np.max(np.abs(uu @ r @ uu.conj().T - s)),
            "energy": abs(float(energy(cells_bf, hb_c)) - float(energy(rho_b, hb))),
            "ergotropy": abs(float(ergotropy(cells_bf, hb_c)) - float(ergotropy(rho_b, hb))),
            "n_photons": abs(
                float(np.trace(r @ bf["number"]).real) - float(dq.expect(m.number(), state).real)
            ),
            "dicke": np.max(np.abs(dd - dd_bf)),
            # weight of the j = N/2 sector: the first N + 1 Schur rows of the brute-force cells
            "symmetric": abs(
                float(np.trace(_schur_cells(cells_bf, u)[: n + 1, : n + 1]).real)
                - float(dq.expect(m.symmetric_projector(), state).real)
            ),
        }
        errs = {k: max(errs[k], float(new.get(k, 0.0))) for k in errs}
    return errs


# --------------------------------------------------------------------- checks
def check_structure() -> CheckResult:
    """Degeneracies, operator algebra in the stored basis and trace preservation of L.

    * ``sum_j d_j (2j + 1) = 2^N`` for ``N <= 12``;
    * ``[J+, J-] = 2 Jz`` and the ``H_B`` spectrum ``k omega0`` with multiplicity
      ``C(N, k)`` in the stored ``2^N n_c`` basis (``N = 4``, ``n_max = 2``);
    * the full Liouvillian (all rates on, random ``g``) annihilates the trace functional
      ``sum_j tr sigma_j`` (exact trace preservation, float64 construction).
    """
    errs = []
    for n in range(1, 13):
        errs.append(abs(sum(degeneracy(n, tj) * (tj + 1) for tj in sector_spins(n)) - 2**n))
    m = DickeBatteryLocal(
        n_cells=4, n_photons=1, n_max=2, rwa=False, kappa=0.1, gamma=0.2, gamma_phi=0.3,
        gamma_local=0.4, gamma_phi_local=0.5, pumping_local=0.6,
    )  # fmt: skip
    jp, jm, jz = (np.asarray(op.to_jax()) for op in (m.jplus(), m.jminus(), m.jz()))
    errs.append(float(np.max(np.abs(jp @ jm - jm @ jp - 2 * jz))))
    levels = np.sort(np.diag(np.asarray(m.local_battery_hamiltonian().to_jax()).real))
    expected = np.concatenate([np.full(math.comb(4, k), float(k)) for k in range(5)])
    errs.append(float(np.max(np.abs(levels - expected * m.omega0))))
    lay = layout(m.n_cells, m.dim_cavity)
    tr = np.zeros(lay.size)
    for vo, b in zip(lay.vec_offsets, lay.block_sizes, strict=True):
        tr[vo + np.arange(b) * (b + 1)] = 1.0
    lv = np.asarray(m.liouvillian(m.hamiltonian(jnp.asarray([0.37]))), dtype=np.complex128)
    trace_err = float(np.max(np.abs(tr @ lv)))
    errs.append(trace_err)
    return _res(
        "structure",
        "algebra",
        float(max(errs)),
        1e-5 if not _x64() else 1e-12,
        f"sum_j d_j(2j+1) = 2^N (N <= 12), [J+, J-] = 2Jz and H_B multiplicities C(N, k) "
        f"(N=4), |trace functional of L|={trace_err:.1e} with every rate on",
    )


def check_brute_force_agreement() -> CheckResult:
    """Release gate: brute-force ``2^N (x) cavity`` simulation, N = 2, 3, 4, random pwc g(t).

    All six dissipators on (cavity loss, collective emission and dephasing, local
    emission, dephasing and pumping), detuned, TC and Dicke couplings. Compared after
    every step: the full joint state (Schur basis), energy, ergotropy of the full cell
    state, photon number, the reduced cell density in the Dicke basis and the weight of the
    symmetric sector ``j = N/2`` (``symmetric`` observable, against the weight of the
    brute-force cell state on the first ``N + 1`` Schur vectors); the trace of the symmetric
    projector must equal ``(N + 1)(n_max + 1)``.
    """
    configs = (
        {
            "n_cells": 2,
            "rwa": False,
            "n_photons": 1,
            "n_max": 3,
            "kappa": 0.1,
            "gamma": 0.07,
            "gamma_phi": 0.05,
        },
        {"n_cells": 3, "n_photons": 2, "n_max": 3, "kappa": 0.05},
        {"n_cells": 4, "n_photons": 1, "n_max": 1, "gamma": 0.03},
    )
    horizon, dt, bound = 6, 0.3, 0.6
    worst, parts = 0.0, []
    for k, cfg in enumerate(configs):
        m = DickeBatteryLocal(
            omega0=0.9, omega_c=1.2, gamma_local=0.13, gamma_phi_local=0.08, pumping_local=0.04,
            **cfg,
        )  # fmt: skip
        acts = _f32_actions(_random_actions(10 + k, horizon, bound))
        errs = _brute_vs_model(m, acts, dt)
        worst = max(worst, *errs.values())
        parts.append(f"N={m.n_cells}: " + ", ".join(f"{name} {v:.1e}" for name, v in errs.items()))
    return _res(
        "brute_force_agreement",
        "cross_check",
        worst,
        _agreement_tol(),
        f"max deviation from the explicit 2^N x cavity simulation over {horizon} random steps "
        f"({_precision()}): " + "; ".join(parts),
    )


def _full_space_ergotropy(rho_sym: np.ndarray, m: DickeBatteryLocal) -> float:
    """Ergotropy over all unitaries of the 2^N cell space for a state in the j = N/2 sector.

    The ``(N + 1)`` eigenvalues of the symmetric-sector state, padded with zeros to
    ``2^N``, are paired with the ``2^N`` levels ``k omega0`` (multiplicity ``C(N, k)``).
    """
    n = m.n_cells
    r = np.zeros(2**n)
    r[: n + 1] = np.linalg.eigvalsh(rho_sym)
    levels = np.concatenate([np.full(math.comb(n, k), k * m.omega0) for k in range(n + 1)])
    e = float(np.real(np.trace(rho_sym @ np.diag(m.omega0 * np.arange(n + 1.0)))))
    return e - float(np.sum(np.sort(r)[::-1] * np.sort(levels)))


def check_collective_limit() -> CheckResult:
    """Only collective processes: identical to ``DickeBattery`` (battery/Dicke-v0 model).

    Env rollouts of the same random pulse: open TC (N = 3, kappa, gamma, gamma_phi) and
    closed Dicke (N = 2, rwa=False, all rates zero). The reference ``DickeBattery`` is
    stepped with the exact ``Expm`` propagator. Compared per step: energy, ergotropy of
    the reduced cell state, photon number, the reward, and the final ``j = N/2`` block
    against the reference density matrix; the weight outside ``j = N/2`` must stay 0.

    The ergotropy reference is the full-space (``2^N``) ergotropy of the reference cell
    state: ``DickeBattery`` itself reports the ergotropy over unitaries of the
    ``(N + 1)``-dimensional symmetric subspace, which is smaller; the gap is reported.
    """
    horizon, dt, bound = 8, 0.3, 0.5
    cases = (
        {
            "n_cells": 3,
            "kappa": 0.1,
            "gamma": 0.05,
            "gamma_phi": 0.03,
            "omega0": 1.1,
            "omega_c": 0.9,
        },
        {"n_cells": 2, "rwa": False, "n_photons": 1, "n_max": 4},
    )
    worst, parts = 0.0, []
    for k, kw in enumerate(cases):
        loc = DickeBatteryLocal(**kw)
        ref = DickeBattery(**kw)
        actions = _random_actions(20 + k, horizon, bound)
        env = BatteryCharging(model=loc, horizon=horizon, dt=dt, action_bound=bound)
        state, ts = env.reset(jax.random.PRNGKey(0))
        q = ref.initial_state()
        hb_ref = ref.battery_reduced(q)[1]
        e_err = w_err = n_err = r_err = sym_err = w_gap = 0.0
        e_prev = 0.0
        for a in actions:
            state, ts = env.step(state, a)
            q = ref.evolve(q, ref.hamiltonian(jnp.asarray(a, dtype=jnp.float32)), dt, method=EXPM)
            q = dq.asqarray(q.to_jax(), dims=(ref.dim_battery, ref.dim_cavity))
            rho_b = dq.ptrace(q, 0)
            e_ref = float(energy(rho_b, hb_ref))
            e_err = max(e_err, abs(float(ts.info["energy"]) - e_ref))
            w_full = _full_space_ergotropy(np.asarray(rho_b.to_jax()), loc)
            w_err = max(w_err, abs(float(ts.info["ergotropy"]) - w_full))
            w_gap = max(w_gap, w_full - float(ergotropy(rho_b, hb_ref)))
            n_err = max(
                n_err, abs(float(ts.info["n_photons"]) - float(dq.expect(ref.number(), q).real))
            )
            r_err = max(r_err, abs(float(ts.reward) - (e_ref - e_prev)))
            sym_err = max(sym_err, abs(float(ts.info["symmetric"]) - 1.0))
            e_prev = e_ref
        b = loc.layout.block_sizes[0]
        blk = np.asarray(state.qstate.to_jax())[:b, :b]
        s_err = float(np.max(np.abs(blk - np.asarray(q.todm().to_jax()))))
        errs = (e_err, w_err, n_err, r_err, sym_err, s_err)
        worst = max(worst, *errs)
        kind = "open TC" if loc.rwa else "closed Dicke"
        parts.append(
            f"{kind} N={loc.n_cells}: energy {e_err:.1e}, ergotropy {w_err:.1e}, photons "
            f"{n_err:.1e}, reward {r_err:.1e}, 1 - symmetric {sym_err:.1e}, state {s_err:.1e} "
            f"(full-space minus symmetric-subspace ergotropy up to {w_gap:.3f})"
        )
    return _res(
        "collective_limit",
        "cross_check",
        worst,
        _agreement_tol(),
        f"max deviation from DickeBattery (Expm) over {horizon} random steps "
        f"({_precision()}): " + "; ".join(parts),
    )


def _random_block_state(m: DickeBatteryLocal, seed: int) -> Array:
    """Random permutation-invariant cell state: random PSD block in every sector, trace 1."""
    rng = np.random.default_rng(seed)
    lay = m.layout
    vec = np.zeros(lay.size, dtype=complex)
    for vo, b in zip(lay.vec_offsets, lay.block_sizes, strict=True):
        x = rng.normal(size=(b, b)) + 1j * rng.normal(size=(b, b))
        vec[vo : vo + b * b] = (x @ x.conj().T).reshape(-1)
    tr = sum(
        np.trace(vec[vo : vo + b * b].reshape(b, b)).real
        for vo, b in zip(lay.vec_offsets, lay.block_sizes, strict=True)
    )
    return jnp.asarray(vec / tr)


def _to_piqs_order(rho_d: np.ndarray, n_cells: int) -> np.ndarray:
    """Reverse ``m`` inside each block (PIQS orders ``m = j, ..., -j``)."""
    perm, i = [], 0
    for tj in sector_spins(n_cells):
        perm.extend(range(i + tj, i - 1, -1))
        i += tj + 1
    return rho_d[np.ix_(perm, perm)]


def check_piqs_cross_check() -> CheckResult:
    """Cells only (no cavity): model vs QuTiP PIQS ``Dicke(...).liouvillian()`` + ``mesolve``.

    N = 3 and 4, every local and collective cell process on, drift ``omega0 (Jz + N/2)``,
    a random permutation-invariant initial state with coherences in every ``j`` sector.
    PIQS evaluates its rates in single precision (``cdef float`` in ``_piqs.pyx``), so the
    tolerance is 1e-5 even in float64.
    """
    try:
        import qutip as qt
        from qutip import piqs
    except ImportError:
        return skipped(
            "piqs_cross_check", "cross_check", 1e-5, "not run: qutip is not installed", MODEL
        )
    rates = {
        "gamma": 0.11,
        "gamma_phi": 0.07,
        "gamma_local": 0.2,
        "gamma_phi_local": 0.15,
        "pumping_local": 0.05,
    }
    times = np.array([0.0, 0.7, 1.5, 3.0])
    worst, parts = 0.0, []
    for n in (3, 4):
        m = DickeBatteryLocal(n_cells=n, n_photons=0, n_max=0, omega0=1.3, **rates)
        vec0 = _random_block_state(m, seed=n)
        state = m.from_blocks(vec0)
        rho0 = _to_piqs_order(np.asarray(m.dicke_density(state), dtype=np.complex128), n)
        h = m.omega0 * (piqs.jspin(n, "z", basis="dicke") + n / 2.0 * qt.qeye(rho0.shape[0]))
        system = piqs.Dicke(
            n,
            hamiltonian=h,
            emission=m.gamma_local,
            dephasing=m.gamma_phi_local,
            pumping=m.pumping_local,
            collective_emission=m.gamma,
            collective_dephasing=m.gamma_phi,
        )
        opts = {"atol": 1e-12, "rtol": 1e-10, "nsteps": 100_000, "progress_bar": ""}
        res = qt.mesolve(system.liouvillian(), qt.Qobj(rho0), times, options=opts)
        h_model = m.drift()
        err, t_prev = 0.0, 0.0
        for t, rho_qt in zip(times[1:], res.states[1:], strict=True):
            state = m.evolve(state, h_model, float(t - t_prev))
            t_prev = t
            mine = _to_piqs_order(np.asarray(m.dicke_density(state), dtype=np.complex128), n)
            err = max(err, float(np.max(np.abs(mine - rho_qt.full()))))
        worst = max(worst, err)
        parts.append(f"N={n}: {err:.1e}")
    return _res(
        "piqs_cross_check",
        "cross_check",
        worst,
        1e-5,
        "max |sigma_model - sigma_piqs| (Dicke basis) at t = 0.7, 1.5, 3.0 from a random "
        "block state, all cell processes on (" + ", ".join(parts) + f"; qutip {qt.__version__})",
    )


def _excited_state(m: DickeBatteryLocal) -> dq.QArray:
    lay = m.layout
    vec = np.zeros(lay.size)
    b = lay.block_sizes[0]
    vec[b * b - 1] = 1.0  # j = N/2, m = +N/2, Fock 0 (n_max = 0)
    return m.from_blocks(jnp.asarray(vec))


def _populations(m: DickeBatteryLocal, state: dq.QArray) -> np.ndarray:
    """Probability of k excited cells, k = 0..N, from the Dicke blocks."""
    d = np.real(np.diag(np.asarray(m.dicke_density(state))))
    p, i = np.zeros(m.n_cells + 1), 0
    for tj in sector_spins(m.n_cells):
        for a in range(tj + 1):
            p[round(a - tj / 2.0 + m.n_cells / 2.0)] += d[i + a]
        i += tj + 1
    return p


def _purity(state: dq.QArray) -> float:
    rho = state.to_jax()
    return float(jnp.real(jnp.trace(rho @ rho)))


def check_local_emission_relaxation() -> CheckResult:
    """Local emission alone, g = 0: every cell relaxes independently at gamma_local.

    From all cells excited (N = 4) the state is ``(p|e><e| + (1-p)|g><g|)^(x)N`` with
    ``p = exp(-gamma t)``: ``E = N omega0 p``, the number of excited cells is binomial
    ``C(N, k) p^k (1-p)^(N-k)`` (populations of all ``j`` sectors), the purity is
    ``(p^2 + (1-p)^2)^N`` and the ergotropy is ``N omega0 max(0, 2p - 1)``. Collective
    emission would instead pass through the superradiant cascade and fail the binomial.
    """
    n, gamma, dt = 4, 0.4, 0.25
    m = DickeBatteryLocal(n_cells=n, n_photons=0, n_max=0, gamma_local=gamma, omega0=1.2)
    state = _excited_state(m)
    h = m.drift()
    errs = {"energy": 0.0, "binomial": 0.0, "purity": 0.0, "ergotropy": 0.0}
    for step in range(1, 17):
        state = m.evolve(state, h, dt)
        p = math.exp(-gamma * dt * step)
        rho_b, hb = m.battery_reduced(state)
        binom = np.array([math.comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(n + 1)])
        new = {
            "energy": abs(float(energy(rho_b, hb)) - n * m.omega0 * p),
            "binomial": float(np.max(np.abs(_populations(m, state) - binom))),
            "purity": abs(_purity(state) - (p**2 + (1 - p) ** 2) ** n),
            "ergotropy": abs(float(ergotropy(rho_b, hb)) - n * m.omega0 * max(0.0, 2 * p - 1)),
        }
        errs = {k: max(errs[k], new[k]) for k in errs}
    return _res(
        "local_emission_relaxation",
        "analytic",
        max(errs.values()),
        2e-6 if _x64() else 1e-5,
        "N=4 from all excited, g=0, gamma_local=0.4, t <= 4: "
        + ", ".join(f"{k} {v:.1e}" for k, v in errs.items())
        + " (closed forms of independent single-cell decay at rate gamma_local)",
    )


def check_local_dephasing_and_pumping() -> CheckResult:
    """Closed forms for local dephasing and for pumping plus emission (N = 3, g = 0).

    * Dephasing ``sqrt(gamma_phi_local) sigma_z / 2`` from ``|+>^(x)N``: ``|<J->| =
      (N/2) exp(-gamma_phi_local t / 2)``, purity ``((1 + exp(-gamma_phi_local t))/2)^N``,
      energy constant ``N omega0 / 2``.
    * Pumping ``P`` and emission ``gamma`` from the ground state: every cell has
      ``p(t) = P/(P + gamma) (1 - exp(-(P + gamma) t))``, so ``E = N omega0 p``, purity
      ``(p^2 + (1-p)^2)^N`` and ergotropy ``N omega0 max(0, 2p - 1)`` (positive here
      since ``P > gamma``: tests the degeneracy-aware ergotropy).
    """
    n, dt = 3, 0.25
    gphi = 0.6
    m = DickeBatteryLocal(n_cells=n, n_photons=0, n_max=0, gamma_phi_local=gphi, omega0=0.8)
    lay = m.layout
    b = lay.block_sizes[0]
    psi = np.array([math.sqrt(math.comb(n, k)) / 2 ** (n / 2) for k in range(n + 1)])
    vec = np.zeros(lay.size)
    vec[: b * b] = np.outer(psi, psi).reshape(-1)
    state = m.from_blocks(jnp.asarray(vec))
    h = m.drift()
    e_deph = 0.0
    for step in range(1, 13):
        state = m.evolve(state, h, dt)
        t = dt * step
        coh = abs(complex(dq.expect(m.jminus(), state)))
        rho_b, hb = m.battery_reduced(state)
        e_deph = max(
            e_deph,
            abs(coh - n / 2 * math.exp(-gphi * t / 2)),
            abs(_purity(state) - ((1 + math.exp(-gphi * t)) / 2) ** n),
            abs(float(energy(rho_b, hb)) - n * m.omega0 / 2),
        )
    pump, gam = 0.5, 0.1
    m2 = DickeBatteryLocal(n_cells=n, n_photons=0, n_max=0, pumping_local=pump, gamma_local=gam)
    state = m2.initial_state()
    h2 = m2.drift()
    e_pump = 0.0
    w_last = 0.0
    for step in range(1, 17):
        state = m2.evolve(state, h2, dt)
        t = dt * step
        p = pump / (pump + gam) * (1 - math.exp(-(pump + gam) * t))
        rho_b, hb = m2.battery_reduced(state)
        w_last = float(ergotropy(rho_b, hb))
        e_pump = max(
            e_pump,
            abs(float(energy(rho_b, hb)) - n * m2.omega0 * p),
            abs(_purity(state) - (p**2 + (1 - p) ** 2) ** n),
            abs(w_last - n * m2.omega0 * max(0.0, 2 * p - 1)),
        )
    return _res(
        "local_dephasing_and_pumping",
        "analytic",
        max(e_deph, e_pump),
        2e-6,
        f"N=3, g=0: dephasing (gamma_phi_local=0.6, |<J->|, purity, energy) {e_deph:.1e}; "
        f"pumping 0.5 + emission 0.1 (energy, purity, ergotropy) {e_pump:.1e}; final "
        f"ergotropy {w_last:.3f} omega0",
    )


def _scan_states(env: BatteryCharging, actions: Array) -> tuple[EnvState, Array, Array]:
    state0, _ = env.reset(jax.random.PRNGKey(0))

    def body(state: EnvState, a: Array) -> tuple[EnvState, tuple[Array, Array]]:
        state, ts = env.step(state, a)
        return state, (state.qstate.to_jax(), ts.reward)

    final, (rhos, rewards) = jax.lax.scan(body, state0, actions)
    return final, rhos, rewards


def check_density_matrix_constraints() -> CheckResult:
    """Random rollouts with every process on: Hermitian, trace 1, positive (joint state).

    Open Dicke (N = 3, rwa=False) and open TC with pumping (N = 4), 12 random steps,
    checked after every step on the stored ``2^N n_c`` matrix.
    """
    horizon, bound = 12, 0.5
    rates = {
        "kappa": 0.1,
        "gamma": 0.05,
        "gamma_phi": 0.05,
        "gamma_local": 0.1,
        "gamma_phi_local": 0.1,
    }
    models = (
        DickeBatteryLocal(n_cells=3, rwa=False, n_max=4, **rates),
        DickeBatteryLocal(n_cells=4, n_photons=2, n_max=6, pumping_local=0.05, **rates),
    )
    herm = tr = neg = 0.0
    for k, m in enumerate(models):
        env = BatteryCharging(model=m, horizon=horizon, dt=0.3, action_bound=bound)
        _, rhos, _ = _scan_states(env, _random_actions(30 + k, horizon, bound))
        rhos = np.asarray(rhos, dtype=np.complex128)
        herm = max(herm, float(np.max(np.abs(rhos - np.conj(np.swapaxes(rhos, -1, -2))))))
        tr = max(tr, float(np.max(np.abs(np.trace(rhos, axis1=-2, axis2=-1).real - 1.0))))
        herm_part = 0.5 * (rhos + np.conj(np.swapaxes(rhos, -1, -2)))
        neg = max(neg, max(0.0, -float(np.min(np.linalg.eigvalsh(herm_part)))))
    return _res(
        "density_matrix_constraints",
        "constraints",
        max(herm, tr, neg),
        1e-6 if _x64() else 1e-5,
        f"after every step of {horizon}-step random rollouts (N=3 Dicke, N=4 TC + pumping): "
        f"hermiticity {herm:.1e}, |tr - 1| {tr:.1e}, negativity {neg:.1e}",
    )


def check_expm_step_exactness() -> CheckResult:
    """The control step ``expm(L dt)`` vs an independent Taylor propagator and vs substeps.

    N = 3, n_max = 3, every rate on, ``g = 0.45``, ``dt = 0.4``. The reference applies
    32 substeps of a 30-term Taylor series of ``L dt / 32`` (``||L dt / 32|| < 0.2``, so
    the truncation is far below roundoff); the semigroup check compares one step of
    ``dt`` with eight steps of ``dt / 8``. Both errors are on the compact state vector.
    """
    m = DickeBatteryLocal(
        n_cells=3, n_max=3, rwa=False, kappa=0.1, gamma=0.05, gamma_phi=0.05,
        gamma_local=0.1, gamma_phi_local=0.1, pumping_local=0.05,
    )  # fmt: skip
    dt = 0.4
    h = m.hamiltonian(jnp.asarray([0.45]))
    vec0 = _random_block_state(m, seed=7)
    state0 = m.from_blocks(vec0)
    one = np.asarray(m.to_blocks(m.evolve(state0, h, dt)), dtype=np.complex128)
    lv = np.asarray(m.liouvillian(h), dtype=np.complex128) * (dt / 32)
    v = np.asarray(vec0, dtype=np.complex128)
    for _ in range(32):
        term, acc = v.copy(), v.copy()
        for k in range(1, 31):
            term = lv @ term / k
            acc = acc + term
        v = acc
    taylor = float(np.max(np.abs(one - v)))
    state = state0
    for _ in range(8):
        state = m.evolve(state, h, dt / 8)
    semi = float(np.max(np.abs(one - np.asarray(m.to_blocks(state)))))
    return _res(
        "expm_step_exactness",
        "numerics",
        max(taylor, semi),
        1e-12 if _x64() else 2e-6,
        f"|expm step - Taylor(32 x 30 terms)|={taylor:.1e}, |one step - 8 substeps|={semi:.1e} "
        f"(N=3, Liouvillian dim {m.liouvillian_dim}, {_precision()})",
    )


def check_env_return_equals_gain() -> CheckResult:
    """``BatteryCharging`` return = Q(final) - Q(initial) for energy and ergotropy rewards.

    N = 3, every local process on, 10 random steps. ``Q`` is recomputed independently of
    the env from the Dicke blocks: energy ``tr(sigma H_B)``, ergotropy from the block
    eigenvalues divided by ``d_j`` (:meth:`ergotropy_from_blocks`). The ergotropy reported
    by the env (``battery_reduced``, ``2^N`` cell matrix) must equal the block formula.
    """
    horizon, bound = 10, 0.5
    m = DickeBatteryLocal(
        n_cells=3, n_max=6, gamma_local=0.1, gamma_phi_local=0.1, pumping_local=0.3
    )
    actions = _random_actions(40, horizon, bound)
    errs, finals = {}, {}
    for reward in ("energy", "ergotropy"):
        env = BatteryCharging(model=m, horizon=horizon, dt=0.3, action_bound=bound, reward=reward)
        final, rhos, rewards = _scan_states(env, actions)
        s0 = m.initial_state()
        if reward == "energy":
            hb = m._dicke_hb()

            def q(s: object, hb: Array = hb) -> float:
                return float(jnp.real(jnp.trace(m.dicke_density(s) @ hb)))

        else:

            def q(s: object) -> float:
                return float(m.ergotropy_from_blocks(s))

        errs[reward] = abs(float(jnp.sum(rewards)) - (q(final.qstate) - q(s0)))
        finals[reward] = q(final.qstate)
    w_route = max(
        abs(float(m.ergotropy_from_blocks(r)) - float(ergotropy(*m.battery_reduced(m._qarray(r)))))
        for r in rhos
    )
    return _res(
        "env_return_equals_gain",
        "constraints",
        max(*errs.values(), w_route),
        1e-6 if _x64() else 1e-5,
        f"|return - (Q_T - Q_0)|: energy {errs['energy']:.1e}, ergotropy {errs['ergotropy']:.1e}; "
        f"|ergotropy(2^N matrix) - ergotropy(blocks / d_j)| {w_route:.1e} (N=3, 10 random steps; "
        f"final energy {finals['energy']:.3f}, ergotropy {finals['ergotropy']:.3f} omega0)",
    )


def check_local_cell_dissipator_symmetry() -> CheckResult:
    """N = 1: local emission / dephasing / pumping coincide with the single-spin dissipators.

    For one cell the only sector is ``j = 1/2`` and ``sigma_-^(1) = J-``; the local
    coefficients must equal the generic superoperators ``D[sqrt(r) J-]``,
    ``D[sqrt(r) Jz]`` and ``D[sqrt(r) J+]`` built from matrices (exact in float64).
    """
    jm = np.array([[0.0, 1.0], [0.0, 0.0]])
    jz = np.diag([-0.5, 0.5])
    eye = np.eye(2)

    def sup(op: np.ndarray) -> np.ndarray:
        ldl = op.T @ op
        return np.kron(op, op) - 0.5 * np.kron(ldl, eye) - 0.5 * np.kron(eye, ldl.T)

    errs = [
        np.max(np.abs(local_cell_dissipator(1, 0.3, 0, 0) - 0.3 * sup(jm))),
        np.max(np.abs(local_cell_dissipator(1, 0, 0.3, 0) - 0.3 * sup(jz))),
        np.max(np.abs(local_cell_dissipator(1, 0, 0, 0.3) - 0.3 * sup(jm.T))),
    ]
    return _res(
        "single_cell_dissipators",
        "algebra",
        float(max(errs)),
        1e-12,
        "N=1 local coefficients vs D[J-], D[Jz], D[J+] (emission, dephasing, pumping): "
        + ", ".join(f"{e:.1e}" for e in errs),
    )


CHECKS: tuple[Callable[[], CheckResult], ...] = (
    check_structure,
    check_local_cell_dissipator_symmetry,
    check_brute_force_agreement,
    check_collective_limit,
    check_piqs_cross_check,
    check_local_emission_relaxation,
    check_local_dephasing_and_pumping,
    check_density_matrix_constraints,
    check_expm_step_exactness,
    check_env_return_equals_gain,
)
