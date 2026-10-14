"""Gate targets and gate-quality measures for (possibly leaky) unitaries and channels.

Conventions:

* Basis index 0 is the ground state, index 1 the excited state; a ``k``-level target
  acts on the first ``k`` levels (the computational subspace) of a ``d``-level system.
  Every fidelity and leakage function also accepts ``subspace``: a sequence of ``k``
  basis indices that span the computational subspace, in the order of the target's
  basis (for two qutrits in control (x) target order, ``(0, 1, 3, 4)``). The default
  ``None`` means the first ``k`` levels.
* Average gate fidelity is averaged over Haar-random pure inputs *inside* the
  computational subspace and compares with the ideal output (Wood and Gambetta,
  Phys. Rev. A 97, 032306 (2018), Eqs. (5)-(6))::

      F = [k (1 - L) + k^2 F_pro] / (k (k + 1)) = (k F_pro + 1 - L) / (k + 1)

  with leakage rate ``L = Tr[(1 - P) E(P / k)]`` and process fidelity
  ``F_pro = (1/k^2) sum_ij <i| U_t^dag E(|i><j|) U_t |j>``. For a unitary ``U`` this
  reduces to the closed form of Pedersen, Moller and Molmer, Phys. Lett. A 367, 47
  (2007): ``F = [Tr(M M^dag) + |Tr M|^2] / (k (k + 1))`` with ``M = P_k U_t^dag U P_k``.
  Global phases are irrelevant; relative phases (including Z errors) count.
* Superoperators use dynamiqs' column-stacking vectorisation, ``vec(rho)[i + d j] =
  rho[i, j]`` (checked against ``dq.vectorize`` and ``mesolve``), so
  ``vec(U rho U^dag) = (conj(U) kron U) vec(rho)``.

All functions accept JAX arrays or dynamiqs ``QArray`` objects and broadcast over
leading batch dimensions.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np
from jax import Array

SQ2 = 1.0 / math.sqrt(2.0)

#: Single-qubit targets (computational basis, index 1 = excited). SX follows the IBM
#: convention SX = (1/2) [[1 + i, 1 - i], [1 - i, 1 + i]], so SX @ SX = X.
TARGETS: dict[str, tuple[tuple[complex, ...], ...]] = {
    "I": ((1, 0), (0, 1)),
    "X": ((0, 1), (1, 0)),
    "Y": ((0, -1j), (1j, 0)),
    "Z": ((1, 0), (0, -1)),
    "H": ((SQ2, SQ2), (SQ2, -SQ2)),
    "SX": ((0.5 + 0.5j, 0.5 - 0.5j), (0.5 - 0.5j, 0.5 + 0.5j)),
}


def gate(name: str) -> Array:
    """Named single-qubit target as a ``(2, 2)`` complex array (see :data:`TARGETS`)."""
    try:
        return jnp.asarray(TARGETS[name], dtype=complex)
    except KeyError as e:
        msg = f"unknown gate {name!r}; known: {sorted(TARGETS)}"
        raise ValueError(msg) from e


def rz(theta: float | Array) -> Array:
    """``Rz(theta) = diag(exp(-i theta/2), exp(i theta/2))``."""
    ph = jnp.exp(-0.5j * jnp.asarray(theta))
    return jnp.stack(
        [jnp.stack([ph, jnp.zeros_like(ph)], -1), jnp.stack([jnp.zeros_like(ph), ph.conj()], -1)],
        -2,
    )


def _arr(x: object) -> Array:
    to_jax = getattr(x, "to_jax", None)
    return jnp.asarray(to_jax() if to_jax is not None else x)


def _indices(subspace: Sequence[int] | None, k: int, d: int) -> np.ndarray:
    """Computational basis indices (default ``0 .. k-1``), checked against ``k`` and ``d``."""
    if subspace is None:
        return np.arange(k)
    idx = np.asarray(subspace, dtype=int).reshape(-1)
    if idx.size != k or len(set(idx.tolist())) != k or idx.min() < 0 or idx.max() >= d:
        msg = f"subspace must list {k} distinct indices in [0, {d}), got {list(subspace)}"
        raise ValueError(msg)
    return idx


def _complement(idx: np.ndarray, d: int) -> np.ndarray:
    keep = np.ones(d, dtype=bool)
    keep[idx] = False
    return np.nonzero(keep)[0]


def _block(m: Array, rows: np.ndarray, cols: np.ndarray) -> Array:
    """``m[..., rows, cols]`` as a sub-matrix (static index arrays)."""
    return m[..., rows[:, None], cols[None, :]]


def average_gate_fidelity(
    u: object, target: object, *, subspace: Sequence[int] | None = None
) -> Array:
    """Average gate fidelity of a ``d``-level unitary ``u`` against a ``k``-level ``target``.

    ``F = [Tr(M M^dag) + |Tr M|^2] / (k (k + 1))``, ``M = target^dag P u P``
    (Pedersen et al. 2007), with ``P`` the projector on the first ``k`` levels or on the
    basis states listed in ``subspace`` (in the target's basis order). Population leaked
    out of the subspace lowers ``Tr(M M^dag)`` and ``|Tr M|``. Equal to 1 iff ``u`` acts
    as ``target`` on the subspace up to a global phase. Batched over leading dimensions
    of ``u``.
    """
    u, t = _arr(u), _arr(target)
    k = t.shape[-1]
    if subspace is None:
        m = jnp.conj(t).T @ u[..., :k, :k]
    else:
        idx = _indices(subspace, k, u.shape[-1])
        m = jnp.conj(t).T @ _block(u, idx, idx)
    tr_mm = jnp.sum(jnp.abs(m) ** 2, axis=(-2, -1))
    tr_m = jnp.trace(m, axis1=-2, axis2=-1)
    return (tr_mm + jnp.abs(tr_m) ** 2) / (k * (k + 1))


def leakage(u: object, k: int = 2, *, subspace: Sequence[int] | None = None) -> Array:
    """Leakage rate of a unitary out of the first ``k`` levels (or out of ``subspace``).

    ``L = 1 - (1/k) sum_{i,j<k} |u_ji|^2``, evaluated as the equivalent (for unitary
    ``u``) sum of leaked populations ``(1/k) sum_{i<k, j>=k} |u_ji|^2``, which avoids
    cancellation when ``L`` is tiny and stays non-negative. When ``subspace`` is given,
    ``k = len(subspace)`` and the sums run over those indices and their complement.
    """
    u = _arr(u)
    d = u.shape[-1]
    if subspace is None:
        return jnp.sum(jnp.abs(u[..., k:, :k]) ** 2, axis=(-2, -1)) / k
    idx = _indices(subspace, len(subspace), d)
    out = _complement(idx, d)
    return jnp.sum(jnp.abs(_block(u, out, idx)) ** 2, axis=(-2, -1)) / idx.size


def unitary_to_superop(u: object) -> Array:
    """Column-stacking superoperator of ``rho -> u rho u^dag``: ``conj(u) kron u``."""
    u = _arr(u)
    d = u.shape[-1]
    s = jnp.einsum("...ab,...cd->...acbd", jnp.conj(u), u)
    return s.reshape(*u.shape[:-2], d * d, d * d)


def apply_superop(s: object, rho: object) -> Array:
    """Apply a column-stacking superoperator ``s`` (``d^2 x d^2``) to ``rho`` (``d x d``)."""
    s, rho = _arr(s), _arr(rho)
    d = rho.shape[-1]
    vec = jnp.swapaxes(rho, -1, -2).reshape(*rho.shape[:-2], d * d)
    out = jnp.einsum("...ij,...j->...i", s, vec)
    return jnp.swapaxes(out.reshape(*out.shape[:-1], d, d), -1, -2)


def _channel_tensor(s: Array) -> Array:
    """``E[a, b, c, e] = E(|c><e|)[a, b]`` from a column-stacking superoperator."""
    n = s.shape[-1]
    d = math.isqrt(n)
    # column stacking: row index of S is a + d b, column index c + d e
    t = s.reshape(*s.shape[:-2], d, d, d, d)  # [..., b, a, e, c]
    return jnp.swapaxes(jnp.swapaxes(t, -4, -3), -2, -1)


def _sub_channel(e: Array, k: int, subspace: Sequence[int] | None) -> Array:
    """Restrict ``E[a, b, c, e]`` to the first ``k`` levels or to ``subspace``."""
    if subspace is None:
        return e[..., :k, :k, :k, :k]
    idx = _indices(subspace, k, e.shape[-1])
    return e[(Ellipsis, *np.ix_(idx, idx, idx, idx))]


def superop_leakage(s: object, k: int = 2, *, subspace: Sequence[int] | None = None) -> Array:
    """Leakage rate ``L = Tr[(1 - P) E(P / k)]`` of a channel (Wood and Gambetta 2018, Eq. (2)).

    Equals ``1 - Tr[P E(P / k)]`` for a trace-preserving map and reduces to
    :func:`leakage` for a unitary channel. ``P`` projects on the first ``k`` levels or,
    when given, on ``subspace`` (then ``k = len(subspace)``).
    """
    e = _channel_tensor(_arr(s))
    if subspace is None:
        pops = jnp.einsum("...aacc->...a", e[..., :, :, :k, :k]).real
        return jnp.sum(pops[..., k:], axis=-1) / k
    d = e.shape[-1]
    idx = _indices(subspace, len(subspace), d)
    out = _complement(idx, d)
    sub = e[(Ellipsis, *np.ix_(out, out, idx, idx))]
    pops = jnp.einsum("...aacc->...a", sub).real
    return jnp.sum(pops, axis=-1) / idx.size


def superop_process_fidelity(
    s: object, target: object, *, subspace: Sequence[int] | None = None
) -> Array:
    """Process fidelity on the subspace ``(1/k^2) sum_ij <i|U_t^dag E(|i><j|) U_t|j>``."""
    e = _channel_tensor(_arr(s))
    t = _arr(target)
    k = t.shape[-1]
    ek = _sub_channel(e, k, subspace)
    val = jnp.einsum("ai,bj,...abij->...", jnp.conj(t), t, ek)
    return val.real / (k * k)


def superop_average_gate_fidelity(
    s: object, target: object, *, subspace: Sequence[int] | None = None
) -> Array:
    """Average gate fidelity of a (leaky) channel ``s`` against a ``k``-level ``target``.

    ``F = (k F_pro + 1 - L') / (k + 1)`` with ``1 - L' = Tr[P E(P / k)]`` (Wood and
    Gambetta 2018, Eq. (6)), computed as an exact Haar average over pure inputs in the
    computational subspace (first ``k`` levels, or ``subspace``). Equals
    :func:`average_gate_fidelity` for a unitary channel.
    """
    e = _channel_tensor(_arr(s))
    t = _arr(target)
    k = t.shape[-1]
    ek = _sub_channel(e, k, subspace)
    kept = jnp.einsum("...aacc->...", ek).real  # Tr[P E(P)] = k (1 - L')
    pro = jnp.einsum("ai,bj,...abij->...", jnp.conj(t), t, ek).real  # k^2 F_pro
    return (kept + pro) / (k * (k + 1))
