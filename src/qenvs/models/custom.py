"""Bring-your-own Hamiltonian: a :class:`Model` built from user supplied matrices.

``HamiltonianModel`` wraps a drift Hamiltonian, control operators, optional Lindblad
jump operators, an initial state, named observables and an optional battery Hamiltonian
(all ``hbar = 1``). Inputs may be NumPy arrays, JAX arrays, dynamiqs ``QArray`` objects
or ``qutip.Qobj`` (QuTiP is never imported; a ``Qobj`` is recognised by its ``full()``
method). The arrays are stored as JAX leaves, so the model is a valid equinox pytree and
works unchanged with :class:`~qenvs.envs.StateTransfer`,
:class:`~qenvs.envs.BatteryCharging`, :class:`~qenvs.wrappers.GymnasiumEnv`, ``jit``,
``vmap`` and ``grad``.

What construction guarantees (all raise ``ValueError`` naming the offending operator):

* every operator is a finite square matrix of the same dimension ``n``;
* the drift, every control, every observable and the battery Hamiltonian are Hermitian to
  a relative tolerance of ``1e-8`` (they are then symmetrised exactly, ``(A + A^dag) / 2``);
* the initial state is a normalised ket or a Hermitian, unit-trace, positive semidefinite
  density matrix.

What it cannot guarantee: that the matrices describe the physics you meant (a wrong
sign, a wrong basis ordering, controls that are swapped). See
``docs/guides/custom-models.md`` and :mod:`qenvs.validation.generic`.

State-type convention (same as the built-in models): a closed model (no jump operators)
carries a ket; an open model carries a density matrix. A ket initial state combined with
jump operators becomes a density matrix. A *pure* density matrix without jump operators is
reduced to a ket; a *mixed* density matrix without jump operators is kept as a density
matrix and integrated with the master equation (empty dissipator), so ``is_open`` is true
whenever the model carries a density matrix.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence

import dynamiqs as dq
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from dynamiqs import QArray, TimeQArray
from dynamiqs.method import Method
from jax import Array

from qenvs.models.base import Model

HERMITIAN_RTOL = 1e-8
"""Relative tolerance on ``max|A - A^dag| / max|A|`` for drift, controls, observables, H_B."""

STATE_TOL = 1e-6
"""Absolute tolerance for the norm / trace / Hermiticity / negativity of the initial state."""


def _is_qobj(x: object) -> bool:
    return hasattr(x, "full") and hasattr(x, "dims") and callable(x.full)


def _dims_of(x: object) -> tuple[int, ...] | None:
    """Composite dims carried by a QArray or Qobj, ``None`` when it carries none."""
    raw = None
    if isinstance(x, QArray):
        raw = x.dims
    elif _is_qobj(x):
        raw = x.dims[0]
    if raw is None:
        return None
    try:
        return tuple(int(d) for d in raw)
    except (TypeError, ValueError):
        return None


def _to_numpy(x: object, name: str) -> np.ndarray:
    """Dense complex128 NumPy array from a QArray, Qobj, JAX/NumPy array or nested list."""
    try:
        if isinstance(x, QArray):
            arr = np.asarray(x.to_jax())
        elif _is_qobj(x):
            arr = np.asarray(x.full())
        else:
            arr = np.asarray(x)
        arr = arr.astype(np.complex128)
    except jax.errors.TracerArrayConversionError as e:
        msg = (
            f"{name} is a traced JAX value: HamiltonianModel validates its operators at "
            "construction and must be built from concrete arrays (build it outside jit/vmap)"
        )
        raise ValueError(msg) from e
    except (TypeError, ValueError) as e:
        msg = f"{name} could not be converted to a numeric array: {e}"
        raise ValueError(msg) from e
    if not np.all(np.isfinite(arr)):
        msg = f"{name} contains NaN or inf"
        raise ValueError(msg)
    return arr


def _operator(x: object, name: str, n: int | None = None) -> np.ndarray:
    arr = _to_numpy(x, name)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        msg = f"{name} must be a square matrix, got shape {arr.shape}"
        raise ValueError(msg)
    if n is not None and arr.shape[0] != n:
        msg = f"{name} has dimension {arr.shape[0]} but the drift Hamiltonian has dimension {n}"
        raise ValueError(msg)
    return arr


def _hermitian(x: object, name: str, n: int | None = None) -> np.ndarray:
    """Validated, exactly symmetrised Hermitian operator."""
    arr = _operator(x, name, n)
    asym = float(np.max(np.abs(arr - arr.conj().T), initial=0.0))
    scale = float(np.max(np.abs(arr), initial=0.0))
    if asym > HERMITIAN_RTOL * max(scale, 1e-300):
        msg = (
            f"{name} is not Hermitian: max|A - A^dagger| = {asym:.3e} "
            f"(relative {asym / max(scale, 1e-300):.3e}, tolerance {HERMITIAN_RTOL:.0e}). "
            "Pass the Hermitian operator (for a Hamiltonian term g (a + a^dagger) include "
            "both parts); if the asymmetry is rounding noise, symmetrise with "
            "(A + A.conj().T) / 2 first."
        )
        raise ValueError(msg)
    return 0.5 * (arr + arr.conj().T)


def _is_nested_number_list(x: Sequence[object]) -> bool:
    """True for a list of rows of numbers such as ``[[0, 1], [1, 0]]`` (one matrix, not a list)."""
    if len(x) == 0 or not all(isinstance(r, Sequence) and not isinstance(r, str) for r in x):
        return False
    return all(isinstance(v, (int, float, complex, np.number)) for r in x for v in r)  # type: ignore[attr-defined]


def _operator_list(x: object, name: str) -> list[object]:
    """Normalise ``x`` to a list of operators (single operator, stack or sequence)."""
    if x is None:
        return []
    if isinstance(x, QArray) or _is_qobj(x):
        return [x]
    if isinstance(x, Mapping):
        msg = f"{name} must be an operator or a sequence of operators, not a mapping"
        raise ValueError(msg)
    if isinstance(x, (np.ndarray, jax.Array)):
        if x.ndim == 2:
            return [x]
        if x.ndim == 3:
            return list(x)
        msg = f"{name} must be a matrix or a stack of matrices, got an array of shape {x.shape}"
        raise ValueError(msg)
    if isinstance(x, Sequence) and not isinstance(x, str):
        if _is_nested_number_list(x):
            msg = (
                f"{name} looks like a single matrix written as a nested list; {name} takes a "
                f"sequence of operators, so wrap a single operator in a list: {name}=[A]"
            )
            raise ValueError(msg)
        return list(x)
    msg = f"{name} must be an operator or a sequence of operators, got {type(x).__name__}"
    raise ValueError(msg)


def _initial_state(x: object, n: int) -> np.ndarray:
    """Validated initial state: ket of shape ``(n, 1)`` or density matrix ``(n, n)``."""
    arr = _to_numpy(x, "initial_state")
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim == 2 and arr.shape == (1, n) and n > 1:
        msg = "initial_state is a bra (row vector); pass a ket (column vector) or a density matrix"
        raise ValueError(msg)
    if arr.shape == (n, 1):
        norm2 = float(np.vdot(arr, arr).real)
        if abs(norm2 - 1.0) > STATE_TOL:
            msg = (
                f"initial_state is not normalised: |<psi|psi> - 1| = {abs(norm2 - 1.0):.3e} "
                f"(<psi|psi> = {norm2:.10g}, tolerance {STATE_TOL:.0e}); divide by its norm"
            )
            raise ValueError(msg)
        return arr / math.sqrt(norm2)
    if arr.shape == (n, n):
        herm = float(np.max(np.abs(arr - arr.conj().T), initial=0.0))
        if herm > STATE_TOL:
            msg = (
                f"initial_state density matrix is not Hermitian: max|rho - rho^dagger| = "
                f"{herm:.3e} (tolerance {STATE_TOL:.0e})"
            )
            raise ValueError(msg)
        rho = 0.5 * (arr + arr.conj().T)
        tr = float(np.trace(rho).real)
        if abs(tr - 1.0) > STATE_TOL:
            msg = (
                f"initial_state density matrix has trace {tr:.10g}: |tr - 1| = "
                f"{abs(tr - 1.0):.3e} (tolerance {STATE_TOL:.0e}), expected 1"
            )
            raise ValueError(msg)
        lam_min = float(np.linalg.eigvalsh(rho)[0])
        if lam_min < -STATE_TOL:
            msg = (
                "initial_state density matrix is not positive semidefinite: "
                f"smallest eigenvalue {lam_min:.3e} (tolerance {STATE_TOL:.0e} below zero)"
            )
            raise ValueError(msg)
        return rho / tr
    msg = (
        f"initial_state has shape {arr.shape}; expected a ket ({n}, 1) "
        f"or a density matrix ({n}, {n})"
    )
    raise ValueError(msg)


def _resolve_dims(n: int, dims: Sequence[int] | None, sources: Sequence[object]) -> tuple[int, ...]:
    """Composite dims: explicit ``dims`` wins; otherwise inferred from QArray/Qobj inputs."""
    if dims is not None:
        try:
            out = tuple(int(d) for d in dims)
        except (TypeError, ValueError) as e:
            msg = f"dims must be a sequence of integers, got {dims!r}"
            raise ValueError(msg) from e
        if any(d < 1 for d in out) or math.prod(out) != n:
            msg = f"dims {out} do not multiply to the Hilbert-space dimension {n}"
            raise ValueError(msg)
        return out
    found: tuple[int, ...] | None = None
    for src in sources:
        d = _dims_of(src)
        if d is None or len(d) < 2 or math.prod(d) != n:
            continue
        if found is not None and d != found:
            msg = f"operators carry conflicting dims {found} and {d}; pass dims explicitly"
            raise ValueError(msg)
        found = d
    return found if found is not None else (n,)


class HamiltonianModel(Model):
    """Model defined by user supplied matrices: ``H(t) = H_0 + sum_k u_k(t) H_k`` plus Lindblad ops.

    Build it with ``HamiltonianModel(...)`` or the equivalent
    :meth:`from_operators`; both validate their inputs (see the module docstring).

    Args:
        drift: drift Hamiltonian ``H_0``, Hermitian ``(n, n)``.
        controls: control operators ``H_k``; one operator, a sequence, or a ``(k, n, n)``
            stack. At least one is required (the environments need an action space).
        jump_ops: Lindblad jump operators ``L_j`` (any square ``(n, n)`` matrices, absorbed
            rates included, i.e. pass ``sqrt(gamma) * L``). Empty means closed dynamics.
        initial_state: normalised ket (``(n,)`` or ``(n, 1)``) or density matrix. Defaults
            to the basis state ``|0>`` (index 0).
        observables: mapping name to Hermitian operator, reported in ``info`` by the
            environments.
        battery_hamiltonian: Hermitian ``H_B`` on the full space, needed by
            :class:`~qenvs.envs.BatteryCharging`. The energy and ergotropy are evaluated on
            the full state; for a battery that is only one subsystem of a composite system
            subclass and override ``battery_reduced`` (see the custom-models guide).
        dims: subsystem dimensions of a composite system, e.g. ``(2, 2)`` for two qubits
            (their product must be ``n``). Inferred from ``QArray``/``Qobj`` inputs when
            omitted, else ``(n,)``. Needed for ``dq.ptrace`` on the model's states.

    Raises:
        ValueError: on any violated construction guarantee, naming the operator.

    Attributes:
        drift_matrix: stored ``H_0`` as a complex array.
        control_matrices: stored controls, shape ``(k, n, n)``.
        jump_matrices: stored jump operators, shape ``(m, n, n)``.
        initial_matrix: stored initial state, ket ``(n, 1)`` or density matrix ``(n, n)``.
        observable_matrices: stored observables by name.
        battery_matrix: stored ``H_B`` or ``None``.
        dims: composite subsystem dimensions (static).
    """

    drift_matrix: Array
    control_matrices: Array
    jump_matrices: Array
    initial_matrix: Array
    observable_matrices: dict[str, Array]
    battery_matrix: Array | None
    dims: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        drift: object,
        controls: object,
        jump_ops: object = (),
        initial_state: object | None = None,
        observables: Mapping[str, object] | None = None,
        battery_hamiltonian: object | None = None,
        dims: Sequence[int] | None = None,
    ) -> None:
        h0 = _hermitian(drift, "drift")
        n = h0.shape[0]
        if n < 2:
            msg = f"the Hilbert space must have dimension >= 2, got {n}"
            raise ValueError(msg)
        ctrl_in = _operator_list(controls, "controls")
        if not ctrl_in:
            msg = "at least one control operator is required (the environments need actions)"
            raise ValueError(msg)
        ctrls = [_hermitian(c, f"controls[{k}]", n) for k, c in enumerate(ctrl_in)]
        for k, c in enumerate(ctrls):
            if not np.any(c):
                warnings.warn(
                    f"controls[{k}] is all zero: actions on this channel have no effect",
                    UserWarning,
                    stacklevel=2,
                )
        jump_in = _operator_list(jump_ops, "jump_ops")
        jumps = [_operator(j, f"jump_ops[{k}]", n) for k, j in enumerate(jump_in)]
        if initial_state is None:
            psi0 = np.zeros((n, 1), dtype=np.complex128)
            psi0[0, 0] = 1.0
        else:
            psi0 = _initial_state(initial_state, n)
        if psi0.shape[1] == 1 and jumps:
            psi0 = psi0 @ psi0.conj().T
        elif psi0.shape[1] == n and not jumps:
            lam, vec = np.linalg.eigh(psi0)
            if lam[-1] >= 1.0 - STATE_TOL:  # pure state, no dissipation: keep a ket
                psi0 = vec[:, -1:]
        obs: dict[str, np.ndarray] = {}
        if observables is not None:
            if not isinstance(observables, Mapping):
                msg = "observables must be a mapping name -> operator"
                raise ValueError(msg)
            obs = {str(k): _hermitian(v, f"observables[{k!r}]", n) for k, v in observables.items()}
        hb = (
            None
            if battery_hamiltonian is None
            else _hermitian(battery_hamiltonian, "battery_hamiltonian", n)
        )
        sources = [drift, *ctrl_in, *jump_in]
        self.dims = _resolve_dims(n, dims, sources)
        cdt = jax.dtypes.canonicalize_dtype(complex)
        self.drift_matrix = jnp.asarray(h0, dtype=cdt)
        self.control_matrices = jnp.asarray(np.stack(ctrls), dtype=cdt)
        self.jump_matrices = jnp.asarray(
            np.stack(jumps) if jumps else np.zeros((0, n, n)), dtype=cdt
        )
        self.initial_matrix = jnp.asarray(psi0, dtype=cdt)
        self.observable_matrices = {k: jnp.asarray(v, dtype=cdt) for k, v in obs.items()}
        self.battery_matrix = None if hb is None else jnp.asarray(hb, dtype=cdt)

    @classmethod
    def from_operators(
        cls,
        drift: object,
        controls: object,
        jump_ops: object = (),
        initial_state: object | None = None,
        observables: Mapping[str, object] | None = None,
        battery_hamiltonian: object | None = None,
        dims: Sequence[int] | None = None,
    ) -> HamiltonianModel:
        """Build and validate a model from operators; see the class docstring for the arguments."""
        return cls(
            drift,
            controls,
            jump_ops=jump_ops,
            initial_state=initial_state,
            observables=observables,
            battery_hamiltonian=battery_hamiltonian,
            dims=dims,
        )

    # ------------------------------------------------------------------ Model API
    def _q(self, matrix: Array) -> QArray:
        return dq.asqarray(matrix, dims=self.dims)

    @property
    def dim(self) -> int:
        return int(self.drift_matrix.shape[-1])

    @property
    def is_open(self) -> bool:
        """True when the model carries jump operators or a mixed (density-matrix) state."""
        return bool(self.jump_matrices.shape[0] > 0 or self.initial_matrix.shape[-1] != 1)

    def drift(self) -> QArray:
        return self._q(self.drift_matrix)

    def controls(self) -> list[QArray]:
        return [self._q(c) for c in self.control_matrices]

    def jump_ops(self) -> list[QArray]:
        return [self._q(j) for j in self.jump_matrices]

    def initial_state(self) -> QArray:
        return self._q(self.initial_matrix)

    def observables(self) -> dict[str, QArray]:
        return {k: self._q(v) for k, v in self.observable_matrices.items()}

    def evolve(
        self,
        state: QArray,
        hamiltonian: QArray | TimeQArray,
        duration: float | Array,
        *,
        method: Method | None = None,
    ) -> QArray:
        """Evolve as :meth:`Model.evolve`, keeping the composite ``dims`` on the result.

        dynamiqs 0.3.6 drops the subsystem dims of a density matrix integrated with
        ``Expm`` (``(2, 2)`` becomes ``(4,)``), which breaks ``ptrace`` and the ``lax.scan``
        carry of an environment rollout; they are restored here.
        """
        out = super().evolve(state, hamiltonian, duration, method=method)
        if tuple(out.dims) != self.dims:
            out = dq.asqarray(out.to_jax(), dims=self.dims)
        return out

    def battery_hamiltonian(self) -> QArray:
        if self.battery_matrix is None:
            msg = (
                "this HamiltonianModel was built without battery_hamiltonian; pass "
                "battery_hamiltonian=H_B (Hermitian, same dimension as the drift) to use "
                "BatteryCharging"
            )
            raise NotImplementedError(msg)
        return self._q(self.battery_matrix)
