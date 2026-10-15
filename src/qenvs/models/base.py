"""Base class for physical models.

A :class:`Model` is a pure description of the physics: the drift Hamiltonian,
the control operators the agent can drive, optional Lindblad jump operators,
the initial state and a dictionary of named observables. It knows nothing about
rewards, horizons or action spaces; those live in :mod:`qenvs.envs`.

Every model ships with analytic reference results that the validation test-suite
checks (see ``tests/``). That is the contract behind "physically validated".
"""

from __future__ import annotations

import abc
import dataclasses
import math
from typing import ClassVar

import dynamiqs as dq
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from dynamiqs import QArray, TimeQArray
from dynamiqs.method import Method, Tsit5
from jax import Array

DEFAULT_METHOD: Method = Tsit5()

STRUCTURAL = (
    "structural size or selector: it fixes array shapes or the initial state, so it cannot "
    "change inside jit or vmap"
)
"""Reason given when a structural field is used as a randomisable parameter."""

DISSIPATION = (
    "dissipation rate: whether it is zero decides the closed (ket) or open (density matrix) "
    "code path, so it is not a randomisable parameter"
)
"""Reason given when a dissipation rate is used as a randomisable parameter."""

HERMITIAN_CHECK_RTOL = 1e-8
"""Relative tolerance of the Hermiticity check of :meth:`Model.check_parameters`."""


@dataclasses.dataclass(frozen=True)
class ParameterSpec:
    """Declaration of one randomisable Hamiltonian parameter of a model.

    Attributes:
        unit: unit of the parameter value, for example ``"GHz"`` or ``"rad/ns"``. A
            distribution that states its own unit must agree with it.
        support: closed interval ``(lo, hi)`` of physically valid values; an end may be
            infinite (a transmon anharmonicity lives in ``(-inf, 0]``, a coupling in
            ``[0, inf)``, a detuning on the whole real line).
        w_ref: model reference width in the unit of the parameter: a typical device
            to device or calibration to calibration spread. It is used only to
            normalise the oracle observation ``(value - nominal) / w_ref`` and as the
            recommended scale in the docs, never to draw samples. ``0`` means "no
            reference scale": such a parameter is left out of the oracle observation.
    """

    unit: str
    support: tuple[float, float] = (-math.inf, math.inf)
    w_ref: float = 0.0

    def __post_init__(self) -> None:
        lo, hi = self.support
        lo, hi, w = float(lo), float(hi), float(self.w_ref)
        if math.isnan(lo) or math.isnan(hi) or lo > hi:
            msg = f"support must be a closed interval (lo, hi) with lo <= hi, got {self.support!r}"
            raise ValueError(msg)
        if not (math.isfinite(w) and w >= 0.0):
            msg = f"w_ref must be a finite number >= 0, got {self.w_ref!r}"
            raise ValueError(msg)
        if not isinstance(self.unit, str):
            msg = f"unit must be a string, got {self.unit!r}"
            raise ValueError(msg)
        object.__setattr__(self, "support", (lo, hi))
        object.__setattr__(self, "w_ref", w)

    def contains(self, value: float) -> bool:
        """Whether ``value`` lies in the closed support interval."""
        return self.support[0] <= value <= self.support[1]


class Model(eqx.Module):
    """Abstract physical model.

    Subclasses are immutable equinox modules (JAX pytrees), so they can be
    jitted, vmapped and differentiated through.

    Parameter contract (used by :class:`~qenvs.wrappers.randomization.ParameterRandomization`):
    a model declares its randomisable Hamiltonian parameters in :meth:`parameter_specs`
    and implements :meth:`with_parameters` (a pure function, safe inside ``jit`` and
    ``vmap``). The default ``with_parameters`` replaces the fields of the same name, so a
    model whose parameters are fields only has to declare them. Dissipation rates and
    structural sizes are never declared: naming one raises an error with the reason
    (``non_parameters`` maps such field names to the reason).
    """

    non_parameters: ClassVar[dict[str, str]] = {}

    @property
    @abc.abstractmethod
    def dim(self) -> int:
        """Hilbert-space dimension."""

    @abc.abstractmethod
    def drift(self) -> QArray:
        """Drift (uncontrolled) Hamiltonian ``H_0``."""

    @abc.abstractmethod
    def controls(self) -> list[QArray]:
        """Control operators ``H_k``; the total Hamiltonian is ``H_0 + sum_k u_k(t) H_k``."""

    def jump_ops(self) -> list[QArray]:
        """Lindblad jump operators. Empty list means closed (unitary) dynamics."""
        return []

    @abc.abstractmethod
    def initial_state(self) -> QArray:
        """Initial state (ket for closed systems, density matrix otherwise)."""

    def observables(self) -> dict[str, QArray]:
        """Named Hermitian observables reported by environments (may be empty)."""
        return {}

    def battery_hamiltonian(self) -> QArray:
        """Energy-storage Hamiltonian ``H_B`` (ground energy 0); battery models override."""
        msg = f"{type(self).__name__} does not define a battery Hamiltonian"
        raise NotImplementedError(msg)

    def battery_reduced(self, qstate: QArray) -> tuple[QArray, QArray]:
        """Reduced battery state and the battery Hamiltonian acting on it.

        Energy and ergotropy of a battery are properties of the battery
        subsystem alone. Models with ancillas (a charger cavity, a bath mode)
        override this to partial-trace them out and return ``H_B`` on the
        battery space. The default returns the full state and
        :meth:`battery_hamiltonian`.
        """
        return qstate, self.battery_hamiltonian()

    # ------------------------------------------------------------------ parameter contract
    def parameter_specs(self) -> dict[str, ParameterSpec]:
        """Randomisable Hamiltonian parameters: name to :class:`ParameterSpec` (default: none)."""
        return {}

    def parameter_values(self) -> dict[str, Array | float]:
        """Current (nominal) value of every declared parameter, in the unit of its spec."""
        return {name: getattr(self, name) for name in self.parameter_specs()}

    def _unknown_parameter_error(self, name: str) -> ValueError:
        declared = sorted(self.parameter_specs())
        reason = self.non_parameters.get(name)
        why = f": {reason}" if reason else ""
        msg = (
            f"{name!r} is not a randomisable parameter of {type(self).__name__}{why}. "
            f"Declared parameters: {declared}"
        )
        return ValueError(msg)

    def with_parameters(self, **values: Array | float) -> Model:
        """Return a copy of the model with the named parameters replaced (pure, jit and vmap safe).

        Only declared parameters (:meth:`parameter_specs`) may be named; anything else
        (a dissipation rate, ``levels``, a typo) raises ``ValueError`` with the reason. The
        values are not range-checked here (they may be tracers); use
        :meth:`check_parameters` for concrete values. The default implementation swaps the
        fields of the same name with ``equinox.tree_at``.
        """
        specs = self.parameter_specs()
        for name in values:
            if name not in specs:
                raise self._unknown_parameter_error(name)
        if not values:
            return self
        names = tuple(values)
        new = tuple(jnp.asarray(v) for v in values.values())
        return eqx.tree_at(lambda m: tuple(getattr(m, n) for n in names), self, new)

    def check_parameters(self, **values: Array | float) -> None:
        """Validate concrete parameter values before and after applying them.

        Before: every name is declared, every value is a finite real number inside the
        support of its :class:`ParameterSpec`. After: the drift Hamiltonian of
        ``with_parameters(**values)`` is Hermitian (relative tolerance
        ``HERMITIAN_CHECK_RTOL``). Raises ``ValueError`` naming the offending parameter.
        Needs concrete values (call it outside ``jit``).
        """
        specs = self.parameter_specs()
        for name, raw in values.items():
            if name not in specs:
                raise self._unknown_parameter_error(name)
            try:
                arr = np.asarray(raw)
            except jax.errors.TracerArrayConversionError as e:
                msg = f"check_parameters needs concrete values, {name!r} is traced"
                raise ValueError(msg) from e
            if np.iscomplexobj(arr) or arr.ndim != 0 or not np.isfinite(arr):
                msg = f"{name!r} must be a finite real scalar, got {raw!r}"
                raise ValueError(msg)
            lo, hi = specs[name].support
            if not specs[name].contains(float(arr)):
                msg = (
                    f"{name!r} = {float(arr):g} {specs[name].unit} is outside its support "
                    f"[{lo:g}, {hi:g}]"
                )
                raise ValueError(msg)
        h = np.asarray(self.with_parameters(**values).drift().to_jax())
        asym = float(np.max(np.abs(h - h.conj().T), initial=0.0))
        scale = float(np.max(np.abs(h), initial=0.0))
        if asym > HERMITIAN_CHECK_RTOL * max(scale, 1e-300):
            msg = (
                f"the drift Hamiltonian is not Hermitian with parameters {values!r}: "
                f"max|H - H^dagger| = {asym:.3e}"
            )
            raise ValueError(msg)

    def correlation_observables(self) -> dict[str, QArray]:
        """Hermitian second-order observables of the ``"correlations"`` observation mode.

        Default: none. Models define the second moments that matter for their task (the
        Dicke models give the six symmetrised second moments of the collective spin).
        """
        return {}

    # ------------------------------------------------------------------ helpers
    @property
    def n_controls(self) -> int:
        """Number of control channels."""
        return len(self.controls())

    @property
    def is_open(self) -> bool:
        """Whether the model has dissipation."""
        return len(self.jump_ops()) > 0

    def hamiltonian(self, amplitudes: Array) -> QArray:
        """Constant total Hamiltonian for fixed control amplitudes of shape ``(n_controls,)``."""
        h = self.drift()
        for k, hk in enumerate(self.controls()):
            h = h + amplitudes[k] * hk
        return h

    def pwc_hamiltonian(self, times: Array, amplitudes: Array) -> TimeQArray:
        """Piecewise-constant Hamiltonian.

        Args:
            times: interval boundaries, shape ``(N + 1,)``.
            amplitudes: control amplitudes, shape ``(n_controls, N)``.
        """
        h: TimeQArray | QArray = self.drift()
        for k, hk in enumerate(self.controls()):
            h = h + dq.pwc(times, amplitudes[k], hk)
        return h

    def evolve(
        self,
        state: QArray,
        hamiltonian: QArray | TimeQArray,
        duration: float | Array,
        *,
        method: Method | None = None,
    ) -> QArray:
        """Evolve ``state`` under ``hamiltonian`` for ``duration`` and return the final state.

        Uses the Schrodinger equation for closed models and the Lindblad master
        equation otherwise. Density matrices are always accepted.

        The ODE ``method`` (adaptive Tsit5 by default) takes as many internal
        integration steps as needed inside ``duration``: the agent's control
        period and the integrator's time step are independent. For piecewise
        constant Hamiltonians ``dynamiqs.method.Expm()`` is exact.
        """
        tsave = jnp.asarray([0.0, duration])
        method = DEFAULT_METHOD if method is None else method
        if self.is_open:
            # dynamiqs 0.3.6: mesolve(..., method=Expm(), save_states=False).final_state
            # returns the *vectorised* density matrix (n^2, 1); states[-1] is always (n, n).
            res = dq.mesolve(
                hamiltonian, self.jump_ops(), state, tsave, method=method, progress_meter=False
            )
            return res.states[-1]
        res = dq.sesolve(
            hamiltonian, state, tsave, method=method, save_states=False, progress_meter=False
        )
        return res.final_state
