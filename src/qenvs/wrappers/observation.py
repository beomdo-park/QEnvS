"""Observation modes and finite-shot measurement: ``ObservationModel``.

The tasks of QEnvS hand the agent the full quantum state (or the accumulated propagator).
``ObservationModel`` replaces that observation by the kind of information an experiment
provides: expectation values of a fixed list of observables, optionally estimated from a
finite number of shots, or only the elapsed time (open-loop, "blind" control).

Modes (``observation=``):

* ``"state"``: the observation of the wrapped environment, unchanged (default, the
  released behaviour). For gate tasks this is the accumulated propagator or superoperator,
  which an experiment can only get from process tomography, so it is an ideal-simulator
  baseline and not a measurement-statistics observation (see :attr:`observation_kind`).
* ``"expectations"``: ``<O_j>`` for the observables of ``Model.observables()`` of the
  nominal model, in insertion order, then the time fraction ``t / duration``.
* ``"correlations"``: the ``"expectations"`` list followed by the observables of
  ``Model.correlation_observables()`` (second-order correlators; a model without that
  method contributes none), then the time fraction. The common observables get the same
  measurement noise as in ``"expectations"`` (same key leaves).
* ``"time"``: only ``t / duration``.

Whatever the wrapped environments append after the base observation (for example the
oracle parameters of a parameter randomiser) is kept at the end of the new observation.

Finite-shot model (``shots=n``, only for ``"expectations"`` and ``"correlations"``): each
observable is measured ``n`` times in its own eigenbasis, in a separate experiment, on
the ideal evolved state. The outcome counts are drawn from the exact multinomial
distribution of the distinct eigenvalues (degenerate eigenvalues are one outcome),
implemented as a chain of conditional binomial draws, and the estimate is the sample mean
of the outcome eigenvalues. Its variance is exactly ``(<O^2> - <O>^2) / n``. What is
assumed: no measurement back-action on the state, no readout error, no sharing of shots
between commuting observables (the cost is counted conservatively), and the observables
are those of the *nominal* model, so the estimate is of the nominal observable on the
(possibly randomised) physical state. ``shots=None`` returns the exact simulator value.

Information boundary:

1. The policy sees the observation only.
2. In the partial modes (everything but ``"state"``) ``info`` carries no exact value
   (fidelity, leakage, energy, ergotropy, expectations): only the cost keys ``shots_used``
   and ``clip_fraction`` (when a control-error wrapper provides it). The ``"state"`` mode
   keeps the released ``info``.
3. ``diagnostics=True`` adds the exact values as ``info["diagnostics"]``, for evaluation
   and debugging.
4. :meth:`~qenvs.wrappers.Wrapper.info` and :meth:`~qenvs.wrappers.Wrapper.diagnostics` always
   return the exact values; the API cannot stop a caller from feeding them to a policy.
5. The reward is not changed by the mode: it is the exact simulator gain of the task
   quantity, so even in the partial modes the cumulative reward reveals the exact
   performance trajectory. This is the usual simulation-RL assumption and differs from an
   experiment, where the reward would itself be estimated from measurements.

Random numbers: observable ``j`` at observation time ``t`` (``t = 0`` after reset, ``t = k``
after step ``k``) draws from ``keys.measurement_key(episode_key, t, j)``, nothing else.
"""

from __future__ import annotations

from collections.abc import Sequence

import dynamiqs as dq
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from dynamiqs import QArray
from jax import Array

from qenvs import keys
from qenvs.envs.base import Environment, TimeStep
from qenvs.envs.gate import GateControl
from qenvs.wrappers.core import WrappedState, Wrapper

MODES = ("state", "expectations", "correlations", "time")
SHOT_MODES = ("expectations", "correlations")
COST_INFO_KEYS = ("shots_used", "clip_fraction")
"""Keys that stay in ``info`` in the partial observation modes (cost, not performance)."""

MAX_SHOTS = 2**24
"""Largest ``shots``: outcome counts are floats, exact integers up to ``2**24`` in float32."""

DUPLICATE_TOL = 1e-12
"""Two observables closer than this in operator norm are duplicates (rejected)."""

HERMITIAN_RTOL = 1e-10
"""Relative tolerance of ``max|O - O^dag| / max(1, max|O|)`` for an observable."""

DEGENERACY_RTOL = 1e-10
"""Eigenvalues closer than ``DEGENERACY_RTOL * max(1, |O|_max)`` form one outcome."""


# ---------------------------------------------------------------------------- sampling
def outcome_counts(key: Array, probs: Array, shots: int | Array) -> Array:
    """Multinomial outcome counts as an exact chain of conditional binomial draws.

    Outcome ``m`` gets ``Binomial(shots - sum_{l<m} c_l, p_m / sum_{l>=m} p_l)``; the last
    outcome with positive probability gets all that is left, so the counts always sum to
    ``shots``. A probability of exactly 0 gives count 0 and an outcome that has probability
    1 gets all the shots, without calling the binomial sampler (whose inversion branch does
    not terminate for ``p = 0``).

    Args:
        key: PRNG key; the draws of outcome ``m`` use the ``m``-th key of ``split(key, M)``.
        probs: outcome probabilities, shape ``(M,)``, non-negative, summing to 1.
        shots: number of measurements (a positive integer).

    Returns:
        Counts of shape ``(M,)`` (a float array holding integers, dtype of ``probs``).
    """
    probs = jnp.asarray(probs)
    dtype = probs.dtype
    suffix = jnp.cumsum(probs[::-1])[::-1]
    subkeys = jax.random.split(key, probs.shape[0])

    def body(left: Array, xs: tuple[Array, Array, Array]) -> tuple[Array, Array]:
        k, p, s = xs
        q = jnp.where(s > 0, jnp.clip(p / jnp.where(s > 0, s, 1.0), 0.0, 1.0), 0.0)
        interior = (q > 0) & (q < 1)
        draw = jax.random.binomial(k, left, jnp.where(interior, q, 0.5), dtype=dtype)
        c = jnp.where(q >= 1, left, jnp.where(q <= 0, 0.0, draw))
        c = jnp.clip(c, 0.0, left)
        return left - c, c

    _, counts = jax.lax.scan(body, jnp.asarray(shots, dtype=dtype), (subkeys, probs, suffix))
    return counts


# --------------------------------------------------------------------- measured observable
class _Measured(eqx.Module):
    """One observable, eigendecomposed once: operator, distinct eigenvalues, projectors."""

    operator: QArray
    eigenvalues: Array  # (m,) distinct (grouped) eigenvalues, ascending
    eigenvectors: Array  # (n, n) columns, eigenvalue order of eigh
    groups: Array  # (m, n) 0/1, groups[g, i] = 1 if eigenvector i belongs to outcome g

    def probabilities(self, qstate: QArray) -> Array:
        """Outcome probabilities ``Tr(P_g rho)``, shape ``(m,)``.

        Entries below ``64 eps`` are set to 0 and the vector is renormalised: they are
        rounding residue of a state that is an exact eigenstate (an outcome that cannot
        occur must not occur, even in float32).
        """
        x = qstate.to_jax()
        v = self.eigenvectors.astype(x.dtype)
        if x.shape[-1] == 1:  # ket
            pop = jnp.abs((v.conj().T @ x)[:, 0]) ** 2
        else:  # density matrix: diag(V^dag rho V)
            pop = jnp.real(jnp.sum(v.conj() * (x @ v), axis=0))
        p = self.groups.astype(pop.dtype) @ pop
        p = jnp.maximum(p, 0.0)
        p = jnp.where(p < 64 * jnp.finfo(p.dtype).eps, 0.0, p)
        return p / jnp.sum(p)

    def expectation(self, qstate: QArray) -> Array:
        """Exact ``Tr(O rho)`` (the same call as the environments' ``info``)."""
        return dq.expect(self.operator, qstate).real

    def estimate(self, key: Array, qstate: QArray, shots: int) -> Array:
        """Sample mean of the outcome eigenvalues of ``shots`` measurements."""
        p = self.probabilities(qstate)
        counts = outcome_counts(key, p, shots)
        return jnp.sum(counts * self.eigenvalues.astype(p.dtype)) / shots


def _measured(name: str, op: QArray) -> _Measured:
    a = np.asarray(op.to_jax(), dtype=np.complex128)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        msg = f"observable {name!r} must be a square matrix, got shape {a.shape}"
        raise ValueError(msg)
    scale = max(1.0, float(np.max(np.abs(a))))
    asym = float(np.max(np.abs(a - a.conj().T)))
    if asym > HERMITIAN_RTOL * scale:
        msg = (
            f"observable {name!r} is not Hermitian (max|O - O^dag| = {asym:.3e}): a measured "
            "observable needs real eigenvalues. For a complex correlator measure its real and "
            "imaginary parts as two Hermitian observables"
        )
        raise ValueError(msg)
    a = 0.5 * (a + a.conj().T)
    lam, vec = np.linalg.eigh(a)
    tol = DEGENERACY_RTOL * scale
    start = 0
    values: list[float] = []
    members: list[range] = []
    for i in range(1, lam.size + 1):
        if i == lam.size or lam[i] - lam[i - 1] > tol:
            values.append(float(np.mean(lam[start:i])))
            members.append(range(start, i))
            start = i
    groups = np.zeros((len(values), lam.size))
    for g, idx in enumerate(members):
        groups[g, list(idx)] = 1.0
    cdt = jax.dtypes.canonicalize_dtype(complex)
    fdt = jax.dtypes.canonicalize_dtype(float)
    return _Measured(
        operator=op,
        eigenvalues=jnp.asarray(values, dtype=fdt),
        eigenvectors=jnp.asarray(vec, dtype=cdt),
        groups=jnp.asarray(groups, dtype=fdt),
    )


def _check_distinct(items: Sequence[tuple[str, QArray]]) -> None:
    names = [n for n, _ in items]
    dup_names = sorted({n for n in names if names.count(n) > 1})
    if dup_names:
        msg = f"observable names must be unique, got duplicates: {dup_names}"
        raise ValueError(msg)
    mats = [np.asarray(op.to_jax(), dtype=np.complex128) for _, op in items]
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if mats[i].shape != mats[j].shape:
                continue
            if np.linalg.norm(mats[i] - mats[j], 2) <= DUPLICATE_TOL:
                msg = (
                    f"observables {names[i]!r} and {names[j]!r} are the same operator "
                    "(a duplicate would count its shots twice); select one of them with "
                    "observables=[...]"
                )
                raise ValueError(msg)


def observation_kind(env: Environment) -> str:
    """Kind of the ``"state"`` observation of ``env``.

    ``"state"`` for state-transfer and battery tasks, ``"propagator"`` for a closed gate task
    and ``"superoperator"`` for an open one. The last two are accumulated maps that an
    experiment only obtains by process tomography, so the ``"state"`` mode of a gate task is
    an ideal-simulator baseline.
    """
    base = env
    while isinstance(base, Wrapper):
        base = base.env
    if isinstance(base, GateControl):
        return "superoperator" if base.model.is_open else "propagator"
    return "state"


# --------------------------------------------------------------------------- the wrapper
class ObservationModel(Wrapper):
    """Observation mode and finite-shot measurement of an environment (see module docs).

    Args:
        env: environment to wrap (a task, or another wrapper such as a parameter
            randomiser; the wrapper does not change dynamics or rewards).
        observation: ``"state"`` (default), ``"expectations"``, ``"correlations"`` or
            ``"time"``.
        shots: ``None`` (exact expectation values) or a positive integer number of
            measurements per observable and observation. Only allowed with
            ``"expectations"`` and ``"correlations"``.
        diagnostics: if True, ``info["diagnostics"]`` holds the exact values (evaluation and
            debugging only; never feed them to a policy).
        observables: names of the ``Model.observables()`` to measure, in the order given
            (default: all). Needed when the model lists two names for the same operator,
            for example ``energy`` and ``p1`` of a qubit with ``omega0 = 1``, which is
            rejected as a duplicate.

    Raises:
        ValueError: unknown mode, ``shots`` that is not a positive integer or is given for
            ``"state"``/``"time"``, a gate task in ``"expectations"``/``"correlations"``,
            no observables, an unknown name, a non-Hermitian operator, or two observables
            that are the same operator.

    Attributes:
        observation: the mode.
        shots: shots per observable and observation, or ``None``.
        observable_names: names of the measured observables, in observation order.
        observation_kind: what the ``"state"`` mode carries for this task (``"state"``,
            ``"propagator"`` or ``"superoperator"``).
        shots_per_step: ``shots * len(observable_names)``, the measurements of one
            observation (0 without shots).
    """

    observation: str = eqx.field(static=True)
    shots: int | None = eqx.field(static=True)
    with_diagnostics: bool = eqx.field(static=True)
    observable_names: tuple[str, ...] = eqx.field(static=True)
    base_obs_dim: int = eqx.field(static=True)
    measured: tuple[_Measured, ...]

    def __init__(
        self,
        env: Environment,
        observation: str = "state",
        shots: int | None = None,
        diagnostics: bool = False,
        observables: Sequence[str] | None = None,
    ) -> None:
        if observation not in MODES:
            msg = f"observation must be one of {MODES}, got {observation!r}"
            raise ValueError(msg)
        if shots is not None:
            ok = isinstance(shots, (int, np.integer)) and not isinstance(shots, bool)
            if not ok or shots < 1:
                msg = f"shots must be a positive integer or None, got {shots!r}"
                raise ValueError(msg)
            if shots > MAX_SHOTS:
                msg = f"shots must be at most {MAX_SHOTS} (exact float32 counts), got {shots}"
                raise ValueError(msg)
            if observation not in SHOT_MODES:
                msg = (
                    f"shots is only defined for observation in {SHOT_MODES}, got "
                    f"observation={observation!r}; it would be ignored, so it is rejected"
                )
                raise ValueError(msg)
        if not isinstance(diagnostics, (bool, np.bool_)):
            msg = f"diagnostics must be a bool, got {diagnostics!r}"
            raise ValueError(msg)
        base = env
        while isinstance(base, Wrapper):
            base = base.env
        if isinstance(base, GateControl) and observation not in ("state", "time"):
            msg = (
                f"gate tasks support observation 'state' and 'time' only, got {observation!r}: "
                "expectations of a gate need probe input states and a measurement design "
                "(planned for 0.4)"
            )
            raise ValueError(msg)
        if observables is not None and observation not in SHOT_MODES:
            msg = f"observables= selects measured observables; it needs observation in {SHOT_MODES}"
            raise ValueError(msg)

        measured: tuple[_Measured, ...] = ()
        names: tuple[str, ...] = ()
        if observation in SHOT_MODES:
            model = base.model  # type: ignore[attr-defined]
            first = dict(model.observables())
            if observables is not None:
                unknown = [n for n in observables if n not in first]
                if unknown:
                    msg = f"unknown observables {unknown}; the model defines {list(first)}"
                    raise ValueError(msg)
                first = {n: first[n] for n in observables}
            items = list(first.items())
            if observation == "correlations":
                corr = getattr(model, "correlation_observables", None)
                items += list((corr() if corr is not None else {}).items())
            if not items:
                msg = f"{type(model).__name__} defines no observables to measure"
                raise ValueError(msg)
            _check_distinct(items)
            measured = tuple(_measured(n, op) for n, op in items)
            names = tuple(n for n, _ in items)

        self.env = env
        self.observation = observation
        self.shots = None if shots is None else int(shots)
        self.with_diagnostics = bool(diagnostics)
        self.observable_names = names
        self.base_obs_dim = int(base.observation_dim)
        self.measured = measured

    # ------------------------------------------------------------------ properties
    @property
    def observation_kind(self) -> str:
        return observation_kind(self.env)

    @property
    def shots_per_step(self) -> int:
        return 0 if self.shots is None else self.shots * len(self.measured)

    @property
    def observation_dim(self) -> int:
        extra = self.env.observation_dim - self.base_obs_dim
        if self.observation == "state":
            return self.env.observation_dim
        return len(self.measured) + 1 + extra

    # ------------------------------------------------------------------ measurement
    def outcome_probabilities(self, qstate: QArray) -> list[Array]:
        """Outcome probabilities of every measured observable on ``qstate``."""
        return [m.probabilities(qstate) for m in self.measured]

    def measure(self, qstate: QArray, key: Array, t: int | Array) -> Array:
        """The values the policy sees for the measured observables, shape ``(K,)``.

        Args:
            qstate: ideal quantum state of the (possibly randomised) physical system.
            key: episode key; observable ``j`` uses ``keys.measurement_key(key, t, j)``.
            t: observation time (0 after reset, ``k`` after step ``k``).
        """
        if self.shots is None:
            vals = [m.expectation(qstate) for m in self.measured]
        else:
            vals = [
                m.estimate(keys.measurement_key(key, t, j), qstate, self.shots)
                for j, m in enumerate(self.measured)
            ]
        return jnp.stack(vals)

    # ------------------------------------------------------------------ Wrapper hook
    def _transform_timestep(self, state: WrappedState, ts: TimeStep) -> TimeStep:
        info = dict(ts.info)
        if self.observation == "state":
            if self.with_diagnostics:
                info["diagnostics"] = dict(ts.info)
            return TimeStep(ts.observation, ts.reward, ts.done, info)
        obs = ts.observation
        t_frac = obs[self.base_obs_dim - 1 : self.base_obs_dim]
        extra = obs[self.base_obs_dim :]
        parts = []
        if self.measured:
            parts.append(self.measure(state.qstate, state.key, state.step).astype(obs.dtype))
        parts += [t_frac, extra]
        new_info = {k: v for k, v in info.items() if k in COST_INFO_KEYS}
        new_info["shots_used"] = jnp.asarray(self.shots_per_step)
        if self.with_diagnostics:
            new_info["diagnostics"] = info
        return TimeStep(jnp.concatenate(parts), ts.reward, ts.done, new_info)
