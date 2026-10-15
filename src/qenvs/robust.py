"""Robust evaluation of a policy over the distribution an environment samples from.

``evaluate`` runs ``n`` independent episodes of ``env`` under a fixed policy and reports the
physical final performance (mean, lower-tail CVaR, minimum) with bootstrap confidence
intervals where they are validated. The evaluation distribution is whatever the
environment carries: wrap the task in a parameter randomiser, a control-error wrapper or an
observation model first; there is no separate distribution argument. Evaluate a policy on a
distribution other than its training distribution by building the evaluation environment
separately and passing a key that the training stream did not use.

What is reported
----------------
* ``metric``: the physical final performance, read from the exact ``env.info(final_state)``:
  ``"fidelity"`` for gate and state-transfer tasks, the reward quantity (``"energy"`` or
  ``"ergotropy"``) for battery tasks. ``episode_return`` (the training reward summed over the
  episode, a gain ``Q(final) - Q(initial)``) is reported in separate fields; aggregation is
  always over the metric. The metric is *higher is better*, so the CVaR here is the mean of
  the lowest ``ceil(0.1 n)`` episodes, the opposite tail of a loss-based CVaR.
* ``mean``, ``cvar`` (lower 10 percent tail mean) and ``min``. The CVaR estimator has an
  ``O(1/n)`` bias (for U(0, 1) its expectation is exactly ``(k + 1) / (2 (n + 1))``,
  ``k = ceil(0.1 n)``, against the population value 0.05). The minimum is a descriptive
  statistic that falls as ``n`` grows: never compare minima of different ``n``.
* Intervals: non-parametric percentile bootstrap over episodes (``B = 2000``, 2.5 and 97.5
  percentiles). The mean interval is provided for ``n >= 256`` and the CVaR interval for
  ``n >= 1024``; below that only point estimates are returned, with a warning. In a
  Monte Carlo check (1000 independent repetitions each) the empirical coverage at
  ``n = 1024`` was inside [0.92, 0.98] for U(0, 1) and for a gate-fidelity-like
  distribution, but the CVaR interval undercovers for U(0, 1) (0.932, about 2.6 Monte Carlo
  standard errors below the nominal 0.95), so it is approximate and somewhat narrow. That is
  the evidence for those two distributions, not a coverage guarantee for any distribution:
  heavy-tailed, discrete or boundary-heavy performance distributions (for example a
  fidelity piled up at its upper bound 1) are separate validation targets.
* The intervals describe the sampling variation of the evaluation episodes for the given
  policy. Variation across independently trained policies (seeds) is a different source;
  report it separately and do not merge the two into one interval.

Random numbers: ``key`` is required. The episode keys are
``keys.episode_key(fold_in(key, 0), i)`` for ``i = 0, ..., n - 1``, the bootstrap key is
``fold_in(key, 1)`` and a stochastic policy receives ``keys.policy_key(episode_key, t)`` at
decision ``t = 0, 1, ...``. Use a key that does not belong to the training stream.
The outcome of an episode does not depend on the batch it is simulated in, up to the last
bits of floating-point rounding (XLA changes the operation order between batch sizes).
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from qenvs import keys
from qenvs.envs.base import Environment
from qenvs.envs.battery import BatteryCharging
from qenvs.envs.gate import GateControl
from qenvs.envs.state_transfer import StateTransfer
from qenvs.wrappers.core import Wrapper

CVAR_LEVEL = 0.1
"""Fraction of the episodes in the lower tail averaged by :func:`cvar`."""

N_BOOT = 2000
"""Bootstrap replicates ``B`` of the percentile intervals."""

MIN_N_MEAN_INTERVAL = 256
MIN_N_CVAR_INTERVAL = 1024
DEFAULT_SIZES = (256, 1024, 4096)

Policy = Callable[[Array, Array], Array] | Array | np.ndarray | Sequence[Sequence[float]]
Metric = str | Callable[[Any], Any] | None


# ----------------------------------------------------------------------- estimators
def tail_count(n: int, level: float = CVAR_LEVEL) -> int:
    """Number ``k = ceil(level * n)`` of episodes in the lower tail (exact for 0.1)."""
    if level == CVAR_LEVEL:
        return -(-n // 10)  # integer ceil(n / 10), no float rounding of 0.1 * n
    return math.ceil(level * n)


def cvar(values: np.ndarray | Array, axis: int = -1) -> np.ndarray | float:
    """Lower-tail CVaR: mean of the lowest ``ceil(0.1 n)`` entries along ``axis``.

    Higher is better, so this is the average of the worst 10 percent of the episodes.
    Ties are handled exactly (the tail is a multiset of values). Biased by ``O(1/n)``.
    """
    v = np.asarray(values, dtype=float)
    k = tail_count(v.shape[axis])
    low = np.partition(v, k - 1, axis=axis)
    low = np.take(low, np.arange(k), axis=axis)
    out = low.mean(axis=axis)
    return float(out) if np.ndim(out) == 0 else out


def _numpy_rng(key: Array) -> np.random.Generator:
    data = np.asarray(jax.random.key_data(key)).ravel()
    return np.random.default_rng([int(x) for x in data])


def bootstrap_intervals(
    values: np.ndarray | Array,
    key: Array,
    *,
    mean: bool = True,
    tail: bool = True,
    n_boot: int = N_BOOT,
    level: float = 0.95,
) -> dict[str, tuple[float, float]]:
    """Percentile bootstrap intervals of the mean and of the CVaR of ``values``.

    Resamples the ``n`` episode values with replacement ``n_boot`` times (both statistics
    use the same resamples) and returns the ``(1 - level) / 2`` and ``(1 + level) / 2``
    percentiles of each statistic, keyed ``"mean"`` and ``"cvar"``. The resampling indices
    come from a NumPy generator seeded with the data of ``key``, so the result is a
    function of ``(values, key)`` only.
    """
    v = np.asarray(values, dtype=float).ravel()
    n = v.size
    rng = _numpy_rng(key)
    means: list[np.ndarray] = []
    tails: list[np.ndarray] = []
    chunk = max(1, min(n_boot, 4_000_000 // max(n, 1)))
    done = 0
    while done < n_boot:
        c = min(chunk, n_boot - done)
        sample = v[rng.integers(0, n, size=(c, n), dtype=np.int32)]
        if mean:
            means.append(sample.mean(axis=1))
        if tail:
            tails.append(cvar(sample, axis=1))
        done += c
    lo, hi = 100.0 * (1 - level) / 2, 100.0 * (1 + level) / 2
    out: dict[str, tuple[float, float]] = {}
    for name, parts in (("mean", means), ("cvar", tails)):
        if parts:
            a, b = np.percentile(np.concatenate(parts), [lo, hi])
            out[name] = (float(a), float(b))
    return out


# ----------------------------------------------------------------------- metric choice
def _unwrapped(env: Environment) -> Environment:
    return env.unwrapped if isinstance(env, Wrapper) else env


def performance_metric(env: Environment) -> str:
    """Name of the physical final performance of the task behind ``env``.

    Inferred from the unwrapped environment: ``GateControl`` (average gate fidelity, leakage
    included) and ``StateTransfer`` (target-state fidelity) give ``"fidelity"``;
    ``BatteryCharging`` gives its reward quantity, ``"energy"`` or ``"ergotropy"``. Any other
    environment raises ``ValueError``: pass ``metric=`` to :func:`evaluate` explicitly.
    """
    base = _unwrapped(env)
    if isinstance(base, GateControl | StateTransfer):
        return "fidelity"
    if isinstance(base, BatteryCharging):
        return base.reward
    msg = (
        f"no default performance metric for {type(base).__name__}; "
        "pass metric='<info key>' or metric=<function of the final state>"
    )
    raise ValueError(msg)


def _metric_fn(env: Environment, metric: Metric) -> tuple[str, Callable[[Any], Array]]:
    if callable(metric):
        name = getattr(metric, "__name__", "metric")
        return name, lambda state: jnp.asarray(metric(state))
    name = performance_metric(env) if metric is None else str(metric)

    def read(state: Any) -> Array:
        info = env.info(state)
        if name not in info:
            msg = f"metric {name!r} is not a key of env.info(state); available: {sorted(info)}"
            raise KeyError(msg)
        return info[name]

    return name, read


def _policy_fn(env: Environment, policy: Policy) -> Callable[[Array, Array, Array], Array]:
    if callable(policy):
        return lambda obs, key, _t: policy(obs, key)  # type: ignore[operator]
    actions = jnp.asarray(policy)
    expected = (int(env.horizon), int(env.action_dim))
    if actions.shape != expected:
        msg = (
            "an open-loop policy must have shape (horizon, action_dim) = "
            f"{expected}, got {actions.shape}"
        )
        raise ValueError(msg)
    return lambda _obs, _key, t: actions[t]


# ------------------------------------------------------------------------ running
def _runner(
    env: Environment,
    policy: Policy,
    metric: Metric,
    root: Array,
) -> tuple[str, Callable[[np.ndarray], tuple[Array, Array]]]:
    """Jitted function ``indices -> (metric values, episode returns)`` for episodes ``i``."""
    name, read = _metric_fn(env, metric)
    act = _policy_fn(env, policy)
    horizon = int(env.horizon)

    def one(i: Array) -> tuple[Array, Array]:
        ep_key = keys.episode_key(root, i)
        state, ts = env.reset(ep_key)

        def body(carry: Any, t: Array) -> tuple[Any, Array]:
            st, obs = carry
            a = act(obs, keys.policy_key(ep_key, t), t)
            st, ts_new = env.step(st, a)
            return (st, ts_new.observation), ts_new.reward

        (final, _), rewards = jax.lax.scan(body, (state, ts.observation), jnp.arange(horizon))
        return read(final), jnp.sum(rewards)

    return name, jax.jit(jax.vmap(one))


def _collect(
    env: Environment,
    policy: Policy,
    metric: Metric,
    key: Array,
    n: int,
    batch_size: int | None,
) -> tuple[str, np.ndarray, np.ndarray]:
    root = jax.random.fold_in(key, 0)
    name, run = _runner(env, policy, metric, root)
    bs = n if batch_size is None else max(1, min(int(batch_size), n))
    vals: list[np.ndarray] = []
    rets: list[np.ndarray] = []
    for start in range(0, n, bs):
        idx = jnp.asarray(np.arange(start, start + bs), dtype=jnp.uint32)
        v, r = run(idx)
        take = min(bs, n - start)
        vals.append(np.asarray(v, dtype=float)[:take])
        rets.append(np.asarray(r, dtype=float)[:take])
    return name, np.concatenate(vals), np.concatenate(rets)


# ------------------------------------------------------------------------- reports
@dataclass(frozen=True)
class Evaluation:
    """Result of :func:`evaluate`.

    Attributes:
        metric: name of the aggregated performance metric.
        n: number of episodes.
        mean: mean of the metric.
        cvar: mean of the lowest ``ceil(0.1 n)`` metric values (O(1/n) biased estimator).
        min: smallest metric value (descriptive; falls with ``n``).
        mean_interval: 95 percent bootstrap interval of the mean, ``None`` for ``n < 256``.
        cvar_interval: 95 percent bootstrap interval of the CVaR, ``None`` for ``n < 1024``.
        return_mean: mean episode return (training reward), reported apart from the metric.
        return_min: smallest episode return.
        nominal: metric of the same policy on ``nominal_env`` (mean of 2 episodes of an
            environment that is assumed deterministic), ``None`` if not requested.
        nominal_return: its episode return.
        values: per-episode metric, shape ``(n,)``, in episode order.
        returns: per-episode returns, shape ``(n,)``.
    """

    metric: str
    n: int
    mean: float
    cvar: float
    min: float
    mean_interval: tuple[float, float] | None
    cvar_interval: tuple[float, float] | None
    return_mean: float
    return_min: float
    nominal: float | None = None
    nominal_return: float | None = None
    values: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False, compare=False)
    returns: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False, compare=False)

    def as_dict(self) -> dict[str, Any]:
        """Plain-dict summary (floats and lists, without the per-episode arrays)."""
        out = {
            k: getattr(self, k)
            for k in ("metric", "n", "mean", "cvar", "min", "return_mean", "return_min")
        }
        out["mean_interval"] = None if self.mean_interval is None else list(self.mean_interval)
        out["cvar_interval"] = None if self.cvar_interval is None else list(self.cvar_interval)
        out["nominal"] = self.nominal
        out["nominal_return"] = self.nominal_return
        return out


@dataclass(frozen=True)
class Stability:
    """Result of :func:`stability`: the evaluation at several sample sizes.

    Attributes:
        results: ``{n: Evaluation}``; the episodes of the smaller sizes are the first
            episodes of the larger ones (same episode keys).
        mean_spread: ``max - min`` of the mean over the sizes.
        cvar_spread: ``max - min`` of the CVaR over the sizes (the minimum is not compared
            across sizes: it is a descriptive statistic that falls with ``n``).
    """

    results: dict[int, Evaluation]
    mean_spread: float
    cvar_spread: float


def _summarise(
    name: str,
    values: np.ndarray,
    returns: np.ndarray,
    key: Array,
    *,
    warn: bool = True,
) -> Evaluation:
    n = int(values.size)
    intervals: dict[str, tuple[float, float]] = {}
    want_mean = n >= MIN_N_MEAN_INTERVAL
    want_tail = n >= MIN_N_CVAR_INTERVAL
    if want_mean or want_tail:
        intervals = bootstrap_intervals(
            values, jax.random.fold_in(key, 1), mean=want_mean, tail=want_tail
        )
    if warn and not want_mean:
        warnings.warn(
            f"n = {n} < {MIN_N_MEAN_INTERVAL}: no confidence intervals, point estimates only",
            UserWarning,
            stacklevel=3,
        )
    elif warn and not want_tail:
        warnings.warn(
            f"n = {n} < {MIN_N_CVAR_INTERVAL}: the CVaR interval is only checked for "
            f"n >= {MIN_N_CVAR_INTERVAL}; reporting the CVaR point estimate and the mean interval",
            UserWarning,
            stacklevel=3,
        )
    return Evaluation(
        metric=name,
        n=n,
        mean=float(values.mean()),
        cvar=float(cvar(values)),
        min=float(values.min()),
        mean_interval=intervals.get("mean"),
        cvar_interval=intervals.get("cvar"),
        return_mean=float(returns.mean()),
        return_min=float(returns.min()),
        values=values,
        returns=returns,
    )


def _check_key(key: Any) -> None:
    if key is None:
        msg = (
            "an explicit evaluation key is required: pass key=jax.random.PRNGKey(seed) "
            "that the training stream does not use"
        )
        raise ValueError(msg)
    if isinstance(key, int | np.integer) or not hasattr(key, "dtype"):
        msg = "key must be a JAX PRNG key (jax.random.PRNGKey(seed) or jax.random.key(seed))"
        raise ValueError(msg)


def _check_n(n: int, what: str = "n") -> int:
    if isinstance(n, bool) or not isinstance(n, int | np.integer) or n < 1:
        msg = f"{what} must be a positive integer, got {n!r}"
        raise ValueError(msg)
    return int(n)


# ------------------------------------------------------------------------ public API
def evaluate(
    env: Environment,
    policy: Policy,
    key: Array,
    n: int = 1024,
    metric: Metric = None,
    nominal_env: Environment | None = None,
    *,
    batch_size: int | None = None,
) -> Evaluation:
    """Evaluate ``policy`` on ``n`` independent episodes of ``env`` (see the module docs).

    Args:
        env: environment to evaluate in; carries the evaluation distribution (wrappers).
        policy: an open-loop action array of shape ``(horizon, action_dim)`` or a
            JAX-traceable function ``(observation, key) -> action`` (stochastic policies use
            ``key``, a fresh key ``keys.policy_key(episode_key, t)`` at every decision ``t``).
        key: explicit evaluation key (required); see the module docs for the derivation.
        n: number of episodes (default 1024; ``n >= 256`` for a mean interval, ``n >= 1024``
            for a CVaR interval).
        metric: an ``env.info`` key, a function of the final state returning a scalar, or
            ``None`` for :func:`performance_metric`.
        nominal_env: the same task without randomisation. The policy is also run on it for
            2 episodes (it is assumed deterministic; a warning is issued if the two
            episodes differ) and the result is reported as ``Evaluation.nominal``, so that
            robustness bought at the expense of nominal performance is visible.
        batch_size: simulate the episodes in chunks of this size to bound memory (default:
            all at once).

    Returns:
        :class:`Evaluation`.

    Raises:
        ValueError: no key, a non-positive ``n``, an open-loop array of the wrong shape, or
            an environment without a default metric and no ``metric``.
    """
    _check_key(key)
    n = _check_n(n)
    name, values, returns = _collect(env, policy, metric, key, n, batch_size)
    result = _summarise(name, values, returns, key)
    if nominal_env is None:
        return result
    _, nv, nr = _collect(nominal_env, policy, metric, key, 2, None)
    if not np.allclose(nv[0], nv[1], rtol=0.0, atol=1e-9):
        warnings.warn(
            "nominal_env gave different values on two episodes: it is not deterministic, "
            "so Evaluation.nominal is the mean of 2 draws",
            UserWarning,
            stacklevel=2,
        )
    return replace(result, nominal=float(nv.mean()), nominal_return=float(nr.mean()))


def stability(
    env: Environment,
    policy: Policy,
    key: Array,
    sizes: Sequence[int] = DEFAULT_SIZES,
    metric: Metric = None,
    *,
    batch_size: int | None = None,
) -> Stability:
    """Mean and CVaR at several sample sizes, to show whether they have settled.

    One run of ``max(sizes)`` episodes is simulated; the result for a smaller size uses the
    first ``n`` of those episodes (the same episode keys as ``evaluate(..., n=n)``), so the
    sizes are nested and share one bootstrap key. Intervals follow the rules of
    :func:`evaluate` for each size (no interval below its minimum ``n``); no warning is
    issued because showing the small sizes is the purpose of this report.
    """
    _check_key(key)
    sizes = tuple(sorted({_check_n(s, "sizes entry") for s in sizes}))
    if not sizes:
        msg = "sizes must not be empty"
        raise ValueError(msg)
    name, values, returns = _collect(env, policy, metric, key, sizes[-1], batch_size)
    results = {s: _summarise(name, values[:s], returns[:s], key, warn=False) for s in sizes}
    means = [r.mean for r in results.values()]
    cvars = [r.cvar for r in results.values()]
    return Stability(
        results=results,
        mean_spread=float(max(means) - min(means)),
        cvar_spread=float(max(cvars) - min(cvars)),
    )
