"""Endpoint diagnostics for physical piecewise-constant controls.

This low-level helper does not apply environment action bounds or observations.
Metrics are privileged simulator diagnostics, not policy inputs. Only right
endpoints are sampled; their maximum is not a continuous-time maximum.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping

import jax
import jax.numpy as jnp
from dynamiqs import QArray
from dynamiqs.method import Method
from jax import Array

from qenvs.models.base import Model


def rollout_metrics(
    model: Model,
    actions: Array,
    *,
    dt: float,
    metrics: Mapping[str, Callable[[Model, QArray], Array]],
    method: Method,
) -> tuple[Array, dict[str, Array]]:
    """Evaluate scalar metrics at the end of each physical control interval.

    Args:
        model: a concrete model constructed outside JAX transformations.
        actions: real floating array (steps, model.n_controls), in physical units.
            No clipping or normalization is applied, and the function does not cast
            actions itself. JAX does: with jax_enable_x64 off (the JAX default) float64
            actions become float32 on conversion, so enable x64 to keep them.
        dt: positive finite control period, a concrete number fixed outside jit.
        metrics: nonempty mapping of names to pure (model, state) scalar-real
            callbacks. Capture this mapping outside jit, like dt and method.
        method: explicit dynamiqs solver, including its tolerances.

    Returns:
        Times dt through steps*dt and a dict of metric arrays, each shape (steps,).
        No initial sample or state history is returned. Differentiation with
        respect to actions, jit and vmap are supported.

    Raises:
        ValueError: invalid shape, dtype, dt, method, metrics or metric outputs.

    Finite action and output checks belong at the host evaluation boundary;
    this scan does not convert traced arrays to NumPy or hide nonfinite scores.
    """
    actions = jnp.asarray(actions)
    if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] != model.n_controls:
        msg = "actions must have nonempty shape (steps, model.n_controls)"
        raise ValueError(msg)
    if not jnp.issubdtype(actions.dtype, jnp.floating):
        msg = "actions must have a real floating dtype"
        raise ValueError(msg)
    try:
        dt = math.nan if isinstance(dt, (bool, str)) else float(dt)
    except (TypeError, jax.errors.ConcretizationTypeError):
        dt = math.nan  # a traced or non-numeric dt
    if not math.isfinite(dt) or dt <= 0:
        msg = "dt must be a concrete positive finite number, fixed outside jit"
        raise ValueError(msg)
    if not isinstance(method, Method):
        msg = "method must be an explicit dynamiqs solver, e.g. dq.method.Tsit5(rtol=..., atol=...)"
        raise ValueError(msg)
    if (
        not isinstance(metrics, Mapping)
        or not metrics
        or any(not isinstance(k, str) or not callable(f) for k, f in metrics.items())
    ):
        msg = "metrics must be a nonempty mapping of names to callbacks"
        raise ValueError(msg)

    def step(state: QArray, amplitudes: Array) -> tuple[QArray, dict[str, Array]]:
        state = model.evolve(state, model.hamiltonian(amplitudes), dt, method=method)
        values = {name: jnp.asarray(metric(model, state)) for name, metric in metrics.items()}
        for name, value in values.items():
            if value.shape != () or not jnp.issubdtype(value.dtype, jnp.floating):
                msg = f"metric {name!r} must return a real floating scalar"
                raise ValueError(msg)
        return state, values

    _, values = jax.lax.scan(step, model.initial_state(), actions)
    times = (jnp.arange(actions.shape[0]) + 1) * jnp.asarray(dt)
    return times, values
