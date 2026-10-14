"""Plain-data functions behind the CLI and the MCP server.

Everything here returns JSON-serialisable dicts so that both humans and agents
get the same answers from ``qenvs ...`` and from the MCP tools.
"""

from __future__ import annotations

import contextlib
import importlib
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

from qenvs.models.base import Model
from qenvs.registry import make, registry
from qenvs.validation import run_all
from qenvs.validation.core import environment_info

POLICIES = ("reference", "pi-pulse", "zero", "random")


def _param(v: object) -> object:
    """JSON-friendly model parameter (ints and bools keep their type)."""
    if isinstance(v, bool | int | float) or v is None:
        return v
    return str(v)


def list_tasks() -> list[dict[str, object]]:
    """All registered tasks with their one-line descriptions."""
    return [{"id": s.id, "model": s.model, "description": s.description} for s in registry.values()]


def task_info(task_id: str) -> dict[str, object]:
    """Full metadata of one task, including default env dimensions."""
    spec = registry[task_id]
    env = spec.factory()
    return {
        "id": spec.id,
        "model": spec.model,
        "description": spec.description,
        "references": list(spec.references),
        "kwargs": dict(spec.kwargs),
        "units": dict(spec.units),
        "reference_policy": spec.reference is not None,
        "defaults": {
            "horizon": int(env.horizon),
            "dt": float(env.dt),
            "action_dim": int(env.action_dim),
            "action_bound": float(env.action_bound),
            "observation_dim": int(env.observation_dim),
            "open_system": bool(env.model.is_open),
        },
        "usage": {
            "python": f'env = qenvs.make("{spec.id}"); state, ts = env.reset(key); '
            "state, ts = env.step(state, action)",
            "gymnasium": "GymnasiumEnv(env) from qenvs.wrappers",
            "cli": f'qenvs run "{spec.id}" --policy '
            f"{'reference' if spec.reference is not None else 'zero'} --json",
        },
    }


def run_rollout(
    task_id: str,
    policy: str = "pi-pulse",
    seed: int = 0,
    **kwargs: object,
) -> dict[str, object]:
    """Open-loop rollout with a reference policy; returns final metrics."""
    if policy not in POLICIES:
        msg = f"unknown policy {policy!r}; choose from {POLICIES}"
        raise ValueError(msg)
    env = make(task_id, **kwargs)
    shape = (env.horizon, env.action_dim)
    spec = registry[task_id]
    if policy == "reference":
        if spec.reference is None:
            msg = (
                f"task {task_id!r} has no registered reference policy; use pi-pulse, zero or random"
            )
            raise ValueError(msg)
        actions = jnp.asarray(spec.reference(env))
    elif policy == "pi-pulse":
        actions = jnp.full(shape, jnp.pi / (env.horizon * env.dt))
    elif policy == "zero":
        actions = jnp.zeros(shape)
    else:
        key = jax.random.PRNGKey(seed)
        b = env.action_bound
        actions = jax.random.uniform(key, shape, minval=-b, maxval=b)
    actions = jnp.clip(actions, -env.action_bound, env.action_bound)
    state, rewards = env.rollout(actions, jax.random.PRNGKey(seed))
    final_info = {k: float(v) for k, v in env.info(state).items()}
    model_params = {k: _param(v) for k, v in vars(env.model).items()}
    return {
        "task": task_id,
        "model": registry[task_id].model,
        "qenvs_version": environment_info()["qenvs"],
        "policy": policy,
        "seed": seed,
        "kwargs": dict(kwargs),
        "model_params": model_params,
        "units": dict(spec.units),
        "solver": {"method": type(env.method).__name__, "control_dt": float(env.dt)},
        "horizon": int(env.horizon),
        "dt": float(env.dt),
        "duration": float(env.duration),
        "open_system": bool(env.model.is_open),
        "actions": {
            "shape": list(actions.shape),
            "mean_abs": float(jnp.mean(jnp.abs(actions))),
            "integral": [float(x) for x in jnp.sum(actions, axis=0) * env.dt],
        },
        "return": float(rewards.sum()),
        "rewards": [float(r) for r in rewards],
        "final": final_info,
    }


def doctor() -> dict[str, object]:
    """Installation health: versions, backend, optional extras, and a 1-step smoke run."""
    optional: dict[str, str] = {}
    for name in ("gymnasium", "qutip"):
        try:
            mod = __import__(name)
            optional[name] = getattr(mod, "__version__", "installed")
        except ImportError:
            optional[name] = "not installed"
    try:
        env = make("qubit/StateTransfer-v0", horizon=1, dt=0.1)
        state, _ = env.reset(jax.random.PRNGKey(0))
        env.step(state, jnp.zeros(env.action_dim))
        smoke = "ok"
    except Exception as e:
        smoke = f"failed: {type(e).__name__}: {e}"
    info = environment_info()
    return {
        "ok": smoke == "ok",
        "environment": info,
        "optional": optional,
        "smoke": smoke,
        "notes": [
            "GPU: install a CUDA-enabled jax (<0.8) yourself; qenvs only pins the CPU package.",
            "x64 is recommended for validation tolerances: "
            "jax.config.update('jax_enable_x64', True).",
        ],
    }


SOLVER_ONLY_NOTE = (
    "`qutip_solver_cross_check` integrates the model's own matrices: it verifies the solver "
    "stack, not that the matrices are the physics you meant. Pass a `reference` model "
    "(`qenvs validate --model MOD:ATTR --reference MOD:ATTR`) to cross-check the matrices."
)


def fmt_tol(tol: float) -> str:
    """Tolerance for tables: one significant digit when exact (``5e-05``), else two."""
    one = f"{tol:.0e}"
    return one if float(one) == tol else f"{tol:.1e}"


def validation_markdown(out: dict[str, object]) -> str:
    """Render a :func:`validate` result as a Markdown report (one table per model)."""
    checks = out["checks"]
    n_ok = sum(1 for c in checks if c["passed"])
    lines = [
        f"**{n_ok} / {len(checks)} checks passed, {out['n_skipped']} skipped.** "
        "`error` is the measured deviation from the reference, `tol` the acceptance "
        "threshold, `margin` = tol / error (larger is safer).",
        "",
    ]
    models: dict[str, list[dict[str, object]]] = {}
    for c in checks:
        models.setdefault(str(c["model"] or "TwoLevel"), []).append(c)
    for model, rows in models.items():
        lines += [
            f"## {model}",
            "",
            "| check | category | error | tol | margin | status | detail |",
        ]
        lines.append("|---|---|---|---|---|---|---|")
        for c in rows:
            status = "SKIP" if c["skipped"] else ("PASS" if c["passed"] else "FAIL")
            err, tol = float(c["error"]), float(c["tolerance"])
            margin = "inf" if err == 0 else f"{tol / err:.3g}"
            detail = str(c["detail"]).replace("|", "\\|")
            lines.append(
                f"| `{c['name']}` | {c['category']} | {err:.2e} | {fmt_tol(tol)} | {margin} "
                f"| {status} | {detail} |"
            )
        lines.append("")
    if any(c["name"] == "qutip_solver_cross_check" for c in checks):
        lines += [f"Note: {SOLVER_ONLY_NOTE}", ""]
    lines += ["## Environment", "", "| key | value |", "|---|---|"]
    lines += [f"| {k} | `{v}` |" for k, v in out["environment"].items()]
    return "\n".join(lines) + "\n"


def validate() -> dict[str, object]:
    """Run all physics checks and report errors, tolerances and environment."""
    results = [r.to_dict() for r in run_all()]
    return {
        "all_passed": all(r["passed"] or r["skipped"] for r in results),
        "n_skipped": sum(1 for r in results if r["skipped"]),
        "checks": results,
        "environment": environment_info(),
    }


class InvalidModelError(Exception):
    """A ``--model package.module:attr`` spec could not be imported or is not a qenvs Model.

    Reported by the CLI as the JSON error code ``invalid_model``.
    """


def load_model(spec: str) -> Model:
    """Import ``package.module:attr`` and return the :class:`~qenvs.models.Model` it names.

    ``attr`` (dotted names allowed) is either a ``Model`` instance or a zero-argument
    callable returning one. The current working directory is put at the front of ``sys.path``
    while loading (and removed afterwards), so ``qenvs validate --model my_model:MODEL`` works
    next to ``my_model.py`` and a file named like a standard-library module is found first. A
    module the interpreter has already imported (``json``, ``argparse``, ...) stays cached and
    cannot be replaced. The import runs your module: only load code you trust.

    Raises:
        InvalidModelError: malformed spec, import or factory failure, or a result that is
            not a ``Model``.
    """
    module_name, sep, attr = spec.partition(":")
    if not (sep and module_name and attr):
        msg = f"expected 'package.module:attr', got {spec!r}"
        raise InvalidModelError(msg)
    cwd = str(Path.cwd())
    sys.path.insert(0, cwd)  # first, so a file named like a stdlib module is not shadowed
    importlib.invalidate_caches()
    try:
        obj = importlib.import_module(module_name)
        for part in attr.split("."):
            obj = getattr(obj, part)
        if not isinstance(obj, Model) and callable(obj):
            obj = obj()
    except Exception as e:
        msg = f"could not load model {spec!r}: {type(e).__name__}: {e}"
        raise InvalidModelError(msg) from e
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(cwd)  # first occurrence: the entry inserted above
    if not isinstance(obj, Model):
        msg = (
            f"{spec!r} is a {type(obj).__name__}, not a qenvs Model "
            "(expected a Model instance or a zero-argument callable returning one)"
        )
        raise InvalidModelError(msg)
    return obj


def validate_model(model: Model, **kwargs: object) -> dict[str, object]:
    """Run the generic checks on ``model``; same structure as :func:`validate`.

    ``kwargs`` go to :func:`qenvs.validation.generic.model_checks` (``horizon``, ``dt``,
    ``action_bound``, ``seed``, ``method``, ``reference``).
    """
    from qenvs.validation.generic import model_checks

    if not isinstance(model, Model):
        msg = f"expected a qenvs Model, got {type(model).__name__}"
        raise InvalidModelError(msg)
    ref = kwargs.get("reference")
    if ref is not None and not isinstance(ref, Model):
        msg = f"reference must be a qenvs Model, got {type(ref).__name__}"
        raise InvalidModelError(msg)
    results = [r.to_dict() for r in model_checks(model, **kwargs)]
    return {
        "all_passed": all(r["passed"] or r["skipped"] for r in results),
        "n_skipped": sum(1 for r in results if r["skipped"]),
        "checks": results,
        "environment": environment_info(),
    }
