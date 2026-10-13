"""Plain-data functions behind the CLI and the MCP server.

Everything here returns JSON-serialisable dicts so that both humans and agents
get the same answers from ``qenvs ...`` and from the MCP tools.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from qenvs.registry import make, registry
from qenvs.validation import run_all
from qenvs.validation.core import environment_info

UNITS = {
    "energy": "units of omega0 (the model's transition frequency); omega0 = 1 by default",
    "time": "units of 1/omega0; hbar = 1",
    "rates": "decay / dephasing / loss rates in units of omega0",
    "control": "each action component multiplies one control operator of the model "
    "(see `qenvs info <task>`); for a qubit, u sigma_x/2 gives Rabi frequency u",
}

POLICIES = ("pi-pulse", "zero", "random")


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
            "cli": f'qenvs run "{spec.id}" --policy pi-pulse --json',
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
    if policy == "pi-pulse":
        actions = jnp.full(shape, jnp.pi / (env.horizon * env.dt))
    elif policy == "zero":
        actions = jnp.zeros(shape)
    else:
        key = jax.random.PRNGKey(seed)
        b = env.action_bound
        actions = jax.random.uniform(key, shape, minval=-b, maxval=b)
    actions = jnp.clip(actions, -env.action_bound, env.action_bound)
    state, rewards = env.rollout(actions, jax.random.PRNGKey(seed))
    final_info = {k: float(v) for k, v in env._info(state.qstate, state.t).items()}
    model_params = {
        k: float(v) if isinstance(v, (int, float)) else str(v) for k, v in vars(env.model).items()
    }
    return {
        "task": task_id,
        "model": registry[task_id].model,
        "qenvs_version": environment_info()["qenvs"],
        "policy": policy,
        "seed": seed,
        "kwargs": dict(kwargs),
        "model_params": model_params,
        "units": UNITS,
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


def validate() -> dict[str, object]:
    """Run all physics checks and report errors, tolerances and environment."""
    results = [r.to_dict() for r in run_all()]
    return {
        "all_passed": all(r["passed"] or r["skipped"] for r in results),
        "n_skipped": sum(1 for r in results if r["skipped"]),
        "checks": results,
        "environment": environment_info(),
    }
