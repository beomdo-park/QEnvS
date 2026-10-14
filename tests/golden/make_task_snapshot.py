"""Record the behaviour of released tasks in tests/golden/tasks.json.

    python tests/golden/make_task_snapshot.py              # add tasks missing from the file
    python tests/golden/make_task_snapshot.py --check      # print differences, change nothing

Existing entries are never rewritten: a released task must not change (bump its ``-vN``
instead). Run this as part of a release (see RELEASING.md) to add the new tasks.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

PATH = pathlib.Path(__file__).with_name("tasks.json")


def fixed_pulse(env: object) -> jax.Array:
    """Deterministic, task-independent test pulse inside the action bounds."""
    k = jnp.arange(env.horizon)[:, None]
    c = jnp.arange(env.action_dim)[None, :]
    return 0.5 * env.action_bound * jnp.sin(1.0 + 0.7 * k + 1.3 * c)


def snapshot(task_id: str) -> dict[str, object]:
    """Behaviour of one task with its defaults."""
    import qenvs

    spec = qenvs.registry[task_id]
    env = qenvs.make(task_id)
    zeros = jnp.zeros((env.horizon, env.action_dim))
    state, rewards = env.rollout(fixed_pulse(env))
    info = env.info(state)
    out: dict[str, object] = {
        "horizon": int(env.horizon),
        "dt": float(env.dt),
        "action_dim": int(env.action_dim),
        "action_bound": float(env.action_bound),
        "observation_dim": int(env.observation_dim),
        "open_system": bool(env.model.is_open),
        "info_keys": sorted(info),
        "zero_return": float(env.rollout(zeros)[1].sum()),
        "fixed_pulse_return": float(rewards.sum()),
        "fixed_pulse_info": {k: float(v) for k, v in sorted(info.items())},
    }
    if spec.reference is not None:
        out["reference_return"] = float(env.rollout(jnp.asarray(spec.reference(env)))[1].sum())
    return out


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    import qenvs

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--check", action="store_true")
    args = p.parse_args(argv)
    data = json.loads(PATH.read_text()) if PATH.exists() else {}
    added = []
    for task_id in sorted(qenvs.registry):
        if task_id not in data:
            if not args.check:
                data[task_id] = snapshot(task_id)
            added.append(task_id)
    if args.check:
        sys.stdout.write(f"not in snapshot: {added}\n")
        return 0
    PATH.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    sys.stdout.write(f"added {added}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
