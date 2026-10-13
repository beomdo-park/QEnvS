"""Smoke-scale benchmark of the control baselines in ``examples/`` on ``qubit/StateTransfer-v0``.

Runs ``run_grape`` (open-loop Adam, ``examples/grape_jax.py``) and ``run_ppo`` (vectorised
PPO, ``examples/ppo_jax.py``) on the closed task and on the open variant (``gamma1=0.1``)
over several seeds and prints a markdown table::

    python examples/benchmark.py                 # markdown table
    python examples/benchmark.py --json          # raw dict (per-seed results, versions)
    python examples/benchmark.py --out bench.json

Settings are deliberately modest (CPU, a few minutes in total) and not tuned; the numbers
are a regression reference, not a leaderboard. Nothing is written to disk unless ``--out``
is given.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import statistics
import sys
import time
from collections.abc import Sequence

import jax
import jax.numpy as jnp

import qenvs
from qenvs.validation.core import environment_info

try:
    from grape_jax import run_grape
    from ppo_jax import run_ppo
except ImportError:  # imported from outside examples/ (e.g. the tests)
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from grape_jax import run_grape
    from ppo_jax import run_ppo

TASK = "qubit/StateTransfer-v0"
VARIANTS: dict[str, dict[str, object]] = {"closed": {}, "open (gamma1=0.1)": {"gamma1": 0.1}}
# closed-system optimum: any pulse with integral pi gives F = 1 (see qenvs.validation.qubit)
KNOWN_OPTIMUM: dict[str, float | None] = {"closed": 1.0, "open (gamma1=0.1)": None}


def pi_pulse_return(task: str, **task_kwargs: object) -> float:
    """Return of the constant pi-pulse (the closed-system optimum; a reference when open)."""
    env = qenvs.make(task, **task_kwargs)
    actions = jnp.full((env.horizon, env.action_dim), jnp.pi / (env.horizon * env.dt))
    _, rewards = env.rollout(actions)
    return float(rewards.sum())


def cpu_name() -> str:
    """Best-effort CPU model string."""
    try:
        for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def run_benchmark(
    seeds: Sequence[int] = (0, 1, 2),
    *,
    task: str = TASK,
    variants: dict[str, dict[str, object]] | None = None,
    horizon: int = 10,
    dt: float = 0.2,
    grape_steps: int = 200,
    grape_lr: float = 0.1,
    ppo_envs: int = 256,
    ppo_updates: int = 50,
) -> dict[str, object]:
    """Run both baselines on every variant and seed; returns a JSON-serialisable dict."""
    variants = VARIANTS if variants is None else variants
    rows: list[dict[str, object]] = []
    t_start = time.perf_counter()
    for variant, extra in variants.items():
        kw = {"horizon": horizon, "dt": dt, **extra}
        reference = pi_pulse_return(task, **kw)
        for method in ("grape", "ppo"):
            runs: list[dict[str, object]] = []
            t0 = time.perf_counter()
            for seed in seeds:
                if method == "grape":
                    runs.append(run_grape(task, grape_steps, grape_lr, seed, **kw))
                else:
                    runs.append(run_ppo(task, ppo_envs, ppo_updates, seed, **kw))
            wall = time.perf_counter() - t0
            returns = [float(r["final_return"]) for r in runs]  # type: ignore[arg-type]
            rows.append(
                {
                    "variant": variant,
                    "task_kwargs": kw,
                    "method": method,
                    "seeds": list(seeds),
                    "returns": returns,
                    "mean": statistics.fmean(returns),
                    "std": statistics.pstdev(returns) if len(returns) > 1 else 0.0,
                    "known_optimum": KNOWN_OPTIMUM.get(variant),
                    "pi_pulse_return": reference,
                    "wall_time_s": wall,
                    "runs": runs,
                }
            )
    info = environment_info()
    info.update({"cpu": cpu_name(), "cpu_count": str(os.cpu_count())})
    return {
        "task": task,
        "settings": {
            "seeds": list(seeds),
            "horizon": horizon,
            "dt": dt,
            "grape": {"steps": grape_steps, "lr": grape_lr},
            "ppo": {"num_envs": ppo_envs, "updates": ppo_updates},
        },
        "environment": info,
        "rows": rows,
        "total_wall_time_s": time.perf_counter() - t_start,
    }


def format_table(results: dict[str, object]) -> str:
    """Markdown table: task | method | seeds | mean +- std final return | known optimum | wall."""
    lines = [
        "| task | method | seeds | final return (mean +- std) | known optimum | wall-clock |",
        "|---|---|---|---|---|---|",
    ]
    for row in results["rows"]:  # type: ignore[union-attr]
        opt = row["known_optimum"]
        opt_s = f"{opt:.3f}" if opt is not None else f"n/a (pi-pulse {row['pi_pulse_return']:.3f})"
        n = len(row["seeds"])
        lines.append(
            f"| {results['task']} {row['variant']} | {row['method']} | {n} "
            f"| {row['mean']:.4f} +- {row['std']:.4f} | {opt_s} "
            f"| {row['wall_time_s']:.1f} s ({row['wall_time_s'] / n:.1f} s/seed) |"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Command line options."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--task", default=TASK)
    p.add_argument("--seeds", type=int, default=3, help="number of seeds (0, 1, ...)")
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--dt", type=float, default=0.2)
    p.add_argument("--grape-steps", type=int, default=200)
    p.add_argument("--grape-lr", type=float, default=0.1)
    p.add_argument("--ppo-envs", type=int, default=256)
    p.add_argument("--ppo-updates", type=int, default=50)
    p.add_argument("--json", action="store_true", help="print the raw results dict")
    p.add_argument("--out", type=pathlib.Path, default=None, help="also write the dict to PATH")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point (enables JAX x64 like the ``qenvs`` CLI)."""
    jax.config.update("jax_enable_x64", True)
    args = build_parser().parse_args(argv)
    results = run_benchmark(
        range(args.seeds),
        task=args.task,
        horizon=args.horizon,
        dt=args.dt,
        grape_steps=args.grape_steps,
        grape_lr=args.grape_lr,
        ppo_envs=args.ppo_envs,
        ppo_updates=args.ppo_updates,
    )
    if args.out is not None:
        args.out.write_text(json.dumps(results, indent=2))
    if args.json:
        json.dump(results, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        s = results["settings"]
        env = results["environment"]
        sys.stdout.write(
            format_table(results)
            + "\n\nsettings: "
            + f"horizon={s['horizon']} dt={s['dt']} grape={s['grape']} ppo={s['ppo']}\n"  # type: ignore[index]
            + f"environment: python {env['python']}, jax {env['jax']} ({env['jax_backend']}), "  # type: ignore[index]
            + f"dynamiqs {env['dynamiqs']}, qenvs {env['qenvs']}, x64={env['x64']}, "  # type: ignore[index]
            + f"cpu {env['cpu']} ({env['cpu_count']} logical cores)\n"  # type: ignore[index]
            + f"total wall-clock: {results['total_wall_time_s']:.1f} s\n"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
