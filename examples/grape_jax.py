"""GRAPE-like open-loop pulse optimisation through a QEnvS environment with optax.

The pulse is parametrised as ``actions = action_bound * tanh(theta)`` with ``theta`` of
shape ``(horizon, action_dim)``, so every amplitude stays inside the environment's
bounds without clipping. The objective is ``-sum(rewards)`` of ``env.rollout(actions)``
(for ``qubit/StateTransfer-v0`` that is ``F_initial - F_final``) and is minimised with
``optax.adam``. Gradients flow through the dynamiqs ODE solve inside ``env.step``.

Usage::

    python examples/grape_jax.py --task qubit/StateTransfer-v0 --steps 200 --lr 0.1 \
        --set horizon=10 --set dt=0.2 --json

Known optimum for the closed ``qubit/StateTransfer-v0`` task: return 1.0 for any pulse
whose integral ``sum(u dt)`` equals ``pi`` (a pi-pulse).

The optimiser differentiates the environment return directly, for closed and open
systems alike. Since qenvs 0.1.0 the fidelity to a pure target is ``<psi|rho|psi>``,
which is exact and has a finite gradient on density matrices (0.0.1 used the general
Uhlmann formula, whose gradient is NaN for a pure target).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import optax
from jax import Array

import qenvs


def parse_set(items: Sequence[str] | None) -> dict[str, object]:
    """Parse ``--set key=value`` items; values are JSON-decoded when possible (as in ``qenvs``)."""
    out: dict[str, object] = {}
    for item in items or []:
        if "=" not in item:
            msg = f"expected key=value, got {item!r}"
            raise argparse.ArgumentTypeError(msg)
        k, v = item.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def pulse_from_theta(theta: Array, action_bound: float) -> Array:
    """Map unconstrained parameters to bounded control amplitudes."""
    return action_bound * jnp.tanh(theta)


def pulse_summary(actions: Array, dt: float) -> dict[str, list[float]]:
    """Per-channel integral ``sum(u dt)``, mean and maximum absolute amplitude."""
    a = jnp.asarray(actions)
    return {
        "integral": [float(x) for x in a.sum(axis=0) * dt],
        "mean": [float(x) for x in a.mean(axis=0)],
        "max_abs": [float(x) for x in jnp.abs(a).max(axis=0)],
    }


def run_grape(
    task: str,
    steps: int = 200,
    lr: float = 0.1,
    seed: int = 0,
    *,
    init_scale: float = 0.5,
    history_every: int = 10,
    **task_kwargs: object,
) -> dict[str, object]:
    """Optimise an open-loop pulse for ``task`` with Adam and return a JSON-serialisable summary.

    Args:
        task: registered task id, e.g. ``"qubit/StateTransfer-v0"``.
        steps: number of Adam updates.
        lr: Adam learning rate.
        seed: seed of the random initial ``theta`` (``init_scale * N(0, 1)``).
        init_scale: standard deviation of the initial ``theta``. Zero is a stationary point
            of the state-transfer objective, so a random start is required.
        history_every: record the return every this many steps (step 0 and the last step
            are always recorded).
        **task_kwargs: forwarded to ``qenvs.make`` (``horizon``, ``dt``, ``gamma1``, ...).

    Returns:
        Dictionary with ``initial_return``, ``final_return``, ``best_return``, ``history``
        (list of ``{"step", "return"}``), ``wall_time_s``, ``compile_time_s`` and
        ``best_actions`` (integral / mean / max |u| per channel of the best pulse).
    """
    if steps < 1:
        msg = f"steps must be >= 1, got {steps}"
        raise ValueError(msg)
    env = qenvs.make(task, **task_kwargs)
    bound = float(env.action_bound)
    dt = float(env.dt)
    shape = (int(env.horizon), int(env.action_dim))

    def env_return(theta: Array) -> Array:
        _, rewards = env.rollout(pulse_from_theta(theta, bound))
        return rewards.sum()

    objective_name = "return"

    def objective(theta: Array) -> Array:
        return -env_return(theta)

    tx = optax.adam(lr)

    @jax.jit
    def update(theta: Array, opt_state: optax.OptState) -> tuple[Array, optax.OptState, Array]:
        value, grad = jax.value_and_grad(objective)(theta)
        updates, opt_state = tx.update(grad, opt_state, theta)
        return optax.apply_updates(theta, updates), opt_state, -value

    evaluate = jax.jit(env_return)

    t0 = time.perf_counter()
    theta = init_scale * jax.random.normal(jax.random.PRNGKey(seed), shape)
    opt_state = tx.init(theta)
    history: list[dict[str, float]] = []
    best_return, best_theta = -float("inf"), theta
    compile_time = None
    for i in range(steps):
        theta_next, opt_state, ret = update(theta, opt_state)
        ret_f = float(ret)  # objective value at `theta` (before this update)
        if compile_time is None:
            compile_time = time.perf_counter() - t0
            if not (math.isfinite(ret_f) and bool(jnp.all(jnp.isfinite(theta_next)))):
                msg = (
                    f"non-finite objective or gradient at the first step for {task!r} "
                    f"(objective={objective_name}); the task reward may not be differentiable"
                )
                raise RuntimeError(msg)
        if i % history_every == 0:
            history.append({"step": i, "return": ret_f})
        if ret_f > best_return:
            best_return, best_theta = ret_f, theta
        theta = theta_next
    final_return = float(evaluate(theta))
    history.append({"step": steps, "return": final_return})
    if final_return > best_return:
        best_return, best_theta = final_return, theta
    wall = time.perf_counter() - t0

    return {
        "task": task,
        "task_kwargs": dict(task_kwargs),
        "method": "grape-adam",
        "objective": objective_name,
        "steps": steps,
        "lr": lr,
        "seed": seed,
        "horizon": shape[0],
        "dt": dt,
        "action_dim": shape[1],
        "action_bound": bound,
        "open_system": bool(env.model.is_open),
        "initial_return": history[0]["return"],
        "final_return": final_return,
        "best_return": best_return,
        "history": history,
        "wall_time_s": wall,
        "compile_time_s": compile_time,
        "best_actions": pulse_summary(pulse_from_theta(best_theta, bound), dt),
        "final_actions": pulse_summary(pulse_from_theta(theta, bound), dt),
    }


def build_parser() -> argparse.ArgumentParser:
    """Command line options (mirrors ``qenvs run``)."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--task", default="qubit/StateTransfer-v0")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--set", action="append", metavar="KEY=VALUE", help="task kwargs, e.g. --set gamma1=0.1"
    )
    p.add_argument("--json", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point (enables JAX x64 like the ``qenvs`` CLI)."""
    jax.config.update("jax_enable_x64", True)
    args = build_parser().parse_args(argv)
    out = run_grape(args.task, args.steps, args.lr, args.seed, **parse_set(args.set))
    if args.json:
        json.dump(out, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        integral = ", ".join(f"{x:.4f}" for x in out["best_actions"]["integral"])  # type: ignore[index]
        sys.stdout.write(
            f"{out['task']} steps={out['steps']} lr={out['lr']} seed={out['seed']} "
            f"horizon={out['horizon']} dt={out['dt']}\n"
            f"return: initial={out['initial_return']:.6f} final={out['final_return']:.6f} "
            f"best={out['best_return']:.6f}\n"
            f"best pulse integral per channel: [{integral}]  (pi = {jnp.pi:.4f})\n"
            f"wall time: {out['wall_time_s']:.2f} s (compile {out['compile_time_s']:.2f} s)\n"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
