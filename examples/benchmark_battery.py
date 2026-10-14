"""Charging benchmark on the battery tasks: known optimum, physical baseline, GRAPE, PPO.

For every task variant the script reports, per method, the stored energy and the
ergotropy of the battery at the end of the episode (both divided by the maximum
``N omega0``), the control energy ``sum(g^2 dt)`` and the wall-clock time::

    python examples/benchmark_battery.py              # markdown table (about 20 min, CPU)
    python examples/benchmark_battery.py --quick      # smoke settings (seconds)
    python examples/benchmark_battery.py --json --out battery_bench.json

Methods:

* ``optimum``: provable optimum, closed resonant tasks only. For resonant
  Tavis-Cummings ``H(t) = omega0 N_exc + g(t) V`` with ``[N_exc, V] = 0``, so the
  propagator is ``exp(-i omega0 N_exc T) exp(-i theta V)`` with pulse area
  ``theta = sum(g dt)``; the stored energy depends on ``theta`` alone and every
  ``theta`` in ``[-g_max T, g_max T]`` is reachable. The optimum is therefore the
  maximum of ``E(theta)`` over that interval (fine grid, float64). The same argument
  holds for the resonant closed qubit battery, ``H(t) = u(t) sigma_x / 2``.
* ``single pulse``: one rectangular pulse at ``+g_max`` with optimised start step and
  area (a fractional last step realises any area). A physically motivated baseline:
  for closed resonant tasks it reaches the optimum, for open tasks it trades charging
  speed against loss.
* ``GRAPE``: ``examples/grape_jax.py`` (open-loop Adam through ``env.rollout``, 1500 steps).
* ``PPO``: ``examples/ppo_jax.py`` (vectorised PPO, default hyper-parameters, not tuned
  for these tasks). The deterministic policy is evaluated.
* ``random``: mean over uniform random pulses in ``[-g_max, g_max]``.

The numbers are a regression reference for the library at modest, untuned settings,
not a statement about which control method is best in general.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

import qenvs
from qenvs.validation.core import environment_info

try:
    from grape_jax import run_grape
    from ppo_jax import run_ppo
except ImportError:  # imported from outside examples/ (e.g. the tests)
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from grape_jax import run_grape
    from ppo_jax import run_ppo

DICKE_OPEN = {"kappa": 0.1, "gamma": 0.05}
VARIANTS: list[tuple[str, str, dict[str, object]]] = [
    ("Dicke N=2, closed", "battery/Dicke-v0", {"n_cells": 2}),
    ("Dicke N=4, closed", "battery/Dicke-v0", {"n_cells": 4}),
    ("Dicke N=8, closed", "battery/Dicke-v0", {"n_cells": 8}),
    ("Dicke N=2, open", "battery/Dicke-v0", {"n_cells": 2, **DICKE_OPEN}),
    ("Dicke N=4, open", "battery/Dicke-v0", {"n_cells": 4, **DICKE_OPEN}),
    ("Qubit battery, closed", "battery/TwoLevel-v0", {}),
    ("Qubit battery, open", "battery/TwoLevel-v0", {"gamma1": 0.1, "n_th": 0.1}),
]


def n_max_energy(env: object) -> float:
    """Maximum storable energy ``N omega0`` (``omega0`` for the qubit battery)."""
    model = env.model
    return float(getattr(model, "n_cells", 1)) * float(model.omega0)


def evaluate_pulse(env: object, actions: Array) -> dict[str, float]:
    """Final energy, ergotropy and control energy of an open-loop pulse."""
    actions = jnp.clip(jnp.asarray(actions, dtype=jnp.float32), -env.action_bound, env.action_bound)
    state, _ = env.rollout(actions)
    info = env.info(state)
    scale = n_max_energy(env)
    return {
        "energy": float(info["energy"]) / scale,
        "ergotropy": float(info["ergotropy"]) / scale,
        "control_energy": float(jnp.sum(actions**2) * env.dt),
    }


def is_closed_resonant(env: object) -> bool:
    """Whether the provable single-parameter optimum applies."""
    m = env.model
    if m.is_open:
        return False
    if hasattr(m, "n_cells"):
        return bool(m.rwa) and float(m.omega_c) == float(m.omega0)
    return float(m.detuning) == 0.0


def known_optimum(env: object, points: int = 20001) -> dict[str, float] | None:
    """Maximum stored energy over the reachable pulse areas (closed resonant tasks)."""
    if not is_closed_resonant(env):
        return None
    m = env.model
    v = np.asarray(m.controls()[0].to_jax(), dtype=np.complex128)
    psi0 = np.asarray(m.initial_state().to_jax(), dtype=np.complex128).reshape(-1)
    hb = np.asarray(m.battery_hamiltonian().to_jax(), dtype=np.complex128)
    w, u = np.linalg.eigh(v)
    c = u.conj().T @ psi0
    theta_max = float(env.action_bound) * env.horizon * env.dt
    thetas = np.linspace(0.0, theta_max, points)
    psis = u @ (np.exp(-1j * np.outer(w, thetas)) * c[:, None])  # (dim, points)
    energies = np.real(np.einsum("it,ij,jt->t", psis.conj(), hb, psis))
    k = int(np.argmax(energies))
    return {"energy": float(energies[k]) / n_max_energy(env), "theta": float(thetas[k])}


def single_pulse_baseline(env: object, areas: int = 201) -> dict[str, object]:
    """Best rectangular pulse at +g_max over start step and area (vmapped rollouts)."""
    horizon, dt, g = env.horizon, float(env.dt), float(env.action_bound)
    theta_max = g * horizon * dt
    starts = np.arange(horizon)
    thetas = np.linspace(0.0, theta_max, areas)
    pulses = []
    for s in starts:
        for th in thetas:
            n_full = min(int(th / (g * dt)), horizon - s)
            pulse = np.zeros(horizon)
            pulse[s : s + n_full] = g
            rest = th - n_full * g * dt
            if s + n_full < horizon and rest > 0:
                pulse[s + n_full] = rest / dt
            pulses.append(pulse)
    batch = jnp.asarray(np.stack(pulses)[:, :, None], dtype=jnp.float32)

    def final_energy(actions: Array) -> Array:
        state, _ = env.rollout(actions)
        return env.info(state)["energy"]

    energies = np.asarray(jax.jit(jax.vmap(final_energy))(batch))
    best = int(np.argmax(energies))
    out = evaluate_pulse(env, batch[best])
    out["start_step"] = int(starts[best // areas])
    out["area"] = float(jnp.sum(batch[best]) * dt)  # realised area (truncated at the horizon)
    out["candidates"] = len(pulses)
    return out


def random_baseline(env: object, n: int = 256, seed: int = 0) -> dict[str, float]:
    """Mean final energy and ergotropy of uniform random pulses."""
    key = jax.random.PRNGKey(seed)
    b = env.action_bound
    batch = jax.random.uniform(key, (n, env.horizon, env.action_dim), minval=-b, maxval=b)

    def final(actions: Array) -> tuple[Array, Array]:
        state, _ = env.rollout(actions)
        info = env.info(state)
        return info["energy"], info["ergotropy"]

    e, w = jax.jit(jax.vmap(final))(batch)
    scale = n_max_energy(env)
    return {"energy": float(e.mean()) / scale, "ergotropy": float(w.mean()) / scale}


def _stats(xs: Sequence[float]) -> tuple[float, float]:
    return statistics.fmean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def run_variant(
    name: str,
    task: str,
    kwargs: dict[str, object],
    *,
    seeds: Sequence[int],
    grape_steps: int,
    ppo_updates: int,
    ppo_envs: int,
) -> dict[str, object]:
    """All methods on one task variant."""
    env = qenvs.make(task, **kwargs)
    out: dict[str, object] = {
        "variant": name,
        "task": task,
        "kwargs": kwargs,
        "horizon": int(env.horizon),
        "dt": float(env.dt),
        "action_bound": float(env.action_bound),
        "max_energy": n_max_energy(env),
    }
    scale = n_max_energy(env)
    out["optimum"] = known_optimum(env)
    t0 = time.perf_counter()
    out["single_pulse"] = single_pulse_baseline(env)
    out["single_pulse"]["wall_time_s"] = time.perf_counter() - t0
    out["random"] = random_baseline(env)
    for method in ("grape", "ppo"):
        runs = []
        for seed in seeds:
            t0 = time.perf_counter()
            if method == "grape":
                r = run_grape(task, steps=grape_steps, lr=0.1, seed=seed, **kwargs)
                pulse = r["best_pulse"]
            else:
                r = run_ppo(task, num_envs=ppo_envs, updates=ppo_updates, seed=seed, **kwargs)
                pulse = r["policy_pulse"]
            ev = evaluate_pulse(env, jnp.asarray(pulse))
            # The episode starts empty and the reward is the energy gain, so the method's own
            # return must equal the replayed final energy. A mismatch means the method ran on a
            # different environment (e.g. a task kwarg captured by a method argument).
            own = float(r["best_return"] if method == "grape" else r["final_return"]) / scale
            if abs(own - ev["energy"]) > 1e-3:
                msg = (
                    f"{method} on {name}: in-run return {own:.6f} != replayed energy "
                    f"{ev['energy']:.6f}; the method did not run on the benchmark environment"
                )
                raise RuntimeError(msg)
            ev["own_return"] = own
            ev["wall_time_s"] = time.perf_counter() - t0
            ev["seed"] = seed
            runs.append(ev)
        out[method] = {
            "runs": runs,
            **{
                f"{k}_mean_std": _stats([r[k] for r in runs])
                for k in ("energy", "ergotropy", "control_energy", "wall_time_s")
            },
        }
    return out


def markdown(results: list[dict[str, object]]) -> str:
    """Render the benchmark as one markdown table."""
    head = (
        "| task variant | optimum | single pulse | GRAPE | PPO | random |\n"
        "|---|---|---|---|---|---|"
    )
    rows = [head]

    def ms(r: dict[str, object], method: str, key: str = "energy") -> str:
        mean, std = r[method][f"{key}_mean_std"]
        return f"{mean:.4f} +- {std:.4f}"

    for r in results:
        opt = f"{r['optimum']['energy']:.4f}" if r["optimum"] else "unknown"
        sp = r["single_pulse"]
        rows.append(
            f"| {r['variant']} | {opt} | {sp['energy']:.4f} | {ms(r, 'grape')} "
            f"| {ms(r, 'ppo')} | {r['random']['energy']:.4f} |"
        )
    rows.append("")
    rows.append(
        "| task variant | ergotropy: single pulse / GRAPE / PPO | control energy sum(g^2 dt): "
        "single pulse / GRAPE / PPO | wall time per seed (s): GRAPE / PPO |"
    )
    rows.append("|---|---|---|---|")
    for r in results:
        sp = r["single_pulse"]
        g, p = r["grape"], r["ppo"]
        rows.append(
            f"| {r['variant']} | {sp['ergotropy']:.4f} / {g['ergotropy_mean_std'][0]:.4f} / "
            f"{p['ergotropy_mean_std'][0]:.4f} | {sp['control_energy']:.3f} / "
            f"{g['control_energy_mean_std'][0]:.3f} / {p['control_energy_mean_std'][0]:.3f} | "
            f"{g['wall_time_s_mean_std'][0]:.1f} / {p['wall_time_s_mean_std'][0]:.1f} |"
        )
    return "\n".join(rows) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--quick", action="store_true", help="smoke settings (1 seed, few steps)")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--json", action="store_true")
    p.add_argument("--out", type=pathlib.Path)
    args = p.parse_args(argv)
    if args.quick:
        cfg = {"seeds": [0], "grape_steps": 20, "ppo_updates": 2, "ppo_envs": 16}
        variants = VARIANTS[:1] + VARIANTS[3:4]
    else:
        cfg = {
            "seeds": list(range(args.seeds)),
            "grape_steps": 1500,
            "ppo_updates": 50,
            "ppo_envs": 256,
        }
        variants = VARIANTS
    t0 = time.perf_counter()
    results = [run_variant(n, t, kw, **cfg) for n, t, kw in variants]
    report = {
        "settings": cfg,
        "results": results,
        "environment": environment_info(),
        "total_wall_time_s": time.perf_counter() - t0,
    }
    if args.out:
        args.out.write_text(json.dumps(report, indent=2))
    if args.json:
        json.dump(report, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(markdown(results))
        sys.stdout.write(f"\ntotal wall time {report['total_wall_time_s']:.0f} s\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
