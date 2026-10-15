"""Tutorial: calibrate an X gate on a three-level transmon.

The workflow of a pulse engineer, end to end, in hardware units (ns, rad/ns, GHz, us):

1. a plain Gaussian pi pulse and its error budget (infidelity, leakage);
2. DRAG: the first-order coefficient and a sweep around it;
3. GRAPE (gradient through the simulator) from a random start;
4. the same pulses with T1 and T2, against the idle-decoherence reference;
5. exporting the pulse as Rabi frequencies in MHz on the time grid;
6. handing the same task to an RL library through Gymnasium.

Needs ``pip install "qenvs[gym]" optax`` (GRAPE uses optax, step 6 uses gymnasium).

Run::

    python examples/tutorial_transmon_x_gate.py            # about a minute on CPU
    python examples/tutorial_transmon_x_gate.py --out pulse.npz

The numbers quoted on the tutorial page (docs/tutorials/transmon-x-gate.md) come from
this script; ``tests/test_tutorials.py`` runs it in a reduced form.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np

import qenvs
from qenvs.models.transmon import drag_beta, gaussian_amplitude, gaussian_drag

try:
    from grape_jax import run_grape
except ImportError:  # imported from outside examples/ (e.g. the tests)
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from grape_jax import run_grape

TASK = "transmon/XGate-v0"


def final_metrics(env: object, actions: jax.Array) -> dict[str, float]:
    """Gate infidelity, leakage and the populations of |0>, |1>, |2> after the pulse."""
    state, _ = env.rollout(jnp.asarray(actions))
    info = env.info(state)
    return {
        "infidelity": float(1.0 - info["fidelity"]),
        "leakage": float(info["leakage"]),
        **{k: float(v) for k, v in info.items() if k in ("p0", "p1", "p2")},
    }


def gaussian_pulse(env: object, beta: float) -> jax.Array:
    """Lifted Gaussian pi pulse with sigma = T/4 and DRAG coefficient ``beta`` (ns)."""
    horizon, dt = int(env.horizon), float(env.dt)
    sigma = horizon * dt / 4.0
    amp = gaussian_amplitude(jnp.pi, horizon, dt, sigma)
    return jnp.clip(
        gaussian_drag(horizon, dt, amp, sigma, beta), -env.action_bound, env.action_bound
    )


def run(*, grape_steps: int = 300, betas: int = 9) -> dict[str, object]:
    """Execute the tutorial and return every number it reports.

    Runs in float64: the errors of interest are 1e-6 to 1e-4, and a float32 propagator
    accumulates about 1e-5 of rounding error over a 16-step superoperator.
    """
    jax.config.update("jax_enable_x64", True)
    out: dict[str, object] = {}

    # 1. The task: 3-level transmon, alpha/2pi = -0.34 GHz, 16 steps of 2 ns, |Omega|/2pi <= 50 MHz.
    env = qenvs.make(TASK)
    alpha_ghz = float(env.model.anharmonicity_ghz)
    out["task"] = {
        "levels": int(env.model.levels),
        "anharmonicity_ghz": alpha_ghz,
        "gate_ns": float(env.horizon * env.dt),
        "dt_ns": float(env.dt),
        "max_rabi_mhz": float(env.action_bound / (2 * np.pi) * 1e3),
    }

    # 2. Plain Gaussian pi pulse versus DRAG.
    out["gaussian"] = final_metrics(env, gaussian_pulse(env, 0.0))
    beta_phase = drag_beta(alpha_ghz, "phase")
    beta_leak = drag_beta(alpha_ghz, "leakage")
    out["drag_beta_ns"] = {"phase": beta_phase, "leakage": beta_leak}
    out["drag"] = final_metrics(env, gaussian_pulse(env, beta_phase))

    #    A sweep around the first-order value.
    sweep = []
    for b in np.linspace(0.0, 2.0 * beta_phase, betas):
        m = final_metrics(env, gaussian_pulse(env, float(b)))
        sweep.append({"beta_ns": float(b), **m})
    out["drag_sweep"] = sweep
    best = min(sweep, key=lambda r: r["infidelity"])
    out["drag_sweep_best_beta_over_phase_value"] = best["beta_ns"] / beta_phase

    # 3. GRAPE from a random start (gradient of the gate fidelity through the simulator).
    g = run_grape(TASK, steps=grape_steps, lr=0.1, seed=0)
    grape_pulse = jnp.asarray(g["best_pulse"])
    first = next((h["step"] for h in g["history"] if h["return"] > 2 / 3 - 1e-3), None)
    out["grape"] = {
        **final_metrics(env, grape_pulse),
        "steps": grape_steps,
        "first_logged_step_above_F_0.999": first,
        "wall_time_s": g["wall_time_s"],
    }

    # 4. Decoherence: T1 = T2 = 100 us. The env now propagates the superoperator.
    kw = {"t1_us": 100.0, "t2_us": 100.0}
    env_open = qenvs.make(TASK, **kw)
    idle = final_metrics(env_open, jnp.zeros((env_open.horizon, env_open.action_dim)))
    # idle-decoherence reference: 1 - F of doing nothing, measured against the identity
    from qenvs.gates import gate, superop_average_gate_fidelity

    s_idle, _ = env_open.rollout(jnp.zeros((env_open.horizon, env_open.action_dim)))
    f_idle_vs_identity = superop_average_gate_fidelity(s_idle.qstate.to_jax(), gate("I"))
    out["open"] = {
        "drag": final_metrics(env_open, gaussian_pulse(env_open, beta_phase)),
        "grape_closed_pulse": final_metrics(env_open, grape_pulse),
        "idle_reference_infidelity": float(1.0 - f_idle_vs_identity),
        "idle_leakage": idle["leakage"],
    }
    g_open = run_grape(TASK, steps=grape_steps, lr=0.1, seed=0, **kw)
    out["open"]["grape_open"] = final_metrics(env_open, jnp.asarray(g_open["best_pulse"]))

    # 5. Export: Rabi frequencies in MHz on the time grid (what an AWG schedule needs).
    t_ns = (np.arange(env.horizon) + 0.5) * float(env.dt)
    mhz = np.asarray(grape_pulse) / (2 * np.pi) * 1e3
    out["export"] = {
        "t_ns": t_ns.tolist(),
        "omega_i_mhz": mhz[:, 0].tolist(),
        "omega_q_mhz": mhz[:, 1].tolist(),
        "peak_mhz": float(np.abs(mhz).max()),
    }

    # 6. The same task for an RL library.
    from qenvs.wrappers import GymnasiumEnv

    gym_env = GymnasiumEnv(env, normalize_actions=True)
    gym_env.action_space.seed(0)
    obs, _ = gym_env.reset(seed=0)
    total = 0.0
    for _ in range(env.horizon):
        obs, r, terminated, _, info = gym_env.step(gym_env.action_space.sample())
        total += r
    out["gymnasium"] = {
        "observation_dim": int(obs.shape[0]),
        "action_space": str(gym_env.action_space),
        "terminated_at_horizon": bool(terminated),
        "random_policy_final_fidelity": float(info["fidelity"]),
    }
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=pathlib.Path, help="save the GRAPE pulse as .npz (t_ns, MHz)")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)
    res = run()
    if args.out:
        e = res["export"]
        np.savez(
            args.out, t_ns=e["t_ns"], omega_i_mhz=e["omega_i_mhz"], omega_q_mhz=e["omega_q_mhz"]
        )
    if args.json:
        json.dump(res, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        for k in ("task", "drag_beta_ns", "gaussian", "drag", "grape", "open", "gymnasium"):
            sys.stdout.write(f"{k}: {json.dumps(res[k])}\n")
        ratio = res["drag_sweep_best_beta_over_phase_value"]
        sys.stdout.write(f"drag sweep best beta / phase value: {ratio:.3f}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
