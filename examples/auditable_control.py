"""Small reproducible control audit, not a scientific benchmark.

Run with JAX_ENABLE_X64=1 and the dev extra (Optax). Physical controls are tanh(raw)
in dimensionless units. Candidate replay is open-loop; observations are demonstrated
separately. Endpoint peak is not a stopping policy or a continuous-time maximum.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

import dynamiqs as dq
import jax
import jax.numpy as jnp
import numpy as np

from qenvs.envs import BatteryCharging
from qenvs.models import HamiltonianModel
from qenvs.observables import energy, ergotropy
from qenvs.pulses import Pulse
from qenvs.trajectories import rollout_metrics
from qenvs.wrappers import ObservationModel


def make_model(detuning):
    x = np.array([[0.0, 1], [1, 0]])
    z = np.diag([1.0, -1])
    h_b = np.diag([0.0, 1])
    # Local battery emission: L = sqrt(0.01) I tensor |0><1|, not dephasing.
    jump = 0.1 * np.kron(np.eye(2), np.array([[0.0, 1], [0, 0]]))
    return HamiltonianModel(
        drift=detuning * np.kron(np.eye(2), z) + 0.1 * np.kron(x, x),
        controls=[np.kron(np.eye(2), x)],
        jump_ops=[jump],
        dims=(2, 2),
        initial_state=np.array([1.0, 0, 0, 0]),
        battery_subsystems=(1,),
        subsystem_hamiltonian=h_b,
        observables={"battery_energy": np.kron(np.eye(2), h_b)},
    )


def _energy(model, state):
    return energy(*model.battery_reduced(state))


def _work(model, state):
    return ergotropy(*model.battery_reduced(state))


METRICS = {"energy": _energy, "work": _work}


def select_candidates(scores, sources):
    scores = np.asarray(scores, dtype=float)
    sources = np.asarray(sources)
    if (
        scores.ndim != 2
        or min(scores.shape) < 1
        or sources.shape != (scores.shape[1],)
        or sources.dtype.kind not in "iu"
        or not np.isfinite(scores).all()
        or set(sources.tolist()) != set(range(scores.shape[0]))
    ):
        raise ValueError("Expected finite model-by-candidate scores and a source for every model")
    winners = []
    for i in range(scores.shape[0]):
        ids = np.flatnonzero(sources == i)
        winners.append(int(ids[np.argmax(scores[i, ids])]))
    selected = scores[:, winners]
    regrets, reasons = [], []
    for i, row in enumerate(selected):
        if row[i] <= 1e-12:
            regrets.append([None] * len(row))
            reasons.append("reference work <= 1e-12 in dimensionless energy units")
        else:
            regrets.append((1 - row / row[i]).tolist())
            reasons.append(None)
    return {
        "scores": scores.tolist(),
        "sources": sources.tolist(),
        "selected_candidates": winners,
        "selected_matrix": selected.tolist(),
        "regret": regrets,
        "undefined_reason": reasons,
    }


def evaluate_candidates(models, candidates, sources, *, dt, method):
    candidates = np.asarray(candidates)
    if candidates.ndim != 3 or candidates.dtype.kind != "f" or not np.isfinite(candidates).all():
        raise ValueError("Candidates must be finite real floating (candidates, steps, controls)")
    scores, terminal, peaks = [], [], []
    for model in models:
        row, final_row, peak_row = [], [], []
        for actions in candidates:
            times, values = rollout_metrics(
                model, jnp.asarray(actions), dt=dt, metrics=METRICS, method=method
            )
            values = {name: np.asarray(v) for name, v in values.items()}
            if not all(np.isfinite(v).all() for v in values.values()):
                raise ValueError("Nonfinite trajectory diagnostics")
            work = values["work"]
            peak = int(work.argmax())
            row.append(float(work[peak]))
            final_row.append(float(work[-1]))
            peak_row.append(float(times[peak]))
        scores.append(row)
        terminal.append(final_row)
        peaks.append(peak_row)
    result = select_candidates(scores, sources)
    result.update(terminal=terminal, peak_times=peaks)
    return result


def optimize(model, initial_raw, *, iterations, warmup_steps, dt, method, lr=0.05):
    import optax

    if iterations < 1 or not 0 <= warmup_steps <= iterations:
        raise ValueError("Invalid iteration or fixed warmup budget")
    optimizer = optax.adam(lr)
    raw = jnp.asarray(initial_raw)
    state = optimizer.init(raw)
    best, saved, history = -np.inf, None, []

    def objective(parameters, warmup):
        _, values = rollout_metrics(
            model, jnp.tanh(parameters), dt=dt, metrics=METRICS, method=method
        )
        score = values["work"].max()
        loss = -jnp.where(warmup, values["energy"].max(), score)
        return loss, score

    gradient = jax.jit(jax.value_and_grad(objective, has_aux=True))
    for i in range(iterations):
        (_, score), grad = gradient(raw, jnp.asarray(i < warmup_steps))
        updates, state = optimizer.update(grad, state, raw)
        updated = optax.apply_updates(raw, updates)
        if not (np.isfinite(score) and np.isfinite(grad).all() and np.isfinite(updated).all()):
            raise ValueError("Nonfinite optimizer values")
        if float(score) > best:
            best, saved = float(score), np.asarray(jnp.tanh(raw)).tolist()
        history.append(
            {
                "iteration": i + 1,
                "warmup": i < warmup_steps,
                "score": float(score),
                "best_score": best,
                "gradient_norm": float(jnp.linalg.norm(grad)),
                "update_norm": float(jnp.linalg.norm(updated - raw)),
            }
        )
        raw = updated
    return {"actions": saved, "score": best, "history": history}


def source_provenance(module_file):
    path = Path(module_file).resolve()
    digest = hashlib.sha256()
    for source in sorted(path.parent.rglob("*.py")):
        digest.update(str(source.relative_to(path.parent)).encode())
        digest.update(source.read_bytes())
    result = {
        "module_file": str(path),
        "source_sha256": digest.hexdigest(),
        "git_sha": None,
        "git_dirty": None,
    }
    # Require the source layout and repository root to match the imported package.
    if path.parent.name == "qenvs" and path.parent.parent.name == "src":
        root = path.parents[2]

        def git(*args):
            return subprocess.check_output(
                ["git", "-C", str(root), *args], stderr=subprocess.DEVNULL, text=True
            ).strip()

        try:
            if Path(git("rev-parse", "--show-toplevel")).resolve() == root:
                result.update(
                    git_sha=git("rev-parse", "HEAD"), git_dirty=bool(git("status", "--porcelain"))
                )
        except (OSError, subprocess.CalledProcessError):
            pass
    return result


def save_json(path, data):
    serialized = json.dumps(data, indent=2, allow_nan=False)
    with Path(path).open("x") as stream:
        stream.write(serialized + "\n")


def observation_demo(policy):
    env = ObservationModel(
        BatteryCharging(make_model(0.0), horizon=2, dt=0.2),
        observation="expectations",
        observables=["battery_energy"],
    )
    state, timestep = env.reset(jax.random.PRNGKey(0))
    for _ in range(env.horizon):
        # The caller owns diagnostics; the policy is passed only the observation.
        action = policy(timestep.observation)
        state, timestep = env.step(state, action)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    if not jax.config.jax_enable_x64:
        raise ValueError("Set JAX_ENABLE_X64=1 before running this reference example")
    args.out.mkdir(parents=True, exist_ok=False)
    import qenvs

    seed, dt, iterations, warmup = 0, 0.4, 3, 1
    training = dq.method.Tsit5(rtol=1e-8, atol=1e-10)
    precise = dq.method.Tsit5(rtol=1e-10, atol=1e-12)
    initial = np.random.default_rng(seed).normal(0.8, 0.1, size=(2, 3, 1))
    models = [make_model(0.0), make_model(0.2)]
    records, candidates, sources = [], [], []
    for i, model in enumerate(models):
        for raw in initial:
            record = optimize(
                model,
                jnp.asarray(raw),
                iterations=iterations,
                warmup_steps=warmup,
                dt=dt,
                method=training,
            )
            records.append(record)
            candidates.append(record["actions"])
            sources.append(i)
    candidates = np.asarray(candidates)
    result = evaluate_candidates(models, candidates, sources, dt=dt, method=precise)
    manifest = {
        "seed": seed,
        "dt": dt,
        "iterations": iterations,
        "warmup_steps": warmup,
        "lr": 0.05,
        "units": "dimensionless",
        "reduction": "endpoint_peak",
        "model_detunings": [0.0, 0.2],
        "coupling": 0.1,
        "emission_rate": 0.01,
        "training_method": {"name": "Tsit5", "rtol": 1e-8, "atol": 1e-10},
        "evaluation_method": {"name": "Tsit5", "rtol": 1e-10, "atol": 1e-12},
        "dtype": str(candidates.dtype),
        "backend": jax.default_backend(),
        "initial_sha256": hashlib.sha256(initial.tobytes()).hexdigest(),
        "candidates_sha256": hashlib.sha256(candidates.tobytes()).hexdigest(),
        "example_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source": source_provenance(Path(qenvs.__file__)),
        "versions": {
            p: importlib.metadata.version(p)
            for p in ("qenvs", "jax", "jaxlib", "dynamiqs", "numpy", "optax")
        },
    }
    for i, actions in enumerate(candidates):
        pulse = Pulse(
            actions,
            dt_ns=dt,
            units="dimensionless",
            channels=["drive"],
            metadata={"candidate_id": i, "source_model_id": sources[i]},
        )
        pulse.to_json(args.out / f"candidate-{i}.json")
    save_json(args.out / "initial.json", initial.tolist())
    save_json(args.out / "history.json", records)
    save_json(args.out / "manifest.json", manifest)
    save_json(args.out / "evaluation.json", result)
    observation_demo(lambda _obs: jnp.zeros(1))
    print(
        json.dumps(
            {
                "out": str(args.out),
                "candidates": len(candidates),
                "selected": result["selected_candidates"],
            }
        )
    )


if __name__ == "__main__":
    main()
