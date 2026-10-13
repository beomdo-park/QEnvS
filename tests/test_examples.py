"""Control baselines in examples/ (GRAPE-like Adam, vectorised PPO, benchmark table)."""

import json
import math
import pathlib
import sys

import jax
import jax.numpy as jnp
import pytest

EXAMPLES = pathlib.Path(__file__).resolve().parents[1] / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

import benchmark  # noqa: E402
import grape_jax  # noqa: E402
import ppo_jax  # noqa: E402

TASK = "qubit/StateTransfer-v0"


# ------------------------------------------------------------------ GRAPE
def test_grape_reaches_pi_pulse():
    out = grape_jax.run_grape(TASK, steps=60, horizon=10, dt=0.2)
    assert out["final_return"] > 0.95
    assert out["objective"] == "return" and not out["open_system"]
    assert out["final_return"] > out["initial_return"]
    assert out["best_return"] >= out["final_return"]
    # history: step 0, every 10 steps, and the final step 60
    assert [h["step"] for h in out["history"]] == [0, 10, 20, 30, 40, 50, 60]
    assert out["history"][-1]["return"] == out["final_return"]
    # any optimum has |integral(u dt)| = pi; F > 0.95 means within ~0.45 of it
    (integral,) = out["best_actions"]["integral"]
    assert abs(abs(integral) - math.pi) < 0.5
    assert out["best_actions"]["max_abs"][0] <= out["action_bound"] + 1e-6
    assert out["wall_time_s"] > 0
    json.dumps(out)  # JSON-serialisable


def test_grape_open_system_uses_trace_overlap_and_improves():
    # dq.fidelity against a pure target has a NaN gradient for density matrices; the
    # example switches to the (identical) trace overlap and must still learn.
    out = grape_jax.run_grape(TASK, steps=30, horizon=10, dt=0.2, gamma1=0.1)
    assert out["objective"] == "trace_overlap" and out["open_system"]
    assert math.isfinite(out["final_return"])
    assert out["final_return"] > max(out["initial_return"], 0.5)
    pi_pulse = benchmark.pi_pulse_return(TASK, horizon=10, dt=0.2, gamma1=0.1)
    assert out["final_return"] > pi_pulse - 0.05  # close to the pi-pulse reference


def test_grape_pulse_is_bounded_and_set_parsing():
    theta = jnp.array([[-50.0], [0.0], [50.0]])
    u = grape_jax.pulse_from_theta(theta, 2.0)
    assert jnp.all(jnp.abs(u) <= 2.0)
    assert grape_jax.pulse_summary(u, 0.5)["integral"] == pytest.approx([0.0], abs=1e-6)
    assert grape_jax.parse_set(["gamma1=0.1", "horizon=5", "name=abc"]) == {
        "gamma1": 0.1,
        "horizon": 5,
        "name": "abc",
    }
    with pytest.raises(Exception, match="key=value"):
        grape_jax.parse_set(["oops"])


def test_grape_cli_json(capsys):
    assert grape_jax.main(["--steps", "3", "--set", "horizon=3", "--set", "dt=0.2", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["steps"] == 3 and out["horizon"] == 3
    assert math.isfinite(out["final_return"])


# -------------------------------------------------------------------- PPO
def test_ppo_smoke_runs_and_reports_curve():
    out = ppo_jax.run_ppo(TASK, num_envs=16, updates=2, horizon=5, dt=0.2)
    assert len(out["learning_curve"]) == 2
    assert all(math.isfinite(x) for x in out["learning_curve"])
    assert math.isfinite(out["final_return"]) and math.isfinite(out["random_return"])
    assert -1.0 <= out["final_return"] <= 1.0  # return = F_final - F_initial
    assert out["transitions"] == 2 * 16 * 5
    assert out["eval_envs"] == 16
    json.dumps(out)


def test_ppo_minibatch_divisibility_is_checked():
    with pytest.raises(ValueError, match="divisible"):
        ppo_jax.run_ppo(TASK, num_envs=3, updates=1, minibatches=4, horizon=5, dt=0.2)


def test_actor_critic_shapes_and_gaussian():
    model = ppo_jax.ActorCritic(obs_dim=5, action_dim=2, hidden=8, key=jax.random.PRNGKey(0))
    obs = jnp.ones((7, 5))
    mean, value = jax.vmap(model)(obs)
    assert mean.shape == (7, 2) and value.shape == (7,)
    assert jnp.all(jnp.abs(mean) < 0.1)  # small initial mean action
    log_std = jnp.array([0.3, -0.2])
    action = jnp.array([[0.5, -1.0]])
    lp = ppo_jax.gaussian_log_prob(jnp.zeros((1, 2)), log_std, action)
    ref = jax.scipy.stats.norm.logpdf(action, 0.0, jnp.exp(log_std)).sum(-1)
    assert lp == pytest.approx(ref, abs=1e-6)
    ent = ppo_jax.gaussian_entropy(log_std)
    assert ent == pytest.approx(float(jnp.sum(log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e))))


# -------------------------------------------------------------- benchmark
def test_benchmark_table_format():
    results = {
        "task": TASK,
        "rows": [
            {
                "variant": "closed",
                "method": "grape",
                "seeds": [0, 1],
                "mean": 0.99,
                "std": 0.01,
                "known_optimum": 1.0,
                "pi_pulse_return": 1.0,
                "wall_time_s": 4.0,
            },
            {
                "variant": "open (gamma1=0.1)",
                "method": "ppo",
                "seeds": [0],
                "mean": 0.8,
                "std": 0.0,
                "known_optimum": None,
                "pi_pulse_return": 0.85,
                "wall_time_s": 2.0,
            },
        ],
    }
    table = benchmark.format_table(results)
    lines = table.splitlines()
    assert lines[0].startswith("| task | method | seeds |")
    assert "| grape | 2 | 0.9900 +- 0.0100 | 1.000 | 4.0 s (2.0 s/seed) |" in lines[2]
    assert "n/a (pi-pulse 0.850)" in lines[3]
    assert benchmark.pi_pulse_return(TASK, horizon=10, dt=0.2) == pytest.approx(1.0, abs=1e-6)
