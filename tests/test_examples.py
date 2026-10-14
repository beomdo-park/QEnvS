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
import benchmark_battery  # noqa: E402
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


def test_grape_open_system_differentiates_the_return_and_improves():
    # since 0.1.0 the pure-target fidelity has a finite gradient on density matrices,
    # so the open task is optimised through its own return (no surrogate objective).
    out = grape_jax.run_grape(TASK, steps=30, horizon=10, dt=0.2, gamma1=0.1)
    assert out["objective"] == "return" and out["open_system"]
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


def test_battery_known_optimum_matches_closed_forms():
    import qenvs

    # resonant closed TC, N = n = 2: E(theta) peaks at (8/9) N omega0 for theta = pi/sqrt(6)
    env = qenvs.make("battery/Dicke-v0", n_cells=2)
    opt = benchmark_battery.known_optimum(env)
    assert opt["energy"] == pytest.approx(8 / 9, abs=1e-6)
    assert opt["theta"] == pytest.approx(math.pi / math.sqrt(6), abs=1e-3)
    # resonant closed qubit battery: a pi area stores omega0 (reachable since g_max T > pi)
    env_q = qenvs.make("battery/TwoLevel-v0")
    assert benchmark_battery.known_optimum(env_q)["energy"] == pytest.approx(1.0, abs=1e-6)
    # no provable optimum for open or detuned variants
    assert benchmark_battery.known_optimum(qenvs.make("battery/Dicke-v0", kappa=0.1)) is None
    assert benchmark_battery.known_optimum(qenvs.make("battery/Dicke-v0", omega_c=1.2)) is None


def test_battery_single_pulse_reaches_the_closed_optimum():
    import qenvs

    env = qenvs.make("battery/Dicke-v0", n_cells=2)
    sp = benchmark_battery.single_pulse_baseline(env, areas=161)
    assert sp["energy"] == pytest.approx(8 / 9, abs=1e-4)
    assert 0.0 <= sp["ergotropy"] <= sp["energy"] + 1e-6
    assert sp["control_energy"] > 0


def test_battery_benchmark_quick_runs(capsys):
    assert benchmark_battery.main(["--quick", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    closed = out["results"][0]
    assert closed["optimum"]["energy"] == pytest.approx(8 / 9, abs=1e-6)
    for method in ("grape", "ppo"):
        e = closed[method]["energy_mean_std"][0]
        assert math.isfinite(e) and e <= closed["optimum"]["energy"] + 1e-5


def test_example_arguments_never_capture_task_kwargs():
    # A method argument with the same name as a task kwarg silently swallows it (PPO's
    # discount used to be called ``gamma``, the Dicke emission rate).
    import inspect

    import qenvs

    task_kwargs = {k for spec in qenvs.registry.values() for k in spec.kwargs}
    assert "gamma" in task_kwargs  # the case that motivated this test
    for fn in (grape_jax.run_grape, ppo_jax.run_ppo):
        named = {
            n
            for n, prm in inspect.signature(fn).parameters.items()
            if prm.kind is not inspect.Parameter.VAR_KEYWORD
        }
        assert not named & task_kwargs, (fn.__name__, named & task_kwargs)


def test_ppo_forwards_the_dicke_emission_rate():
    out = ppo_jax.run_ppo(
        "battery/Dicke-v0", num_envs=4, updates=1, n_cells=2, gamma=0.05, horizon=3
    )
    assert out["task_kwargs"]["gamma"] == 0.05
    assert out["hyperparameters"]["discount"] == 0.99
