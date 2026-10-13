# Benchmarks

Two reference controllers ship in `examples/` and are exercised by `tests/test_examples.py`:

| Script | Method | What it does |
|---|---|---|
| `examples/grape_jax.py` | GRAPE-like open-loop optimisation | `theta` of shape `(horizon, action_dim)`, `actions = action_bound * tanh(theta)`, objective `-sum(rewards)` of `env.rollout(actions)`, `optax.adam`. Gradients flow through the dynamiqs ODE solve. |
| `examples/ppo_jax.py` | Vectorised PPO (closed loop) | `num_envs` environments stepped with `jax.vmap(env.step)`, `lax.scan` rollouts, GAE, clipped objective, entropy bonus; Gaussian MLP policy and value MLP in `equinox`, optimised with `optax`. The whole run is one jitted `lax.scan` over updates. |
| `examples/benchmark.py` | Driver | Runs both on `qubit/StateTransfer-v0` (closed, and open with `gamma1=0.1`) over 3 seeds and prints the table below. |

The numbers below are **smoke-scale and not tuned**: default hyper-parameters, three seeds,
about one minute of CPU in total. They are a regression reference for the environments and
the examples, not a leaderboard.

## Results

Task `qubit/StateTransfer-v0` with `horizon=10`, `dt=0.2` (episode duration 2.0, `action_bound=2.0`).
The return is $F(\rho_T, \sigma) - F(\rho_0, \sigma)$ with $\sigma = |1\rangle\langle 1|$, so it equals the final
fidelity. Settings: GRAPE `steps=200`, `lr=0.1`, random initial `theta ~ 0.5 N(0, 1)`;
PPO `num_envs=256`, `updates=50` (128,000 transitions), `lr=1e-3`, 4 epochs x 4 minibatches,
`gamma=0.99`, `gae_lambda=0.95`, `clip_eps=0.2`, `ent_coef=0.01`, MLPs 2 x 64 tanh.
Seeds 0, 1, 2. Output of `python examples/benchmark.py` (JAX x64 enabled, as in the `qenvs` CLI):

| task | method | seeds | final return (mean +- std) | known optimum | wall-clock |
|---|---|---|---|---|---|
| qubit/StateTransfer-v0 closed | grape | 3 | 1.0000 +- 0.0000 | 1.000 | 10.7 s (3.6 s/seed) |
| qubit/StateTransfer-v0 closed | ppo | 3 | 0.9997 +- 0.0002 | 1.000 | 16.1 s (5.4 s/seed) |
| qubit/StateTransfer-v0 open (gamma1=0.1) | grape | 3 | 0.9411 +- 0.0002 | n/a (pi-pulse 0.928) | 10.6 s (3.5 s/seed) |
| qubit/StateTransfer-v0 open (gamma1=0.1) | ppo | 3 | 0.9117 +- 0.0016 | n/a (pi-pulse 0.928) | 18.5 s (6.2 s/seed) |

Total wall-clock 60.5 s. "final return" is the return of the optimised pulse (GRAPE) or of the
deterministic policy, mean action, averaged over 256 fresh episodes (PPO); std is over seeds.
Wall-clock includes JIT compilation (1.3 to 3.8 s per run).

* **Known optimum.** Closed system: any pulse with $\int u\,dt = \pi$ gives $F = 1$
  (`qenvs.validation.qubit.check_env_pi_pulse_return`). For the open variant no closed-form
  optimum is known; the constant pi-pulse (`qenvs run ... --policy pi-pulse --set gamma1=0.1`)
  gives 0.928 and is listed as a reference, not as a bound. GRAPE exceeds it (0.941) by
  finishing the transfer slightly faster than the uniform pulse (pulse integral 3.15 > pi).
* **Random baseline.** Uniform random actions in `[-2, 2]` over the same episodes give
  0.10 to 0.13 (closed) and 0.10 to 0.12 (open), reported as `random_return` by `run_ppo`.

### Per-seed detail

GRAPE (return at steps 0 / 50 / 100 / 200, best pulse integral):

| variant | seed | step 0 | step 50 | step 100 | step 200 | integral of u dt |
|---|---|---|---|---|---|---|
| closed | 0 | 0.011 | 0.999 | 1.000 | 1.000000 | +3.1416 |
| closed | 1 | 0.044 | 0.998 | 1.000 | 1.000000 | -3.1416 |
| closed | 2 | 0.052 | 0.998 | 1.000 | 1.000000 | -3.1416 |
| open | 0 | 0.010 | 0.934 | 0.939 | 0.941348 | +3.1475 |
| open | 1 | 0.041 | 0.936 | 0.939 | 0.940894 | -3.1572 |
| open | 2 | 0.046 | 0.932 | 0.938 | 0.941117 | -3.1514 |

Both signs of the pulse are optima (the sign of $u$ only flips the rotation direction). With
`steps=60` the closed run of seed 0 ends at 0.9980 (> 0.95, the test threshold; best iterate
0.99996), and the 200-step run first exceeds 0.999 at step 70 (0.99973), then 0.99999 at
step 100.

PPO (mean episodic return of the stochastic policy during training, every 10 updates, then the
deterministic evaluation):

| variant | seed | random | update 1 | 11 | 21 | 31 | 41 | 50 | deterministic |
|---|---|---|---|---|---|---|---|---|---|
| closed | 0 | 0.114 | 0.091 | 0.965 | 0.987 | 0.990 | 0.994 | 0.995 | 0.99999 |
| closed | 1 | 0.131 | 0.072 | 0.961 | 0.983 | 0.989 | 0.994 | 0.996 | 0.99959 |
| closed | 2 | 0.105 | 0.088 | 0.969 | 0.984 | 0.991 | 0.993 | 0.996 | 0.99960 |
| open | 0 | 0.104 | 0.083 | 0.908 | 0.903 | 0.906 | 0.913 | 0.916 | 0.91221 |
| open | 1 | 0.119 | 0.066 | 0.909 | 0.902 | 0.909 | 0.914 | 0.918 | 0.91342 |
| open | 2 | 0.095 | 0.080 | 0.908 | 0.903 | 0.906 | 0.913 | 0.916 | 0.90957 |

The learned `log_std` ends near -0.5 (std 0.6), so the stochastic curve stays below the
deterministic return. Untuned PPO on the open variant stays 0.017 below the pi-pulse reference
and 0.03 below GRAPE after 50 updates; it is still improving at the end of the run.

### float32

Library users may run without `jax_enable_x64`. The same settings in float32 (seed 0) give
GRAPE 1.0000 (closed) and 0.9403 (open), PPO 0.9994 (closed) and 0.9060 (open), with random
baselines 0.123 and 0.112.

## Differentiability note (open systems)

`StateTransfer` rewards the gain in `dynamiqs.fidelity`. For density matrices that is the
Uhlmann fidelity, computed through an eigen-decomposition of $\sqrt{\rho}\,\sigma\sqrt{\rho}$.
With a pure target this matrix has rank one, so a zero eigenvalue is always present and the
gradient of $\sqrt{\lambda}$ at $\lambda = 0$ is NaN. `jax.grad` through `env.rollout` of the
open task therefore returns NaN (the forward pass is fine; PPO is unaffected because it never
differentiates through the environment). `run_grape` detects an open environment with a pure
`target` and differentiates the trace overlap $\mathrm{tr}(\rho_T\sigma) - \mathrm{tr}(\rho_0\sigma)$
instead, which is identical to the return for a pure target
($F(\rho, |\psi\rangle\langle\psi|) = \langle\psi|\rho|\psi\rangle$; agreement to 1e-16 was checked
on a random pulse). The reported returns are always re-evaluated from `env.rollout` rewards,
and the output dict records which objective was differentiated (`"objective": "return"` or
`"trace_overlap"`). Tasks whose reward is already differentiable (closed systems, energy
rewards) use the return directly.

## Reproduce

```bash
python examples/benchmark.py                        # markdown table (about 1 min on CPU)
python examples/benchmark.py --json --out bench.json
python examples/grape_jax.py --steps 200 --lr 0.1 --set horizon=10 --set dt=0.2 --json
python examples/grape_jax.py --set gamma1=0.1 --set horizon=10 --set dt=0.2
python examples/ppo_jax.py --num-envs 256 --updates 50 --set horizon=10 --set dt=0.2
JAX_PLATFORMS=cpu pytest tests/test_examples.py     # 8 tests, about 30 s on CPU
```

`--set key=value` values are JSON-parsed like in the `qenvs` CLI. `optax` is needed
(`uv sync --extra dev`); everything else is a core dependency.

## Environment

| | |
|---|---|
| python | 3.12.3 |
| jax | 0.7.2 (backend `cpu`, x64 enabled by the example CLIs) |
| dynamiqs | 0.3.6 |
| equinox / optax | 0.13.8 / 0.2.8 |
| qenvs | 0.1.dev6 |
| CPU | AMD Ryzen Threadripper PRO 7975WX (32 cores / 64 threads), Linux 6.14 |

No GPU was used. Timings are for a shared machine and vary by a few tens of percent between runs.

## Known limitations

* Smoke scale only: one task, three seeds, default hyper-parameters, no confidence intervals
  beyond the seed std.
* GRAPE uses a fixed Adam learning rate with a tanh parametrisation; it is not a line-search
  GRAPE and makes no claim of optimality for the open variant.
* PPO uses a state-independent Gaussian `log_std` and clips actions through the environment
  (no tanh squashing); the open-variant result is untuned.
* The trace-overlap objective applies only to environments with a pure `target` attribute;
  open tasks with mixed targets would need a differentiable fidelity in the library.
