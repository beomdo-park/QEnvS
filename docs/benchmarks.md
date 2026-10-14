# Benchmarks

Two reference controllers ship in `examples/` and are exercised by `tests/test_examples.py`. The first part of this page covers the qubit state-transfer task; the [battery charging benchmark](#battery-charging-benchmark) compares them with a provable optimum and a physical baseline on the battery tasks.

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
`discount=0.99`, `gae_lambda=0.95`, `clip_eps=0.2`, `ent_coef=0.01`, MLPs 2 x 64 tanh.
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

## Battery charging benchmark

`examples/benchmark_battery.py` compares five ways of charging the battery tasks, each
evaluated by the stored energy at the end of the episode divided by the maximum
$N\omega_0$ (higher is better, 1 is a fully charged battery). Defaults of every task are
used: `battery/Dicke-v0` has 20 steps of `dt = 0.2` and $|g| \le 0.5\,\omega_0$
(Tavis-Cummings, cavity initially in $|N\rangle$); `battery/TwoLevel-v0` has 20 steps of
`dt = 0.1` and $|u| \le 2\,\omega_0$. Open variants add $\kappa = 0.1$, $\gamma = 0.05$
(Dicke) or $\gamma_1 = 0.1$, $n_{\mathrm{th}} = 0.1$ (qubit).

| method | what it is |
|---|---|
| optimum | Provable optimum, closed resonant tasks only (see below). |
| single pulse | One rectangular pulse at $+g_{\max}$ with optimised start step and area (20 start steps x 201 areas, evaluated in one `vmap`). A physically motivated baseline. |
| GRAPE | `run_grape`, 1500 Adam steps at learning rate 0.1 from a random start ($\theta \sim 0.5\,\mathcal{N}(0, 1)$, amplitude $g_{\max}\tanh\theta$), 3 seeds. |
| PPO | `run_ppo`, 256 environments x 50 updates, default hyper-parameters (not tuned for these tasks), 3 seeds; deterministic policy evaluated. |
| random | Mean over 256 uniform random pulses. |

Every method's own return is compared with a replay of its pulse on the benchmark
environment, and the script stops if they differ by more than $10^{-3}$ (largest observed
difference $1.2\times10^{-7}$).

### What is proven and what is numerical

* **Proven, for every $N$:** on resonance ($\omega_c = \omega_0$, closed, Tavis-Cummings)
  $H(t) = \omega_0 \hat N_{\mathrm{exc}} + g(t)\,V$ with $V = a^\dagger J_- + a J_+$ and
  $[\hat N_{\mathrm{exc}}, V] = 0$. The propagator factorises as
  $e^{-i\omega_0 \hat N_{\mathrm{exc}} T}\, e^{-i\theta V}$ with the pulse area
  $\theta = \sum_k g_k\,\mathrm{dt}$, and the first factor is a global phase on the initial
  Fock state. Every final observable depends on $\theta$ alone, and every
  $\theta \in [-g_{\max}T, g_{\max}T]$ is reachable through the clipped controls. Because
  $V$ and the initial state are real, $E(-\theta) = E(\theta)$, so the optimum is the
  maximum of $E(\theta)$ over $[0, g_{\max}T]$. The same holds for the resonant closed
  qubit battery, $H(t) = u(t)\,\sigma_x/2$, where $E(\theta) = \omega_0\sin^2(\theta/2)$.
* **Closed form:** $N = 1$ (qubit: $\omega_0$ at $\theta = \pi$) and $N = 2$
  ($\tfrac{8}{9}N\omega_0$ at $\theta = \pi/\sqrt{6}$, also the `two_cells_closed_form`
  validation check).
* **Numerical:** for $N = 4$ and $8$ the value is the maximum of $E(\theta)$ on a
  20001-point grid (spacing $10^{-4}$) in float64; an independent float64 computation in
  review gave $0.861460682$ and $0.878688038$.
* **Not covered:** open or detuned variants have no such reduction; their optimum is
  unknown. The methods and the environment run in float32, so values agree with the
  float64 optimum to about $10^{-5}$ and differences below that are noise (GRAPE sits up to
  $4\times10^{-6}$ above the optimum in the table for this reason).

### Stored energy (fraction of $N\omega_0$)

| task variant | optimum | single pulse | GRAPE | PPO | random |
|---|---|---|---|---|---|
| Dicke N=2, closed | 0.8889 | 0.8889 | 0.8889 +- 0.0000 | 0.8875 +- 0.0014 | 0.1188 |
| Dicke N=4, closed | 0.8615 | 0.8615 | 0.8615 +- 0.0000 | 0.8569 +- 0.0027 | 0.1945 |
| Dicke N=8, closed | 0.8787 | 0.8786 | 0.8787 +- 0.0000 | 0.8720 +- 0.0080 | 0.2949 |
| Dicke N=2, open | unknown | 0.6698 | 0.6698 +- 0.0000 | 0.6684 +- 0.0006 | 0.0817 |
| Dicke N=4, open | unknown | 0.6018 | 0.6007 +- 0.0010 | 0.5949 +- 0.0000 | 0.1180 |
| Qubit battery, closed | 1.0000 | 1.0000 | 1.0000 +- 0.0000 | 1.0000 +- 0.0001 | 0.0683 |
| Qubit battery, open | unknown | 0.9283 | 0.9292 +- 0.0000 | 0.8930 +- 0.0003 | 0.0775 |

Mean +- standard deviation over seeds 0, 1, 2.

### Ergotropy, control energy and cost

| task variant | ergotropy: single pulse / GRAPE / PPO | control energy $\sum g^2\,\mathrm{dt}$: single pulse / GRAPE / PPO | wall time per seed (s): GRAPE / PPO |
|---|---|---|---|
| Dicke N=2, closed | 0.8333 / 0.8333 / 0.8252 | 0.632 / 0.640 / 0.949 | 8.1 / 5.4 |
| Dicke N=4, closed | 0.7252 / 0.7248 / 0.7399 | 0.452 / 0.488 / 0.929 | 10.3 / 9.0 |
| Dicke N=8, closed | 0.7728 / 0.7760 / 0.7996 | 0.350 / 0.399 / 0.968 | 31.2 / 70.8 |
| Dicke N=2, open | 0.3396 / 0.3395 / 0.3367 | 0.608 / 0.690 / 0.995 | 17.2 / 16.8 |
| Dicke N=4, open | 0.3133 / 0.2546 / 0.1998 | 0.450 / 0.555 / 1.000 | 95.5 / 99.4 |
| Qubit battery, closed | 1.0000 / 1.0000 / 1.0000 | 6.196 / 6.118 / 5.650 | 6.7 / 4.2 |
| Qubit battery, open | 0.8572 / 0.8610 / 0.7859 | 6.400 / 6.347 / 6.043 | 9.5 / 6.1 |

Ergotropy is evaluated on the reduced battery state; all methods optimised the stored
energy, not the ergotropy. The largest possible control energy on the Dicke tasks is
$20 \times 0.5^2 \times 0.2 = 1$.

### What the numbers show

* **Closed tasks: the reference is exact and GRAPE reaches it.** GRAPE hits the provable
  optimum on every seed for $N = 2, 4, 8$ and the qubit. PPO at default settings is on
  average 0.004 percent (qubit) to 0.76 percent ($N = 8$) below it; the worst single seed
  is 1.8 percent below ($N = 8$, seed 0). The single pulse matches the optimum up to its
  area grid ($4.5\times10^{-5}$ short for $N = 8$, because the grid spacing is 0.01 and
  $\theta^* = 0.6952$).
* **Open Dicke tasks: the single pulse is as good as anything found.** GRAPE reaches the
  single-pulse value for $N = 2$ on every seed. For $N = 4$ one seed reaches it (0.6018)
  and two stop at a local optimum (0.6002). After 300 instead of 1500 steps GRAPE was
  still below the single pulse (0.5975), so the number of optimiser steps matters here.
  For $N = 4$ the best rectangular pulse starts late (step 11) and wins over an early
  start by only 0.0024; the cause is the collective emission $\gamma$ (with $\gamma$
  alone the best start is step 11, with the cavity loss $\kappa$ alone it is step 0). For
  $N = 2$ the best start is step 0.
* **Open qubit task: shaping helps.** GRAPE (0.9292) beats the single rectangular pulse
  (0.9283), the only variant where a time-shaped pulse is measurably better than one
  rectangle.
* **PPO uses bang-bang pulses.** Its Gaussian mean is unbounded and the environment clips
  it, so the deterministic policy saturates at $\pm g_{\max}$: on the Dicke tasks 90 to 100
  percent of its steps sit at the bound (seed 0), with 2 to 10 sign flips, and on the open
  $N = 4$ task its control energy is the maximum, 1.000. The net area is then set mostly by
  the sign pattern, where flipping one saturated step changes it by
  $2 g_{\max}\,\mathrm{dt} = 0.2$, which is coarse: the seed-0 areas are
  $|\theta| = 1.32, 1.00, 0.61$ against $\theta^* = 1.28, 0.92, 0.70$ for $N = 2, 4, 8$
  (closed). On the Dicke tasks it uses 1.4 to 2.4 times the control energy of
  GRAPE and 1.5 to 2.8 times that of the single pulse; nothing in the reward penalises
  control energy, for any of the methods.
* **Energy and ergotropy are different objectives.** In the closed resonant tasks both are
  functions of $\theta$ alone. For $N = 2$ (and the qubit) the two maxima coincide; for
  $N = 4$ and $8$ they do not (energy maximum at $\theta = 0.9195$ and $0.6950$, ergotropy
  maximum at $\theta = 1.016$ and $0.6445$). This is why PPO ends with less energy but more
  ergotropy than GRAPE for $N = 4$ and $8$: its area error happens to point towards the
  ergotropy maximum. PPO does not optimise ergotropy. A `reward="ergotropy"` benchmark is
  a natural next step.
* **Cost.** Wall time grows with $N$ and with the open-system size, through the cost of
  the adaptive ODE solve rather than the dimension alone: PPO takes 5 s for $N = 2$
  closed, 71 s for $N = 8$ closed (state dimension 81) and 99 s for $N = 4$ open
  (Liouvillian dimension 625); GRAPE at 1500 steps takes 7 to 96 s per seed.

Settings: CPU (64 threads, shared machine), float32 (the examples do not enable x64),
jax 0.7.2, dynamiqs 0.3.6, qenvs commit d0c9ab8 (version string
`0.1.1.dev2+gd0c9ab8ed`). Total wall time 1246 s. Reproduce with
`python examples/benchmark_battery.py` (about 20 minutes on CPU) or `--quick` for a smoke
run. An earlier run of this table had wrong PPO numbers for the open Dicke rows because
of an argument-name collision, fixed in d0c9ab8; see the [review log](review-log.md).

## Differentiability note (open systems)

`run_grape` differentiates the environment return directly for closed and open tasks.
In qenvs 0.0.1 the `StateTransfer` reward used the general Uhlmann fidelity, whose
gradient is NaN for a pure target on density matrices (the matrix
$\sqrt{\rho}\,\sigma\sqrt{\rho}$ has rank one, and $d\sqrt{\lambda}/d\lambda$ diverges at
$\lambda = 0$); the first version of this example worked around it with the identical trace
overlap. Since 0.1.0 `qenvs.observables.fidelity` evaluates a pure target as
$\langle\psi|\rho|\psi\rangle$, which is exact and differentiable, so the workaround was
removed. The returns in the table above are unchanged, because both objectives are equal for
a pure target.

## Reproduce

```bash
python examples/benchmark.py                        # markdown table (about 1 min on CPU)
python examples/benchmark.py --json --out bench.json
python examples/grape_jax.py --steps 200 --lr 0.1 --set horizon=10 --set dt=0.2 --json
python examples/grape_jax.py --set gamma1=0.1 --set horizon=10 --set dt=0.2
python examples/ppo_jax.py --num-envs 256 --updates 50 --set horizon=10 --set dt=0.2
JAX_PLATFORMS=cpu pytest tests/test_examples.py     # 13 tests, about 55 s on CPU
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
* Gradients through a *mixed* target still use the general Uhlmann fidelity, whose
  gradient can be singular; all registered tasks use pure targets.
