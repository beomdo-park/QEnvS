# Observation modes, finite shots and robust evaluation

The tasks of QEnvS hand the agent the full quantum state. An experiment cannot do that: it gets
expectation values of a few observables, each estimated from a finite number of shots. This page
describes `ObservationModel` (observation modes and finite-shot measurement), the information
boundary that comes with it, and `qenvs.robust` (evaluation of a policy over the distribution an
environment samples from, with the statistics that are validated and the scope of that
validation).

```python
import jax
import qenvs
from qenvs import robust

env = qenvs.make("battery/Dicke-v0", n_cells=4, observation="expectations", shots=1000)
env.observation_dim  # 5: energy, n_photons, excitation, top_fock, t / T
```

`make(task_id, observation=..., shots=..., diagnostics=..., observables=...)` wraps the task in
`qenvs.wrappers.observation.ObservationModel` if any of these is given; `make(task_id)` returns
the unwrapped task exactly as before. The wrapper changes the observation and the `info` only:
dynamics and rewards are those of the wrapped environment.

## Observation modes

| `observation=` | Observation | `shots` | What it is |
|---|---|---|---|
| `"state"` (default) | the task's own observation: flattened state, or accumulated propagator / superoperator for gate tasks, plus `t / T` | not allowed | ideal-simulator baseline, identical to the released behaviour |
| `"expectations"` | `<O_j>` of `Model.observables()` in insertion order, then `t / T` | allowed | measurement-statistics partial observation |
| `"correlations"` | the `"expectations"` list, then `Model.correlation_observables()`, then `t / T` | allowed | partial observation with second-order correlators |
| `"time"` | `t / T` only | not allowed | open-loop ("blind") control |

* The measured observables are those of the **nominal** model at construction, so under parameter
  randomisation the estimate is of the nominal observable on the physical state.
* In `"correlations"` the common observables get the same measurement noise as in
  `"expectations"` (they use the same random-number leaves), which reduces the variance of a
  comparison between the two modes.
* Entries that other wrappers append to the observation (for example the oracle parameters of
  `ParameterRandomization`) stay at the end of the observation, whether that wrapper is inside or
  outside the observation model.
* Observables must be Hermitian (a complex correlator is measured as its real and imaginary
  parts, two observables). Two observables that are the same operator are rejected, because
  they would count their shots twice. Some built-in tasks list the same operator under two names
  (for example `energy` and `p1` of a qubit with `omega0 = 1`); choose with
  `observables=("energy",)`, which keeps the order you give.
* Gate tasks (`GateControl`) support `"state"` and `"time"` only. `"expectations"` of a gate
  needs probe input states and a measurement design, which is planned for 0.4. The attribute
  `observation_kind` of an `ObservationModel` tells what `"state"` carries: `"state"`, `"propagator"` (closed gate task) or
  `"superoperator"` (open gate task). The last two are accumulated maps that an experiment only
  obtains by process tomography, so the `"state"` mode of a gate task is an ideal-simulator
  baseline, not a measurement.

## Finite shots and their cost

`shots=n` means: each observable is measured `n` times in its own eigenbasis, in a separate
experiment, and the estimate is the mean of the outcome eigenvalues. The outcome counts are drawn
from the exact multinomial distribution of the distinct eigenvalues (degenerate eigenvalues are one
outcome), as a chain of conditional binomial draws. The estimator is unbiased and its variance is
exactly `(<O^2> - <O>^2) / n`; an observable on one of its eigenstates is measured without noise.
`shots=None` returns the exact expectation value.

What is assumed, and not modelled:

* no measurement back-action: the quantum state is not changed by the observation, so this is
  an ideal state evolution combined with a noisy observation, not closed-loop feedback on a
  collapsing state;
* no readout error;
* observables are measured in separate experiments, so non-commuting observables are fine. In a
  real device commuting observables can share shots; this is not modelled, which counts the cost
  conservatively;
* `shots` is per observable and per observation;
* outcome probabilities below 64 machine epsilons (1.4e-14 in float64, 7.6e-6 in float32) are
  treated as 0 and the rest renormalised: they are rounding residue of an eigenstate, and an
  outcome that cannot occur must not occur. A genuine outcome probability below that threshold
  is therefore not sampled; it matters only for `shots` far larger than the inverse of it.

The shot budget of one observation is `shots * K` with `K` the number of observables,
reported in `info["shots_used"]` (0 without shots) and as `env.shots_per_step`. An episode of
`H` steps observes `H + 1` times (reset included), so for Dicke `N = 4` with `shots=1000`
that is 4000 measurements per observation and 84 000 per 20-step episode. Whether that is
affordable on a device, where every observation is a fresh experiment repeated `n` times, is the
user's judgement.

Random numbers: observable `j` at observation time `t` (`t = 0` after `reset`, `t = k` after step
`k`) draws only from `keys.measurement_key(episode_key, t, j)`. Changing `shots`, the number of
observables or any other wrapper does not change the draws of the other leaves.

## Information boundary

1. The policy input is the observation and nothing else.
2. `info` in the partial modes (`"expectations"`, `"correlations"`, `"time"`) contains no exact
   value (no fidelity, leakage, energy, ergotropy or exact expectation value): only the cost
   keys `shots_used` and `clip_fraction`. The `"state"` mode keeps the released `info`.
3. `diagnostics=True` adds the exact values as `info["diagnostics"]`, for evaluation and
   debugging.
4. `env.info(state)` (any environment) and `env.diagnostics(state)` (wrapped environments) always
   return the exact values. The API cannot
   prevent a caller from feeding them to a policy; keep training loops and evaluation code
   separate (as the examples do).
5. **The reward is not changed by the mode.** It is the exact simulator gain of the task quantity
   (fidelity, energy or ergotropy), so even in the partial modes the cumulative reward reveals the
   exact performance trajectory. This is the usual assumption of simulation-based RL and it differs
   from an experiment, where the reward would itself have to be estimated from measurements
   (planned for 0.4). Report results of partial-observation training with this caveat.

With the Gymnasium adapters the `info` dictionary can contain a nested dictionary
(`info["diagnostics"]`); `GymnasiumEnv` returns it as a nested dictionary of numpy arrays and
`GymnasiumVectorEnv` as a nested dictionary of batched arrays with its own `_key` masks.

## Episode keys

One run key makes every episode reproducible: episode `i` uses
`keys.episode_key(run_key, i) = fold_in(run_key, i)`.

* `qenvs.robust.evaluate` uses `fold_in(key, 0)` as the run key of its episodes.
* `GymnasiumEnv(env, seed=s)`: run key `PRNGKey(s)`, the `i`-th episode since the last seeding is
  episode `i` (`reset(seed=...)` restarts at 0).
* `GymnasiumVectorEnv(env, num_envs, seed=s)`: one run key `PRNGKey(s)`; the `j`-th episode of
  sub-environment `b` is episode `i = j * num_envs + b`. The vector env therefore replays the
  episodes of a single env of the same seed in interleaved order. (Before 0.3 sub-environment `b`
  used the stream of `GymnasiumEnv(seed=seed + b)`. Deterministic tasks ignore the key, so their
  behaviour is unchanged; only user-defined environments with random resets see different
  samples.)

## Evaluating a policy

```python
from qenvs.distributions import Normal
from qenvs.wrappers.control import ControlErrors

task = qenvs.make("transmon/XGate-v0")
actions = qenvs.registry["transmon/XGate-v0"].reference(task)  # a known-good open-loop pulse
noisy = ControlErrors(task, amplitude_scale=Normal(std=0.02), step_noise=Normal(std=0.01))
result = robust.evaluate(noisy, actions, jax.random.PRNGKey(123), n=1024, nominal_env=task)
result.mean, result.cvar, result.min  # final fidelity: mean, lower-tail CVaR, minimum
result.mean_interval, result.cvar_interval  # 95 percent percentile bootstrap
result.nominal  # the same policy on the nominal task
result.return_mean  # the training reward summed over the episode, reported separately
```

* The evaluation distribution is whatever the environment carries: wrap the task in a parameter
  randomiser, a control-error wrapper or an observation model first. To evaluate a policy on a
  distribution other than its training distribution, build the evaluation environment separately.
  Use an evaluation key that the training stream does not use.
* `policy` is an open-loop array of shape `(horizon, action_dim)` or a JAX-traceable function
  `(observation, key) -> action`. A stochastic policy receives a different key at every decision,
  `keys.policy_key(episode_key, t) = fold_in(fold_in(episode_key, 3), t)`, `t = 0, 1, ...`,
  so no random sample is reused within an episode and the episode stays reproducible. The leaf is
  used only by `evaluate`; with RL libraries the library manages its own random numbers.
* `metric=None` evaluates `performance_metric(env)`: `"fidelity"` for gate tasks (average gate
  fidelity, leakage counted as error) and for state-transfer tasks (target-state fidelity), and
  the reward quantity (`"energy"` or `"ergotropy"`) of battery tasks, read from the exact
  `env.info(final_state)`. It can be another `info` key or a function of the final state.
* `n` defaults to 1024. `batch_size` simulates the episodes in chunks to bound memory.
  The outcome of an episode does not depend on the batch it is simulated in, up to the last bits
  of floating-point rounding.
* `episode_return` is the training reward summed over the episode, a gain `Q(final) - Q(initial)`.
  It is reported (`return_mean`, `return_min`, `returns`) but never aggregated in place of the
  metric.

### Statistics and their scope

* The CVaR is the mean of the lowest `ceil(0.1 n)` episodes (the integer ceiling of `n / 10`).
  The metric is higher-is-better, so this tail is the opposite of a loss-based CVaR. The
  estimator has an `O(1/n)` bias: for U(0, 1) its expectation is exactly `(k + 1) / (2 (n + 1))`
  with `k = ceil(0.1 n)`, against the population value 0.05.
* The minimum is a descriptive statistic that falls as `n` grows. Do not compare minima of
  different `n`.
* Intervals: percentile bootstrap over episodes, `B = 2000`, 2.5 and 97.5 percentiles. The mean
  interval is provided for `n >= 256` and the CVaR interval for `n >= 1024`. Below that,
  `evaluate` returns point estimates with a warning (`n < 256`: no interval; `256 <= n < 1024`: the
  mean interval and the CVaR point estimate).
* Measured coverage of the 95 percent intervals at `n = 1024`, `B = 2000`, 1000 independent
  repetitions with fixed keys (the slow test `test_bootstrap_interval_coverage`):

  | distribution | mean interval | CVaR interval |
  |---|---|---|
  | U(0, 1) | 0.954 | 0.932 |
  | gate-fidelity-like, `F = 1 - (2/3) sin^2(pi e / 2)`, `e ~ N(0, 0.05^2)` | 0.950 | 0.944 |

  All four values (the mean and CVaR intervals, for both distributions) are inside the test's
  acceptance window [0.92, 0.98], which is set at about 4 binomial standard errors (0.0069 at
  0.95) around 0.95 and applies to each of the four coverages separately. These are empirical
  coverages observed in this experiment on the two tested distributions. For U(0, 1) the CVaR
  interval undercovered in this setting: 0.932 is about 2.6 standard errors below the nominal
  0.95, hard to dismiss as noise, though one experiment of 1000 repetitions does not pin down a
  general bias. For the gate-fidelity-like distribution 0.944 is within one standard error of
  nominal. Passing the window [0.92, 0.98] does not show that the interval is calibrated; treat
  the CVaR interval as approximate and somewhat narrow. A stronger calibration claim would need more distributions,
  more repetitions and a proper test (or a different interval method).
  This is evidence **for these two distributions**, not a coverage guarantee for any
  distribution. Heavy-tailed, discrete or boundary-heavy performance distributions (for example a
  fidelity piled up at its upper bound 1) are separate validation targets: check the coverage on a
  distribution of that kind before trusting a CVaR interval.
* The intervals describe the sampling variation of the evaluation episodes for a given policy.
  The variation between independently trained policies (training seeds) is another source: report
  it separately and do not merge the two into one interval.
* `stability(env, policy, key, sizes=(256, 1024, 4096))` evaluates once at the largest size and
  reports the mean and CVaR for nested prefixes (the same episode keys as `evaluate(n=...)`), with
  the spread of each across the sizes. The prefixes are nested, so the spread understates the
  sampling variation; it is a rough check that the estimates have stopped moving, not a
  convergence test.
* `nominal_env` runs the policy on the nominal (not randomised) task so that robustness bought at
  the price of nominal performance is visible. It is assumed deterministic and run for two
  episodes; a warning is issued if the two differ.

## Batch execution: time and memory

Measured on 4 CPU cores (`taskset -c 0-3`) with float64, 1024 episodes through `robust.evaluate`
(one jitted `vmap` of a `lax.scan` over the horizon). "First" includes compilation, "second"
is a repeat with a new key. Peak resident memory of the whole process, which is 298 MB right
after importing `qenvs` and jax. The wrapped environments carry all the wrappers named; the bare
rows run the unwrapped task with the same policy.

| Task and wrappers | Horizon | First | Second | Peak RSS |
|---|---|---|---|---|
| `qubit/StateTransfer-v0`, bare | 20 | 1.1 s | 0.4 s | 490 MB |
| same, parameter randomisation + control errors + `expectations` with 1000 shots | 20 | 2.0 s | 1.1 s | 564 MB |
| `battery/Dicke-v0` `n_cells=4`, bare | 20 | 1.0 s | 0.4 s | 468 MB |
| same, control errors + `expectations` with 1000 shots (4 observables) | 20 | 2.4 s | 1.6 s | 515 MB |
| `transmon/XGate-v0`, bare | 16 | 0.9 s | 0.4 s | 483 MB |
| same, parameter randomisation + control errors | 16 | 1.6 s | 1.0 s | 531 MB |

The same qubit run in float32 takes 1.8 s and 1.0 s with a peak of 566 MB. The wrappers cost
roughly a factor 2.5 to 4 in run time on these small tasks, where the unwrapped step is already
cheap; the absolute numbers depend on the task, so measure your own with `time.perf_counter`
and `resource.getrusage`.

Empirical checks on 1024 episodes (tests `tests/test_batch.py`): the leaves of the key tree, the
measurement noise of different observables and steps, and neighbouring episodes are uncorrelated
within `|r| <= 4 / sqrt(n)`. Uncorrelatedness is a sanity check against key reuse, not a proof of
statistical independence.
