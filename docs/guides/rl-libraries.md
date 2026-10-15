# Using QEnvS with RL libraries

QEnvS offers three interfaces to the same physics. They share the reward, observation and
termination semantics of the functional environment, so results are comparable.

| Interface | Class | Use it when |
|---|---|---|
| Functional JAX | `qenvs.make(...)`: `reset(key)`, `step(state, action)` | JAX-native training (PPO/SAC written in JAX, evolution strategies, GRAPE), `jit`/`vmap`/`grad` over episodes. Fastest, differentiable. |
| Gymnasium | `qenvs.wrappers.GymnasiumEnv` | Single-environment libraries and quick experiments (SB3 `DummyVecEnv`, `check_env`, notebooks). |
| Gymnasium vector | `qenvs.wrappers.vector.GymnasiumVectorEnv` | Batched numpy training loops (CleanRL, SB3 through a vec adapter, any `gymnasium.vector` consumer). One jitted `vmap` steps all sub-environments. |

Install the Gymnasium extra with `pip install 'qenvs[gym]'` (gymnasium >= 1.0).

## Normalised actions

All QEnvS tasks have a symmetric control bound `env.action_bound` (2.0 for
`qubit/StateTransfer-v0`). Gymnasium's checker warns about non-normalised Box action spaces and
SB3 and CleanRL assume `[-1, 1]`. Pass `normalize_actions=True`:

```python
env = GymnasiumEnv(qenvs.make("qubit/StateTransfer-v0"), normalize_actions=True)
env.action_space  # Box(-1.0, 1.0, (1,), float32)
```

The action is multiplied by `env.action_bound` before stepping, exactly (a test compares
`normalize_actions=True` with the unscaled wrapper bit for bit). The environment still clips to
the bound, so an out-of-range policy output saturates at `+-action_bound`. The observation space
is unchanged: it is `Box(-inf, inf)` (the quantum state components are bounded by 1 in modulus,
but the space is not declared that way), so the checker's unbounded-observation warning remains.

## Reward and termination semantics

* The reward at each step is the fidelity gain (for state transfer), so the undiscounted
  return of an episode equals the final fidelity minus the initial fidelity.
* Every episode ends with `terminated=True` after exactly `env.horizon` steps.
  `truncated` is **never** True.
  QEnvS tasks are fixed-duration control problems: the total pulse time is part of the task and
  the return is defined at that final time. Reaching the horizon is therefore a genuine terminal
  state, not a time-limit cut-off of a longer process. Value bootstrapping must stop there, and
  that is what `terminated=True` tells the library (a time-limit `truncated=True` would make
  PPO/SAC bootstrap from a state that has no future). Do not wrap QEnvS environments in
  `TimeLimit` and expect different behaviour.
* Time-dependent policies are possible: the observation contains the normalised time `t/T`.

## Vector environment and autoreset

`GymnasiumVectorEnv(env, num_envs, *, seed=0, normalize_actions=False)` implements the
gymnasium>=1.0 `VectorEnv` API with numpy inputs and outputs:

* `observation_space` / `action_space` are batched, `single_observation_space` /
  `single_action_space` describe one sub-environment.
* `reset(seed=s)` returns `(obs[num_envs, obs_dim], infos)`; `step(actions[num_envs, action_dim])`
  returns `(obs, reward, terminated, truncated, infos)` as float32, float32, bool, bool.
* Every returned array (observations, rewards, `terminated`, `truncated` and each info value) is
  a writable numpy copy, so in-place edits and `gymnasium.wrappers.vector` wrappers such as
  `ClipReward`, `VectorizeTransformReward` and `VectorizeTransformObservation` work (tested).
* `step` needs actions of shape `(num_envs, action_dim)` exactly. A 1-D action of shape
  `(num_envs,)` is rejected with `ValueError("actions must have shape ...")`, there is no implicit
  reshape.
* Infos are dicts of batched arrays (`infos["fidelity"]` has shape `(num_envs,)`) with the usual
  `_key` boolean masks (all True, since every sub-environment reports every key).
* Episode keys follow `qenvs.keys.episode_key(run_key, i) = fold_in(PRNGKey(seed), i)`. The
  `j`-th episode of sub-environment `b` is episode `i = j * num_envs + b`, so the vector
  environment replays the episodes of `GymnasiumEnv(seed=seed)` in interleaved order (tested,
  including a random-reset toy environment). Since 0.3; in 0.2 sub-environment `b` followed
  `GymnasiumEnv(seed=seed + b)`. The built-in tasks reset deterministically, so their episodes
  are unchanged.

Autoreset mode is `AutoresetMode.NEXT_STEP` (declared in `metadata["autoreset_mode"]` on
gymnasium >= 1.1, where the enum exists; the version in the development venv is 1.4.0).
On the step where an episode ends you receive the true final observation and
`terminated=True`. On the next call the action of that sub-environment is ignored and you
receive the new initial observation, reward 0, `terminated=False`, `truncated=False`.
An episode therefore occupies `horizon + 1` calls. Libraries that expect `SAME_STEP` must not be
used without adaptation. Calling `reset()` while an autoreset is pending discards it: the next
`step` uses its action normally (tested). Gymnasium 1.0 has no `AutoresetMode` enum: there
`NEXT_STEP` is the only behaviour and nothing needs declaring, so the class omits the metadata
key. The gymnasium 1.0 path is tested: both wrapper test files (25 tests) pass on gymnasium 1.0.0
with jax 0.4.36 (the CI lowest-versions environment).

`gymnasium.wrappers.vector.RecordEpisodeStatistics` works and reports `episode["r"]`,
`episode["l"]` (tested).

## CleanRL-style loop (tested)

This script is executed in the test suite as a random-policy loop (`test_cleanrl_style_loop_three_episodes`)
and was run standalone to produce the output below.

```python
import numpy as np

import qenvs
from qenvs.wrappers.vector import GymnasiumVectorEnv

num_envs = 8
envs = GymnasiumVectorEnv(
    qenvs.make("qubit/StateTransfer-v0", horizon=20, dt=0.1),
    num_envs,
    seed=0,
    normalize_actions=True,
)
envs.action_space.seed(0)  # makes the random policy, and so the output below, deterministic
obs, _ = envs.reset(seed=0)
episodic_return = np.zeros(num_envs)
finished = []
for _ in range(3 * 21):  # 3 episodes (20 steps + 1 autoreset step each)
    actions = envs.action_space.sample()  # replace by policy(obs)
    obs, rewards, terminations, truncations, infos = envs.step(actions)
    episodic_return += rewards
    for i in np.flatnonzero(terminations | truncations):
        finished.append((episodic_return[i], infos["fidelity"][i]))
        episodic_return[i] = 0.0
print(len(finished), "episodes, mean return", np.mean([r for r, _ in finished]))
```

Output: `24 episodes, mean return 0.0686...` (3 episodes times 8 environments, random actions
from `action_space.seed(0)`; the mean return equals the mean final fidelity here, 0.0686...).
Note that the reward on the autoreset call is 0, so summing rewards across it is safe.

## Stable-Baselines3 (untested)

SB3 is not installed in the development environment, so the following is **untested**. It uses
the single-environment wrapper because SB3 builds its own vectorisation.

```python
import qenvs
from qenvs.wrappers import GymnasiumEnv
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env


def make():
    return GymnasiumEnv(qenvs.make("qubit/StateTransfer-v0"), normalize_actions=True)


env = make_vec_env(make, n_envs=4)
model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=20_000)
```

The single-environment wrapper steps one ODE solve per call (about 4k steps/s on CPU). For
throughput prefer the vector class with a library that consumes `gymnasium.vector` directly, or
the functional API.

## Measured throughput

`qubit/StateTransfer-v0` with defaults (horizon 20, dt 0.1), `normalize_actions=True`, zero
actions, CPU only (`JAX_PLATFORMS=cpu`, x64 off), steady state after compilation, on a shared
64-core machine while other jobs were running, so treat the numbers as order of magnitude:

| Interface | num_envs | steps/s |
|---|---|---|
| `GymnasiumEnv` | 1 | 4.2e3 |
| `GymnasiumVectorEnv` | 1 | 8.6e3 |
| `GymnasiumVectorEnv` | 64 | 2.2e5 |
| `GymnasiumVectorEnv` | 1024 | 1.2e6 |
| functional `jit(vmap(step))` | 1024 | 1.4e6 |

The vector class costs up to tens of percent over the raw functional `vmap` at large batch (host
transfers and copies, autoreset selection); repeated measurements here gave between no
measurable gap and about 15 percent, and the machine is shared, so the gap is not resolved more
finely than that. Note that the vector step also evaluates `reset` for every
sub-environment each call (selected with `where`), which is negligible for the bundled tasks
but would matter for a task with an expensive `reset`.

## Limitations

* CPU numbers only; no GPU measurement was made.
* The vector class is not a `jax.Array` interface: for end-to-end JAX training avoid the
  numpy round trip and use the functional API.
* `render` is not supported; `close` releases nothing, it only sets `closed=True`.
