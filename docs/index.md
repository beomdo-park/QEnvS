<p align="center"><img src="assets/qenvs_logo_1200.png" alt="QEnvS" width="440"></p>

**Physically validated quantum control environments for reinforcement learning and optimal control.**

*Reusable physics. Reproducible control.*

QEnvS is a model zoo of open-quantum-system control problems (quantum batteries,
qubit state transfer, later thermal machines) with a uniform environment API.
Every model ships with analytic validation checks that run as tests and are
printed by `qenvs validate`. Built on [dynamiqs](https://github.com/dynamiqs/dynamiqs)
(JAX), so environments are `jit`-able, `vmap`-able and differentiable.

## Install

```bash
pip install qenvs            # core
pip install 'qenvs[gym]'     # + Gymnasium adapter
```

## 60-second tour

```python
import jax, jax.numpy as jnp
import qenvs

env = qenvs.make("qubit/StateTransfer-v0", horizon=10, dt=0.2)

# pi-pulse: integral of u dt = pi  ->  |0> -> |1>
actions = jnp.full((env.horizon, env.action_dim), jnp.pi / (env.horizon * env.dt))
state, rewards = env.rollout(actions)
print(float(rewards.sum()))  # 1.0  (= F_final - F_initial)

# gradient-based control through the same environment
grad = jax.grad(lambda a: -env.rollout(a)[1].sum())(jnp.zeros_like(actions))

# batch 1024 episodes
keys = jax.random.split(jax.random.PRNGKey(0), 1024)
states, ts = jax.vmap(env.reset)(keys)
states, ts = jax.jit(jax.vmap(env.step))(states, jnp.ones((1024, env.action_dim)))
```

Gymnasium (stable-baselines3, CleanRL, ...):

```python
from qenvs.wrappers import GymnasiumEnv

gym_env = GymnasiumEnv(env)
obs, info = gym_env.reset(seed=0)
```

## Command line and agents

```bash
qenvs list --json
qenvs info qubit/StateTransfer-v0 --json
qenvs run qubit/StateTransfer-v0 --policy pi-pulse --set gamma1=0.1 --json
qenvs validate            # per-check error vs tolerance, solver, versions
qenvs doctor              # install, backend, optional extras, smoke run
```

An LLM-oriented index of these docs is served at [`/llms.txt`](llms.txt).

## Validation philosophy

An RL interface does not make a simulator physically correct. Each model in
QEnvS is accepted only with checks in `qenvs.validation`:

| Category | Example |
|---|---|
| analytic | Rabi oscillation `P1 = sin^2(ut/2)`, pi-pulse inversion |
| open_system | T1 exponential decay, T_phi coherence decay |
| thermodynamics | ergotropy of reference states |
| conservation | `<H>` constant under static Hamiltonian (closed) |
| constraints | density matrix Hermitian, trace 1, positive semidefinite after random driving |
| numerics | adaptive Tsit5 vs exact `Expm` agree |
| control | pi-pulse action sequence returns 1 through the environment |

The agent's control period `dt` and the integrator's internal step are
independent: the ODE solver adapts inside each control interval.

## Citation

If you use QEnvS, please cite the software DOI
[10.5281/zenodo.23359969](https://doi.org/10.5281/zenodo.23359969) (all versions; version
0.1.0 is [10.5281/zenodo.23359970](https://doi.org/10.5281/zenodo.23359970)). A BibTeX
entry and `CITATION.cff` are in the
[repository](https://github.com/beomdo-park/QEnvS#citation).
