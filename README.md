# QEnvS

**Physically validated quantum control environments for reinforcement learning and optimal control.**

*Reusable physics. Reproducible control.*

[![ci](https://github.com/beomdo-park/qenvs/actions/workflows/ci.yml/badge.svg)](https://github.com/beomdo-park/qenvs/actions/workflows/ci.yml)
[![docs](https://github.com/beomdo-park/qenvs/actions/workflows/docs.yml/badge.svg)](https://beomdo-park.github.io/qenvs)
[![PyPI](https://img.shields.io/pypi/v/qenvs)](https://pypi.org/project/qenvs/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

QEnvS (Quantum Environment Suite) is a model zoo of open-quantum-system control
problems, quantum batteries, qubit state transfer, thermal machines, packaged so
that you can plug them into an RL agent or a gradient-based optimiser in a few
lines, instead of re-deriving the physics for every paper.

Every model ships with **analytic reference results that run as tests**
(pi-pulse inversion, exponential decay, ergotropy bounds, collective-charging
scaling, ...). If a model is in QEnvS, its physics has been checked against
closed-form solutions and against an independent simulator (QuTiP).

Built on [dynamiqs](https://github.com/dynamiqs/dynamiqs) (JAX): environments are
`jit`-able, `vmap`-able across thousands of trajectories, and differentiable, so
the same model serves RL (PPO/SAC), GRAPE-style optimal control and MPC.

## Install

```bash
pip install qenvs            # core: jax + dynamiqs
pip install 'qenvs[gym]'     # + gymnasium adapter (stable-baselines3, CleanRL, ...)
```

## Quickstart

Functional JAX API (batch 1024 episodes on one GPU):

```python
import jax, jax.numpy as jnp
import qenvs

env = qenvs.make("qubit/StateTransfer-v0", horizon=10, dt=0.2)

keys = jax.random.split(jax.random.PRNGKey(0), 1024)
states, ts = jax.vmap(env.reset)(keys)
actions = jnp.ones((1024, env.action_dim))
states, ts = jax.jit(jax.vmap(env.step))(states, actions)
print(ts.reward.shape, ts.info["fidelity"].shape)  # (1024,) (1024,)
```

Gradient-based control through the same environment:

```python
def loss(actions):
    _, rewards = env.rollout(actions)
    return -rewards.sum()  # = -(F_final - F_initial)


grad = jax.grad(loss)(jnp.zeros((env.horizon, env.action_dim)))
```

Gymnasium for classic RL libraries:

```python
from qenvs.wrappers import GymnasiumEnv
from stable_baselines3 import PPO

gym_env = GymnasiumEnv(env)
PPO("MlpPolicy", gym_env).learn(20_000)
```

Command line, for humans and coding agents (every command has `--json`;
`uvx qenvs ...` works without installing):

```bash
qenvs list --json
qenvs info qubit/StateTransfer-v0 --json       # physics, kwargs, references, usage
qenvs run qubit/StateTransfer-v0 --policy pi-pulse --set gamma1=0.1 --json
qenvs validate                                  # per-check error vs tolerance + versions
qenvs doctor                                    # install / backend / optional extras check
```

Repository conventions for coding agents are in [AGENTS.md](AGENTS.md); an LLM
index of the docs is at [`/llms.txt`](https://beomdo-park.github.io/qenvs/llms.txt).

## Models and tasks

| Task id | Model | Dissipation | Validated against | Status |
|---|---|---|---|---|
| `qubit/StateTransfer-v0` | driven qubit | T1, T_phi (Lindblad) | Rabi formula, pi-pulse, T1/T_phi decay, ergotropy, energy conservation, rho constraints, solver convergence | available |
| `battery/TwoLevel-v0` | open qubit battery | T1, T_phi, thermal | ergotropy/power bounds | planned |
| `battery/Dicke-v0` | N qubits + cavity (Dicke / Tavis-Cummings) | cavity loss, dephasing | collective charging scaling | planned |
| `transfer/LambdaSTIRAP-v0` | three-level Lambda | spontaneous emission | adiabatic STIRAP transfer | planned |

See the [roadmap](docs/roadmap.md).

## Why not X?

* **qutip-qoc / GRAPE / CRAB** optimise a pulse for one fixed model; QEnvS gives
  you the model as a reusable, tested object plus an environment interface.
* **qiskit-gym, qgym** are circuit-level (synthesis, routing); QEnvS is
  Hamiltonian/pulse-level with open-system dynamics.
* **Paper repositories** (e.g. ICML 2025 RL-for-quantum-control code) are
  reproductions of one paper; QEnvS is the shared, maintained layer under them.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). New models must come with analytic
validation tests.

## Citation

See [CITATION.cff](CITATION.cff). A Zenodo DOI will be attached to the first release.

## License

Apache-2.0. Third-party attributions are listed in [NOTICE](NOTICE).
