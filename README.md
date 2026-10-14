<p align="center">
  <img src="docs/assets/qenvs_logo_1200.png" alt="QEnvS" width="480">
</p>

**Physically validated quantum control environments for reinforcement learning and optimal control.**

*Reusable physics. Reproducible control.*

[![ci](https://github.com/beomdo-park/QEnvS/actions/workflows/ci.yml/badge.svg)](https://github.com/beomdo-park/QEnvS/actions/workflows/ci.yml)
[![docs](https://github.com/beomdo-park/QEnvS/actions/workflows/docs.yml/badge.svg)](https://beomdo-park.github.io/QEnvS)
[![PyPI](https://img.shields.io/pypi/v/qenvs)](https://pypi.org/project/qenvs/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23359969.svg)](https://doi.org/10.5281/zenodo.23359969)
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
index of the docs is at [`/llms.txt`](https://beomdo-park.github.io/QEnvS/llms.txt).

## Architecture

```mermaid
flowchart LR
    M["Model<br/>drift, controls, jump ops,<br/>initial state, observables"] --> E["Environment<br/>reset / step / rollout<br/>jit, vmap, grad"]
    E --> G["GymnasiumEnv<br/>SB3, CleanRL"]
    E --> R["registry.make()<br/>namespace/Name-vN"]
    M --> V["validation.CHECKS<br/>analytic, constraints, numerics"]
    V --> C["qenvs validate / tests"]
    E --> O["GRAPE via jax.grad<br/>PPO via vmap"]
```

The physics lives only in `Model`; everything above it is an adapter. A model is
merged only with checks in `qenvs.validation` that compare it against closed-form
results, physical constraints and an independent simulator (QuTiP).

## Models and tasks

| Task id | Model | Dissipation | Validated against | Status |
|---|---|---|---|---|
| `qubit/StateTransfer-v0` | driven qubit | T1, T_phi (Lindblad) | Rabi formula, pi-pulse, T1/T_phi decay, ergotropy, energy conservation, rho constraints, solver convergence | available |
| `battery/TwoLevel-v0` | open qubit battery (`QubitBattery`) | thermal bath (T1, n_th), T_phi | thermal steady state and stationary Gibbs state, pi-pulse energy = ergotropy = omega0, charging-power bound, QuTiP cross-check, rho constraints, solver convergence | available |
| `battery/Dicke-v0` | N cells + cavity (Tavis-Cummings / Dicke, symmetric subspace) | cavity loss, collective emission and dephasing | Jaynes-Cummings Rabi formula, excitation conservation, collective charging scaling (Ferraro 2018), RWA vs counter-rotating terms, QuTiP cross-check, rho constraints, operator algebra | available |
| `transfer/LambdaSTIRAP-v0` | three-level Lambda (pump + Stokes) | spontaneous emission from the excited level | dark state, two-level Rabi limit, STIRAP transfer (P3 > 0.99, P2 < 0.05), intuitive-order and diabatic counterexamples, emission robustness, QuTiP cross-check | available |

Each model has a page under `docs/models/` with its Hamiltonian, assumptions and validation table. See the [roadmap](docs/roadmap.md).

## Baselines

`examples/grape_jax.py` (open-loop Adam, GRAPE-style, through `env.rollout`),
`examples/ppo_jax.py` (vectorised PPO, equinox + optax) and `examples/benchmark.py`
(table in [docs/benchmarks.md](docs/benchmarks.md)). Closed `qubit/StateTransfer-v0`,
3 seeds, about a minute on CPU: GRAPE 1.0000, PPO 0.9997 +- 0.0002 (optimum 1.0).

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

Software DOI (all versions): [10.5281/zenodo.23359969](https://doi.org/10.5281/zenodo.23359969).
Version 0.1.0: [10.5281/zenodo.23359970](https://doi.org/10.5281/zenodo.23359970).
Machine-readable metadata is in [CITATION.cff](CITATION.cff) (GitHub shows a "Cite this
repository" button).

```bibtex
@software{park_qenvs,
  author  = {Park, Beomdo},
  title   = {{QEnvS}: physically validated quantum control environments for
             reinforcement learning and optimal control},
  year    = {2026},
  version = {0.1.0},
  doi     = {10.5281/zenodo.23359969},
  url     = {https://github.com/beomdo-park/QEnvS}
}
```

## License

Apache-2.0. Third-party attributions are listed in [NOTICE](NOTICE).
