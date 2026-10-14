# Getting started

## Install

```bash
pip install qenvs              # core: jax (CPU), dynamiqs, equinox, numpy
pip install 'qenvs[gym]'       # + Gymnasium adapter for SB3, CleanRL, ...
pip install 'qenvs[dev]'       # + QuTiP cross-checks, pytest, optax (examples)
```

or, without installing anything permanently:

```bash
uvx qenvs doctor
uvx qenvs list
```

### GPU

QEnvS runs on whatever backend your jax install provides. For NVIDIA GPUs install a
CUDA-enabled jax inside the supported range, for example:

```bash
pip install qenvs 'jax[cuda12]<0.11'      # or jax[cuda13]
qenvs doctor                               # jax_backend should read "gpu"
```

The supported versions are listed on [Stability and versioning](stability.md).

## Check the installation

```bash
qenvs doctor          # versions, backend, optional extras, a one-step smoke run
qenvs validate        # every physics check: measured error against its tolerance
```

`qenvs validate` should end with every check passed. The same report is published for
each documentation build on the [Validation report](validation.md) page.

## First run

```bash
qenvs list                                         # registered tasks
qenvs info battery/Dicke-v0                        # physics, kwargs, defaults, references
qenvs run qubit/StateTransfer-v0 --policy pi-pulse --json
```

From Python:

```python
import jax.numpy as jnp
import qenvs

env = qenvs.make("qubit/StateTransfer-v0", horizon=10, dt=0.2)
actions = jnp.full((env.horizon, env.action_dim), jnp.pi / (env.horizon * env.dt))
state, rewards = env.rollout(actions)
print(float(rewards.sum()))   # 1.0: the pi-pulse transfers |0> to |1>
```

## Precision

Library code runs in jax's default float32. The command-line tool enables float64
(`jax_enable_x64`) so that `qenvs validate` reports double-precision errors. Enable it in
your own scripts when you need it:

```python
import jax
jax.config.update("jax_enable_x64", True)   # before creating arrays
```

Every validation check is designed to pass in both precisions.

## Troubleshooting

| symptom | cause and fix |
|---|---|
| `TypeError: unsupported operand type(s) for \|: 'types.UnionType' and 'str'` on import | jax 0.11 or newer with dynamiqs 0.3.6. Install `jax<0.11`. |
| `qenvs doctor` shows `jax_backend: cpu` on a GPU machine | the CPU-only jaxlib is installed; install `jax[cuda12]<0.11` (or `cuda13`). |
| QuTiP cross-checks show `SKIP` | QuTiP is not importable; install `qenvs[dev]` (QuTiP normally comes with dynamiqs). |
| A Dicke run with `rwa=False` changes when `n_max` grows | the Fock cutoff is not converged; see the cutoff table on the [Dicke page](models/dicke.md). |
