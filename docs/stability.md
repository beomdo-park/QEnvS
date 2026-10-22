# Stability and versioning

QEnvS is in the 0.x series (`Development Status :: 3 - Alpha`). This page states what you
can rely on when you build on it.

## What is versioned how

| thing | rule |
|---|---|
| Task ids (`namespace/Name-vN`) | A released task id never changes behaviour: reward, observation, action space, defaults and physics stay fixed. Any change that could alter learning or optimisation results gets a new suffix (`-v1`), and the old id keeps working for at least one minor release. |
| Validation checks | Checks can be added at any time. A check is never loosened silently: a tolerance change is listed in the changelog with its reason. |
| Python API (`qenvs.*`) | Semantic versioning with the 0.x convention. A minor release (0.3 to 0.4) is the only place for changes that can break existing code: removed or renamed names, changed signatures or defaults, changed behaviour of existing calls. A patch release (0.3.1 to 0.3.2) carries fixes and backward-compatible additions: new functions, classes, modules, tasks or checks, and new keyword arguments whose default keeps the old behaviour. Every removal or rename is announced with a `DeprecationWarning` for at least one minor release first. |
| CLI and `--json` output | Commands, flags and the keys of the JSON documents follow the same rule as the Python API. New keys may appear in any release; consumers should ignore unknown keys. Error codes (`unknown_task`, `invalid_argument`, ...) are stable. |
| Examples (`examples/`) | Reference scripts, not part of the package (not in the wheel). Their arguments and outputs may change in any release. |

Public API means everything importable without a leading underscore from `qenvs`,
`qenvs.models` (including `qenvs.models.transmon` and `qenvs.models.custom`),
`qenvs.envs` (including `qenvs.envs.gate`), `qenvs.gates`, `qenvs.wrappers` (including
`qenvs.wrappers.vector`, `qenvs.wrappers.randomization`, `qenvs.wrappers.control`,
`qenvs.wrappers.observation`), `qenvs.observables`, `qenvs.pulses`, `qenvs.distributions`,
`qenvs.robust`, `qenvs.trajectories`, `qenvs.keys`, `qenvs.validation` (including
`qenvs.validation.generic`) and
`qenvs.registry`. Use `Environment.info(state)` for
diagnostics; methods starting with `_` (for example `Environment._info`) are internal.
`qenvs.toolkit` is the plain-dict layer behind the CLI and follows the CLI rules below.

Up to 0.3.1 a patch release carried fixes only. From 0.3.2 on (maintainer decision,
2026-10-22) backward-compatible additions may also ship in a patch release, so the minor
number moves only for breaking changes. See
[Releasing](https://github.com/beomdo-park/QEnvS/blob/main/RELEASING.md).

## Supported environments

| item | supported | tested in CI |
|---|---|---|
| Python | 3.11, 3.12, 3.13 | all three on Linux, 3.12 on macOS |
| jax | `>=0.4.36, !=0.7.0, !=0.7.1, <0.11` | lowest (0.4.36) and newest allowed (0.10.x) |
| dynamiqs | `>=0.3.6` | 0.3.6 |
| OS | Linux, macOS; Windows untested | Ubuntu, macOS |
| Accelerators | CPU tested; GPU through a CUDA-enabled jax install | CPU only |

The upper bound on jax exists because dynamiqs 0.3.6 does not import with jax 0.11 (fixed
upstream, not yet released). It will be lifted when a fixed dynamiqs is released.

## Numerical reproducibility

Results are deterministic for a fixed seed, jax version, platform and precision. They are
not bit-identical across CPU and GPU or across jax versions; the validation tolerances are
chosen to hold in float32 and float64 on every supported combination.
