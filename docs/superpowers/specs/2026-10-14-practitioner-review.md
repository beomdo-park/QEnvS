# Practitioner review (2026-10-14)

Question: could a pulse or control engineer at a hardware lab (IBM Quantum as the example)
use QEnvS for real work today? Review of v0.1.0, then the plan for the next round.

## Findings

| priority | gap | evidence | decision |
|---|---|---|---|
| P0 | Dependency pin `jax<0.8` while jax 0.11 is current | dynamiqs 0.3.6 imports and passes the full suite on jax 0.8.3, 0.9.2 and 0.10.2; only 0.11 fails (`types.UnionType \| str`, fixed upstream in dynamiqs#1140, unreleased) | widen to `<0.11`; CI tests lowest and highest versions (done, e488441) |
| P0 | No hardware model | only dimensionless two-level, Dicke and Lambda models; no leakage level, no I/Q drive, no GHz / ns units | add a transmon (levels, anharmonicity, I/Q drive, T1/T2 in us, ns time) |
| P0 | No gate tasks | every task is state transfer or charging; gate calibration (X, SX) is the daily work of a pulse engineer | add `GateControl` on propagators / superoperators with leakage-aware average gate fidelity |
| P0 | No bring-your-own Hamiltonian | users must subclass `Model`, and get no physics checks for their own system | add `HamiltonianModel.from_operators` and generic checks, `qenvs validate --model` |
| P1 | RL-library friction | actions not normalised to [-1, 1] (gymnasium warns, SB3 expects it); no vector env | `normalize_actions`, `GymnasiumVectorEnv` |
| P1 | No tutorials | adoption depends on a worked example in the user's own domain | transmon gate tutorial after the transmon lands |
| P2 | CI coverage | only Linux, Python 3.11/3.12, newest versions | add 3.13, macOS, lowest-direct resolution (done, e488441); `py.typed` |
| P2 | Lowest versions untested | `qutip` 5.0 imports `setuptools` at runtime without declaring it | `qutip>=5.1`; lower bounds on every direct dependency (done) |

Out of scope this round (recorded, not started): two-qubit gates (cross-resonance / ECR),
pulse export to qiskit-dynamics signals, import of device properties from a Qiskit
backend, GPU throughput benchmark (shared GPU), measurement-based observations.

## Architecture

Unchanged: `Model` (physics) -> functional `Environment` -> adapters -> registry. All new
work is additive: new models, a new environment class, a new generic validation module,
new wrapper options. No public API is removed or changed in a breaking way.

## Verification plan

Each feature is implemented in new files and reviewed adversarially (physics derivation,
mutation testing in a private copy, API compliance). The lead integrates the wiring
(registry, `__init__` exports, docs navigation), runs the full suite in float32 and
float64, and records the review in `docs/review-log.md`.
