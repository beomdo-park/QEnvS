# Changelog

All notable changes to QEnvS are listed here. Task ids carry their own version
suffix (`-vN`); a released task is never changed in place.

## [Unreleased]

### Changed

- Scope public physics-validation claims to the checks available for each model and record
  the repository-backed round 6 review findings and statistical limitations.
- Add serial float32 validation, weekly newest-dependency tests and pull-request documentation
  builds to CI. Documentation deployment remains restricted to the main branch.

## 0.3.2a1 (candidate, unreleased)

- Document the maintainer-approved additive API exception for this alpha only.
- Check subsystem rewards and trajectory endpoints, batching and gradients in
  both CI precision modes with explicit float32 tolerances.

- Custom Hamiltonian models can declare battery subsystems and a local Hamiltonian,
  with reduced-state work and a consistent full-space energy operator.
- New endpoint trajectory diagnostics preserve physical action precision and expose
  explicit solver settings without changing environment rollout or robust metrics.
- Add a small auditable control example with all-candidate replay, fixed warmup,
  optimizer traces and source provenance. No released task behavior changes.

## [0.3.0] - 2026-10-16

### Added

- `transmon/CRGate-v0`: two-qubit cross-resonance gate on two coupled multi-level transmons
  (`qenvs.models.two_transmon.TwoTransmon`: Duffing oscillators with exchange coupling, CR
  drive on the control and a cancellation drive on the target, optional T1/T2), target
  ZX(pi/2) on the computational subspace with leakage, and a calibrated flat-top reference
  pulse. `GateControl` and `qenvs.gates` gain a backward-compatible `subspace` argument.
- `battery/DickeLocal-v0`: the Dicke battery with local emission, dephasing and pumping on
  every cell (`qenvs.models.dicke_local.DickeBatteryLocal`, permutation-invariant method of
  Shammah et al. 2018, all spin sectors), checked against a brute-force 2^N simulation to
  1e-8. `battery/Dicke-v0` is unchanged.
- `qenvs.pulses`: an immutable `Pulse` record with units and provenance, lossless JSON and
  NPZ round trips, exact resampling of piecewise-constant pulses to an AWG sample period,
  padding, rad/ns and MHz conversion; `Transmon.from_properties` reads T1, T2, frequency
  and anharmonicity from a plain dict with explicit units or Qiskit-style SI values.
- Realistic RL conditions (design: `docs/superpowers/specs/2026-10-15-realistic-rl-conditions-proposal.md`):
  `qenvs.wrappers.ParameterRandomization` (episode-fixed model parameters drawn from
  `Normal`, `Uniform` or `TruncatedNormal` offsets with support checks at construction, optional
  oracle observation), `qenvs.wrappers.ControlErrors` (gain and offset errors per episode,
  relative noise per step, `info["clip_fraction"]`), `qenvs.wrappers.ObservationModel` and
  `qenvs.make(..., observation=, shots=, diagnostics=, observables=)` (modes `state`,
  `expectations`, `correlations`, `time`; exact finite-shot measurement in the eigenbasis of each
  observable; exact values only in `info["diagnostics"]` when asked for), and
  `qenvs.robust.evaluate` and `stability` (mean, CVaR of the lowest 10 percent, percentile
  bootstrap intervals, episode return reported separately). Every random number comes from a
  fixed key tree (`qenvs.keys`). The model contract gains `parameter_specs`, `parameter_values`,
  `with_parameters`, `check_parameters` and `correlation_observables` (all additive), and
  `HamiltonianModel.from_operators(..., parameters=...)` adds randomisable terms
  `H0 + sum_k p_k O_k`. `qenvs.wrappers` no longer needs gymnasium for these classes.
- Release safety rails: an API gate (`tests/test_api_stability.py`, griffe diff against the
  last release tag), a behaviour snapshot of every released task
  (`tests/golden/tasks.json`), and `RELEASING.md` with a release-candidate process.

### Changed

- `GymnasiumEnv` and `GymnasiumVectorEnv` derive episode keys as
  `fold_in(PRNGKey(seed), i)`; in the vector environment the `j`-th episode of sub-environment
  `b` is episode `i = j * num_envs + b`. Environments with random resets therefore see different
  samples than in 0.2 (sub-environment `b` used to follow `GymnasiumEnv(seed=seed + b)`). The
  built-in tasks reset deterministically and are unchanged (behaviour snapshot).

### Deprecated

- `qenvs.toolkit.UNITS`, removed in 0.2.0 without a deprecation period, is restored and
  emits a `DeprecationWarning`; it will be removed in 0.4. Units are per task: use
  `qenvs.registry[task_id].units`.

## [0.2.0] - 2026-10-15

### Added

- `transmon/XGate-v0` and `transmon/SXGate-v0`: single-qubit gate synthesis on a
  multi-level transmon (`qenvs.models.transmon.Transmon`: anharmonicity, I/Q drive, RWA)
  in hardware units (time in ns, angular frequency in rad/ns, kwargs in GHz and
  microseconds), closed or with T1/T2. Gaussian DRAG pulse helpers with the DRAG
  coefficients confirmed numerically, and a DRAG reference pulse per task.
- `qenvs.envs.gate.GateControl`: gate environment on the accumulated propagator (closed)
  or superoperator (open); reward is the gain of the leakage-aware average gate fidelity.
- `qenvs.gates`: X, Y, Z, H, SX (IBM convention) and Rz targets; average gate fidelity
  (Pedersen et al. 2007) and leakage for unitaries; process fidelity, average gate
  fidelity and leakage for leaky channels (Wood and Gambetta 2018).
- `qenvs.models.custom.HamiltonianModel.from_operators`: bring your own Hamiltonian from
  NumPy, JAX, dynamiqs or QuTiP operators, validated at construction.
- `qenvs.validation.generic.model_checks` and `qenvs validate --model module:attr`
  (JSON error code `invalid_model`): physics and numerics checks for any model.
- `GymnasiumEnv(normalize_actions=True)` and `qenvs.wrappers.vector.GymnasiumVectorEnv`
  (batched gymnasium VectorEnv on one `jit(vmap(step))`).
- Guides: bring your own Hamiltonian, RL libraries. Model page for the transmon.
- Per-task units metadata in the registry (`TaskSpec.units`).
- `qenvs run --policy reference` runs a task's registered reference solution (pi pulse,
  STIRAP pair, Gaussian DRAG); `qenvs info` reports units and whether one exists.
- `Environment.info(state)`: public access to the diagnostics of a state.
- Tutorial: an X gate on a transmon (`examples/tutorial_transmon_x_gate.py`).
- 14 transmon validation checks (67 in total), all passing in float32 and float64.
- `examples/benchmark_battery.py`: charging benchmark on the battery tasks with a
  provable optimum for closed resonant variants, a single-pulse physical baseline,
  GRAPE, PPO and random pulses; results in `docs/benchmarks.md`.
- `run_grape` returns the optimised pulse (`best_pulse`) and `run_ppo` the deterministic
  policy pulse (`policy_pulse`).
- Review log page with the counterexamples found in adversarial review and the checks
  that now guard against them.

### Changed

- Dependency range widened: `jax>=0.4.36,<0.11` (was `<0.8`); jax 0.8 to 0.10 are tested.
  Lower bounds added for `equinox`, `numpy` and the dev tools; `qutip>=5.1` (5.0 imports
  setuptools at runtime without declaring it).
- CI tests Python 3.11, 3.12, 3.13 on Linux, Python 3.12 on macOS, and the lowest allowed
  versions of every direct dependency.
- The package ships a `py.typed` marker.

### Fixed

- Six test assertions were always true (`.isket` / `.isdm` not called); they now check,
  which exposed and corrected a stale expectation in the Lambda task test.
- `qenvs run` reported integer model parameters such as `levels` as floats.
- `run_ppo`: the discount factor was named `gamma`, which captured the `gamma` emission
  rate of `battery/Dicke-v0` so that PPO trained on the wrong environment. Renamed to
  `discount`; a test rejects any example argument that shares a name with a task kwarg.

## [0.1.0] - 2026-10-14

First release with a model zoo.

### Added

- `battery/TwoLevel-v0`: open qubit battery (`QubitBattery`) with a thermal bath
  (T1 at occupation `n_th`) and pure dephasing; energy or ergotropy reward.
- `battery/Dicke-v0`: collective battery (`DickeBattery`), N cells and one cavity
  mode in the permutation-symmetric subspace, Tavis-Cummings (`rwa=True`) or full
  Dicke (`rwa=False`) coupling, collective cavity loss, emission and dephasing.
  Energy and ergotropy are evaluated on the reduced cell state.
- `transfer/LambdaSTIRAP-v0`: three-level Lambda system (`LambdaSystem`) with pump
  and Stokes controls and spontaneous emission; `stirap_pulses` helper.
- `BatteryCharging` environment shared by all battery models, and the
  `Model.battery_reduced` hook for composite batteries.
- 44 new validation checks (53 in total) with closed-form references, physical
  constraints, solver convergence and QuTiP cross-checks; every check passes in
  float32 and float64. `qenvs validate` reports skipped checks (QuTiP not
  installed) as SKIP.
- Baselines: `examples/grape_jax.py` (open-loop Adam through `env.rollout`),
  `examples/ppo_jax.py` (vectorised PPO), `examples/benchmark.py`, and the
  benchmark page.
- Model documentation pages with Hamiltonians, assumptions, validity limits and
  measured validation tables.
- Validation report page regenerated by every documentation build, and
  `qenvs validate --markdown`.
- Logo, favicon and architecture diagram.

### Fixed

- Fidelity to a pure target is now `<t|rho|t>`, which is exact and has a finite
  gradient on density matrices (the general square-root formula returned NaN
  gradients for open systems).
- `qenvs run` works for battery tasks.

## [0.0.1] - 2026-10-13

Skeleton release: `qubit/StateTransfer-v0`, functional environment API,
Gymnasium adapter, JSON command-line interface, physics validation framework.

[0.3.0]: https://github.com/beomdo-park/QEnvS/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/beomdo-park/QEnvS/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/beomdo-park/QEnvS/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/beomdo-park/QEnvS/releases/tag/v0.0.1
