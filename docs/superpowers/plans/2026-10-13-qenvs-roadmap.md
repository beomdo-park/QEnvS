# QEnvS implementation plan

Phase 0 is complete (this commit). Each later phase is one PR-sized unit with a
validation gate; do not start a phase before the previous gate is green in CI.

## Phase 0: skeleton (done 2026-10-13)

- [x] src layout, hatchling + hatch-vcs, ruff, pytest, pre-commit
- [x] ci.yml (ruff, codespell, pytest 3.11/3.12, build), release.yml (PyPI OIDC),
      docs.yml (mkdocs -> Pages), dependabot
- [x] `Model`, `TwoLevel`, `StateTransfer`, `GymnasiumEnv`, registry
- [x] analytic tests: Rabi, pi-pulse, T1 decay, T_phi dephasing, ergotropy,
      gymnasium `check_env`, jit/vmap/grad smoke
- [ ] manual one-time steps (owner): PyPI pending publisher, Zenodo toggle

## Phase 1: `battery/TwoLevel-v0`

1. `models/two_level.py`: thermal bath jump ops (`n_th`), `battery_hamiltonian`.
2. `envs/battery.py`: `BatteryCharging` env; reward options `energy`, `ergotropy`,
   `power` (gain per dt); terminal bonus optional.
3. Tests: steady-state population `n_th/(2 n_th + 1)`; ergotropy monotone under
   resonant drive without loss; QuTiP `mesolve` cross-check on a random pulse.
4. Port the author's single-qubit charging scripts (`250809_SingleQubit_QB_*`)
   as an example notebook.

## Phase 2: `battery/Dicke-v0`

1. `models/dicke.py`: N TLS + single cavity mode, Tavis-Cummings and Dicke
   (counter-rotating) variants; controls = cavity drive amplitude, coupling g(t),
   detuning. Permutation-symmetric Liouvillian for open N (PIQS algorithm,
   re-implemented in JAX; cite Shammah et al. 2018, BSD-3 via QuTiP).
2. Tests: closed N=1 reduces to Jaynes-Cummings Rabi; charging power scaling
   with N matches Ferraro et al. 2018 regime; PIQS cross-check for N=4, open.
3. Port `250730_Dicke_TLS_Charging_optimization` SAC env as a task variant.

## Phase 3: `transfer/LambdaSTIRAP-v0`

1. `models/lambda_system.py`: three-level Lambda, pump/Stokes controls,
   spontaneous emission from the excited level.
2. Tests: counter-intuitive Gaussian pulse pair reaches transfer > 0.99 in the
   adiabatic limit; intuitive order fails; emission-free dark state invariant.

## Phase 4: baselines and benchmark page

1. `examples/ppo_jax.py` (PureJaxRL-style, Apache-2.0 attribution) vmapped over
   1024 envs; `examples/grape_jax.py` using `env.rollout` + optax.
2. Docs page with a table: task, method, seeds, mean final metric, wall-clock.
3. CI smoke: 200-step PPO run must not crash (no accuracy assertion).

## Phase 4b: agent adapters (optional)

* MCP server as a thin adapter over `qenvs.toolkit` (extra `[mcp]`), only if
  there is demand; the plain-dict toolkit already keeps this a ~40-line change.
* `docs/agent-guide.md`: how an agent discovers, runs and interprets results.

## Phase 5: release

1. `git tag v0.1.0`; verify PyPI + GitHub release + Pages.
2. Enable Zenodo GitHub integration, add DOI badge.
3. JOSS paper draft (`paper/paper.md`), target submission after phase 4.

## Owner one-time setup (cannot be automated)

* PyPI: account -> Publishing -> add pending publisher
  (project `qenvs`, owner `beomdo-park`, repo `QEnvS`, workflow `release.yml`,
  environment `pypi`).
* GitHub: Settings -> Environments -> create `pypi` (optionally require reviewer).
* Zenodo: log in with GitHub, toggle `beomdo-park/QEnvS`.
