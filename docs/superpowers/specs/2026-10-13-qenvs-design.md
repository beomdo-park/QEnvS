# QEnvS design spec (2026-10-13)

## Intent

Build an open-source library of **physically validated open-quantum-system
control models** (quantum batteries, qubit state transfer, later thermal
machines) exposed through a uniform environment interface, so that setting up
an RL / optimal-control study costs minutes instead of weeks, and so that the
underlying physics is reused rather than re-derived per paper.

Primary user: the author's own Dicke quantum-battery optimisation work (K-Quantum
Hub / IonQ proposal). Secondary users: RL-for-quantum-control researchers who
currently hand-write JAX environments per paper (e.g. ICML 2025 RL4qcWpc).

Success criteria for v0.1:

* `pip install qenvs` works on Linux/macOS CPU; GPU via the user's own jax install.
* Three validated models (two-level battery, Dicke battery, Lambda transfer),
  each with analytic tests and a QuTiP cross-check.
* One RL baseline and one gradient baseline reproduce a known optimum
  (pi-pulse; collective charging advantage) from the docs.
* Zero paid infrastructure.

## Landscape (why this does not already exist)

* Circuit-level RL environments exist (qiskit-gym, qgym) but no Hamiltonian/
  open-system environment library.
* Optimal-control suites (qutip-qoc, QuOCS, C3, qopt) optimise pulses for one
  model; they do not provide reusable tested models or an RL interface.
* Quantum-battery code is per-paper QuTiP scripts; the only "package"
  (openquantumbattery) is a Streamlit demo with no Dicke model or control API.
* Simulator engines are mature (dynamiqs, qiskit-dynamics, QuTiP 5); the gap is
  the model + environment layer above them.

## Architecture (four layers)

```
qenvs.models      Model(eqx.Module): drift, controls, jump_ops, initial_state,
                  observables, hamiltonian(), pwc_hamiltonian(), evolve()
qenvs.envs        Environment (functional, Jumanji-style): reset(key) /
                  step(state, action) -> (EnvState, TimeStep); rollout() via scan
qenvs.wrappers    GymnasiumEnv (numpy I/O, jit-compiled step) for SB3/CleanRL
qenvs.registry    make("TaskId-vN", **kwargs); versioned ids
qenvs.observables energy, ergotropy, fidelity (thermodynamics helpers)
```

Decisions:

* **Engine: dynamiqs** (Apache-2.0, JAX, Lindblad + SME, differentiable,
  batching). Pin `jax<0.8` until dynamiqs supports jax 0.8+.
* **Functional API first, Gymnasium second.** Functional gives vmap/jit/grad;
  Gymnasium is a thin adapter so classic libraries work.
* **Dense reward = fidelity gain per step** for state transfer, so the return is
  the final fidelity minus the initial. Battery tasks will use ergotropy /
  power rewards (phase 1).
* **Validation = tests, not docs.** A model is merged only with closed-form
  checks. `tests/` doubles as the physics certificate.
* **Observation = flattened state + time.** Full-state observation keeps v0
  simple; partial/measurement-based observations are a later task variant.

## Distribution and infrastructure (all free tiers)

| Need | Choice | Cost | Alternative rejected |
|---|---|---|---|
| Package index | PyPI, Trusted Publishing (OIDC) | free, no tokens | API tokens (rotation burden) |
| CI | GitHub Actions (public repo: unlimited) | free | none |
| Docs | mkdocs-material on GitHub Pages | free | ReadTheDocs (also free, but second service) |
| Versioning | hatch-vcs from git tags | none | manual version bumps |
| DOI | Zenodo GitHub integration | free | none |
| Coverage | none in v0 | none | Codecov (free for OSS, add later if useful) |
| Deps bot | Dependabot, monthly, actions only | free | none |

## Licensing

Apache-2.0 (matches JAX, dynamiqs, Jumanji, qiskit-gym ecosystem).

Reuse policy:

* dynamiqs (Apache-2.0): tooling conventions adapted; attributed in NOTICE.
* Jumanji (Apache-2.0): API design pattern only; attributed in NOTICE.
* QuTiP (BSD-3): used as test oracle; attributed in NOTICE.
* RL4qcWpc (no LICENSE file): **no code copied**; ideas cited in docs only.

## Risks

* dynamiqs/jax version drift (already hit once). Mitigation: pin + CI matrix.
* Small community. Mitigation: author is the first user; JOSS paper for citability.
* Physics correctness is the whole value. Mitigation: analytic tests + QuTiP
  cross-checks are required, not optional.
