# Roadmap

| Phase | Deliverable | Validation gate | Status |
|---|---|---|---|
| 0 | Package skeleton, CI, PyPI trusted publishing, docs, `qubit/StateTransfer-v0` | Rabi / pi-pulse / decay / ergotropy tests green | done (2026-10-13) |
| 1 | `battery/TwoLevel-v0`: open qubit battery, thermal bath, energy or ergotropy reward (charging power reported in `info`) | analytic steady state, QuTiP cross-check | done (2026-10-13) |
| 2 | `battery/Dicke-v0`: N two-level systems + cavity (Dicke, Tavis-Cummings), permutation-symmetric solver for N up to ~50 | collective charging scaling vs. Ferraro et al. 2018; QuTiP cross-check (PIQS later) | done (2026-10-13), collective dissipators only |
| 3 | `transfer/LambdaSTIRAP-v0`: three-level state transfer with spontaneous emission | adiabatic STIRAP limit | done (2026-10-13) |
| 4 | Baselines: PPO (JAX, vmapped), PPO via SB3 quickstart, GRAPE via `jax.grad`; benchmark tables in docs (qubit transfer, battery charging with a provable optimum) | reproducible seeds, CI smoke run | done (2026-10-13), smoke-scale |
| 5 | v0.1.0 on PyPI, Zenodo DOI | release workflow, attestations | done (2026-10-14) |
| 6 | Practitioner features: transmon gate tasks in hardware units (`transmon/XGate-v0`, `transmon/SXGate-v0`), leakage-aware gate fidelity, bring-your-own Hamiltonian with generic checks, Gymnasium vector env, tutorial, jax up to 0.10, CI on Python 3.11 to 3.13, macOS and lowest dependencies | gate-fidelity closed forms, DRAG coefficient, QuTiP cross-check; mutation-tested generic checks | done (2026-10-14) |
| 7 | JOSS paper draft; two-qubit gates (cross-resonance); local dissipation for Dicke (PIQS-style); pulse import and export for hardware toolchains | | next |

Out of scope for v0.x: pulse-level hardware backends, stochastic master equations
(measurement feedback), many-body models beyond permutation symmetry.
