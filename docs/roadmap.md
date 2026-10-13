# Roadmap

| Phase | Deliverable | Validation gate | Status |
|---|---|---|---|
| 0 | Package skeleton, CI, PyPI trusted publishing, docs, `qubit/StateTransfer-v0` | Rabi / pi-pulse / decay / ergotropy tests green | done (2026-10-13) |
| 1 | `battery/TwoLevel-v0`: open qubit battery, thermal bath, energy / ergotropy / power rewards | analytic steady state, QuTiP cross-check | done (2026-10-13) |
| 2 | `battery/Dicke-v0`: N two-level systems + cavity (Dicke, Tavis-Cummings), permutation-symmetric solver for N up to ~50 | collective charging scaling vs. Ferraro et al. 2018; QuTiP cross-check (PIQS later) | done (2026-10-13), collective dissipators only |
| 3 | `transfer/LambdaSTIRAP-v0`: three-level state transfer with spontaneous emission | adiabatic STIRAP limit | done (2026-10-13) |
| 4 | Baselines: PPO (JAX, vmapped), SAC via SB3, GRAPE via `jax.grad`; benchmark table in docs | reproducible seeds, CI smoke run | done (2026-10-13), smoke-scale |
| 5 | v0.1.0 on PyPI, Zenodo DOI, JOSS paper draft | | |

Out of scope for v0.x: pulse-level hardware backends, stochastic master equations
(measurement feedback), many-body models beyond permutation symmetry.
