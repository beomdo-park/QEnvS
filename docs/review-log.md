# Review log

QEnvS models are accepted only after adversarial review: independent reviewers try to
refute the physics, re-derive the reference formulas, and mutate the code to see whether
the tests notice. This page records what those reviews found (four rounds so far), the counterexample that
showed each problem, the fix, and the check that now guards against it. Every number below
was reproduced independently of the code under review (numpy or scipy, or QuTiP 5).

## Round 1 (2026-10-13): first review of the model zoo

Four implementers wrote the qubit battery, the Dicke battery, the Lambda system and the
baselines; seven reviewers checked them (two per model, one for the baselines).

| model | verdict | finding | counterexample before the fix | fix and guard |
|---|---|---|---|---|
| Dicke battery | refuted 2/2 | Ergotropy was computed on the full cells+cavity state with $H_B \otimes 1$. For closed dynamics the global state is pure, so this ergotropy equals the energy, and `reward="ergotropy"` duplicated `reward="energy"`. | $N = n = 1$, $g = 0.5$, $t = \pi/2$: energy $0.5\,\omega_0$, reported ergotropy $0.5\,\omega_0$; the reduced cell state is $\mathrm{diag}(1/2, 1/2)$, whose ergotropy is $0$. | `Model.battery_reduced` traces out the cavity; `BatteryCharging` evaluates both quantities on the reduced state. Guarded by `reduced_ergotropy` (closed form, tolerance $10^{-6}$). Reference point: $N = 2$, $g = 0.5$, $t = 1$ gives energy $0.784$ and reduced ergotropy $0.277$. |
| Dicke battery | (same review) | The cavity frequency, $\omega_0 \ne 1$ and the sign of the detuning were never exercised. Four mutations passed every test: cavity term built with $\omega_0$ instead of $\omega_c$, $H_B$ without $\omega_0$, flipped detuning sign, task factory dropping `omega_c`. The ergotropy bug above was likewise invisible to the tests. | All four mutants passed the original 23 tests. | `detuned_jaynes_cummings` (closed form, including $\langle V\rangle = (d/g)P_e$, which is odd in the detuning), `omega0_scaling`, `qutip_detuned`, `reduced_ergotropy`, and a test that every task kwarg reaches the model. A later review re-ran the mutants: all caught. |
| Dicke battery | (same review) | The Fock cutoff for `rwa=False` was not validated, and the default was far from converged at the default coupling bound. | $N = 3$, constant $g = 0.5$, 20 steps of `dt = 0.2`: return $1.7545$ at the default $n_{\max} = n + 4$ against $1.5302$ converged ($n + 24$), 15 percent too high. | Warning in the task description and the model page, the `top_fock` observable (population of the highest Fock level), and `cutoff_convergence` for weak coupling. Refined in round 2. |
| Dicke battery | (same review) | Wording overstated the literature: "charging time $\sim N^{-1/2}$" stated as fact. | Fitted exponent of the first-peak time is $-0.40$ for $N = 1 \dots 48$ in this protocol. | Task description and docs now report the measured exponent and state that the $N^{-1/2}$ law is not established here. |
| Lambda system | refuted 1/2 | The task kwarg `delta_two_photon` could be dropped by the task factory without any test failing. | Mutant passed all 24 tests. | `test_task_factory_forwards_detunings_and_gamma` compares the model drift with $\mathrm{diag}(0, \Delta, \delta)$. The 26 other mutations of the reviewers were caught. |
| Qubit battery | refuted 1/2 | The new module was not wired into the registry and the validation list, so `qenvs.make` and `qenvs validate` did not see it (an integration step, not physics). | `qenvs.make("battery/TwoLevel-v0")` raised `KeyError` on a clean checkout. | Wired in; the CLI now lists and validates every task. All 14 physics mutations of the reviewers were caught. |
| Baselines | 0/1 | `jax.grad` through the open qubit task returned NaN: the Uhlmann fidelity differentiates $\sqrt{\lambda}$ at a zero eigenvalue when the target is pure. | `jax.grad` of the open `qubit/StateTransfer-v0` return: NaN in 0.0.1. | `qenvs.observables.fidelity` evaluates a pure target as $\langle\psi\rvert\rho\lvert\psi\rangle$ (exact and differentiable); after the fix the same gradient is finite. The GRAPE example no longer needs its workaround. |

## Round 2 (2026-10-14): re-review of the Dicke battery

Two reviewers: one re-derived the physics and recomputed every number on the model page;
one mutated the code in an isolated copy (12 requested mutations plus 4 extra).

| verdict | finding | counterexample | fix and guard |
|---|---|---|---|
| physics reviewer: refuted | The remedy "set $n_{\max} \ge n + 24$" for `rwa=False` was a generalisation from $N = 3$. | $N = 6$, constant $g = 0.5$: return $2.7524$ at $n + 24$ against $2.6043$ at $n + 56$ (top-Fock population $6\times10^{-3}$ at $n + 24$). | The model page lists the required cutoff per $N$ and states the criterion: raise `n_max` until `top_fock` stays below about $10^{-4}$. Measured weak-coupling errors of the default cutoff ($g = 0.1$): $8.5\times10^{-6}$, $1.1\times10^{-4}$, $5.2\times10^{-4}$, $4.2\times10^{-3}$, $4.3\times10^{-2}$ for $N = 2, 3, 4, 6, 8$. |
| mutation reviewer: not refuted | All 16 mutations were caught, but the jump operators were caught only by the QuTiP cross-checks, and the detuning sign without QuTiP only by a structural test. | Reversed cavity-loss or emission operator and a wrong dephasing prefactor: caught only with QuTiP installed. | `collective_dissipators`: closed-form Lindblad dynamics for $N = 1$ ($e^{-\kappa t}$, $e^{-\gamma t}$, $e^{-\gamma_\phi t/2}$), tolerance $2\times10^{-6}$; the detuning sign is pinned by $\langle V\rangle = (d/g)P_e$. Re-checked with runtime mutants: all five caught without QuTiP. |
| mutation reviewer | Thin float32 margins (1.4x to 2.4x) on three checks. | `detuned_jaynes_cummings` $7.2\times10^{-7}$ against $10^{-6}$ in float32. | Tolerances of these three checks widened (they measure roundoff or solver accuracy, and a physics bug moves them by orders of magnitude more); every check passes in float32 and float64. |
| physics reviewer | Ergotropy is a sparse reward for Tavis-Cummings charging from a Fock state. | $N = 2$, constant $g = 0.3$ for 6 steps: ergotropy return exactly $0$ with an exactly zero gradient. | Documented on the model page (use `reward="energy"` for early training). |

The decision rule for two votes is that a tie is decided by the maintainer: here the code
was correct and the refuted claim was documentation, so the documentation was fixed and the
release went ahead.

## Round 3 (2026-10-14): review of the battery charging benchmark

Two reviewers: one checked the optimality argument and every published claim against the
raw results; one reviewed the benchmark code and reproduced numbers independently.

| verdict | finding | counterexample | fix and guard |
|---|---|---|---|
| code reviewer: refuted | Argument-name collision. `run_ppo` called its discount factor `gamma`, so the Dicke collective emission rate `gamma=0.05` was captured by it: PPO trained on the environment without emission and with discount 0.05, and its pulse was then evaluated on the correct environment, which hid the mismatch. | First-step energy for $g = -0.5$, $N = 4$, $\kappa = 0.1$: $0.154791$ inside `run_ppo`, which is the value without emission; the intended environment gives $0.151771$. | Discount renamed to `discount`. `test_example_arguments_never_capture_task_kwargs` fails if any example argument shares a name with a registered task kwarg, and the benchmark now raises if a method's own return differs from the replayed energy by more than $10^{-3}$. The whole table was re-run on the fixed commit: the open Dicke PPO values moved from 0.6623 to 0.6684 ($N = 2$) and from 0.5840 to 0.5949 ($N = 4$). |
| claims reviewer: refuted | The stated gap of PPO to the optimum ("0.2 to 1.6 percent") matched neither the means nor the worst seed. | Means: 0.004 percent (qubit) to 0.76 percent ($N = 8$); worst seed 1.8 percent. | The text quotes the mean range and the worst seed separately. |
| claims reviewer | Optimality argument re-derived and checked numerically: $[\hat N_{\mathrm{exc}}, V] = 0$, the initial state is an $\hat N_{\mathrm{exc}}$ eigenstate, every pulse area is reachable through clipping and float32 casting, $E(-\theta) = E(\theta)$. | Independent float64 optima $0.888888889$, $0.861460682$, $0.878688038$, $1$ agree with the published ones. | No change needed; the page now separates what is proven (the reduction, for every $N$) from what is numerical (the maximum for $N = 4, 8$). |
| claims reviewer | "The single pulse beats GRAPE on open tasks" was an artefact of stopping GRAPE at 300 steps. | $N = 4$ open: GRAPE gives $0.5978$ after 300 steps and $0.6018$ after 1500, the single-pulse value. | GRAPE runs 1500 steps in the benchmark. It then reaches the single-pulse value for $N = 2$ on every seed and for $N = 4$ on one seed of three (the other two stop at 0.6002). |
| claims reviewer | Interpretation of the late-starting pulse and of energy against ergotropy needed qualifiers. | Late start wins by only $0.0024$ for $N = 4$ open and is caused by the emission rate $\gamma$ (with $\gamma$ alone the best start is step 11, with $\kappa$ alone step 0); for $N = 2$ open the best start is step 0. The energy and ergotropy optima coincide for $N = 2$ and differ for $N = 4$ and $8$. | Both statements are qualified on the benchmark page. |

## Round 4 (2026-10-14): practitioner features

Four reviewers checked the transmon gate tasks (physics; mutation testing), the
bring-your-own-Hamiltonian checks, and the Gymnasium wrappers. All four refuted at first;
the findings were fixed and the previously surviving mutants re-run. A read-only QA sweep
of the rest of the repository followed.

| area | finding | counterexample | fix and guard |
|---|---|---|---|
| Transmon (physics review) | The info observables of `GateControl` are documented as the current map applied to the initial state, but no test fixed the orientation of the closed-system propagator. | Replacing `U @ psi0` by `U.T @ psi0` passed every test; on a random 3-level pulse it changes $p_1$ from 0.158 to 0.260. | Tests against an independent numpy propagator (closed) and Liouvillian (open), and the check `info_observables_orientation`. |
| Transmon (mutation review) | The model page claimed that fidelity and leakage are unchanged under a row/column-stacking swap of the superoperator. True for leakage and for real targets only: the swap maps $U$ to $\bar U$. | Fidelity against SX changes under the swap (0.554 against 0.354 on a random unitary), while X, Y, Z and H do not ($\bar Y = -Y$ is a global phase). | Docs corrected; a test pins the difference. The SX target and its adjoint are now told apart by the check `sx_orientation`. 11 of the 12 requested mutants were caught at first; the twelfth is mathematically equivalent. |
| Custom models | A tolerance of the battery-consistency check was 10x tighter than the norm check it stands for, so valid models failed on long rollouts; and 13 of 32 mutants of the generic checks survived. | Built-in `TwoLevel()` at horizon 400 failed while every other check passed. | Rollout tolerances scale with the integrated time; near-threshold tests kill all 24 mutants defined in the fix. The self-consistency QuTiP check is now named `qutip_solver_cross_check` and states that it does not verify the physics; `--reference` compares against an independent model. |
| Gymnasium | The wrappers returned read-only numpy views of jax arrays. | `gymnasium.wrappers.vector.ClipReward` failed with "assignment destination is read-only". | Writable copies; tests run the official vector wrappers, also on gymnasium 1.0.0 with jax 0.4.36. |
| QA sweep | Six test assertions used `.isket` / `.isdm` without calling them, so they were always true. Fixing them exposed one stale expectation: the Lambda task keeps its pure target as a ket for open systems since 0.1.0, and the test still expected a density matrix. | `bool(state.isket)` is `True` for any state. | Assertions call the methods; the expectation is updated. 27 other findings (stale text, wrong file paths, inconsistent numbers, a missing import in a tutorial snippet) were fixed. One finding was rejected after checking: the GitHub releases do carry the PEP 740 attestation files. |

## How to reproduce

* The current validation numbers are regenerated by every documentation build:
  [Validation report](validation.md), or locally `qenvs validate --markdown`.
* The counterexamples above are small enough to recompute with plain numpy; the model
  pages list the exact parameters.
