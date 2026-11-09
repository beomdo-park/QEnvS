# Review log

QEnvS models are accepted only after adversarial review: independent reviewers try to
refute the physics, re-derive the reference formulas, and mutate the code to see whether
the tests notice. This page records what those reviews found (six rounds so far), the counterexample that
showed each problem, the fix, and the check that now guards against it. Physics references
use independent formulas or simulators where noted; statistical coverage measurements and
API regression checks have the scope and limitations stated in each round.

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

## Round 5 (2026-10-15): 0.3 features (pulses, cross-resonance, local dissipation)

Five reviewers checked the pulse I/O module (one review), the cross-resonance task and the
Dicke battery with local dissipation (physics review and mutation review each). The pulse
module was not refuted; both reviews of each new task refuted at first. Fixes were made by
separate agents, and the lead re-injected each bug behind a major finding into a copy of
the tree to confirm that the new test now fails.

| area | finding | counterexample | fix and guard |
|---|---|---|---|
| Cross-resonance (physics review) | The defaults table called $\Delta = f_c - f_t = -0.1$ GHz "inside the straddling regime $\alpha_c < \Delta < 0$". Straddling is $0 < \Delta < -\alpha_c$, so the default is not straddling. The reference pulse also failed on the straddling side. | First-order ZX coefficient $\lvert\alpha_c/(\Delta(\alpha_c+\Delta))\rvert$: 7.7 per GHz at $-0.1$ GHz against 14.2 per GHz at $+0.1$ GHz. `reference_actions` at `detuning_ghz=0.1` raised "no flat-top width reaches a pi/2 conditional rotation". | Docs corrected; the default stays and the reason is stated. The reference takes the amplitude sign from the first-order $\nu_{ZX}$ (F = 0.9963 at $+0.1$ GHz). Tests `test_first_order_zx_straddling_regime`, `test_reference_positive_detuning_uses_negative_amplitude`. |
| Cross-resonance (mutation review) | Dropping `subspace` from the open-system leakage of `GateControl` passed every test. | Reference pulse with T1 = T2 = 100 us: leakage 0.124 instead of 1.41e-4. | `test_open_env_leakage_uses_the_computational_subspace`; with the bug re-injected it fails (0.124 > 3e-4). The operand order of `to_qubit_frame` (unitary and superoperator) is pinned by `test_to_qubit_frame_operand_order`. 56 of 60 mutants were caught before the fix. |
| Cross-resonance (rejected) | A reviewer proposed the cancellation ratio $J/\Delta$ for `levels=2`, on the grounds that the IX term is $-J\Omega/\Delta$ for qubits. | For qubits there is no first-order IX term: $\nu_{IX} = -J\Omega/(\Delta+\alpha_c) \to 0$ as $\alpha_c \to -\infty$. With ratio 0 the IX angle is about 0; with $J/\Delta$ it is large. | `levels=2` uses ratio 0; its reference reaches F = 0.9948 (the three-level ratio gave 0.9644 on two levels). `test_reference_two_level_model`. |
| Local dissipation (both reviews) | The default Fock cutoff $n_{\max} = n_{\mathrm{photons}} + N$ was documented as exact with local pumping. Pumping is a continuous source: a cell can emit into the cavity and be pumped again, so no finite cutoff is exact. | $N = 1$, $n_{\mathrm{photons}} = 0$, pumping 0.3, $g = 0.5$ for 40 steps: energy 30 to 50 percent off at the default cutoff. | With pumping, `n_max` must be given (error at construction); `info["top_fock"]` is the convergence diagnostic. Convergence is measured separately from the brute-force gate: energy change from $n_{\max}$ to $n_{\max}+4$ is 6.7e-2 (2 to 6), 1.9e-3 (4 to 8), 1.1e-5 (6 to 10). With the guard re-injected, `test_pumping_requires_an_explicit_cutoff` and `test_task_default_contract` fail. The task default has no pumping, where the cutoff is exact. |
| Local dissipation (mutation review) | The weight of the symmetric sector (`symmetric` in `info`) was untested. | Dropping the last basis index of the $j = N/2$ projector survived all 10 checks. | `brute_force_agreement` compares the symmetric weight with the brute-force state and the projector trace with $(N+1)(n_{\max}+1)$. Caches are read-only. Thin float32 tolerances were widened; the x64 1e-8 brute-force gate is unchanged. |
| Local dissipation (docs) | The cost was described as polynomial in $N$; only the Liouvillian dimension is. Gradient memory was not documented. | `jit(grad(rollout))` peak memory 8.8 GB at $N = 4$, horizon 20, and 39.5 GB at $N = 5$, horizon 20. | Docs state the $O(4^N)$ state, ergotropy and observation cost and the memory table. The defaults (local emission and dephasing 0.05, no pumping) are pinned by `test_task_default_contract` and described as a benchmark setting, not a standard experimental value. |
| Pulses | Not refuted; 40 of 43 mutants caught. A time-reversing `to_actions` survived because the resampling gate built both runs from it. | `samples[::-1]` passed all 27 tests. | The gate's reference is built from the raw actions, and `to_actions` is asserted equal to the input. Also fixed: stale `unpadded_n_samples` after resampling, complex input silently truncated, the padding caveat for open systems (1.2e-4 at T1 = 30 us, T2 = 25 us). |

## Round 6 (2026-10-15): realistic RL conditions

The parameter-randomisation, control-error, finite-shot observation and robust-evaluation
implementation in commit `598d66c` received an adversarial API and statistics review. The
private mutation patches were not retained, so the repository does not claim a reproducible
total mutation count for this round. The table records the surviving gaps and the regression
guards that are present in the released tree.

| area | finding | counterexample | fix and guard |
|---|---|---|---|
| Parameter scope | CR detuning and Dicke `omega0` cannot use the generic episode-randomisation contract without changing the target. | CR detuning changes the episode frame while the target remains in the nominal end frame, leaving an uncorrectable phase. Dicke `omega0` is also the energy unit, so sampling it rescales the reward and evaluation metric. | The parameters are refused with the physical reason. `test_unsupported_parameters_are_refused_with_reason` pins the refusal and `test_neighbours_of_the_refused_parameters_stay_supported` confirms that CR single-transmon detuning and Dicke `omega_c` remain available. |
| Differentiation | `DickeBatteryLocal.ergotropy_from_blocks` converted `omega0` through host NumPy, which could reject a traced value. | Differentiating the fixed-state ergotropy with respect to `omega0` failed instead of returning the expected linear derivative. | JAX multiplication keeps `omega0` traced. `test_ergotropy_from_blocks_accepts_a_traced_omega0`, `test_ergotropy_from_blocks_is_differentiable_in_omega0` and the float32 comparison guard it. |
| Public API and sampling dtype | Removing wrapper names from `qenvs.wrappers.__all__` or ignoring a distribution's requested dtype survived the earlier tests. | The objects remained importable internally while public star imports lost them; a float64 request could silently return float32. | `test_public_exports_of_qenvs_wrappers` and `test_samples_have_the_requested_dtype` pin both contracts. |
| Tail statistics | A floating-point tail-count formula and a bootstrap implementation that averaged the tail directly were not tied to the public CVaR definition. | Boundary sizes such as $n=1024$ require exactly 103 samples at level 0.1, and an independently replayed bootstrap could disagree with the reported CVaR interval. | `test_tail_count_is_exact_at_round_n` uses integer arithmetic at boundary sizes. `test_bootstrap_cvar_uses_the_cvar_definition` replays the same resamples through the public point estimator. |
| Interval claim | Passing the predeclared coverage gate did not establish calibrated 95 percent CVaR intervals. | With $n=1024$, 2000 bootstrap resamples and 1000 repetitions, observed CVaR coverage was 0.932 for Uniform(0, 1) and 0.944 for the fidelity-shaped distribution. | The guide calls the interval approximate and slightly narrow, and limits the evidence to the two tested distributions. The coverage test and warning say "checked", not "validated". |

## Alpha follow-up (2026-10-19): subsystem rewards and endpoint diagnostics

The lead reviewed the alpha changes and prior release history directly, as requested
by the maintainer. Existing focused float64 tests did not cover the new APIs in the
float32 CI job. Five analytic cases now check Bell-state local rewards, nonadjacent
subsystem embedding and partial trace, endpoint times, jit, vmap and smooth energy
gradients in both precision modes. Local results: five cases passed in float32;
49 related tests passed in float64, including the API compatibility gate. Tolerances
were declared before execution, without relaxing existing float64 tests.

The fixes-only patch policy also needed an explicit exception for the requested
0.3.2a1 additive APIs. RELEASING.md records the alpha-only exception. These local
results do not establish full platform or lowest-dependency compatibility; the
hosted matrix and documentation build must pass before tagging.

## Round 7 (2026-10-22): 0.3.2 release candidate

Two independent reviews of the 0.3.2 changes, physics first and then the public API with
mutation testing, found no blocker. The API diff against v0.3.0 (griffe) has no breaking
change and the released task snapshot is unchanged. Every surviving mutant in the table was
injected again after the fixes and now fails a test: the two physics-round mutants (row 2)
and the six API-round mutants (rows 3 to 7). The patches are described here rather than kept
as files. The alpha-only API exception above was replaced by the versioning rule in
[stability](stability.md): a patch release may carry backward-compatible additions.

| area | finding | counterexample | fix and guard |
|---|---|---|---|
| Precision claim | The `rollout_metrics` docs said float64 actions are always kept. | With `jax_enable_x64` off (the JAX default), JAX converts float64 actions to float32. | The docstring, guide and changelog state the conversion. Float64-only trajectory tests carry an x64 marker, and the float32 tolerances in `tests/test_alpha_precision.py` are derived from the float32 epsilon and the solver tolerances. |
| Conjugation and transpose | Real symmetric cases cannot tell a conjugated stored subsystem Hamiltonian or a transposed embedding from the correct one. | Both mutants passed every existing test. | `test_complex_subsystem_hamiltonian_and_coherent_state_on_nonadjacent_factors` uses a complex Hermitian Hamiltonian and a random complex state. |
| Embedding permutation | Using the forward permutation where its inverse is needed survived. | The tested factors (0, 2) of dims (2, 3, 2) give the permutation (0, 2, 1), which is its own inverse. | `test_trailing_subsystems_partial_trace_and_embedding` traces out the leading factor, keeping (1, 2) and (2,), against einsum and index oracles. |
| Ground-energy gate | A one-sided check of the ground energy survived. | A subsystem Hamiltonian with ground energy $-1$ was accepted. | `test_ground_energy_gate` rejects diag(-1, 0) and accepts a shift of $-10^{-9}$, inside the $10^{-8}$ tolerance. |
| Factory compatibility | `from_operators` passed the new keywords unconditionally. | A subclass with the 0.3.0 `__init__` signature failed with a `TypeError` through the factory. | The keywords are forwarded only when given; `test_factory_keeps_subclasses_with_the_030_init_signature`. |
| Physical amplitudes | Clipping the actions inside `rollout_metrics` survived. | An amplitude of 3.0 clipped to 2 gives the energy $\sin^2(0.4) = 0.152$ instead of $\sin^2(0.6) = 0.319$. | `test_physical_amplitudes_are_not_clipped`. |
| Audit reductions | Reporting the peak as the terminal value, or the last time as the peak time, survived. | The evaluation test checked only shapes and finiteness. | `test_evaluation_matrix_matches_direct_rollouts` compares every model and candidate with a direct rollout, including a candidate whose peak is not the last step; `test_main_writes_the_audit_record` runs the example end to end. |
| Argument checks | `rollout_metrics` accepted arguments that failed later or changed the output type. | A traced `dt` raised a JAX concretization error, an integer `dt` gave integer times, and `method=None` reached the solver. | A bool, string or traced `dt`, a `method` that is not a dynamiqs solver and `metrics` that is not a mapping raise `ValueError`; an integer `dt` becomes float. Tests in `tests/test_trajectories.py`. |
| Release process | The release workflow did not require tested code, and a final tag on the commit of the last rc was built as the rc. | The first `v0.3.2` build came out as 0.3.2rc1 (see the changelog). | The tagged commit must be on main, which takes only pull requests with passing checks, and the build takes its version from the pushed tag. |

## How to reproduce

* The current validation numbers are regenerated by every documentation build:
  [Validation report](validation.md), or locally `qenvs validate --markdown`.
* The counterexamples above are small enough to recompute with plain numpy; the model
  pages list the exact parameters.
