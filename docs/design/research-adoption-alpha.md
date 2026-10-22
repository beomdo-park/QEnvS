# QEnvS research adoption alpha

Status: Implemented
Owner: @beomdo-park
Issue: https://github.com/beomdo-park/QEnvS/issues/8
Target: 0.3.2a1 alpha, released as 0.3.2

## Problem

QEnvS already provides Model.battery_reduced, HamiltonianModel, ObservationModel,
Pulse metadata and robust.evaluate. The missing integration is an explicit path from
a custom composite model to auditable trajectory evaluation. Do not duplicate the
existing physical solver or introduce a new optimizer framework.

The application maintainer requested subsystem reward configuration, unambiguous
dissipation semantics and observation boundaries first, followed by endpoint metrics,
precise candidate evaluation, optimization history and source provenance.
These requests are requirements, not evidence that existing calculations are wrong.

Maintenance issue #6 and draft PR #7 target 0.3.1. This additive API work is separate.

## Decision

The first alpha contains two additive APIs and a generic integration example.
All names and signatures below are proposed. Implementation accepted by the maintainer on 2026-10-19.

### Explicit subsystem reward

Extend HamiltonianModel with optional keyword-only battery_subsystems and
subsystem_hamiltonian arguments. They must be supplied together and must not be
combined with the existing battery_hamiltonian argument. With neither supplied,
preserve the existing behavior exactly.
Expose the same new keyword-only arguments through from_operators without moving
or removing any existing positional arguments.

battery_subsystems is a nonempty tuple of distinct, increasing subsystem indices
in dims. Reject bool, noninteger, repeated and out-of-range indices. Preserve the
original tensor-factor order. subsystem_hamiltonian must be finite and Hermitian,
with dimension equal to the product of retained dims. Validate it at construction,
outside jit, with the existing operator validators.
For the new subsystem path, require ground energy zero to absolute tolerance
1e-8 * max(1, spectral norm). Reject a shifted Hamiltonian rather than silently
shifting its spectrum. This additional check does not alter legacy construction.

battery_reduced returns the retained density matrix and the subsystem Hamiltonian.
battery_hamiltonian returns that Hamiltonian embedded in the full Hilbert space,
with identities on discarded factors. Thus stored energy remains consistent with
the base Model contract while ergotropy uses the reduced state. Reuse the existing
partial-trace machinery. Do not equate reduced-state and global ergotropy.

### Explicit endpoint trajectory diagnostics

Add qenvs.trajectories.rollout_metrics(model, actions, *, dt, metrics, method).
This is a low-level physical-control API, not a new Environment rollout contract.
actions has shape (steps, model.n_controls) and contains physical amplitudes.
It is not normalized, clipped or silently cast to float32. The caller performs
any normalization before calling it. dt is a positive finite scalar, method is
an explicit dynamiqs method, and metrics maps names to pure JAX-compatible
functions (model, qstate) -> real scalar.
dt, method and the metric mapping are fixed configuration captured outside jit;
the first alpha promises differentiation over actions, not over dt or callbacks.

Return (times, values), where times has shape (steps,) and values maps each metric
name to an array of shape (steps,). Sample only right endpoints dt through steps*dt;
exclude the initial state. Reject empty sequences, wrong control dimensions,
invalid dt and an empty metric mapping. Shapes are checked at tracing time;
finite actions and outputs are checked by the host evaluation example, not by
Python conversions inside jit. No full state trajectory is retained in the output.
Nonfinite runtime values are not converted to successful scores.

Use Model.hamiltonian, Model.evolve and lax.scan. Keep requested action precision
subject to JAX configuration. Support jit, vmap and grad with matching eager output.
The example records the selected method and tolerances in its manifest. These
remain distinct from the output sampling period.

Peak and terminal reductions remain caller choices. The example explicitly names
endpoint_peak and terminal, records the first peak time on ties, and explains that
endpoint_peak is not a continuous-time maximum or an executable stopping policy.
Do not modify robust.evaluate, which currently reports final-state performance.

### Generic integration example and metadata

Use a small, independently constructed bipartite system, not an unpublished
application configuration. Name each jump channel and write its operator and
rate convention in the example. Existing Dicke gamma means collective emission;
gamma_phi means collective dephasing. Do not rename or reinterpret released kwargs.
A universal named-channel API is deferred until a second use case establishes it.

Use existing ObservationModel for expectation observations. A policy receives only
the observation, never state, exact diagnostics or model parameters. Document that
exact simulator reward can itself reveal performance, so this is not a measurement
feedback simulator. Test the actual policy argument boundary, not just field names.

The cross-evaluation example uses fixed open-loop action arrays, replayed unchanged
on each evaluation model. It does not rerun feedback policies on new observations.
The observation-boundary check is a separate small wrapper example and test.

Save S[i,j], the endpoint-peak ergotropy of candidate j on model i, and the source
model ID source[j] for every candidate. For each source c, choose winner[c] as the
first argmax of S[c,j] among candidates with source[j] == c. Then C[i,c] =
S[i,winner[c]]. Reference[i] = C[i,i], not the maximum across all source groups.
Report regret[i,c] = 1 - C[i,c]/Reference[i]. For this example, Reference <= 1e-12
in declared dimensionless energy units yields null regret and an explicit reason.
This threshold is an example convention, not a universal physical tolerance.
Negative regret is allowed. Save the full S matrix, source IDs and selected IDs.
Saved optimizer scores must correspond to saved actions before the update.

The included example-local optimizer trace records phase, score, best score,
gradient norm and update norm separately. Warmup length is fixed independently
of total iterations. The example and its tests are required alpha deliverables;
Optax remains an optional installation extra, not a core dependency. Trace stability
does not establish global optimality or convergence.

Reuse Pulse serialization where applicable. Store application settings, action
units, reduction rule, solver method/tolerances, dtype, backend, seeds, input hashes,
dependency versions and actual imported source provenance. Git SHA and dirty status
are optional for wheel installations and explicitly unavailable there. An installed
version string alone is not source provenance. Output creation refuses overwrite.
This metadata stays example-local in the first alpha; no general experiment store.

## Validation gates

1. Subsystem constructor failures and unchanged legacy behavior. Compare a small
   bipartite entangled state against a manual NumPy partial trace and spectrum.
   Verify equal full/reduced energy and intentionally different global/reduced work.
2. Analytic one-cell oscillation: endpoint times and energy match the closed form.
   Test a charge/discharge pulse where endpoint peak differs from terminal value.
   Compare with an independently assembled QuTiP problem in the optional test extra.
3. Compare eager, jit and vmap trajectories and a smooth energy gradient against
   finite differences. Do not use a degenerate ergotropy spectrum as a smooth
   gradient oracle. Precision-specific tolerances must be documented from the
   analytic error scale before inspecting application results.
4. Regress action precision using distinguishable float64 amplitudes under x64.
   Verify failure for invalid shapes and host rejection of nonfinite candidates.
5. Synthetic candidate matrix catches transposed axes, missing candidates and
   a winner chosen with a different precision. Verify zero-denominator handling.
6. Short/long optimizer runs with identical seeds and fixed warmup share their
   common prefix. Re-evaluate saved actions to verify saved-score consistency.
7. Confirm policy receives only declared observations. Test manifest generation
   with both a Git checkout and an installed package without .git.
8. Run affected tests only on the shared server, then existing API stability and
   released-task snapshot gates. Start only at load1 <= 50, at most 4 CPU cores,
   no GPU, and at most 12 cores across the user's processes.

## Implementation sequence

1. Accept this design, reconcile open GitHub claims, then open a dedicated issue
   and claim. Create an isolated branch/worktree; use an independent environment.
2. Add failing subsystem tests, implement the minimal constructor extension and
   run the affected custom-model and battery tests.
3. Add failing trajectory tests, implement the scan helper and run analytic,
   gradient and precision checks.
4. Add the generic example and focused audit tests. Review for unpublished content.
5. Open a draft PR with Done, Next, Decisions and Open questions. Before any push
   or CI dispatch, inspect repository visibility, runner types, duplicate runs,
   job history and billing access. Estimate jobs and runner minutes separately
   from billing. Keep max-parallel 2 and test artifact retention 7 days.
6. Build isolated wheel and sdist with a proposed 0.3.2a1 version only after checks;
   install the wheel into a separate environment and execute the small example.
   Tagging, publishing and merging are separate final actions.

## Out of scope

New scientific claims, paper gates, application preparation recipes and results,
new physical models, new optimizers, arbitrary stopping-policy APIs, default reward
changes, changes to released task IDs, automatic migration of active research,
hardware control, GPU benchmarking and new core dependencies.

## Implementation notes

Implementation authorized on 2026-10-19. Target 0.3.2a1 preserves existing API behavior.

On 2026-10-19 the maintainer authorized lead self-review and closing the remaining
precision and version-policy gaps. The 0.3.2a1 additive API exception is scoped to
the alpha in RELEASING.md. Five analytic precision cases now run in the ordinary
float64 matrix and the serial float32 job. No solver or physics implementation
changed during this follow-up.

For 0.3.2 (2026-10-22) the versioning rule changed instead of the alpha exception: patch
releases may carry backward-compatible additions (docs/stability.md). A physics review and
an API and mutation review found no blocker. Deviations from the design, with reasons:
`from_operators` forwards the subsystem arguments only when given, because unconditional
forwarding broke subclasses with the 0.3.0 `__init__` signature; `rollout_metrics` also
rejects a bool, string or traced `dt`, a non-dynamiqs `method` and non-mapping `metrics`,
which previously failed later with unrelated errors or ran with integer times.
