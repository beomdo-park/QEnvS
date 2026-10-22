# AGENTS.md

QEnvS (`qenvs`): physically validated quantum control environments on JAX + dynamiqs.
Read this before changing anything.

## Before you write code (people and agents)

Several contributors work on QEnvS asynchronously, each with their own agents. GitHub is the
only place where work is coordinated; do not start from a local guess about what is free.

1. Find the issue. `gh issue list --label "in progress"` and `gh pr list` show what is being
   worked on. If your task overlaps, comment there instead of writing a second implementation.
2. Claim it. Comment on the issue, assign yourself, add the `in progress` label, and open a draft
   PR within a day (the draft PR is the lock). A claim with no activity for 14 days may be taken
   over after a comment on the issue. No issue yet: open one first.
3. Design first when it matters. A change to the public API, to physics, or to a released task
   needs a short design in `docs/design/<slug>.md` (template below) marked `Accepted` by a
   maintainer before implementation. Implement the accepted design; do not redesign it. Record
   every deviation with its reason under "Implementation notes" in the same file.
4. Keep the diff to the claim: no drive-by refactors or formatting of other files. Search for an
   existing helper before writing one.
5. Hand off in the PR description, not in your head: Done, Next, Decisions, Open questions (the PR
   template has the fields), so the next person or agent continues without re-deriving.
6. Shared machines: run only the tests you touched (`pytest tests/test_x.py`), keep `-n 4` or
   less, and let CI run the full suite. The test conftest pins each pytest process to a few
   cores (`QENVS_TEST_CPUS`).

## Project resource policy

The `.33` host and EcoAI-Thor are shared laboratory servers, not private runners. This policy is
specific to QEnvS work on those two machines.

- Before starting a test, benchmark, training run or agent workflow, check CPU load, process count,
  GPU utilisation and GPU memory use. Defer the job when another lab user is active or the host is
  already busy.
- Keep the combined QEnvS CPU use from the lead process and all agents at 12 cores or fewer on each
  server. Keep pytest at `-n 4` or fewer. On `.33`, do not launch new work when the 1-minute load
  average is above 50.
- Run at most one GPU-consuming QEnvS job per server unless the user explicitly approves more. Do
  not reserve most of the GPU or unified memory when a smaller batch, device or serial run works.
- Do not launch unattended sweeps, parallel training runs or full-suite repetitions on either
  server without explicit user approval. Prefer targeted local checks and let GitHub CI run the
  ordinary full test matrix.
- If a required run would exceed these limits, stop and report the estimated CPU, GPU, memory and
  duration before asking for approval.

Design template (`docs/design/<slug>.md`): Status (Proposed, Accepted, Implemented,
Superseded), Owner, Issue, Problem, Decision (API, physics, validation gates), Out of scope,
Implementation notes.

## GitHub Actions budget

- Account usage is shared with other projects. Before pushing or dispatching CI, inspect active
  runs and recent job durations, estimate runner time, and check billing usage when accessible.
  If billing access is unavailable, report it rather than claiming a remaining allowance.
- Public standard-runner execution is free; still account separately for concurrency, artifact
  storage and any paid runner use. Do not duplicate a PR matrix with a manual full-matrix run.
- Weekly and manual dependency checks run only the newest-dependency job. The regular matrix
  runs in parallel (standard runners are free for this public repository). A pull request that
  changes only documentation runs only the tests that read documentation; push CI on main runs
  the full suite. Jobs have timeouts and test artifacts expire after seven days.
- Batch fixes before pushing, rerun only failed jobs when possible, and defer optional paid work
  when account-wide remaining usage cannot be established. Do not change billing permissions or
  account spending limits automatically.

## Commands

```bash
uv sync --extra dev                 # install (Python 3.11 to 3.13)
uv run pytest -n 4                  # tests (physics checks live in src/qenvs/validation/<model>.py)
uv run ruff check . && uv run ruff format .
uv run qenvs validate               # prints per-check error vs tolerance + versions
uv run mkdocs build --strict        # docs (needs --extra docs)
```

## Discover the library without reading source

```bash
qenvs list --json
qenvs info qubit/StateTransfer-v0 --json      # physics, kwargs, defaults, references, usage
qenvs run qubit/StateTransfer-v0 --policy pi-pulse --set gamma1=0.1 --json
qenvs validate --json
qenvs doctor --json                           # versions, backend, optional extras, smoke run
```

With `--json`, failures are `{"error": {"code": "unknown_task" | "invalid_argument" | "invalid_model", "message": ...}}`
and exit code 1. Docs index for LLMs: https://beomdo-park.github.io/QEnvS/llms.txt

## Layout

- `src/qenvs/models/` physics only (`Model`: drift, controls, jump_ops, initial_state, observables)
- `src/qenvs/envs/` functional envs (`reset(key)`, `step(state, action)` -> `(EnvState, TimeStep)`)
- `src/qenvs/wrappers/` Gymnasium adapters (`GymnasiumEnv`, batched `GymnasiumVectorEnv`)
- `src/qenvs/gates.py` gate targets and leakage-aware gate fidelity (unitaries and superoperators)
- `src/qenvs/models/custom.py` `HamiltonianModel` (bring your own operators); `src/qenvs/validation/generic.py` checks that run on any model (`qenvs validate --model mod:attr`)
- `src/qenvs/registry.py` core registry; `src/qenvs/tasks/<model>.py` registers task ids `namespace/Name-vN` with metadata
- `src/qenvs/validation/<model>.py` analytic / constraint / numerics checks exposing `CHECKS` (single source for tests + CLI); wired in `validation/__init__.py` `MODULES`
- `src/qenvs/toolkit.py` plain-dict API behind `cli.py` (an MCP adapter can sit on it later)

## Rules

1. A new model ships with `validation/<model>.py` checks (closed-form result, physical constraints,
   solver convergence, QuTiP cross-check) and a `tasks/<model>.py` registry entry with references.
   No checks, no merge.
2. Never change a released task's reward/observation/defaults; bump `-vN`.
3. Core deps stay `jax`, `dynamiqs`, `equinox`, `numpy`. Anything else is an optional extra.
4. Index 1 is the excited state. Do not use `dq.sigmam()`/`sigmap()` for relaxation (QuTiP spin
   convention is reversed); build `|0><1|` explicitly.
5. `jax<0.11` pin exists because dynamiqs 0.3.6 fails to import on jax 0.11 (fixed upstream in
   dynamiqs#1140, unreleased). CI tests the lowest and the highest allowed versions.
6. Do not copy code from repositories without an OSI license (e.g. RL4qcWpc). Attribute reuse in `NOTICE`.
7. Silence solver progress bars (`progress_meter=False`) so `--json` output stays clean.
8. Plain ASCII punctuation in code, comments, docs and notebooks: no middle dot (U+00B7), no en/em dash
   (U+2013/U+2014), no ` -- ` as a prose separator. Use a colon, comma or a new sentence. CI greps for
   the middle dot and the two dashes; ` -- ` is checked in review.
