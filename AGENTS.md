# AGENTS.md

QEnvS (`qenvs`): physically validated quantum control environments on JAX + dynamiqs.
Read this before changing anything.

## Commands

```bash
uv sync --extra dev                 # install (Python 3.11/3.12)
uv run pytest -n auto               # tests (physics checks live in src/qenvs/validation.py)
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

With `--json`, failures are `{"error": {"code": "unknown_task" | "invalid_argument", "message": ...}}`
and exit code 1. Docs index for LLMs: https://beomdo-park.github.io/QEnvS/llms.txt

## Layout

- `src/qenvs/models/` physics only (`Model`: drift, controls, jump_ops, initial_state, observables)
- `src/qenvs/envs/` functional envs (`reset(key)`, `step(state, action)` -> `(EnvState, TimeStep)`)
- `src/qenvs/wrappers/` Gymnasium adapter
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
5. `jax<0.8` pin exists because dynamiqs 0.3.6 fails to import on jax>=0.8. Re-test before loosening.
6. Do not copy code from repositories without an OSI license (e.g. RL4qcWpc). Attribute reuse in `NOTICE`.
7. Silence solver progress bars (`progress_meter=False`) so `--json` output stays clean.
8. Plain ASCII punctuation in code, comments, docs and notebooks: no middle dot (U+00B7), no en/em dash
   (U+2013/U+2014), no ` -- ` as a prose separator. Use a colon, comma or a new sentence. CI greps for this.
