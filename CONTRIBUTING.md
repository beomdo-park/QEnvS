# Contributing

Participation follows the [code of conduct](CODE_OF_CONDUCT.md). Report security issues privately as described in [SECURITY.md](SECURITY.md).

```bash
git clone https://github.com/beomdo-park/QEnvS && cd QEnvS
uv sync --extra dev          # or: pip install -e '.[dev]'
uv run pre-commit install
uv run pytest -n auto
```

## Adding a model

1. Subclass `qenvs.models.Model` in `src/qenvs/models/<name>.py`: implement
   `dim`, `drift`, `controls`, `initial_state`; override `jump_ops` and
   `observables` as needed.
2. Add **validation checks** in `src/qenvs/validation/<name>.py`, exposed as a
   `CHECKS` tuple and listed in `MODULES` in `src/qenvs/validation/__init__.py`:
   at least one closed-form result (decay rate, Rabi frequency, known optimum,
   scaling law), physical constraints, solver convergence and, where possible, a
   cross-check against QuTiP. `tests/test_<name>.py` runs them and tests the API.
   The same checks are what `qenvs validate` prints.
3. Register a task in `src/qenvs/tasks/<name>.py` with a `-v0` suffix, its kwargs,
   units, references and, if one exists, a reference solution; import the module in
   `src/qenvs/tasks/__init__.py`.
4. Add the model to the table in `README.md` and write `docs/models/<name>.md`.

A model without validation checks will not be merged.

## Versioning of task ids

Bump the suffix (`-v0` -> `-v1`) whenever a change could alter learning results
(reward, observation, defaults). Never silently change a released task.

## Releases

See [RELEASING.md](RELEASING.md): minor releases go through a release candidate, an API
gate and a behaviour snapshot of released tasks.
