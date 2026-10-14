# Contributing

Participation follows the [code of conduct](CODE_OF_CONDUCT.md). Report security issues privately as described in [SECURITY.md](SECURITY.md).

```bash
git clone https://github.com/beomdo-park/QEnvS && cd qenvs
uv sync --extra dev          # or: pip install -e '.[dev]'
uv run pre-commit install
uv run pytest -n auto
```

## Adding a model

1. Subclass `qenvs.models.Model` in `src/qenvs/models/<name>.py`: implement
   `dim`, `drift`, `controls`, `initial_state`; override `jump_ops` and
   `observables` as needed.
2. Add **analytic validation tests** in `tests/test_<name>.py`. At least one
   closed-form result (decay rate, Rabi frequency, known optimum, scaling law)
   and, where possible, a cross-check against QuTiP.
3. Register a task in `src/qenvs/registry.py` with a `-v0` suffix.
4. Add the model to the table in `README.md` with its references.

A model without reference tests will not be merged.

## Versioning of task ids

Bump the suffix (`-v0` -> `-v1`) whenever a change could alter learning results
(reward, observation, defaults). Never silently change a released task.

## Releases

Tag `vX.Y.Z` on `main`; the `release.yml` workflow builds and publishes to PyPI
via Trusted Publishing and creates a GitHub release.
