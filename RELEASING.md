# Releasing QEnvS

Releases are cut from `main` by pushing a tag; `release.yml` builds, publishes to PyPI
through Trusted Publishing and creates the GitHub release; Zenodo archives it and mints a
version DOI. Minor releases (0.x.0) go through a release candidate first.

Pre-release tags (`vX.Y.ZaN` alpha, `bN` beta, `rcN` release candidate) are published to
PyPI only: the workflow creates no GitHub release for them, so Zenodo mints no DOI for a
candidate. An alpha may be cut before the feature freeze to test packaging and a fresh
install of the features that are already done; features can still follow an alpha, not an
rc.

## 1. Feature freeze

- [ ] Every new model has validation checks wired into `validation.MODULES`, a model page
      and a changelog entry; the adversarial review is recorded in `docs/review-log.md`.
- [ ] `CHANGELOG.md`: move "Unreleased" items under the new version; list deprecations and
      migration notes.
- [ ] API gate: `tests/test_api_stability.py` passes. Intended breaking changes are in its
      `ALLOWED` dict, announced in the changelog, and have had a deprecation period.
- [ ] Task snapshot: `python tests/golden/make_task_snapshot.py` adds the new tasks to
      `tests/golden/tasks.json` (existing entries are never rewritten); commit the file.

## 2. Release candidate

- [ ] Tag `vX.Y.0rc1` on `main` and push it. PyPI marks it as a pre-release, so
      `pip install qenvs` keeps installing the previous version; testers use
      `pip install --pre qenvs==X.Y.0rc1`.
- [ ] CI is green on the full matrix (Python 3.11 to 3.13, macOS, lowest dependencies).
- [ ] Fresh environment: `uvx --from qenvs==X.Y.0rc1 qenvs doctor`, `qenvs validate`,
      `qenvs list`, and the tutorial script run without errors.
- [ ] Docs build with `mkdocs build --strict`; the validation report shows every check
      passing.
- [ ] Any fix after rc1 is released as rc2, and so on. No new features after the freeze.

## 3. Final release

- [ ] Maintainer GO after looking at the release candidate.
- [ ] Bump `version` and `date-released` in `CITATION.cff`; tag `vX.Y.0` on the commit that
      was tested as the last rc (plus only the changelog and citation edits).
- [ ] Check PyPI, the GitHub release (with `*.publish.attestation` files) and the Zenodo
      record; add the version DOI to `README.md`, `CITATION.cff` and `docs/index.md`.

Patch releases (X.Y.Z) carry fixes only and may skip the release candidate.

For the maintainer-approved `0.3.2a1` experiment, backward-compatible, opt-in Python
API additions are an explicit exception to the fixes-only patch policy. Existing
call signatures and released task behavior remain stable. This exception covers
the alpha only; a final release needs a separate versioning decision. The alpha
still requires the compatibility matrix, float32 checks and documentation build
to pass on the candidate commit before tagging.
