# Releasing QEnvS

Releases are cut from `main` by pushing a tag; `release.yml` builds, publishes to PyPI
through Trusted Publishing and creates the GitHub release; Zenodo archives it and mints a
version DOI. Minor releases (0.x.0), and patch releases that add features, go through a
release candidate first.

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
      was tested as the last rc (plus only the changelog and citation edits). The build takes
      its version from the pushed tag, so the final tag may share a commit with the last rc.
- [ ] Check PyPI, the GitHub release (with `*.publish.attestation` files) and the Zenodo
      record; add the version DOI to `README.md`, `CITATION.cff` and `docs/index.md`.

Patch releases (X.Y.Z) carry fixes and backward-compatible additions (see
[stability](docs/stability.md); a change that can break existing code waits for the next
minor release). A patch with additions goes through a release candidate and the same
checks as a minor release; a fixes-only patch may skip the candidate. Either way the
compatibility matrix, the float32 checks and the documentation build pass on the tagged
commit.

For an alpha, update the installation instructions, changelog and CITATION.cff
version/date together. Keep the concept DOI and historic final-release DOI
identifiers; do not assign a previous version DOI to the alpha. Publish matching
user and API documentation after integration.
