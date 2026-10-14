"""API gate: no unannounced breaking change against the last final release.

Compares the public API of the working tree with the latest ``vX.Y.Z`` tag using griffe.
Intended breaking changes must be listed in ``ALLOWED`` with a reason (and announced in
CHANGELOG.md with a deprecation period, see docs/stability.md). Changes of constant values
(``ATTRIBUTE_CHANGED_VALUE``, e.g. a tuple gaining an entry) are not breaking and ignored.
"""

from __future__ import annotations

import os
import pathlib
import re
import subprocess

import pytest

griffe = pytest.importorskip("griffe")

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: "object path": "reason" for every intended breaking change since the last release.
ALLOWED: dict[str, str] = {}

IGNORED_KINDS = {"ATTRIBUTE_CHANGED_VALUE"}


def _last_release_tag() -> str | None:
    try:
        out = subprocess.run(
            ["git", "tag", "--list", "v*", "--merged", "HEAD", "--sort=-v:refname"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
        head_tags = subprocess.run(
            ["git", "tag", "--points-at", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        return None
    finals = [t for t in out if re.fullmatch(r"v\d+\.\d+\.\d+", t) and t not in head_tags]
    return finals[0] if finals else None


def test_no_unannounced_breaking_changes():
    tag = _last_release_tag()
    if tag is None:
        if os.environ.get("CI"):
            pytest.fail("no release tag found; CI must check out the full history (fetch-depth: 0)")
        pytest.skip("not a git checkout with release tags")
    old = griffe.load_git("qenvs", ref=tag, repo=ROOT, search_paths=["src"])
    new = griffe.load("qenvs", search_paths=[str(ROOT / "src")])
    breaks = [b for b in griffe.find_breaking_changes(old, new) if b.kind.name not in IGNORED_KINDS]
    unexpected = [f"{b.kind.name}: {b.obj.path}" for b in breaks if b.obj.path not in ALLOWED]
    assert not unexpected, f"breaking changes against {tag} not in ALLOWED: {unexpected}"


def test_removed_toolkit_units_is_back_as_a_deprecated_alias():
    from qenvs import toolkit

    with pytest.warns(DeprecationWarning, match="per task"):
        units = toolkit.UNITS
    assert "time" in units
    with pytest.raises(AttributeError):
        toolkit.does_not_exist  # noqa: B018
