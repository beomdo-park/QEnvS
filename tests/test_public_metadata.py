"""Contracts for public validation claims and release metadata."""

from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _text(path: str) -> str:
    return (ROOT / path).read_text()


def test_readme_scopes_independent_simulator_claims():
    readme = " ".join(_text("README.md").split())
    assert "where an equivalent independent formulation is available" in readme
    assert (
        "its physics has been checked against closed-form solutions and against an independent "
        "simulator (QuTiP)"
    ) not in readme


def test_review_log_records_repository_backed_round_6_evidence():
    review = " ".join(_text("docs/review-log.md").split())
    for evidence in (
        "## Round 6",
        "598d66c",
        "test_public_exports_of_qenvs_wrappers",
        "test_samples_have_the_requested_dtype",
        "test_tail_count_is_exact_at_round_n",
        "test_bootstrap_cvar_uses_the_cvar_definition",
        "0.932",
        "The private mutation patches were not retained",
    ):
        assert evidence in review


def test_zenodo_description_covers_v03_scope():
    metadata = json.loads(_text(".zenodo.json"))
    description = metadata["description"].lower()
    for term in ("transmon", "local dissipation", "finite-shot", "robust evaluation"):
        assert term in description


def test_changelog_links_v03_comparison():
    changelog = _text("CHANGELOG.md")
    assert "[0.3.0]: https://github.com/beomdo-park/QEnvS/compare/v0.2.0...v0.3.0" in changelog
