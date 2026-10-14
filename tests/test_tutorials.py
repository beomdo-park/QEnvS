"""The tutorial scripts in examples/ run and reproduce their headline claims."""

import pathlib
import sys

import pytest

EXAMPLES = pathlib.Path(__file__).resolve().parents[1] / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

import tutorial_transmon_x_gate as tutorial  # noqa: E402


@pytest.fixture(scope="module")
def result():
    return tutorial.run(grape_steps=60, betas=5)


def test_drag_beats_the_plain_gaussian(result):
    assert result["gaussian"]["infidelity"] == pytest.approx(9.07e-4, rel=2e-2)
    assert result["drag"]["infidelity"] < result["gaussian"]["infidelity"] / 100
    # 5-point sweep over [0, 2 beta_phase]: the minimum is at the first-order value
    assert result["drag_sweep_best_beta_over_phase_value"] == pytest.approx(1.0)


def test_grape_reaches_the_closed_gate(result):
    assert result["grape"]["infidelity"] < 1e-3
    assert result["grape"]["leakage"] < 1e-4


def test_decoherence_dominates_with_100us_coherence(result):
    o = result["open"]
    assert o["idle_reference_infidelity"] == pytest.approx(1.600e-4, rel=1e-2)
    assert o["idle_reference_infidelity"] < o["drag"]["infidelity"] < 1.8e-4


def test_export_and_gymnasium(result):
    e = result["export"]
    assert len(e["t_ns"]) == len(e["omega_i_mhz"]) == 16
    assert e["peak_mhz"] <= 50.0 + 1e-6
    g = result["gymnasium"]
    assert g["terminated_at_horizon"] and g["observation_dim"] == 19
