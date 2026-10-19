"""Every physics check in qenvs.validation must pass; numbers are reported by `qenvs validate`."""

import os

import jax
import pytest

from qenvs import validation


def test_validation_precision_matches_request():
    requested = os.environ.get("QENVS_TEST_X64", "1")
    assert bool(jax.config.jax_enable_x64) is (requested != "0")


@pytest.mark.parametrize("check", validation.CHECKS, ids=lambda c: c.__name__)
def test_physics_check_passes(check):
    r = check()
    assert r.passed, f"{r.name}: error={r.error:.3e} > tol={r.tolerance:.0e} ({r.detail})"
