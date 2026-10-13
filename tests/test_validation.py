"""Every physics check in qenvs.validation must pass; numbers are reported by `qenvs validate`."""

import pytest

from qenvs import validation


@pytest.mark.parametrize("check", validation.CHECKS, ids=lambda c: c.__name__)
def test_physics_check_passes(check):
    r = check()
    assert r.passed, f"{r.name}: error={r.error:.3e} > tol={r.tolerance:.0e} ({r.detail})"
