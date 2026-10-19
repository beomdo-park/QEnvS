"""Subsystem work must not include work available only from joint unitaries."""

import jax
import numpy as np
import pytest

from qenvs.models import HamiltonianModel
from qenvs.observables import energy, ergotropy


def make(**kwargs):
    options = {
        "drift": np.zeros((4, 4)),
        "controls": [np.eye(4)],
        "dims": (2, 2),
        "initial_state": np.array([1, 0, 0, 1]) / np.sqrt(2),
        "battery_subsystems": (1,),
        "subsystem_hamiltonian": np.diag([0.0, 1.0]),
    }
    options.update(kwargs)
    return HamiltonianModel(**options)


@pytest.mark.parametrize("density", [False, True])
def test_bell_local_work_is_zero_while_global_work_is_half(density):
    ket = np.array([1, 0, 0, 1]) / np.sqrt(2)
    model = make(initial_state=np.outer(ket, ket) if density else ket)
    state = model.initial_state()
    rho, h = jax.jit(lambda s: model.battery_reduced(s))(state)
    np.testing.assert_allclose(rho.to_jax(), np.eye(2) / 2, atol=1e-14)
    assert float(energy(rho, h)) == pytest.approx(0.5)
    assert float(ergotropy(rho, h)) == pytest.approx(0, abs=1e-14)
    assert float(energy(state, model.battery_hamiltonian())) == pytest.approx(0.5)
    assert float(ergotropy(state, model.battery_hamiltonian())) == pytest.approx(0.5)


def test_nonadjacent_subsystems_embedding_and_factory():
    h = np.array([[0, 0, 0, 0], [0, 2, 1, 0], [0, 1, 2, 0], [0, 0, 0, 4.0]])
    rho = np.diag(np.arange(1, 13) / 78)
    model = HamiltonianModel.from_operators(
        np.zeros((12, 12)),
        [np.eye(12)],
        initial_state=rho,
        dims=(2, 3, 2),
        battery_subsystems=(0, 2),
        subsystem_hamiltonian=h,
    )
    reduced, hb = model.battery_reduced(model.initial_state())
    np.testing.assert_allclose(reduced.to_jax(), np.diag([9, 12, 27, 30]) / 78)
    np.testing.assert_allclose(hb.to_jax(), h)
    full = model.battery_hamiltonian().to_jax()
    for i in range(12):
        a, b, c = np.unravel_index(i, (2, 3, 2))
        for j in range(12):
            d, e, f = np.unravel_index(j, (2, 3, 2))
            assert full[i, j] == (h[2 * a + c, 2 * d + f] if b == e else 0)
    assert float(energy(reduced, hb)) == pytest.approx(
        float(energy(model.initial_state(), model.battery_hamiltonian()))
    )


@pytest.mark.parametrize("indices", [(), (True,), (1.0,), (2,), (-1,), (1, 1), (1, 0)])
def test_invalid_subsystems_rejected(indices):
    with pytest.raises(ValueError, match="battery_subsystems"):
        make(battery_subsystems=indices)


@pytest.mark.parametrize(
    "changes",
    [
        {"subsystem_hamiltonian": None},
        {"battery_subsystems": None},
        {"battery_hamiltonian": np.eye(4)},
        {"subsystem_hamiltonian": np.eye(3)},
        {"subsystem_hamiltonian": np.eye(2)},
        {"subsystem_hamiltonian": np.array([[0, 1], [0, 1]])},
        {"subsystem_hamiltonian": np.diag([0, np.nan])},
    ],
)
def test_invalid_subsystem_hamiltonian_rejected(changes):
    with pytest.raises(ValueError):
        make(**changes)


def test_all_subsystems_and_legacy_behavior():
    h = np.diag([0.0, 1, 2, 3])
    model = make(battery_subsystems=(0, 1), subsystem_hamiltonian=h)
    rho, hb = model.battery_reduced(model.initial_state())
    np.testing.assert_allclose(rho.to_jax(), model.initial_state().todm().to_jax())
    np.testing.assert_allclose(hb.to_jax(), h)
    legacy = HamiltonianModel(np.zeros((4, 4)), [np.eye(4)], (), None, None, h, (2, 2))
    legacy_rho, legacy_h = legacy.battery_reduced(legacy.initial_state())
    np.testing.assert_allclose(legacy_rho.to_jax(), legacy.initial_state().to_jax())
    np.testing.assert_allclose(legacy_h.to_jax(), h)
