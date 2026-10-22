"""Precision gates for subsystem rewards and endpoint trajectories.

Run in both CI precision modes. Tolerances: 2e-7 for algebra, 2e-6 for integrated
energies and 5e-6 for gradients. They follow the float32 error scale: float32 epsilon
1.2e-7 for algebra, and the Tsit5 tolerance used here (rtol 1e-6, atol 1e-7) for integrated
quantities. Measured float32 errors (independent review, 2026-10-22) are at most 3e-8, so
the margins are 7 to 200 times, while a 2e-4 relative error in the action scale now fails.
These catch joint-work substitution, tensor-order errors, shifted endpoint
sampling and lost action derivatives without relying on float64 rounding.
"""

import dynamiqs as dq
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qenvs.models import HamiltonianModel
from qenvs.observables import energy, ergotropy
from qenvs.trajectories import rollout_metrics


@pytest.mark.parametrize("density", [False, True])
def test_bell_local_reward_and_precision(density):
    ket = np.array([1, 0, 0, 1]) / np.sqrt(2)
    model = HamiltonianModel(
        np.zeros((4, 4)),
        [np.eye(4)],
        initial_state=np.outer(ket, ket) if density else ket,
        dims=(2, 2),
        battery_subsystems=(1,),
        subsystem_hamiltonian=np.diag([0.0, 1.0]),
    )
    rho, h = jax.jit(lambda s: model.battery_reduced(s))(model.initial_state())
    dtype = jnp.complex128 if jax.config.jax_enable_x64 else jnp.complex64
    assert rho.to_jax().dtype == dtype
    assert h.to_jax().dtype == dtype
    np.testing.assert_allclose(rho.to_jax(), np.eye(2) / 2, rtol=0, atol=2e-7)
    np.testing.assert_allclose([energy(rho, h), ergotropy(rho, h)], [0.5, 0], atol=2e-7)


def test_nonadjacent_embedding_and_partial_trace():
    h = np.array([[0, 0, 0, 0], [0, 2, 1, 0], [0, 1, 2, 0], [0, 0, 0, 4.0]])
    model = HamiltonianModel.from_operators(
        np.zeros((12, 12)),
        [np.eye(12)],
        initial_state=np.diag(np.arange(1, 13) / 78),
        dims=(2, 3, 2),
        battery_subsystems=(0, 2),
        subsystem_hamiltonian=h,
    )
    rho, hb = model.battery_reduced(model.initial_state())
    np.testing.assert_allclose(rho.to_jax(), np.diag([9, 12, 27, 30]) / 78, atol=2e-7)
    # Independent basis-index oracle: spectator indices must match.
    full = np.asarray(model.battery_hamiltonian().to_jax())
    for i in range(12):
        a, b, c = np.unravel_index(i, (2, 3, 2))
        for j in range(12):
            d, e, f = np.unravel_index(j, (2, 3, 2))
            assert full[i, j] == (h[2 * a + c, 2 * d + f] if b == e else 0)
    assert float(energy(rho, hb)) == pytest.approx(198 / 78, abs=2e-6)


@pytest.mark.parametrize("mixed", [False, True])
def test_endpoint_energy_jit_vmap_and_gradient(mixed):
    model = HamiltonianModel(
        np.zeros((2, 2)),
        [np.array([[0, 1], [1, 0]])],
        initial_state=np.diag([0.8, 0.2]) if mixed else np.array([1.0, 0]),
        battery_hamiltonian=np.diag([0.0, 1]),
    )

    def run(actions):
        return rollout_metrics(
            model,
            actions,
            dt=0.2,
            metrics={"energy": lambda m, s: energy(s, m.battery_hamiltonian())},
            method=dq.method.Tsit5(rtol=1e-6, atol=1e-7),
        )

    actions = jnp.array([[0.7], [0.4]])
    times, values = run(actions)
    dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    assert times.dtype == dtype
    assert values["energy"].dtype == dtype
    np.testing.assert_allclose(times, [0.2, 0.4], rtol=0, atol=2e-7)
    offset, scale = (0.2, 0.6) if mixed else (0.0, 1.0)
    expected = offset + scale * np.sin([0.14, 0.22]) ** 2
    np.testing.assert_allclose(values["energy"], expected, rtol=0, atol=2e-6)

    def final(a):
        return run(a)[1]["energy"][-1]

    np.testing.assert_allclose(jax.jit(final)(actions), expected[-1], rtol=0, atol=2e-6)
    np.testing.assert_allclose(
        jax.vmap(final)(jnp.stack([actions, 2 * actions])),
        offset + scale * np.sin([0.22, 0.44]) ** 2,
        rtol=0,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        jax.grad(final)(actions),
        np.full((2, 1), scale * 0.2 * np.sin(0.44)),
        rtol=0,
        atol=5e-6,
    )
