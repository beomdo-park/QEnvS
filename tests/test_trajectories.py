"""Analytic endpoint and differentiation gates for physical-amplitude trajectories."""

import importlib

import dynamiqs as dq
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qenvs.models import HamiltonianModel
from qenvs.observables import energy

METHOD = dq.method.Tsit5(rtol=1e-10, atol=1e-12)


def model(mixed=False):
    return HamiltonianModel(
        np.zeros((2, 2)),
        [np.array([[0, 1], [1, 0]])],
        initial_state=np.diag([0.8, 0.2]) if mixed else np.array([1.0, 0]),
        battery_hamiltonian=np.diag([0.0, 1]),
    )


def run(m, actions, dt=0.2, metrics=None):
    module = importlib.import_module("qenvs.trajectories")
    return module.rollout_metrics(
        m,
        actions,
        dt=dt,
        metrics=metrics
        if metrics is not None
        else {"energy": lambda m, s: energy(s, m.battery_hamiltonian())},
        method=METHOD,
    )


needs_x64 = pytest.mark.skipif(
    not jax.config.jax_enable_x64, reason="float64 expectations (1e-8); run with x64"
)


@needs_x64
@pytest.mark.parametrize("mixed", [False, True])
def test_analytic_endpoints_and_qutip(mixed):
    m = model(mixed)
    times, values = run(m, jnp.ones((2, 1)), dt=np.pi / 2)
    np.testing.assert_allclose(times, [np.pi / 2, np.pi], atol=1e-14)
    np.testing.assert_allclose(values["energy"], [0.8, 0.2] if mixed else [1, 0], atol=1e-8)
    qt = pytest.importorskip("qutip")
    rho = qt.Qobj(np.diag([0.8, 0.2])) if mixed else qt.basis(2, 0)
    result = qt.mesolve(
        qt.sigmax(),
        rho,
        [0, np.pi / 2, np.pi],
        e_ops=[qt.Qobj(np.diag([0.0, 1]))],
        options={"atol": 1e-12, "rtol": 1e-10},
    )
    np.testing.assert_allclose(values["energy"], result.expect[0][1:], atol=1e-8)


@needs_x64
def test_jit_vmap_and_action_gradient():
    m = model()

    def f(a):
        return run(m, a)[1]["energy"][-1]

    a = jnp.array([[0.7]])
    assert float(f(a)) == pytest.approx(np.sin(0.14) ** 2, abs=1e-8)
    np.testing.assert_allclose(jax.jit(f)(a), f(a), atol=1e-8)
    np.testing.assert_allclose(
        jax.vmap(f)(jnp.stack([a, 2 * a])), [np.sin(0.14) ** 2, np.sin(0.28) ** 2], atol=1e-8
    )
    grad = float(jax.grad(f)(a)[0, 0])
    assert grad == pytest.approx(0.2 * np.sin(0.28), abs=1e-7)
    fd = float((f(a + 1e-4) - f(a - 1e-4)) / 2e-4)
    assert grad == pytest.approx(fd, abs=1e-6)


@needs_x64
def test_float64_actions_are_not_rounded():
    # This double isolates action forwarding from numerical integration error.
    class IdentityEvolution:
        n_controls = 1

        def initial_state(self):
            return jnp.array(0.0)

        def hamiltonian(self, a):
            return a[0]

        def evolve(self, state, h, duration, *, method):
            assert method is METHOD
            return h

    a = jnp.array([[1 + 2**-40]])
    _, v = run(IdentityEvolution(), a, metrics={"value": lambda m, s: s})
    assert float(v["value"][0]) == 1 + 2**-40
    assert float(v["value"][0]) != 1.0


@pytest.mark.parametrize(
    "actions,dt",
    [
        (np.zeros((0, 1)), 0.2),
        (np.zeros(2), 0.2),
        (np.zeros((2, 2)), 0.2),
        (np.zeros((1, 1)), 0),
        (np.zeros((1, 1)), -1),
        (np.zeros((1, 1)), float("nan")),
        (np.zeros((1, 1)), float("inf")),
        (np.zeros((1, 1)), True),
        (np.zeros((1, 1)), "0.2"),
        (np.zeros((1, 1)), None),
        (np.ones((1, 1), dtype=complex), 0.2),
    ],
)
def test_invalid_actions_and_dt(actions, dt):
    with pytest.raises(ValueError):
        run(model(), actions, dt=dt)


def test_empty_or_nonscalar_metrics_rejected():
    with pytest.raises(ValueError):
        run(model(), jnp.ones((1, 1)), metrics={})
    with pytest.raises(ValueError):
        run(model(), jnp.ones((1, 1)), metrics={"rho": lambda m, s: s.to_jax()})
    with pytest.raises(ValueError):
        run(model(), jnp.ones((1, 1)), metrics={"complex": lambda m, s: jnp.array(1j)})


def test_physical_amplitudes_are_not_clipped():
    # an environment would clip 3.0 to its bound of 2, giving sin^2(0.4) = 0.152
    _, values = run(model(), jnp.array([[3.0]]))
    assert float(values["energy"][0]) == pytest.approx(np.sin(0.6) ** 2, abs=1e-6)


def test_dt_must_be_concrete_and_is_coerced_to_float():
    times, _ = run(model(), jnp.ones((2, 1)), dt=1)
    assert jnp.issubdtype(times.dtype, jnp.floating)
    np.testing.assert_allclose(times, [1, 2])
    with pytest.raises(ValueError, match="concrete"):
        jax.jit(lambda dt: run(model(), jnp.ones((1, 1)), dt=dt)[0])(0.2)


def test_method_and_metrics_types_checked():
    module = importlib.import_module("qenvs.trajectories")
    metrics = {"energy": lambda m, s: energy(s, m.battery_hamiltonian())}
    with pytest.raises(ValueError, match="method"):
        module.rollout_metrics(model(), jnp.ones((1, 1)), dt=0.2, metrics=metrics, method=None)
    with pytest.raises(ValueError, match="mapping"):
        run(model(), jnp.ones((1, 1)), metrics=[("energy", metrics["energy"])])
