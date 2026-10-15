"""Dicke battery with local dissipation: physics checks (qenvs.validation.dicke_local) and API."""

import dynamiqs as dq
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import qenvs
import qenvs.tasks.dicke_local  # registers battery/DickeLocal-v0
from qenvs.envs import BatteryCharging
from qenvs.models.dicke import DickeBattery
from qenvs.models.dicke_local import (
    MAX_LIOUVILLIAN_DIM,
    DickeBatteryLocal,
    degeneracy,
    sector_spins,
)
from qenvs.registry import registry as task_registry
from qenvs.validation import dicke_local as validation


@pytest.mark.parametrize("check", validation.CHECKS, ids=lambda c: c.__name__)
def test_physics_check_passes(check):
    r = check()
    assert r.model == "DickeBatteryLocal"
    assert r.passed, f"{r.name}: error={r.error:.3e} > tol={r.tolerance:.0e} ({r.detail})"


# ------------------------------------------------------------------ model API
def test_sectors_and_dimensions():
    m = DickeBatteryLocal(n_cells=4, n_photons=1)
    assert m.sectors == ((2.0, 1), (1.0, 3), (0.0, 2))
    assert (m.dim_cells, m.dim_cavity, m.dim) == (16, 2, 32)
    assert m.liouvillian_dim == (25 + 9 + 1) * 4
    assert m.n_controls == 1 and m.is_open
    assert sector_spins(5) == (5, 3, 1)
    assert [degeneracy(5, tj) for tj in (5, 3, 1)] == [1, 4, 5]


def test_cutoff_defaults():
    assert DickeBatteryLocal().cutoff == 3  # n_photons: exact for rwa=True without pumping
    assert DickeBatteryLocal(kappa=0.1, gamma_local=0.2).cutoff == 3  # dissipation only lowers
    assert DickeBatteryLocal(rwa=False).cutoff == 7  # n_photons + 4, no guarantee
    assert DickeBatteryLocal(n_cells=2, n_photons=1, n_max=4).cutoff == 4
    # with pumping the excitation number is unbounded: no finite default is exact
    assert DickeBatteryLocal(pumping_local=0.1, n_max=6).cutoff == 6


def test_pumping_requires_an_explicit_cutoff():
    for kw in ({}, {"rwa": False}, {"n_cells": 2, "n_photons": 0}):
        with pytest.raises(ValueError, match=r"n_max must be given.*top_fock"):
            DickeBatteryLocal(pumping_local=0.1, **kw)
    with pytest.raises(ValueError, match="no finite Fock cutoff is exact"):
        qenvs.make("battery/DickeLocal-v0", pumping_local=0.05)
    qenvs.make("battery/DickeLocal-v0", pumping_local=0.05, n_max=4)  # explicit cutoff is fine
    DickeBatteryLocal(pumping_local=0.0)  # zero rate: the default stays exact


def _pumped_run(n_cells, n_max, pumping, steps=40, g=0.5):
    env = qenvs.make(
        "battery/DickeLocal-v0",
        n_cells=n_cells,
        n_photons=0,
        n_max=n_max,
        pumping_local=pumping,
        gamma_local=0.05,
        gamma_phi_local=0.0,
        horizon=steps,
    )
    state, _ = env.rollout(jnp.full((steps, 1), g))
    info = env.info(state)
    return float(info["energy"]), float(info["top_fock"])


def test_pumping_truncation_error_shrinks_and_top_fock_flags_it():
    # Old default n_photons + N = 1 for N = 1, n_photons = 0: energies off by tens of percent.
    ref_energy, ref_top = _pumped_run(1, 12, 0.3)
    assert ref_top < 1e-10
    results = {n: _pumped_run(1, n, 0.3) for n in (1, 2, 4, 6, 8)}
    errors = {n: abs(e - ref_energy) for n, (e, _) in results.items()}
    tops = {n: t for n, (_, t) in results.items()}
    assert errors[1] > 0.2 * ref_energy  # too small a cutoff is badly wrong, not subtly
    for small, large in ((1, 2), (2, 4), (4, 6), (6, 8)):
        assert errors[large] < errors[small]  # error shrinks as n_max grows
        assert tops[large] < tops[small]  # and so does the top-Fock population
    assert errors[8] < 1e-6 and tops[8] < 1e-5
    # the monitor flags the bad cutoffs and bounds the error it causes
    assert tops[1] > 0.1 and tops[2] > 1e-2 and tops[6] < 1e-3
    for n in (1, 2, 4, 6):
        assert errors[n] < tops[n]
    # n_max versus n_max + 4 (the comparison the error message recommends)
    assert abs(results[2][0] - results[6][0]) > 1e-2  # unconverged: the two cutoffs disagree
    assert abs(results[4][0] - results[8][0]) < 3e-3 and tops[4] > 1e-4  # top_fock still warns
    assert abs(results[6][0] - results[8][0]) < 1e-4


def test_pumping_truncation_two_cells():
    e_small, top_small = _pumped_run(2, 2, 0.2, steps=30)
    e_mid, top_mid = _pumped_run(2, 6, 0.2, steps=30)
    e_ref, top_ref = _pumped_run(2, 10, 0.2, steps=30)
    assert abs(e_mid - e_ref) < abs(e_small - e_ref)
    assert top_ref < top_mid < top_small
    assert top_small > 1e-2 and abs(e_small - e_ref) > 1e-2


def test_ergotropy_from_blocks_accepts_a_traced_omega0():
    import equinox as eqx

    env = qenvs.make("battery/DickeLocal-v0", n_cells=2)
    state, _ = env.rollout(jnp.full((env.horizon, env.action_dim), 0.5))  # charged cells
    model, rho = env.model, state.qstate

    def ergo(w):
        return eqx.tree_at(lambda m: m.omega0, model, w).ergotropy_from_blocks(rho)

    eager = ergo(1.3)
    assert float(eager) > 0.1  # a charged state, so the comparison is not 0 == 0
    traced = jax.jit(ergo)(1.3)
    np.testing.assert_allclose(np.asarray(traced), np.asarray(eager), rtol=1e-12, atol=1e-12)
    # omega0 scales the levels and the energy alike, so the ergotropy scales linearly
    np.testing.assert_allclose(np.asarray(ergo(2.6)), 2.0 * np.asarray(eager), rtol=1e-12)


def _charged_ergotropy(w):
    """Ergotropy from the blocks of a charged two-cell state, with omega0 replaced by w."""
    import equinox as eqx

    env = qenvs.make("battery/DickeLocal-v0", n_cells=2)
    state, _ = env.rollout(jnp.full((env.horizon, env.action_dim), 0.5))
    return eqx.tree_at(lambda m: m.omega0, env.model, w).ergotropy_from_blocks(state.qstate)


def test_ergotropy_from_blocks_is_differentiable_in_omega0():
    # the ergotropy is linear in omega0 at a fixed state, so d/d omega0 = value / omega0
    value = float(_charged_ergotropy(1.3))
    grad = float(jax.jit(jax.grad(_charged_ergotropy))(1.3))
    assert value > 0.1
    np.testing.assert_allclose(grad, value / 1.3, rtol=1e-10)


def test_ergotropy_from_blocks_float32_matches_float64():
    # float32 runs in a fresh process: switching x64 off inside this one leaves int32 state
    # behind in jax that breaks later x64 tests (order dependent), as the other float32 tests
    # of this repository also avoid
    import os
    import subprocess
    import sys

    ref = float(_charged_ergotropy(1.3))
    assert ref > 0.1  # charged state: the relative error has a safe denominator
    code = (
        "import jax; jax.config.update('jax_enable_x64', False)\n"
        "import jax.numpy as jnp, equinox as eqx, qenvs\n"
        "env = qenvs.make('battery/DickeLocal-v0', n_cells=2)\n"
        "state, _ = env.rollout(jnp.full((env.horizon, env.action_dim), 0.5))\n"
        "f = lambda w: eqx.tree_at(lambda m: m.omega0, env.model, w)"
        ".ergotropy_from_blocks(state.qstate)\n"
        "x = jax.jit(f)(1.3)\n"
        "print(x.dtype, float(x))\n"
    )
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "JAX_ENABLE_X64": "0"}
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=600
    )
    assert out.returncode == 0, out.stderr[-2000:]
    dtype, value = out.stdout.split()[-2:]
    assert dtype == "float32"
    rel = abs(float(value) - ref) / abs(ref)
    print(f"float32 vs float64 relative error of the ergotropy: {rel:.2e}")
    assert rel < 1e-4, rel


def test_invalid_configurations_rejected():
    with pytest.raises(ValueError, match="n_cells"):
        DickeBatteryLocal(n_cells=0)
    with pytest.raises(ValueError, match="n_max"):
        DickeBatteryLocal(n_cells=2, n_photons=3, n_max=2)
    for name in DickeBatteryLocal.RATES:
        with pytest.raises(ValueError, match=name):
            DickeBatteryLocal(**{name: -0.1})
    with pytest.raises(ValueError, match="Liouvillian dimension"):
        DickeBatteryLocal(n_cells=6)  # 84 * 49 = 4116 > limit
    with pytest.raises(ValueError, match="stored state dimension"):
        DickeBatteryLocal(n_cells=8, n_photons=1, n_max=2)  # 256 * 3 = 768 > 512
    assert DickeBatteryLocal(n_cells=5).liouvillian_dim <= MAX_LIOUVILLIAN_DIM


def test_initial_state_and_observables():
    m = DickeBatteryLocal(n_cells=2, n_photons=2)
    rho = m.initial_state()
    assert rho.isdm() and rho.dims == (4, 3)
    obs = {k: float(dq.expect(op, rho).real) for k, op in m.observables().items()}
    assert obs == pytest.approx(
        {"energy": 0.0, "n_photons": 2.0, "excitation": 2.0, "top_fock": 1.0, "symmetric": 1.0}
    )
    rho_b, hb = m.battery_reduced(rho)
    assert rho_b.shape == (4, 4)
    np.testing.assert_allclose(np.diag(np.asarray(hb.to_jax())).real, [0, 1, 2, 1])


def test_jump_ops_and_time_dependent_hamiltonian_are_rejected():
    m = DickeBatteryLocal(n_cells=2, gamma_local=0.1)
    with pytest.raises(NotImplementedError):
        m.jump_ops()
    times = jnp.asarray([0.0, 0.1, 0.2])
    h = m.pwc_hamiltonian(times, jnp.ones((1, 2)))
    with pytest.raises(TypeError, match="constant"):
        m.evolve(m.initial_state(), h, 0.2)


def test_symmetric_projector_trace_and_weight():
    m = DickeBatteryLocal(n_cells=3, n_photons=1, n_max=2)
    proj = np.asarray(m.symmetric_projector().to_jax())
    assert np.trace(proj).real == (m.n_cells + 1) * m.dim_cavity  # rank 4 * 3
    np.testing.assert_allclose(proj @ proj, proj, atol=0)  # projector
    # weight on j = N/2 of a block state equals the trace of the first block
    vec = np.asarray(validation._random_block_state(m, seed=2))
    lay = m.layout
    b = lay.block_sizes[0]
    state = m.from_blocks(jnp.asarray(vec))
    expected = np.trace(vec[: b * b].reshape(b, b)).real
    assert float(dq.expect(m.symmetric_projector(), state).real) == pytest.approx(expected)


def test_dissipator_cache_is_read_only():
    m = DickeBatteryLocal(n_cells=2, gamma_local=0.1)
    d = m.dissipator()
    assert not d.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        d[0, 0] = 1.0
    assert m.dissipator() is d or np.array_equal(m.dissipator(), d)


def test_blocks_round_trip():
    m = DickeBatteryLocal(n_cells=3, n_photons=1)
    vec = validation._random_block_state(m, seed=1)
    back = m.to_blocks(m.from_blocks(vec))
    np.testing.assert_allclose(np.asarray(back), np.asarray(vec), atol=1e-12)
    assert float(jnp.trace(m.dicke_density(m.from_blocks(vec))).real) == pytest.approx(1.0)


def test_schur_transform_is_unitary_and_matches_degeneracies():
    for n in (2, 3, 4):
        u = validation.schur_transform(n)
        np.testing.assert_allclose(u @ u.conj().T, np.eye(2**n), atol=1e-12)


def test_local_dephasing_breaks_symmetric_subspace():
    env = qenvs.make("battery/DickeLocal-v0", n_cells=3, gamma_local=0.0, gamma_phi_local=0.4)
    state, _rewards = env.rollout(jnp.full((env.horizon, 1), 0.3))
    info = env.info(state)
    assert 0.0 < float(info["symmetric"]) < 0.99
    assert float(info["purity"]) < 1.0


# ------------------------------------------------------------------ env API
def test_registered_task_defaults_and_kwargs():
    env = qenvs.make("battery/DickeLocal-v0")
    m = env.model
    assert isinstance(env, BatteryCharging) and isinstance(m, DickeBatteryLocal)
    assert (env.horizon, env.dt, env.action_bound, env.reward) == (20, 0.2, 0.5, "energy")
    assert (m.n_cells, m.cutoff, m.gamma_local, m.gamma_phi_local, m.pumping_local) == (
        3,
        3,
        0.05,
        0.05,
        0.0,
    )
    assert env.observation_dim == 2 * 32**2 + 1
    kw = {
        "n_cells": 2,
        "n_photons": 1,
        "n_max": 3,
        "omega0": 1.1,
        "omega_c": 1.2,
        "rwa": False,
        "kappa": 0.01,
        "gamma": 0.02,
        "gamma_phi": 0.03,
        "gamma_local": 0.04,
        "gamma_phi_local": 0.06,
        "pumping_local": 0.07,
    }
    m2 = qenvs.make("battery/DickeLocal-v0", **kw).model
    assert {k: getattr(m2, k) for k in kw} == kw
    spec = task_registry["battery/DickeLocal-v0"]
    assert set(kw) <= set(spec.kwargs) and "rates" in spec.units
    assert any("Shammah" in r for r in spec.references)


def test_task_default_contract():
    import pathlib

    env = qenvs.make("battery/DickeLocal-v0")
    m = env.model
    assert m.gamma_local == 0.05 and m.gamma_phi_local == 0.05 and m.pumping_local == 0.0
    assert m.cutoff == m.photons == 3  # without pumping the default cutoff is exact
    # explicit values reach the model unchanged
    m2 = qenvs.make("battery/DickeLocal-v0", gamma_local=0.2, gamma_phi_local=0.0).model
    assert (m2.gamma_local, m2.gamma_phi_local, m2.pumping_local) == (0.2, 0.0, 0.0)
    m3 = qenvs.make("battery/DickeLocal-v0", n_cells=2, kappa=0.1, rwa=False).model
    assert (m3.gamma_local, m3.gamma_phi_local, m3.cutoff) == (0.05, 0.05, 2 + 4)
    # pumping without n_max fails at construction, before any step; with n_max it runs
    with pytest.raises(ValueError, match="n_max must be given"):
        qenvs.make("battery/DickeLocal-v0", pumping_local=0.1)
    env_p = qenvs.make("battery/DickeLocal-v0", n_cells=2, pumping_local=0.1, n_max=3, horizon=2)
    state, _ = env_p.rollout(jnp.full((2, 1), 0.2))
    assert np.isfinite(float(env_p.info(state)["energy"]))
    # the stated defaults match the code: kwargs text and the docs defaults table
    spec = task_registry["battery/DickeLocal-v0"]
    assert "default 0.05" in spec.kwargs["gamma_local"]
    assert "default 0.05" in spec.kwargs["gamma_phi_local"]
    assert "default 0)" in spec.kwargs["pumping_local"]
    assert "default 3" in spec.kwargs["n_cells"] and "default 20" in spec.kwargs["horizon"]
    assert "default 0.2" in spec.kwargs["dt"] and "0.5" in spec.kwargs["action_bound"]
    doc = (pathlib.Path(__file__).parents[1] / "docs/models/dicke_local.md").read_text()
    assert "| `gamma_local` | 0.0 / **0.05** |" in doc
    assert "| `gamma_phi_local` | 0.0 / **0.05** |" in doc
    assert "| `pumping_local` | 0.0 |" in doc
    assert (env.horizon, env.dt, env.action_bound) == (20, 0.2, 0.5)


def test_single_cell_local_equals_collective():
    # N = 1: sigma_- = J-, sigma_z / 2 = Jz, so local and collective processes coincide
    actions = jax.random.uniform(jax.random.PRNGKey(5), (6, 1), minval=-0.5, maxval=0.5)
    loc = BatteryCharging(
        model=DickeBatteryLocal(n_cells=1, gamma_local=0.2, gamma_phi_local=0.3, kappa=0.1),
        horizon=6,
        dt=0.3,
        action_bound=0.5,
    )
    ref = BatteryCharging(
        model=DickeBattery(n_cells=1, gamma=0.2, gamma_phi=0.3, kappa=0.1),
        horizon=6,
        dt=0.3,
        action_bound=0.5,
        method=dq.method.Tsit5(rtol=1e-10, atol=1e-12),
    )
    s_loc, r_loc = loc.rollout(actions)
    s_ref, r_ref = ref.rollout(actions)
    np.testing.assert_allclose(np.asarray(r_loc), np.asarray(r_ref), atol=1e-8)
    np.testing.assert_allclose(
        np.asarray(s_loc.qstate.to_jax()), np.asarray(s_ref.qstate.to_jax()), atol=1e-8
    )


def test_step_is_jit_and_vmap_compatible():
    env = qenvs.make("battery/DickeLocal-v0", n_cells=2, horizon=3)
    keys = jax.random.split(jax.random.PRNGKey(1), 4)
    states, _ = jax.vmap(env.reset)(keys)
    actions = jnp.linspace(-0.5, 0.5, 4)[:, None]
    states, ts = jax.jit(jax.vmap(env.step))(states, actions)
    assert ts.observation.shape == (4, env.observation_dim)
    assert ts.info["ergotropy"].shape == (4,)
    e = np.asarray(ts.info["energy"])
    np.testing.assert_allclose(e, e[::-1], atol=1e-6)  # g -> -g gauge symmetry


def test_rollout_is_differentiable():
    env = qenvs.make("battery/DickeLocal-v0", n_cells=2, horizon=4, kappa=0.05)

    def loss(actions):
        _, r = env.rollout(actions)
        return -r.sum()

    a = jnp.full((4, 1), 0.3)
    g = jax.grad(loss)(a)
    assert jnp.all(jnp.isfinite(g))
    assert jnp.all(g < 0)  # more coupling charges faster early in the episode
    eps = 1e-3  # the env casts actions to float32, so a tiny step would hit roundoff
    da = jnp.zeros_like(a).at[1, 0].set(eps)
    fd = (loss(a + da) - loss(a - da)) / (2 * eps)
    assert float(g[1, 0]) == pytest.approx(float(fd), rel=1e-4)
