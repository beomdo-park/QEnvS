"""Gate-fidelity tools (qenvs.gates), the GateControl environment and the transmon gate tasks."""

import math
import pathlib
import sys

import dynamiqs as dq
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm

import qenvs
import qenvs.tasks.transmon as transmon_tasks
from qenvs.envs.gate import GateControl
from qenvs.gates import (
    TARGETS,
    apply_superop,
    average_gate_fidelity,
    gate,
    leakage,
    rz,
    superop_average_gate_fidelity,
    superop_leakage,
    superop_process_fidelity,
    unitary_to_superop,
)
from qenvs.models.transmon import Transmon

EXAMPLES = pathlib.Path(__file__).resolve().parents[1] / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

import grape_jax  # noqa: E402


def _random_unitary(rng, d):
    z = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def _random_dm(rng, d):
    x = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    rho = x @ x.conj().T
    return rho / np.trace(rho)


# ---------------------------------------------------- independent numpy / scipy references
TWO_PI = 2 * math.pi


def _np_hamiltonian(levels, anh_ghz, det_ghz, wi, wq):
    a = np.diag(np.sqrt(np.arange(1.0, levels)), 1).astype(complex)
    n = np.diag(np.arange(float(levels))).astype(complex)
    drift = TWO_PI * det_ghz * n + 0.5 * TWO_PI * anh_ghz * n @ (n - np.eye(levels))
    return drift + 0.5 * wi * (a + a.conj().T) + 0.5j * wq * (a.conj().T - a)


def _np_closed_map(levels, anh_ghz, det_ghz, actions, dt):
    """Accumulated propagator of the piecewise-constant H, built with scipy only."""
    u = np.eye(levels, dtype=complex)
    for wi, wq in np.asarray(actions, dtype=float):
        u = expm(-1j * dt * _np_hamiltonian(levels, anh_ghz, det_ghz, wi, wq)) @ u
    return u


def _np_open_map(levels, anh_ghz, det_ghz, t1_us, t2_us, actions, dt):
    """Column-stacking superoperator of the piecewise-constant Lindbladian (scipy only)."""
    a = np.diag(np.sqrt(np.arange(1.0, levels)), 1).astype(complex)
    n = np.diag(np.arange(float(levels))).astype(complex)
    g1 = 1 / (1000 * t1_us)
    gphi = 1 / (1000 * t2_us) - 0.5 * g1
    eye = np.eye(levels)
    diss = np.zeros((levels**2, levels**2), dtype=complex)
    for c in (math.sqrt(g1) * a, math.sqrt(2 * gphi) * n):
        cdc = c.conj().T @ c
        diss += np.kron(c.conj(), c) - 0.5 * np.kron(eye, cdc) - 0.5 * np.kron(cdc.T, eye)
    s = np.eye(levels**2, dtype=complex)
    for wi, wq in np.asarray(actions, dtype=float):
        h = _np_hamiltonian(levels, anh_ghz, det_ghz, wi, wq)
        liouv = -1j * (np.kron(eye, h) - np.kron(h.T, eye)) + diss
        s = expm(dt * liouv) @ s
    return s


def _vec(rho):
    return rho.reshape(-1, order="F")


def _unvec(v, d):
    return v.reshape(d, d, order="F")


def _pedersen(u, target):
    k = target.shape[0]
    m = target.conj().T @ u[:k, :k]
    return (np.sum(np.abs(m) ** 2) + abs(np.trace(m)) ** 2) / (k * (k + 1))


def _mixed_actions(seed, horizon=16, amp=0.3):
    """Random I/Q pulse with both quadratures nonzero (float64, not float32-representable)."""
    return np.random.default_rng(seed).uniform(-amp, amp, size=(horizon, 2))


def _random_open_channel(seed=0):
    model = Transmon(t1_us=0.03, t2_us=0.02)
    env = GateControl(model=model, target=gate("X"))
    a = jax.random.uniform(jax.random.PRNGKey(seed), (env.horizon, 2), minval=-0.3, maxval=0.3)
    state, _ = env.rollout(a)
    return np.asarray(state.qstate.to_jax())


# ------------------------------------------------------------------ gate targets
def test_targets_are_unitary_and_sx_squares_to_x():
    for name in TARGETS:
        g = np.asarray(gate(name))
        assert np.allclose(g @ g.conj().T, np.eye(2)), name
    sx = np.asarray(gate("SX"))
    assert np.allclose(sx, 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]))
    assert np.allclose(sx @ sx, np.asarray(gate("X")))
    assert np.allclose(
        np.asarray(rz(math.pi / 2)), np.diag(np.exp([-0.25j * math.pi, 0.25j * math.pi]))
    )
    with pytest.raises(ValueError):
        gate("CNOT")


# ------------------------------------------------------------------ fidelity tools
def test_vectorisation_convention_matches_mesolve():
    """dynamiqs' mepropagator uses column stacking, vec(rho)[i + d j] = rho[i, j]."""
    m = Transmon(t1_us=0.02, t2_us=0.015, detuning_ghz=0.01)
    h = m.hamiltonian(jnp.array([0.2, -0.1]))
    s = dq.mepropagator(h, m.jump_ops(), [0.0, 3.0], progress_meter=False).final_propagator
    rho = _random_dm(np.random.default_rng(1), 3)
    ref = dq.mesolve(
        h,
        m.jump_ops(),
        dq.asqarray(rho),
        [0.0, 3.0],
        method=dq.method.Tsit5(rtol=1e-10, atol=1e-10),
        progress_meter=False,
    ).states[-1]
    out = np.asarray(apply_superop(s, rho))
    assert np.max(np.abs(out - np.asarray(ref.to_jax()))) < 1e-8
    assert np.allclose(np.asarray(dq.vectorize(dq.asqarray(rho)).to_jax()).ravel(), rho.ravel("F"))
    # unitary channel: vec(U rho U^dag) = (conj(U) kron U) vec(rho)
    u = _random_unitary(np.random.default_rng(2), 3)
    s_u = np.asarray(unitary_to_superop(u))
    assert np.allclose(s_u, np.kron(u.conj(), u))
    assert np.allclose(np.asarray(apply_superop(s_u, rho)), u @ rho @ u.conj().T)


def test_superop_fidelity_equals_unitary_formula_for_leaky_unitaries():
    rng = np.random.default_rng(3)
    for name in ("X", "SX", "H"):
        for _ in range(3):
            u = _random_unitary(rng, 3)  # leaky: mixes |2> into the subspace
            s = unitary_to_superop(u)
            f_u = float(average_gate_fidelity(u, gate(name)))
            f_s = float(superop_average_gate_fidelity(s, gate(name)))
            assert f_s == pytest.approx(f_u, abs=1e-12)
            assert float(superop_leakage(s, 2)) == pytest.approx(float(leakage(u, 2)), abs=1e-12)
            # spec form of the leakage: 1 - (1/k) sum_{i,j<k} |U_ji|^2
            assert float(leakage(u, 2)) == pytest.approx(1 - np.sum(np.abs(u[:2, :2]) ** 2) / 2)


def test_average_fidelity_of_leaky_channel_equals_state_average():
    """F of an open 3-level channel = mean over the 6 Pauli eigenstates (a qubit 2-design).

    This checks the Wood-Gambetta definition (average over computational inputs only,
    output projected on the ideal state) without using the closed form.
    """
    s = _random_open_channel()
    target = np.asarray(gate("SX"))
    states = [
        np.array([1, 0]),
        np.array([0, 1]),
        np.array([1, 1]) / math.sqrt(2),
        np.array([1, -1]) / math.sqrt(2),
        np.array([1, 1j]) / math.sqrt(2),
        np.array([1, -1j]) / math.sqrt(2),
    ]
    vals, kept = [], []
    for psi in states:
        rho = np.zeros((3, 3), complex)
        rho[:2, :2] = np.outer(psi, psi.conj())
        out = np.asarray(apply_superop(s, rho))
        ideal = np.concatenate([target @ psi, [0.0]])
        vals.append(np.real(ideal.conj() @ out @ ideal))
        kept.append(np.real(np.trace(out[:2, :2])))
    f = float(superop_average_gate_fidelity(s, target))
    assert f == pytest.approx(np.mean(vals), abs=1e-10)
    lk = float(superop_leakage(s, 2))
    assert lk == pytest.approx(1 - np.mean(kept), abs=1e-10)
    assert 1e-4 < lk < 0.5  # the channel really leaks
    # Wood and Gambetta (2018) Eq. (6): F = (k F_pro + 1 - L) / (k + 1)
    f_pro = float(superop_process_fidelity(s, target))
    assert f == pytest.approx((2 * f_pro + 1 - lk) / 3, abs=1e-10)


def test_fidelity_is_batched_and_phase_invariant():
    u = jnp.stack([jnp.eye(3), jnp.exp(0.7j) * jnp.eye(3)])
    f = average_gate_fidelity(u, gate("I"))
    assert f.shape == (2,)
    assert np.allclose(np.asarray(f), 1.0)
    assert float(average_gate_fidelity(rz(0.4), rz(0.4))) == pytest.approx(1.0)


# ------------------------------------------------------------------ environment
def test_reset_identity_info_keys_and_dims():
    env = qenvs.make("transmon/XGate-v0")
    state, ts = env.reset(jax.random.PRNGKey(0))
    assert np.allclose(np.asarray(state.qstate.to_jax()), np.eye(3))
    assert set(ts.info) == {"p0", "p1", "p2", "state_leakage", "fidelity", "leakage"}
    assert float(ts.info["fidelity"]) == pytest.approx(1 / 3)
    assert float(ts.info["leakage"]) == 0.0 and float(ts.info["p0"]) == 1.0
    assert env.action_dim == 2 and env.observation_dim == 2 * 9 + 1
    assert ts.observation.shape == (env.observation_dim,)
    assert (env.horizon, env.dt, env.duration) == (16, 2.0, 32.0)
    assert env.action_bound == pytest.approx(2 * math.pi * 0.05)
    open_env = qenvs.make("transmon/SXGate-v0", t1_us=100.0, t2_us=100.0)
    state, ts = open_env.reset(jax.random.PRNGKey(0))
    assert open_env.observation_dim == 2 * 81 + 1
    assert float(ts.info["fidelity"]) == pytest.approx(2 / 3)
    assert set(ts.info) == {"p0", "p1", "p2", "state_leakage", "fidelity", "leakage"}


def test_info_observables_follow_the_map():
    """A pi pulse on |0> of an ideal qubit ends with p1 = 1 and the X-gate fidelity 1."""
    env = qenvs.make("transmon/XGate-v0", levels=2, horizon=4, dt=2.5)
    w = math.pi / 10
    state, rewards = env.rollout(jnp.tile(jnp.array([w, 0.0]), (4, 1)))
    info = env._info(state.qstate)
    assert float(info["p1"]) == pytest.approx(1.0, abs=1e-9)
    assert float(info["fidelity"]) == pytest.approx(1.0, abs=1e-9)
    assert float(rewards.sum()) == pytest.approx(2 / 3, abs=1e-9)


ANH, DET, DT = -0.3, 0.01, 2.0


def _env(levels=3, det=DET, target="X", horizon=16, initial="0", **kw):
    model = Transmon(levels=levels, anharmonicity_ghz=ANH, detuning_ghz=det, initial=initial, **kw)
    return GateControl(model=model, target=gate(target), horizon=horizon, dt=DT, action_bound=1.0)


@pytest.mark.parametrize(("initial", "det"), [("0", 0.0), ("1", DET)])
def test_closed_info_is_the_column_of_the_initial_level(initial, det):
    """info p_k = |U[k, i0]|^2 for an asymmetric U (a transposed map would give |U[i0, k]|^2)."""
    actions = _mixed_actions(0)
    env = _env(det=det, initial=initial)
    state, _ = env.rollout(jnp.asarray(actions))
    u = _np_closed_map(3, ANH, det, actions, DT)
    i0 = int(initial)
    pops, wrong = np.abs(u[:, i0]) ** 2, np.abs(u[i0, :]) ** 2
    assert np.max(np.abs(pops - wrong)) > 1e-2  # the pulse really breaks the symmetry
    assert np.allclose(np.asarray(state.qstate.to_jax()), u, atol=1e-10, rtol=0)
    info = env._info(state.qstate)
    for k in range(3):
        assert float(info[f"p{k}"]) == pytest.approx(pops[k], abs=1e-10)
    assert float(info["state_leakage"]) == pytest.approx(pops[2], abs=1e-10)
    assert float(info["state_leakage"]) == pytest.approx(float(info["p2"]), abs=1e-14)
    # gate leakage uses the subspace columns, a different quantity from the state leakage
    assert float(info["leakage"]) == pytest.approx(np.sum(np.abs(u[2, :2]) ** 2) / 2, abs=1e-10)
    psi = np.asarray(env.evolved_state(state.qstate).to_jax())
    assert np.allclose(psi.ravel(), u[:, i0], atol=1e-10, rtol=0)  # the amplitudes, with phase


def _mesolve_chain(rho, h_list, c_ops, dt):
    out = dq.asqarray(rho)
    for h in h_list:
        out = dq.mesolve(
            dq.asqarray(h),
            c_ops,
            out,
            [0.0, dt],
            method=dq.method.Tsit5(rtol=1e-10, atol=1e-10),
            progress_meter=False,
        ).states[-1]
    return np.asarray(out.to_jax())


def test_open_info_and_superop_action_follow_independent_references():
    """Open env: info p_k = diag(S vec(rho0)); S applied to a non-diagonal rho matches numpy
    (Liouvillian exponentials) and dq.mesolve."""
    t1, t2 = 0.05, 0.04
    actions = _mixed_actions(1)
    env = _env(initial="1", t1_us=t1, t2_us=t2)
    state, _ = env.rollout(jnp.asarray(actions))
    s_np = _np_open_map(3, ANH, DET, t1, t2, actions, DT)
    assert np.allclose(np.asarray(state.qstate.to_jax()), s_np, atol=1e-10, rtol=0)
    rho0 = np.zeros((3, 3), complex)
    rho0[1, 1] = 1.0
    rho = _unvec(s_np @ _vec(rho0), 3)
    assert np.max(np.abs(rho - rho.T)) > 1e-2  # complex, non-symmetric output
    info = env._info(state.qstate)
    for k in range(3):
        assert float(info[f"p{k}"]) == pytest.approx(rho[k, k].real, abs=1e-10)
    assert float(info["state_leakage"]) == pytest.approx(rho[2, 2].real, abs=1e-10)
    evolved = np.asarray(env.evolved_state(state.qstate).to_jax())
    assert np.allclose(evolved, rho, atol=1e-10, rtol=0)  # includes the coherences
    # a non-diagonal input exposes any transposition of S or of the reshape
    rho_nd = _random_dm(np.random.default_rng(7), 3)
    out = np.asarray(apply_superop(state.qstate, rho_nd))
    assert np.allclose(out, _unvec(s_np @ _vec(rho_nd), 3), atol=1e-10, rtol=0)
    assert np.max(np.abs(out - out.T)) > 1e-2
    # mesolve over the same piecewise-constant pulse (4 steps, separate env)
    env4 = _env(initial="1", t1_us=t1, t2_us=t2, horizon=4)
    state4, _ = env4.rollout(jnp.asarray(actions[:4]))
    g1, gphi = 1 / (1000 * t1), 1 / (1000 * t2) - 0.5 / (1000 * t1)
    a = np.diag(np.sqrt([1.0, 2.0]), 1).astype(complex)
    n = np.diag([0.0, 1.0, 2.0]).astype(complex)
    c_ops = [dq.asqarray(math.sqrt(g1) * a), dq.asqarray(math.sqrt(2 * gphi) * n)]
    h_list = [_np_hamiltonian(3, ANH, DET, wi, wq) for wi, wq in actions[:4]]
    ref = _mesolve_chain(rho_nd, h_list, c_ops, DT)
    assert np.max(np.abs(np.asarray(apply_superop(state4.qstate, rho_nd)) - ref)) < 1e-7


def test_actions_are_clipped():
    env = qenvs.make("transmon/XGate-v0")
    s0, _ = env.reset(jax.random.PRNGKey(0))
    big, _ = env.step(s0, jnp.array([10.0, -10.0]))
    edge, _ = env.step(s0, jnp.array([env.action_bound, -env.action_bound]))
    assert np.allclose(np.asarray(big.qstate.to_jax()), np.asarray(edge.qstate.to_jax()))


def test_done_flag_and_observation_content():
    """done is False before the horizon and True at it; obs = [Re(U), Im(U), t / T]."""
    horizon = 4
    actions = _mixed_actions(2, horizon)
    env = _env(horizon=horizon)
    state, ts0 = env.reset(jax.random.PRNGKey(0))
    assert not bool(ts0.done)
    assert float(ts0.observation[-1]) == 0.0
    assert np.allclose(np.asarray(ts0.observation[:9]), np.eye(3).ravel())
    for k in range(1, horizon + 1):
        state, ts = env.step(state, jnp.asarray(actions[k - 1]))
        assert bool(ts.done) is (k == horizon)
        u = _np_closed_map(3, ANH, DET, actions[:k], DT)
        obs = np.asarray(ts.observation)
        assert obs.shape == (2 * 9 + 1,)
        assert np.allclose(obs[:9], u.real.ravel(), atol=1e-10, rtol=0)
        assert np.allclose(obs[9:18], u.imag.ravel(), atol=1e-10, rtol=0)
        assert obs[-1] == pytest.approx(k / horizon, abs=1e-12)
        assert int(state.step) == k and float(state.t) == pytest.approx(k * DT)
    assert obs[-1] == 1.0  # exactly 1 at the horizon


def test_open_observation_content_and_done():
    t1, t2 = 0.05, 0.04
    actions = _mixed_actions(3, 2)
    env = _env(horizon=2, t1_us=t1, t2_us=t2)
    state, _ = env.reset(jax.random.PRNGKey(0))
    state, ts1 = env.step(state, jnp.asarray(actions[0]))
    assert not bool(ts1.done)
    state, ts2 = env.step(state, jnp.asarray(actions[1]))
    assert bool(ts2.done)
    s = _np_open_map(3, ANH, DET, t1, t2, actions, DT)
    obs = np.asarray(ts2.observation)
    assert obs.shape == (2 * 81 + 1,)
    assert np.allclose(obs[:81], s.real.ravel(), atol=1e-10, rtol=0)
    assert np.allclose(obs[81:162], s.imag.ravel(), atol=1e-10, rtol=0)
    assert obs[-1] == 1.0


def test_fidelity_against_asymmetric_target_matches_pedersen():
    """Ry(pi/2) = [[c, -s], [s, c]] is real and not symmetric: a transposed target changes F."""
    c = s = 1 / math.sqrt(2)
    ry = np.array([[c, -s], [s, c]], dtype=complex)
    horizon = 4
    actions = _mixed_actions(4, horizon)
    env = GateControl(
        model=Transmon(levels=3, anharmonicity_ghz=ANH, detuning_ghz=DET),
        target=jnp.asarray(ry),
        horizon=horizon,
        dt=DT,
        action_bound=1.0,
    )
    state, _ = env.reset(jax.random.PRNGKey(0))
    f_prev = _pedersen(np.eye(3, dtype=complex), ry)
    for k in range(1, horizon + 1):
        state, ts = env.step(state, jnp.asarray(actions[k - 1]))
        u = _np_closed_map(3, ANH, DET, actions[:k], DT)
        f = _pedersen(u, ry)
        assert float(ts.info["fidelity"]) == pytest.approx(f, abs=1e-10)
        assert float(env.fidelity(state.qstate)) == pytest.approx(f, abs=1e-10)
        assert float(ts.reward) == pytest.approx(f - f_prev, abs=1e-10)
        f_prev = f
    assert abs(f - _pedersen(u, ry.T)) > 1e-3
    # open env with negligible decoherence: same fidelity through the superoperator path
    open_env = GateControl(
        model=Transmon(levels=3, anharmonicity_ghz=ANH, detuning_ghz=DET, t1_us=1e6, t2_us=1e6),
        target=jnp.asarray(ry),
        horizon=horizon,
        dt=DT,
        action_bound=1.0,
    )
    s_state, _ = open_env.rollout(jnp.asarray(actions))
    assert float(open_env.fidelity(s_state.qstate)) == pytest.approx(f, abs=1e-6)


def test_row_stacking_swap_maps_u_to_conj_u_and_changes_complex_targets():
    """Pins the docs claim: the row/column swap leaves F for X, Y, Z, H and the leakage unchanged,
    but changes F for targets with conj(T) not proportional to T (SX, Rz); the unitary
    formula and the superoperator formula agree only for the column-stacking superoperator."""
    u = _random_unitary(np.random.default_rng(2), 3)
    s_col = np.asarray(unitary_to_superop(u))
    s_row = np.kron(u, u.conj())  # superoperator of rho -> U rho U^dag under ROW stacking
    rho = _random_dm(np.random.default_rng(5), 3)
    assert np.allclose((s_row @ rho.reshape(-1)).reshape(3, 3), u @ rho @ u.conj().T)
    assert not np.allclose(np.asarray(apply_superop(s_row, rho)), u @ rho @ u.conj().T)
    assert np.allclose(np.asarray(apply_superop(s_col, rho)), u @ rho @ u.conj().T)
    assert float(superop_leakage(s_row, 2)) == pytest.approx(float(superop_leakage(s_col, 2)))
    for name in ("X", "Y", "Z", "H"):  # conj(T) = T up to a global phase
        t = gate(name)
        f_col = float(superop_average_gate_fidelity(s_col, t))
        f_row = float(superop_average_gate_fidelity(s_row, t))
        assert f_row == pytest.approx(f_col, abs=1e-12), name
    for t, name in ((gate("SX"), "SX"), (rz(0.7), "Rz(0.7)")):
        f_col = float(superop_average_gate_fidelity(s_col, t))
        f_row = float(superop_average_gate_fidelity(s_row, t))
        assert f_col == pytest.approx(float(average_gate_fidelity(u, t)), abs=1e-12)
        assert f_row == pytest.approx(float(average_gate_fidelity(u.conj(), t)), abs=1e-12)
        assert abs(f_row - f_col) > 1e-2, name
    # concretely for SX on this unitary: 0.55355 (column stacking) versus 0.35396 (swapped)
    t = gate("SX")
    assert float(superop_average_gate_fidelity(s_col, t)) == pytest.approx(0.55355, abs=1e-5)
    assert float(superop_average_gate_fidelity(s_row, t)) == pytest.approx(0.35396, abs=1e-5)


@pytest.mark.parametrize("open_system", [False, True])
def test_step_outputs_are_double_precision_under_x64(open_system):
    """The conftest enables x64: no float32 cast of actions, maps, observations or rewards."""
    kw = {"t1_us": 0.05, "t2_us": 0.04} if open_system else {}
    env = _env(horizon=2, **kw)
    state, ts0 = env.reset(jax.random.PRNGKey(0))
    assert jnp.asarray(1.0).dtype == jnp.float64
    assert state.qstate.to_jax().dtype == jnp.complex128 and ts0.observation.dtype == jnp.float64
    a = _mixed_actions(6, 1)[0]
    assert a.dtype == np.float64 and np.float64(np.float32(a[0])) != a[0]
    new, ts = env.step(state, jnp.asarray(a))
    assert new.qstate.to_jax().dtype == jnp.complex128
    assert ts.observation.dtype == jnp.float64 and ts.reward.dtype == jnp.float64
    assert all(v.dtype == jnp.float64 for v in ts.info.values())
    # no precision lost on the way in: float32 rounding of the action would show at 1e-8
    ref = (
        _np_open_map(3, ANH, DET, 0.05, 0.04, a[None], DT)
        if open_system
        else _np_closed_map(3, ANH, DET, a[None], DT)
    )
    assert np.max(np.abs(np.asarray(new.qstate.to_jax()) - ref)) < 1e-12


@pytest.mark.parametrize("open_system", [False, True])
def test_step_is_jit_and_vmap_compatible(open_system):
    kw = {"t1_us": 50.0, "t2_us": 40.0} if open_system else {}
    env = qenvs.make("transmon/XGate-v0", horizon=3, **kw)
    keys = jax.random.split(jax.random.PRNGKey(1), 5)
    states, ts = jax.vmap(env.reset)(keys)
    actions = jnp.linspace(-0.2, 0.2, 10).reshape(5, 2)
    states, ts = jax.jit(jax.vmap(env.step))(states, actions)
    assert ts.observation.shape == (5, env.observation_dim)
    assert ts.reward.shape == (5,) and ts.done.shape == (5,)
    for k in ("fidelity", "leakage", "p0", "p1", "p2", "state_leakage"):
        assert ts.info[k].shape == (5,)
    assert bool(jnp.all(jnp.isfinite(ts.reward)))


@pytest.mark.parametrize("open_system", [False, True])
def test_rollout_is_differentiable(open_system):
    kw = {"t1_us": 50.0, "t2_us": 40.0} if open_system else {}
    env = qenvs.make("transmon/SXGate-v0", horizon=6, **kw)

    def loss(actions):
        _, r = env.rollout(actions)
        return -r.sum()

    a = 0.1 * jax.random.normal(jax.random.PRNGKey(3), (6, 2))
    g = jax.grad(loss)(a)
    assert g.shape == a.shape
    assert bool(jnp.all(jnp.isfinite(g)))
    assert float(jnp.abs(g).max()) > 0.0
    # finite-difference check of one component
    eps = 1e-5
    e = jnp.zeros_like(a).at[2, 1].set(eps)
    fd = (loss(a + e) - loss(a - e)) / (2 * eps)
    assert float(g[2, 1]) == pytest.approx(float(fd), rel=1e-4, abs=1e-8)


def test_target_validation():
    with pytest.raises(ValueError):
        GateControl(model=Transmon(levels=2), target=jnp.eye(3))
    with pytest.raises(ValueError):
        GateControl(model=Transmon(), target=jnp.ones((2, 3)))
    env = GateControl(model=Transmon(), target=dq.asqarray(np.asarray(gate("X"))))
    assert env.n_comp == 2


# ------------------------------------------------------------------ tasks
def test_tasks_registered_with_metadata():
    for tid in ("transmon/XGate-v0", "transmon/SXGate-v0"):
        spec = qenvs.registry[tid]
        assert "ns" in spec.description and "rad/ns" in spec.description
        assert set(spec.kwargs) == {
            "horizon",
            "dt",
            "action_bound",
            "max_rabi_ghz",
            "levels",
            "anharmonicity_ghz",
            "detuning_ghz",
            "t1_us",
            "t2_us",
            "method",
        }
        assert any("Krantz" in r for r in spec.references)
    assert set(transmon_tasks.UNITS) == {"time", "frequency", "control", "coherence"}


def test_task_kwargs_reach_model_and_env():
    """Every kwarg is forwarded (non-default values, so a dropped kwarg fails)."""
    env = qenvs.make(
        "transmon/XGate-v0",
        levels=4,
        anharmonicity_ghz=-0.21,
        detuning_ghz=0.003,
        t1_us=123.0,
        t2_us=77.0,
        horizon=7,
        dt=1.5,
        action_bound=0.9,
        method=dq.method.Tsit5(),
    )
    m = env.model
    assert (m.levels, m.anharmonicity_ghz, m.detuning_ghz, m.t1_us, m.t2_us) == (
        4,
        -0.21,
        0.003,
        123.0,
        77.0,
    )
    assert (env.horizon, env.dt, env.action_bound) == (7, 1.5, 0.9)
    assert isinstance(env.method, dq.method.Tsit5)
    assert m.is_open and env.observation_dim == 2 * 16**2 + 1
    rabi = qenvs.make("transmon/SXGate-v0", max_rabi_ghz=0.02)
    assert rabi.action_bound == pytest.approx(2 * math.pi * 0.02)
    assert np.allclose(np.asarray(rabi.target), np.asarray(gate("SX")))
    with pytest.raises(ValueError):
        qenvs.make("transmon/XGate-v0", max_rabi_ghz=0.02, action_bound=0.1)
    with pytest.raises(ValueError):
        qenvs.make("transmon/XGate-v0", t1_us=10.0, t2_us=30.0)
    with pytest.raises(TypeError):
        qenvs.make("transmon/XGate-v0", gamma1=0.1)
    # defaults: closed 3-level transmon
    d = qenvs.make("transmon/XGate-v0").model
    assert (d.levels, d.anharmonicity_ghz, d.detuning_ghz, d.t1_us, d.t2_us) == (
        3,
        -0.34,
        0.0,
        None,
        None,
    )


def test_grape_reaches_high_fidelity_x_gate():
    out = grape_jax.run_grape("transmon/XGate-v0", steps=100, lr=0.1, seed=0)
    f_final = out["final_return"] + 1 / 3
    assert f_final > 0.999
    first = next(h["step"] for h in out["history"] if h["return"] + 1 / 3 > 0.999)
    assert first <= 100
    assert out["best_actions"]["max_abs"][0] <= out["action_bound"] + 1e-9


def test_grape_open_sx_gate_is_coherence_limited():
    """With T1 = T2 = 100 us the optimum stays near the idle-decoherence reference (32 ns gate)."""
    out = grape_jax.run_grape("transmon/SXGate-v0", steps=150, lr=0.1, t1_us=100.0, t2_us=100.0)
    f_final = out["final_return"] + 2 / 3
    t, t1, t2 = 0.032, 100.0, 100.0
    idle = (2 * (1 + math.exp(-t / t1) + 2 * math.exp(-t / t2)) / 4 + 1) / 3
    assert 0.999 < f_final < 1.0
    assert 1 - f_final > 0.5 * (1 - idle)


@pytest.mark.parametrize(
    ("tid", "f0"), [("transmon/XGate-v0", 1 / 3), ("transmon/SXGate-v0", 2 / 3)]
)
def test_reference_drag_pulse_baseline(tid, f0):
    env = qenvs.make(tid)
    actions = transmon_tasks.reference_actions(tid, env)
    assert actions.shape == (env.horizon, 2)
    assert float(jnp.abs(actions).max()) <= env.action_bound
    state, rewards = env.rollout(actions)
    f = float(env.fidelity(state.qstate))
    assert 1 - f < 1e-5  # measured 4.9e-6 (X) and 1.8e-6 (SX)
    assert float(rewards.sum()) == pytest.approx(f - f0, abs=1e-9)
    with pytest.raises(ValueError):
        transmon_tasks.reference_actions("qubit/StateTransfer-v0", env)
