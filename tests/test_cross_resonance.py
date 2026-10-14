"""Two coupled transmons, the subspace option of qenvs.gates and the transmon/CRGate-v0 task."""

import math
import os
import pathlib
import subprocess
import sys

import dynamiqs as dq
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm

import qenvs
import qenvs.tasks.cross_resonance as cr
from qenvs.envs.gate import GateControl
from qenvs.gates import (
    average_gate_fidelity,
    gate,
    leakage,
    superop_average_gate_fidelity,
    superop_leakage,
    superop_process_fidelity,
    unitary_to_superop,
)
from qenvs.models.transmon import Transmon
from qenvs.models.two_transmon import (
    TwoTransmon,
    static_zz_second_order,
    to_qubit_frame,
    zx_rate_first_order,
)
from qenvs.validation import cross_resonance as validation

EXAMPLES = pathlib.Path(__file__).resolve().parents[1] / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

import grape_jax  # noqa: E402

TWO_PI = 2 * math.pi
IDX = (0, 1, 3, 4)


def _random_unitary(rng, d):
    z = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


@pytest.fixture(scope="module")
def reference():
    """Default task, its calibrated reference pulse and the resulting closed fidelity."""
    env = qenvs.make("transmon/CRGate-v0")
    actions = cr.reference_actions(env)
    state, rewards = env.rollout(actions)
    return env, actions, state, rewards


# ------------------------------------------------------------------ physics checks
@pytest.mark.parametrize("check", validation.CHECKS, ids=lambda c: c.__name__)
def test_physics_check_passes(check):
    r = check()
    assert r.model == "TwoTransmon"
    assert not r.skipped, r.detail
    assert r.passed, f"{r.name}: error={r.error:.3e} > tol={r.tolerance:.0e} ({r.detail})"


@pytest.mark.slow
def test_physics_checks_pass_in_float32():
    """The conftest enables x64; the same tolerances must hold in single precision."""
    code = (
        "import jax; jax.config.update('jax_enable_x64', False)\n"
        "from qenvs.validation import cross_resonance as v\n"
        "bad = [r for r in (c() for c in v.CHECKS) if not (r.passed or r.skipped)]\n"
        "assert not bad, [(r.name, r.error, r.tolerance) for r in bad]\n"
        "import jax.numpy as jnp; assert jnp.asarray(1.0).dtype == jnp.float32\n"
    )
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "JAX_ENABLE_X64": "0"}
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=180
    )
    assert out.returncode == 0, out.stderr[-2000:]


# ------------------------------------------------------------------ model
def test_model_operators_match_an_independent_construction():
    m = TwoTransmon(
        levels=3,
        detuning_ghz=-0.08,
        anharmonicity_c_ghz=-0.33,
        anharmonicity_t_ghz=-0.29,
        coupling_ghz=0.004,
    )
    low = np.diag([1.0, math.sqrt(2.0)], 1)
    a, b = np.kron(low, np.eye(3)), np.kron(np.eye(3), low)
    nc, nt = a.T @ a, b.T @ b
    h0 = (
        TWO_PI * -0.08 * nc
        + 0.5 * TWO_PI * -0.33 * nc @ (nc - np.eye(9))
        + 0.5 * TWO_PI * -0.29 * nt @ (nt - np.eye(9))
        + TWO_PI * 0.004 * (a.T @ b + a @ b.T)
    )
    assert np.allclose(np.asarray(m.drift().to_jax()), h0, atol=1e-14)
    ctrl = [0.5 * (a + a.T), 0.5j * (a.T - a), 0.5 * (b + b.T), 0.5j * (b.T - b)]
    for got, want in zip(m.controls(), ctrl, strict=True):
        assert np.allclose(np.asarray(got.to_jax()), want, atol=1e-15)
    assert m.computational_indices == IDX and m.dim == 9 and m.n_controls == 4
    assert not m.is_open


def test_jump_operators_per_transmon():
    m = TwoTransmon(t1_c_us=0.05, t2_c_us=0.04, t1_t_us=0.08)
    ops = [np.asarray(j.to_jax()) for j in m.jump_ops()]
    a, b = (np.asarray(o.to_jax()) for o in m.ops())
    nc = a.conj().T @ a
    g1c, g1t = 1 / 50.0, 1 / 80.0
    gphic = 1 / 40.0 - 0.5 * g1c
    want = [math.sqrt(g1c) * a, math.sqrt(2 * gphic) * nc, math.sqrt(g1t) * b]
    assert len(ops) == 3
    for got, w in zip(ops, want, strict=True):
        assert np.allclose(got, w, atol=1e-14)
    assert m.is_open and m.initial_state().to_jax().shape == (9, 9)


def test_model_validation_errors():
    with pytest.raises(ValueError):
        TwoTransmon(levels=1)
    with pytest.raises(ValueError):
        TwoTransmon(initial="02")
    with pytest.raises(ValueError):
        TwoTransmon(t1_t_us=10.0, t2_t_us=30.0)
    with pytest.raises(ValueError):
        TwoTransmon(t1_c_us=-1.0)


def test_static_zz_second_order_scaling():
    """The remainder of the second-order formula shrinks as (J / Delta)^2 (x64)."""
    rels = []
    for j in (0.001, 0.003):
        m = TwoTransmon(anharmonicity_t_ghz=-0.3, coupling_ghz=j)
        exact = float(m.static_zz()) / TWO_PI
        rels.append(exact / static_zz_second_order(-0.1, -0.34, -0.3, j) - 1)
    assert 6 < rels[1] / rels[0] < 12  # about 9
    # sign at the defaults (Delta = -0.1 GHz, alpha = -0.34 GHz): 1/(Delta - alpha_t) = +1/0.24
    # and -1/(Delta + alpha_c) = +1/0.44 add, so zeta > 0. (Delta = -0.1 GHz is not in the
    # straddling regime 0 < Delta < -alpha_c = 0.34 GHz.)
    assert float(TwoTransmon().static_zz()) > 0


def test_first_order_rates_formula_values():
    zx, ix = zx_rate_first_order(-0.1, -0.34, 0.003, 0.05)
    assert zx == pytest.approx(-(0.003 * 0.05 / -0.1) * (-0.34 / -0.44))
    assert ix == pytest.approx(-0.003 * 0.05 / -0.44)
    assert zx > 0 and ix > 0


def test_first_order_zx_straddling_regime():
    """Straddling is 0 < Delta < -alpha_c (target between w_c + alpha_c and w_c), which does
    not contain the default Delta = -0.1 GHz: there |alpha_c / (Delta (alpha_c + Delta))| is
    7.7 per GHz against 14.2 per GHz at Delta = +0.1 GHz, and nu_ZX has the opposite sign."""
    j, rabi, ac = 0.003, 0.05, -0.34
    zx_d, ix_d = zx_rate_first_order(-0.1, ac, j, rabi)
    zx_s, ix_s = zx_rate_first_order(0.1, ac, j, rabi)
    # nu_ZX = -J Omega * alpha_c / (Delta (alpha_c + Delta))
    assert zx_d / (j * rabi) == pytest.approx(-ac / (-0.1 * (ac - 0.1)), rel=1e-12)
    assert abs(zx_d) / (j * rabi) == pytest.approx(7.727, abs=1e-3)
    assert abs(zx_s) / (j * rabi) == pytest.approx(14.167, abs=1e-3)
    assert zx_d > 0 > zx_s and abs(zx_s) > 1.8 * abs(zx_d)
    assert ix_d > 0 and ix_s > 0  # -J Omega / (Delta + alpha_c), Delta + alpha_c < 0 in both
    m = TwoTransmon()
    assert not (0.0 < m.detuning_ghz < -m.anharmonicity_c_ghz)  # the default is not straddling
    assert 0.0 < 0.1 < -m.anharmonicity_c_ghz


@pytest.mark.parametrize("duration", [37.0, 51.6])
def test_frame_change_sign(duration):
    """J = 0, no drive: exp(+i Delta n_c T) U_r is the identity on the qubit states (closed and
    open); the opposite sign is not."""
    m = TwoTransmon(coupling_ghz=0.0)
    h = np.asarray(m.drift().to_jax())
    u = expm(-1j * h * duration)
    uq = np.asarray(to_qubit_frame(m, u, duration))
    assert np.allclose(uq[np.ix_(IDX, IDX)], np.eye(4), atol=1e-12)
    wrong = np.asarray(m.frame_unitary(-duration)) @ u
    assert np.max(np.abs(wrong[np.ix_(IDX, IDX)] - np.eye(4))) > 0.5
    # superoperator path
    s = np.asarray(unitary_to_superop(u))
    sq = to_qubit_frame(m, s, duration)
    assert float(superop_average_gate_fidelity(sq, np.eye(4), subspace=IDX)) == pytest.approx(
        1.0, abs=1e-12
    )
    # the control alone, driven at J = 0, matches a single transmon detuned by Delta in its
    # rotating frame, followed by the same frame change exp(+i Delta n T)
    acts = np.random.default_rng(0).uniform(-0.2, 0.2, size=(5, 2))
    env2 = GateControl(model=TwoTransmon(coupling_ghz=0.0), target=jnp.eye(4), horizon=5, dt=2.0)
    full = np.concatenate([acts, np.zeros((5, 2))], axis=1)
    state, _ = env2.rollout(jnp.asarray(full))
    single = GateControl(model=Transmon(detuning_ghz=-0.1), target=jnp.eye(2), horizon=5, dt=2.0)
    s1, _ = single.rollout(jnp.asarray(acts))
    uc = np.diag(np.exp(1j * TWO_PI * -0.1 * np.arange(3) * 10.0)) @ np.asarray(s1.qstate.to_jax())
    u2 = np.asarray(to_qubit_frame(env2.model, state.qstate.to_jax(), 10.0))
    n = np.arange(3)
    ut_idle = np.diag(np.exp(-1j * 0.5 * TWO_PI * -0.34 * n * (n - 1) * 10.0))  # anharmonic phase
    assert np.allclose(u2, np.kron(uc, ut_idle), atol=1e-9)


def test_to_qubit_frame_operand_order():
    """U_q = F U_r and S_q = S_F S_r with the frame on the left (F is diagonal but does not
    commute with a generic map, so m @ F differs)."""
    m = TwoTransmon()
    duration = 37.0
    f = np.asarray(m.frame_unitary(duration))
    rng = np.random.default_rng(11)
    u = _random_unitary(rng, 9)
    got = np.asarray(to_qubit_frame(m, u, duration))
    assert np.allclose(got, f @ u, atol=1e-14)
    assert np.max(np.abs(got - u @ f)) > 0.1  # the mutant m @ f is far away
    s = rng.normal(size=(81, 81)) + 1j * rng.normal(size=(81, 81))
    sf = np.asarray(unitary_to_superop(f))
    got_s = np.asarray(to_qubit_frame(m, s, duration))
    assert np.allclose(got_s, sf @ s, atol=1e-12)
    assert np.max(np.abs(got_s - s @ sf)) > 1.0  # the mutant m @ unitary_to_superop(f)


# ------------------------------------------------------------------ gates: subspace keyword
def test_subspace_default_is_the_first_k_levels():
    rng = np.random.default_rng(3)
    for d, k in ((3, 2), (9, 4)):
        u = _random_unitary(rng, d)
        t = _random_unitary(rng, k)
        s = unitary_to_superop(u)
        first = tuple(range(k))
        assert float(average_gate_fidelity(u, t)) == float(
            average_gate_fidelity(u, t, subspace=first)
        )
        assert float(leakage(u, k)) == pytest.approx(float(leakage(u, subspace=first)), abs=1e-15)
        assert float(superop_average_gate_fidelity(s, t)) == pytest.approx(
            float(superop_average_gate_fidelity(s, t, subspace=first)), abs=1e-14
        )
        assert float(superop_leakage(s, k)) == pytest.approx(
            float(superop_leakage(s, subspace=first)), abs=1e-14
        )
        assert float(superop_process_fidelity(s, t)) == pytest.approx(
            float(superop_process_fidelity(s, t, subspace=first)), abs=1e-14
        )


def test_subspace_selects_and_orders_states():
    """F with subspace idx equals F of the permuted unitary on the first k levels."""
    rng = np.random.default_rng(4)
    u = _random_unitary(rng, 9)
    t = _random_unitary(rng, 4)
    perm = list(IDX) + [i for i in range(9) if i not in IDX]
    up = u[np.ix_(perm, perm)]
    f = float(average_gate_fidelity(u, t, subspace=IDX))
    assert f == pytest.approx(float(average_gate_fidelity(up, t)), abs=1e-14)
    lk = float(leakage(u, subspace=IDX))
    assert lk == pytest.approx(float(leakage(up, 4)), abs=1e-14)
    assert lk == pytest.approx(1 - np.sum(np.abs(u[np.ix_(IDX, IDX)]) ** 2) / 4, abs=1e-14)
    s = unitary_to_superop(u)
    assert float(superop_average_gate_fidelity(s, t, subspace=IDX)) == pytest.approx(f, abs=1e-12)
    assert float(superop_leakage(s, subspace=IDX)) == pytest.approx(lk, abs=1e-12)
    # order matters: reversing the subspace is a different embedding
    rev = tuple(reversed(IDX))
    assert abs(float(average_gate_fidelity(u, t, subspace=rev)) - f) > 1e-3
    # batched over leading dimensions
    batch = jnp.stack([jnp.asarray(u), jnp.eye(9, dtype=complex)])
    fb = average_gate_fidelity(batch, t, subspace=IDX)
    assert fb.shape == (2,) and float(fb[0]) == pytest.approx(f, abs=1e-14)


def test_subspace_validation():
    u = jnp.eye(9)
    with pytest.raises(ValueError):
        average_gate_fidelity(u, jnp.eye(4), subspace=(0, 1, 3))
    with pytest.raises(ValueError):
        average_gate_fidelity(u, jnp.eye(4), subspace=(0, 1, 1, 4))
    with pytest.raises(ValueError):
        leakage(u, subspace=(0, 1, 3, 9))
    m = TwoTransmon()
    with pytest.raises(ValueError):
        GateControl(model=m, target=jnp.eye(4), subspace=(0, 1, 3))
    with pytest.raises(ValueError):
        GateControl(model=m, target=jnp.eye(4), subspace=(0, 1, 3, 12))
    env = GateControl(model=m, target=jnp.eye(4), subspace=np.array([0, 1, 3, 4]))
    assert env.subspace == IDX and isinstance(env.subspace[0], int)
    # transmon task untouched: subspace None
    assert qenvs.make("transmon/XGate-v0").subspace is None


def test_gate_env_subspace_equals_explicit_first_levels():
    """GateControl(subspace=(0, 1)) on a transmon reproduces the default behaviour."""
    a = np.random.default_rng(5).uniform(-0.3, 0.3, size=(6, 2))
    e0 = GateControl(model=Transmon(), target=gate("SX"), horizon=6)
    e1 = GateControl(model=Transmon(), target=gate("SX"), horizon=6, subspace=(0, 1))
    s0, r0 = e0.rollout(jnp.asarray(a))
    s1, r1 = e1.rollout(jnp.asarray(a))
    assert np.allclose(np.asarray(r0), np.asarray(r1), atol=1e-15)
    assert float(e0.leakage(s0.qstate)) == pytest.approx(float(e1.leakage(s1.qstate)), abs=1e-15)


# ------------------------------------------------------------------ targets and conventions
def test_zx_gate_and_qiskit_rzx_convention():
    """zx_gate(theta) in control (x) target order equals Qiskit's RZXGate(theta) on (q0, q1)
    after reordering Qiskit's little-endian basis |q1 q0>. The Qiskit matrix below is
    transcribed from the RZXGate documentation (Qiskit is not imported here)."""
    for theta in (math.pi / 2, 0.7):
        c, s = math.cos(theta / 2), math.sin(theta / 2)
        qiskit = np.array(
            [
                [c, 0, -1j * s, 0],
                [0, c, 0, 1j * s],
                [-1j * s, 0, c, 0],
                [0, 1j * s, 0, c],
            ]
        )
        swap = np.eye(4)[[0, 2, 1, 3]]  # |q1 q0> <-> |q0 q1>
        ours = np.asarray(cr.zx_gate(theta))
        assert np.allclose(swap @ qiskit @ swap, ours, atol=1e-15)
        assert not np.allclose(qiskit, ours)
    z, x = np.diag([1, -1]), np.array([[0, 1], [1, 0]])
    zx = np.asarray(cr.zx_gate())
    assert np.allclose(zx, (np.eye(4) - 1j * np.kron(z, x)) / math.sqrt(2))
    emb = np.eye(9, dtype=complex)
    emb[np.ix_(IDX, IDX)] = zx
    th = np.asarray(cr.conditional_angles(jnp.asarray(emb), TwoTransmon()))
    assert np.allclose(th, [math.pi / 2, -math.pi / 2], atol=1e-12)


def test_cr_target_pulls_back_the_frame_change():
    m = TwoTransmon()
    for duration in (320.0, 51.6):
        t = np.asarray(cr.cr_target(m, duration))
        phi = TWO_PI * -0.1 * duration
        d = np.diag([1, 1, np.exp(1j * phi), np.exp(1j * phi)])
        assert np.allclose(d @ t, np.asarray(cr.zx_gate()), atol=1e-12)
        # F(identity) = (2 + cos(Delta T)) / 5
        f0 = float(average_gate_fidelity(np.eye(4), t))
        assert f0 == pytest.approx((2 + math.cos(phi)) / 5, abs=1e-12)


# ------------------------------------------------------------------ task
def test_task_registered_with_metadata():
    spec = qenvs.registry["transmon/CRGate-v0"]
    assert spec.model == "qenvs.models.two_transmon.TwoTransmon"
    assert "ns" in spec.description and "rad/ns" in spec.description
    assert set(spec.kwargs) == {
        "horizon",
        "dt",
        "action_bound",
        "max_rabi_ghz",
        "levels",
        "detuning_ghz",
        "anharmonicity_c_ghz",
        "anharmonicity_t_ghz",
        "coupling_ghz",
        "t1_c_us",
        "t2_c_us",
        "t1_t_us",
        "t2_t_us",
        "method",
    }
    assert any("Magesan" in r for r in spec.references)
    assert spec.units == cr.UNITS and spec.reference is cr.reference_actions


def test_reset_dims_and_info():
    env = qenvs.make("transmon/CRGate-v0")
    _, ts = env.reset(jax.random.PRNGKey(0))
    assert (env.horizon, env.dt, env.duration) == (64, 5.0, 320.0)
    assert env.action_bound == pytest.approx(TWO_PI * 0.06)
    assert env.action_dim == 4 and env.map_dim == 9 and env.observation_dim == 2 * 81 + 1
    assert set(ts.info) == {"p00", "p01", "p10", "p11", "state_leakage", "fidelity", "leakage"}
    assert float(ts.info["fidelity"]) == pytest.approx(0.6)  # Delta T = -64 pi: D = 1
    assert float(ts.info["p00"]) == 1.0 and float(ts.info["leakage"]) == 0.0
    open_env = qenvs.make("transmon/CRGate-v0", t1_c_us=100.0, t2_t_us=80.0, horizon=4)
    _, ts = open_env.reset(jax.random.PRNGKey(0))
    assert open_env.observation_dim == 2 * 81**2 + 1
    assert float(ts.info["fidelity"]) == pytest.approx((2 + math.cos(TWO_PI * -0.1 * 20)) / 5)


def test_task_kwargs_reach_model_and_env():
    env = qenvs.make(
        "transmon/CRGate-v0",
        levels=4,
        detuning_ghz=-0.07,
        anharmonicity_c_ghz=-0.31,
        anharmonicity_t_ghz=-0.32,
        coupling_ghz=0.0021,
        t1_c_us=101.0,
        t2_c_us=81.0,
        t1_t_us=102.0,
        t2_t_us=82.0,
        horizon=7,
        dt=3.5,
        action_bound=0.9,
        method=dq.method.Tsit5(),
    )
    m = env.model
    got = (
        m.levels,
        m.detuning_ghz,
        m.anharmonicity_c_ghz,
        m.anharmonicity_t_ghz,
        m.coupling_ghz,
        m.t1_c_us,
        m.t2_c_us,
        m.t1_t_us,
        m.t2_t_us,
    )
    assert got == (4, -0.07, -0.31, -0.32, 0.0021, 101.0, 81.0, 102.0, 82.0)
    assert (env.horizon, env.dt, env.action_bound) == (7, 3.5, 0.9)
    assert isinstance(env.method, dq.method.Tsit5)
    assert env.subspace == (0, 1, 4, 5)  # levels 4
    assert np.allclose(np.asarray(env.target), np.asarray(cr.cr_target(m, 24.5)))
    assert m.is_open and env.observation_dim == 2 * 16**4 + 1
    rabi = qenvs.make("transmon/CRGate-v0", max_rabi_ghz=0.02)
    assert rabi.action_bound == pytest.approx(TWO_PI * 0.02)
    with pytest.raises(ValueError):
        qenvs.make("transmon/CRGate-v0", max_rabi_ghz=0.02, action_bound=0.1)
    with pytest.raises(ValueError):
        qenvs.make("transmon/CRGate-v0", t1_c_us=10.0, t2_c_us=30.0)
    with pytest.raises(TypeError):
        qenvs.make("transmon/CRGate-v0", gamma1=0.1)
    d = qenvs.make("transmon/CRGate-v0").model
    assert (d.levels, d.detuning_ghz, d.anharmonicity_c_ghz, d.anharmonicity_t_ghz) == (
        3,
        -0.1,
        -0.34,
        -0.34,
    )
    assert (d.coupling_ghz, d.t1_c_us, d.t2_c_us, d.t1_t_us, d.t2_t_us) == (
        0.003,
        None,
        None,
        None,
        None,
    )
    two = qenvs.make("transmon/CRGate-v0", levels=2, horizon=3)
    assert two.subspace == (0, 1, 2, 3) and two.model.dim == 4


def test_info_populations_follow_the_map():
    """info p_xy = |U[idx_xy, idx_10]|^2 for initial '10' (a transposed map would differ)."""
    a = np.random.default_rng(6).uniform(-0.3, 0.3, size=(6, 4))
    m = TwoTransmon(initial="10", coupling_ghz=0.02)
    env = GateControl(model=m, target=cr.cr_target(m, 6.0), horizon=6, dt=1.0, subspace=IDX)
    state, _ = env.rollout(jnp.asarray(a))
    u = np.asarray(state.qstate.to_jax())
    info = env.info(state)
    col = np.abs(u[:, 3]) ** 2
    row = np.abs(u[3, :]) ** 2
    assert np.max(np.abs(col - row)) > 1e-3
    for name, i in zip(("p00", "p01", "p10", "p11"), IDX, strict=True):
        assert float(info[name]) == pytest.approx(col[i], abs=1e-12)
    assert float(info["state_leakage"]) == pytest.approx(1 - col[list(IDX)].sum(), abs=1e-12)


# ------------------------------------------------------------------ reference and GRAPE
def test_reference_pulse(reference):
    env, actions, state, rewards = reference
    assert actions.shape == (64, 4)
    assert float(jnp.abs(actions).max()) <= env.action_bound
    a = np.asarray(actions)
    assert np.all(a[:, 1] == 0) and np.all(a[:, 3] == 0)
    ratio = 0.003 / (-0.1 - 0.34)
    on = a[:, 0] > 0
    assert np.allclose(a[on, 2] / a[on, 0], ratio)
    f = float(env.fidelity(state.qstate))
    assert f == pytest.approx(0.98432, abs=2e-4)  # measured 0.984316 (x64, CPU)
    assert float(env.leakage(state.qstate)) < 2e-4  # measured 1.3e-4
    assert float(rewards.sum()) == pytest.approx(f - 0.6, abs=1e-9)
    uq = to_qubit_frame(env.model, state.qstate.to_jax(), env.duration)
    th = np.asarray(cr.conditional_angles(uq, env.model))
    assert (th[0] - th[1]) / 2 == pytest.approx(math.pi / 2, abs=1e-4)
    # with T1 = T2 = 100 us on both transmons the same pulse loses about 0.4 percent more
    noisy = qenvs.make(
        "transmon/CRGate-v0", t1_c_us=100.0, t2_c_us=100.0, t1_t_us=100.0, t2_t_us=100.0
    )
    s_open, _ = noisy.rollout(actions)
    f_open = float(noisy.fidelity(s_open.qstate))
    assert 0.978 < f_open < f  # measured 0.98057


def test_reference_requires_feasible_bound_and_two_transmon():
    with pytest.raises(ValueError):
        cr.reference_actions(qenvs.make("transmon/CRGate-v0", max_rabi_ghz=0.002, horizon=16))
    with pytest.raises(ValueError):
        cr.reference_actions(qenvs.make("transmon/XGate-v0"))


def test_open_env_leakage_uses_the_computational_subspace(reference):
    """With T1 / T2 the superoperator leakage must be taken on the 4 qubit states of the
    9-level space. Without the subspace the first 4 levels (|00>, |01>, |02>, |10>) are
    treated as computational and the leakage of the reference pulse is 0.124, not 1.4e-4."""
    env, actions, state, _ = reference
    noisy = qenvs.make(
        "transmon/CRGate-v0", t1_c_us=100.0, t2_c_us=100.0, t1_t_us=100.0, t2_t_us=100.0
    )
    s_open, _ = noisy.rollout(actions)
    lk_open = float(noisy.leakage(s_open.qstate))
    lk_closed = float(env.leakage(state.qstate))
    assert lk_open < 3e-4  # measured 1.41e-4
    assert abs(lk_open - lk_closed) < 3e-5  # T1 / T2 = 100 us adds about 7e-6 (closed 1.34e-4)
    assert lk_open == pytest.approx(
        float(superop_leakage(s_open.qstate, 4, subspace=IDX)), abs=1e-12
    )
    assert float(superop_leakage(s_open.qstate, 4)) > 0.1  # without the subspace
    _, ts = noisy.reset(jax.random.PRNGKey(0))
    assert float(ts.info["leakage"]) == 0.0


def test_reference_positive_detuning_uses_negative_amplitude():
    """Delta = +0.1 GHz (straddling, 0 < Delta < -alpha_c): nu_ZX < 0, so the CR amplitude that
    gives the +pi/2 conditional rotation is negative; the reference must find it."""
    env = qenvs.make("transmon/CRGate-v0", detuning_ghz=0.1)
    actions = cr.reference_actions(env)
    a = np.asarray(actions)
    assert float(jnp.abs(actions).max()) <= env.action_bound
    on = a[:, 0] != 0
    assert np.all(a[on, 0] < 0) and np.all(a[:, 1] == 0) and np.all(a[:, 3] == 0)
    assert np.allclose(a[on, 2] / a[on, 0], 0.003 / (0.1 - 0.34))
    state, _ = env.rollout(actions)
    f = float(env.fidelity(state.qstate))
    assert f > 0.99  # measured 0.99626 (x64, CPU), pulse of 180 ns, |A| / 2 pi = 46.3 MHz
    th = np.asarray(
        cr.conditional_angles(
            to_qubit_frame(env.model, state.qstate.to_jax(), env.duration), env.model
        )
    )
    assert (th[0] - th[1]) / 2 == pytest.approx(math.pi / 2, abs=1e-4)
    # the amplitude sign follows the first-order nu_ZX
    assert cr.cr_amplitude_sign(env.model) == -1.0
    assert cr.cr_amplitude_sign(TwoTransmon()) == 1.0


def test_reference_two_level_model():
    """levels = 2: dressing the control qubit by the drive gives H_eff = -(J Omega / 2 Delta)
    Z X and no IX term, so the cancellation ratio is 0 (the alpha_c -> -inf limit of
    J / (Delta + alpha_c)), not the three-level -0.0068. The reference is at least as good as
    the three-level one (0.99476 against 0.98432) and the sign search works for Delta > 0."""
    assert cr.cancellation_ratio(TwoTransmon(levels=2)) == 0.0
    assert cr.cancellation_ratio(TwoTransmon()) == pytest.approx(0.003 / -0.44)
    f = {}
    for kw in ({}, {"detuning_ghz": 0.1}):
        env = qenvs.make("transmon/CRGate-v0", levels=2, **kw)
        actions = cr.reference_actions(env)
        assert np.all(np.asarray(actions)[:, 2] == 0)
        state, _ = env.rollout(actions)
        f[bool(kw)] = float(env.fidelity(state.qstate))
        assert float(env.leakage(state.qstate)) < 1e-12
    f3 = 0.98432  # three-level default, see test_reference_pulse
    assert f[False] > f3 - 0.005 and f[True] > f3 - 0.005  # measured 0.99476 and 0.99476
    # the three-level ratio is the wrong tone for two levels: it leaves an IX term
    env = qenvs.make("transmon/CRGate-v0", levels=2)
    base = cr.reference_actions(env)
    wrong = base.at[:, 2].set(base[:, 0] * (0.003 / (-0.1 - 0.34)))
    f_wrong = float(env.fidelity(env.rollout(wrong)[0].qstate))
    assert f_wrong < f[False] - 0.01


def test_reference_unequal_anharmonicities_and_noisy_env():
    """The tone ratio uses the control anharmonicity, J / (Delta + alpha_c), and the reference
    is calibrated on the closed model: a noisy env gives the same actions as the closed one."""
    kw = {"anharmonicity_c_ghz": -0.30, "anharmonicity_t_ghz": -0.34}
    closed = qenvs.make("transmon/CRGate-v0", **kw)
    noisy = qenvs.make(
        "transmon/CRGate-v0", t1_c_us=100.0, t2_c_us=80.0, t1_t_us=90.0, t2_t_us=70.0, **kw
    )
    a_closed = np.asarray(cr.reference_actions(closed))
    a_noisy = np.asarray(cr.reference_actions(noisy))
    assert np.array_equal(a_closed, a_noisy)
    on = a_closed[:, 0] > 0
    ratio = a_closed[on, 2] / a_closed[on, 0]
    assert np.allclose(ratio, 0.003 / (-0.1 - 0.30))  # alpha_c, not alpha_t (-0.0068)
    assert not np.allclose(ratio, 0.003 / (-0.1 - 0.34))
    state, _ = closed.rollout(jnp.asarray(a_closed))
    assert float(closed.fidelity(state.qstate)) > 0.975  # measured 0.98134 (x64, CPU)


@pytest.mark.slow
def test_grape_improves_on_the_reference(reference):
    env, _, state, _ = reference
    f_ref = float(env.fidelity(state.qstate))
    out = grape_jax.run_grape("transmon/CRGate-v0", steps=150, lr=0.05, seed=0, history_every=50)
    f_best = out["best_return"] + 0.6
    f_final = out["final_return"] + 0.6
    # measured (x64, CPU): best 0.99805, final 0.98894 (Adam oscillates), reference 0.98432
    assert f_best > f_ref + 0.01
    assert f_final > f_ref
    assert out["best_actions"]["max_abs"][0] <= out["action_bound"] + 1e-9


@pytest.mark.parametrize("open_system", [False, True])
def test_step_is_jit_vmap_and_grad_compatible(open_system):
    kw = {"t1_c_us": 50.0, "t2_t_us": 40.0} if open_system else {}
    env = qenvs.make("transmon/CRGate-v0", horizon=3, **kw)
    keys = jax.random.split(jax.random.PRNGKey(1), 3)
    states, _ = jax.vmap(env.reset)(keys)
    actions = jnp.linspace(-0.2, 0.2, 12).reshape(3, 4)
    states, ts = jax.jit(jax.vmap(env.step))(states, actions)
    assert ts.observation.shape == (3, env.observation_dim) and ts.reward.shape == (3,)
    for k in ("fidelity", "leakage", "p00", "p11", "state_leakage"):
        assert ts.info[k].shape == (3,)

    def loss(a):
        _, r = env.rollout(a)
        return -r.sum()

    a = 0.1 * jax.random.normal(jax.random.PRNGKey(3), (3, 4))
    g = jax.grad(loss)(a)
    assert bool(jnp.all(jnp.isfinite(g))) and float(jnp.abs(g).max()) > 0
    eps = 1e-5
    e = jnp.zeros_like(a).at[1, 2].set(eps)
    fd = (loss(a + e) - loss(a - e)) / (2 * eps)
    assert float(g[1, 2]) == pytest.approx(float(fd), rel=1e-4, abs=1e-8)
