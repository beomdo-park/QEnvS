"""Validation checks for the three-level Lambda system (STIRAP), closed and open.

Reference pulses for the STIRAP checks: horizon 100, ``dt = 0.6`` (``T = 60``),
peak Rabi frequency ``Omega_0 = 1``, Gaussian ``1/e`` half-width 12
(``Omega_0 * width = 12``, adiabatic) and half-delay ``tau = 6``. The
piecewise-constant sampling of the environment reproduces the continuous-time
Gaussian solution to better than ``1e-4`` in ``P_3``, so the numbers below are
properties of STIRAP itself, not of the control grid.
"""

from __future__ import annotations

from collections.abc import Callable

import dynamiqs as dq
import jax
import jax.numpy as jnp
import numpy as np
from dynamiqs.method import Method

from qenvs.envs.state_transfer import StateTransfer
from qenvs.models.lambda_system import LambdaSystem, stirap_pulses
from qenvs.validation.core import CheckResult, skipped
from qenvs.validation.core import result as _result

MODEL = "LambdaSystem"

# Reference STIRAP pulse pair shared by the transfer checks.
_HORIZON = 100
_DT = 0.6
_OMEGA0 = 1.0
_TAU = 6.0
_WIDTH = 12.0
_OPEN_GAMMA = 0.5 * _OMEGA0


# -------------------------------------------------------------------- helpers
def _env(
    gamma: float = 0.0,
    *,
    horizon: int,
    dt: float,
    action_bound: float = 1.0,
    method: Method | None = None,
    **model_kw: float,
) -> StateTransfer:
    """Lambda-system state-transfer env towards |3> (ket if closed, dm if open)."""
    model = LambdaSystem(gamma=gamma, **model_kw)
    target = dq.basis(3, 2)
    target = target.todm() if model.is_open else target
    extra = {} if method is None else {"method": method}
    return StateTransfer(
        model=model, target=target, horizon=horizon, dt=dt, action_bound=action_bound, **extra
    )


def _stirap_actions(order: str) -> jax.Array:
    return stirap_pulses(_HORIZON, _DT, _OMEGA0, _TAU, _WIDTH, order=order)


def _trajectory(
    env: StateTransfer, actions: jax.Array
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Repeated ``env.step``; returns per-step ``(P2, P3, reward)`` arrays."""
    step = jax.jit(lambda s, a: env.step(s, a))
    state, _ = env.reset(jax.random.PRNGKey(0))
    p2, p3, rewards = [], [], []
    for a in actions:
        state, ts = step(state, a)
        p2.append(float(ts.info["p2"]))
        p3.append(float(ts.info["p3"]))
        rewards.append(float(ts.reward))
    return np.asarray(p2), np.asarray(p3), np.asarray(rewards)


def _final_p3(env: StateTransfer, actions: jax.Array) -> float:
    state, _ = env.rollout(actions)
    return float(dq.expect(env.model.observables()["p3"], state.qstate).real)


# --------------------------------------------------------------------- checks
def check_dark_state() -> CheckResult:
    """|D> ~ Omega_s |1> - Omega_p |3> is an exact zero-eigenvector of H (delta = 0, any Delta)."""
    rng = np.random.default_rng(11)
    err = 0.0
    for _ in range(6):
        # float32-representable amplitudes so the matrix-vector product below is exact in
        # float64 whatever precision the model was built with.
        omega_p, omega_s, delta = rng.uniform(-2.0, 2.0, 3).astype(np.float32)
        m = LambdaSystem(delta_one_photon=float(delta))
        h = np.asarray(m.hamiltonian(jnp.array([omega_p, omega_s])).to_jax(), dtype=np.complex128)
        dark = np.array([omega_s, 0.0, -omega_p], dtype=np.complex128)
        dark /= np.linalg.norm(dark)
        err = max(err, float(np.linalg.norm(h @ dark)))
    return _result(
        "dark_state_zero_eigenvector",
        "analytic",
        err,
        1e-10,
        "max ||H |D>|| over 6 random (Omega_p, Omega_s, Delta), delta = 0",
        model=MODEL,
    )


def check_adiabatic_eigenvalues() -> CheckResult:
    """Eigenvalues of H (delta = 0): 0 and (Delta +- sqrt(Delta^2 + Omega_p^2 + Omega_s^2)) / 2."""
    rng = np.random.default_rng(12)
    err = 0.0
    for _ in range(6):
        omega_p, omega_s, delta = rng.uniform(-2.0, 2.0, 3).astype(np.float32)
        m = LambdaSystem(delta_one_photon=float(delta))
        h = np.asarray(m.hamiltonian(jnp.array([omega_p, omega_s])).to_jax(), dtype=np.complex128)
        d, p, s = float(delta), float(omega_p), float(omega_s)
        root = np.sqrt(d * d + p * p + s * s)
        ref = np.sort([0.0, 0.5 * (d + root), 0.5 * (d - root)])
        err = max(err, float(np.max(np.abs(np.linalg.eigvalsh(h) - ref))))
    return _result(
        "adiabatic_state_eigenvalues",
        "analytic",
        err,
        1e-9,
        "max |eig(H) - {0, (Delta +- Omega_gen)/2}| over 6 random parameter sets",
        model=MODEL,
    )


def check_two_level_limit() -> CheckResult:
    """Omega_s = 0, Delta = 0, constant Omega_p: P2(t) = sin^2(Omega_p t / 2)."""
    u = 1.3
    m = LambdaSystem()
    ts = jnp.linspace(0.0, 2 * jnp.pi / u, 17)
    res = dq.sesolve(
        m.hamiltonian(jnp.array([u, 0.0])),
        m.initial_state(),
        ts,
        method=dq.method.Tsit5(rtol=1e-9, atol=1e-11),
        progress_meter=False,
    )
    p2 = np.asarray(dq.expect(m.observables()["p2"], res.states).real)
    err = np.max(np.abs(p2 - np.sin(u * np.asarray(ts) / 2) ** 2))
    return _result(
        "two_level_limit_rabi",
        "analytic",
        err,
        1e-6,
        "max |P2 - sin^2(Omega_p t / 2)| over one Rabi period, Stokes off",
        model=MODEL,
    )


def check_spontaneous_emission_branching() -> CheckResult:
    """Open system from |2>, no drive: P2 = exp(-gamma t), P1 = P3 = (1 - exp(-gamma t)) / 2."""
    g = 0.4
    m = LambdaSystem(gamma=g, initial="2")
    ts = jnp.linspace(0.0, 6.0, 13)
    res = dq.mesolve(
        m.drift(),
        m.jump_ops(),
        m.initial_state(),
        ts,
        method=dq.method.Tsit5(rtol=1e-9, atol=1e-11),
        progress_meter=False,
    )
    obs = m.observables()
    p1, p2, p3 = (np.asarray(dq.expect(obs[k], res.states).real) for k in ("p1", "p2", "p3"))
    decay = np.exp(-g * np.asarray(ts))
    err = max(
        np.max(np.abs(p2 - decay)),
        np.max(np.abs(p1 - 0.5 * (1 - decay))),
        np.max(np.abs(p3 - 0.5 * (1 - decay))),
    )
    return _result(
        "spontaneous_emission_branching",
        "open_system",
        err,
        1e-6,
        "max |P_i - analytic| for P2 = exp(-gamma t), P1 = P3 = (1 - exp(-gamma t)) / 2",
        model=MODEL,
    )


def check_stirap_transfer() -> CheckResult:
    """Closed STIRAP (counter-intuitive order, adiabatic): final P3 > 0.99."""
    env = _env(horizon=_HORIZON, dt=_DT)
    p2, p3, _ = _trajectory(env, _stirap_actions("counterintuitive"))
    return _result(
        "stirap_transfer",
        "control",
        1.0 - p3[-1],
        1e-2,
        f"1 - P3(T) for Omega_0*width={_OMEGA0 * _WIDTH:g}, tau/width={_TAU / _WIDTH:g}; "
        f"max_t P2={p2.max():.3e}",
        model=MODEL,
    )


def check_stirap_intermediate_population() -> CheckResult:
    """Closed STIRAP (counter-intuitive order): the lossy state |2> stays below 5 %."""
    env = _env(horizon=_HORIZON, dt=_DT)
    p2, p3, _ = _trajectory(env, _stirap_actions("counterintuitive"))
    return _result(
        "stirap_intermediate_population",
        "control",
        p2.max(),
        5e-2,
        f"max_t P2 along the counter-intuitive sequence; final P3={p3[-1]:.4f}",
        model=MODEL,
    )


def check_stirap_intuitive_order() -> CheckResult:
    """Intuitive order (pump first) populates |2> strongly: max_t P2 > 0.3.

    With the pump first, |1> is not the dark state but an equal superposition of the
    two bright states, so the population oscillates through |2| (P2 reaches ~1) and
    the final P3 = cos^2(phi) depends on the accumulated pulse area rather than on
    adiabaticity. The intermediate population is therefore the robust discriminator.
    """
    env = _env(horizon=_HORIZON, dt=_DT)
    p2, p3, _ = _trajectory(env, _stirap_actions("intuitive"))
    return _result(
        "stirap_intuitive_order",
        "control",
        1.0 - p2.max(),
        0.7,
        f"1 - max_t P2 (passes iff max P2 > 0.3); max P2={p2.max():.3f}, final P3={p3[-1]:.3f}",
        model=MODEL,
    )


def check_stirap_emission_robustness() -> CheckResult:
    """With gamma = 0.5 Omega_0, STIRAP still reaches P3 > 0.95 and beats the intuitive order."""
    env = _env(_OPEN_GAMMA, horizon=_HORIZON, dt=_DT)
    p3_ci = _final_p3(env, _stirap_actions("counterintuitive"))
    p3_in = _final_p3(env, _stirap_actions("intuitive"))
    err = 1.0 - p3_ci if p3_ci > p3_in else 1.0
    return _result(
        "stirap_emission_robustness",
        "open_system",
        err,
        5e-2,
        f"1 - P3(T) with gamma={_OPEN_GAMMA:g}: counter-intuitive P3={p3_ci:.4f}, "
        f"intuitive P3={p3_in:.4f} (error forced to 1 unless counter-intuitive is better)",
        model=MODEL,
    )


def check_adiabaticity_sensitivity() -> CheckResult:
    """Omega_0 * width = 1.5 (diabatic): the same pulse shapes give P3 < 0.9."""
    horizon, dt, width, tau = 100, 0.1, 1.5, 1.0
    env = _env(horizon=horizon, dt=dt)
    p3 = _final_p3(env, stirap_pulses(horizon, dt, _OMEGA0, tau, width))
    return _result(
        "stirap_adiabaticity_sensitivity",
        "control",
        p3,
        0.9,
        f"P3(T) for Omega_0*width=1.5, tau/width={tau / width:.2g} (adiabatic case: "
        f"Omega_0*width={_OMEGA0 * _WIDTH:g} gives > 0.99)",
        model=MODEL,
    )


def _random_open_env(horizon: int, dt: float, seed: int) -> tuple[StateTransfer, np.ndarray]:
    env = _env(0.3, horizon=horizon, dt=dt, delta_one_photon=0.2, delta_two_photon=0.1)
    rng = np.random.default_rng(seed)
    actions = rng.uniform(-1.0, 1.0, (horizon, 2)).astype(np.float32)
    return env, actions


def check_qutip_cross_check() -> CheckResult:
    """Open Lambda system under random piecewise-constant pump/Stokes vs QuTiP 5 mesolve."""
    try:
        import qutip as qt
    except ImportError:
        return skipped(
            "qutip_cross_check",
            "cross_check",
            1e-5,
            "not run: qutip is not installed (pip install 'qenvs[dev]')",
            model=MODEL,
        )

    horizon, dt = 12, 0.3
    env, actions = _random_open_env(horizon, dt, seed=3)
    model = env.model
    state, _ = env.rollout(jnp.asarray(actions))
    rho_dq = np.asarray(state.qstate.to_jax())

    k1, k2, k3 = (qt.basis(3, i) for i in range(3))
    h0 = model.delta_one_photon * k2 * k2.dag() + model.delta_two_photon * k3 * k3.dag()
    hp = 0.5 * (k1 * k2.dag() + k2 * k1.dag())
    hs = 0.5 * (k3 * k2.dag() + k2 * k3.dag())
    rate = np.sqrt(model.gamma / 2.0)
    c_ops = [rate * k1 * k2.dag(), rate * k3 * k2.dag()]
    options = {"atol": 1e-10, "rtol": 1e-10, "nsteps": 100_000, "store_final_state": True}
    rho = k1 * k1.dag()
    for omega_p, omega_s in actions:
        h = h0 + float(omega_p) * hp + float(omega_s) * hs
        rho = qt.mesolve(h, rho, [0.0, dt], c_ops, options=options).final_state
    err = np.max(np.abs(rho_dq - np.asarray(rho.full())))
    return _result(
        "qutip_cross_check",
        "cross_check",
        err,
        1e-5,
        f"max |rho_qenvs - rho_qutip| after {horizon} random pump/Stokes steps "
        f"(gamma={model.gamma:g}, Delta={model.delta_one_photon:g}, "
        f"delta={model.delta_two_photon:g}), QuTiP {qt.__version__}",
        model=MODEL,
    )


def check_density_matrix_constraints() -> CheckResult:
    """Open system under random driving: rho stays Hermitian, trace-1 and positive."""
    env, actions = _random_open_env(15, 0.3, seed=7)
    state, _ = env.rollout(jnp.asarray(actions))
    rho = np.asarray(state.qstate.to_jax())
    herm = np.max(np.abs(rho - rho.conj().T))
    tr = abs(np.trace(rho).real - 1.0)
    neg = max(0.0, -float(np.min(np.linalg.eigvalsh(rho))))
    err = max(herm, tr, neg)
    return _result(
        "density_matrix_constraints",
        "constraints",
        err,
        1e-6,
        f"hermiticity={herm:.2e}, |tr-1|={tr:.2e}, negativity={neg:.2e}",
        model=MODEL,
    )


def check_solver_convergence() -> CheckResult:
    """Adaptive Tsit5 vs exact matrix exponential on the open random-driving rollout."""
    env_a, actions = _random_open_env(12, 0.3, seed=5)
    env_b = _env(
        env_a.model.gamma,
        horizon=env_a.horizon,
        dt=env_a.dt,
        method=dq.method.Expm(),
        delta_one_photon=env_a.model.delta_one_photon,
        delta_two_photon=env_a.model.delta_two_photon,
    )
    sa, _ = env_a.rollout(jnp.asarray(actions))
    sb, _ = env_b.rollout(jnp.asarray(actions))
    err = np.max(np.abs(np.asarray(sa.qstate.to_jax()) - np.asarray(sb.qstate.to_jax())))
    return _result(
        "solver_convergence",
        "numerics",
        err,
        1e-6,
        "max |rho_Tsit5 - rho_Expm| at episode end",
        model=MODEL,
    )


def check_env_stirap_return() -> CheckResult:
    """Functional env: the STIRAP return equals F(final) - F(initial)."""
    env = _env(horizon=_HORIZON, dt=_DT)
    actions = _stirap_actions("counterintuitive")
    state, rewards = env.rollout(actions)
    f_final = float(dq.fidelity(state.qstate, env.target).real)
    f_initial = float(dq.fidelity(env.model.initial_state(), env.target).real)
    ret = float(np.sum(np.asarray(rewards, dtype=np.float64)))
    err = abs(ret - (f_final - f_initial))
    return _result(
        "env_stirap_return",
        "control",
        err,
        1e-6,
        f"|sum(rewards) - (F_final - F_initial)|, F_final={f_final:.4f}, F_initial={f_initial:g}",
        model=MODEL,
    )


CHECKS: tuple[Callable[[], CheckResult], ...] = (
    check_dark_state,
    check_adiabatic_eigenvalues,
    check_two_level_limit,
    check_spontaneous_emission_branching,
    check_stirap_transfer,
    check_stirap_intermediate_population,
    check_stirap_intuitive_order,
    check_stirap_emission_robustness,
    check_adiabaticity_sensitivity,
    check_qutip_cross_check,
    check_density_matrix_constraints,
    check_solver_convergence,
    check_env_stirap_return,
)
