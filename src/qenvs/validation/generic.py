"""Generic physics and numerics checks for any :class:`~qenvs.models.Model`.

:func:`model_checks` is the entry point used by ``qenvs validate --model`` and by
:func:`qenvs.toolkit.validate_model`. It needs no closed-form answer, so it runs on
models you wrote yourself (typically :class:`~qenvs.models.custom.HamiltonianModel`).

What the checks establish: the operators are Hermitian and consistently shaped, the
initial state is valid, the integrator preserves the norm (closed) or keeps ``rho``
Hermitian, unit-trace and positive semidefinite (open) along a random piecewise-constant
rollout *through the real environment code*, the default solver agrees with a much tighter
one (and with the exact matrix exponential where that is cheap), QuTiP integrates the same
matrices to the same states, and, if a battery Hamiltonian is given, the thermodynamic
bounds ``0 <= ergotropy <= energy - E_min`` hold.

What they cannot establish: that the matrices are the physics you meant. A sign error in a
coupling, swapped controls, a reversed basis ordering or a wrong rate are valid physics of
a different system and pass every check below. Without ``reference=`` the QuTiP check
(named ``qutip_solver_cross_check``) is built from the same matrices as the model, so it
validates the solver stack, not the matrices; pass ``reference=`` (an independently built
model of the intended physics) to cross-check the matrices too (``qutip_cross_check``). See
``docs/guides/custom-models.md``.

Time step rule (``dt=None``): with ``b = action_bound``,

    w = ||H_0|| + b * sum_k ||H_k|| + sum_j ||L_j||^2,        dt = 1 / w,

with ``||.||`` the spectral (2-) norm and ``w = 0`` mapped to ``dt = 1``. ``w`` bounds the
fastest rate in the dynamics, so each control interval rotates the state by about one
radian at most and ``horizon`` steps span ``horizon / w``. Models whose drift is much
larger than the controls and rates (a lab-frame Hamiltonian) get a tiny ``dt`` and a
short episode; pass ``dt`` explicitly for those.

Tolerances hold in float32 and float64 (see the constants below). The accuracy of the
default adaptive solver (``rtol = atol = 1e-6``) sets the floor of the numerics checks,
not the floating-point precision.

Tolerance scaling rule: the rollout-based checks (norm or density matrix, solver, QuTiP,
battery consistency, ergotropy bounds) accumulate integrator error along the episode, so
their tolerance is the base constant times

    s = max(1, horizon * dt / (REFERENCE_STEPS * dt_default)),        REFERENCE_STEPS = 12,

with ``dt_default = default_dt(model, action_bound)``. ``s`` is the integrated time in units of
the model's fastest rate: 1 for the default rollout (12 steps at the default ``dt``), and
proportional to the episode length or to a coarser ``dt`` beyond it. A valid model therefore
passes at ``horizon=400`` or ``dt = 3 * default``; the static checks (``structure``,
``hermiticity``, ``initial_state_valid``) are not scaled. The reported ``tolerance`` is the
scaled one and the detail states the factor.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Callable

import dynamiqs as dq
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from dynamiqs.method import Expm, Method, Tsit5

from qenvs.envs.state_transfer import StateTransfer
from qenvs.models.base import DEFAULT_METHOD, Model
from qenvs.observables import energy, ergotropy
from qenvs.validation.core import CheckResult, skipped
from qenvs.validation.core import result as _result

TOL_HERMITIAN = 1e-6
"""Relative asymmetry ``max|A - A^dag| / max|A|`` of the operators."""
TOL_STATE = 1e-5
"""Norm, trace, Hermiticity and negativity of the initial state."""
TOL_CONSERVATION = 5e-5
"""Norm drift (closed) or Hermiticity, trace and negativity (open) along the rollout."""
TOL_SOLVER = 5e-5
"""Maximum entry-wise state difference between the method under test and a reference solver."""
TOL_QUTIP = 5e-5
"""Maximum entry-wise state difference to QuTiP along the rollout."""
TOL_THERMO = 1e-5
"""Violation of ``0 <= ergotropy <= energy - E_min``, relative to the width of the H_B spectrum."""
EXPM_MAX_DIM_OPEN = 16
"""Largest Hilbert dimension for which the exact Liouvillian exponential is used (open)."""
REFERENCE_STEPS = 12
"""Rollout length (the default ``horizon``) at the default ``dt`` for which the base
tolerances are calibrated; longer or coarser rollouts scale them (see the module doc)."""
JIT_MIN_HORIZON = 48
"""Rollouts longer than this compile the exact-propagator step instead of dispatching it eagerly."""
N_RANDOM_STATES = 8
"""Random density matrices of the battery space added to the ergotropy bound check."""


def _x64() -> bool:
    return bool(jax.config.jax_enable_x64)


def default_dt(model: Model, action_bound: float = 1.0) -> float:
    """Control period used by :func:`model_checks` when ``dt`` is not given (see module doc)."""

    def norm(op: object) -> float:
        return float(np.linalg.norm(np.asarray(op.to_jax()), 2))  # type: ignore[attr-defined]

    w = norm(model.drift())
    w += abs(action_bound) * sum(norm(c) for c in model.controls())
    w += sum(norm(j) ** 2 for j in model.jump_ops())
    return 1.0 / w if w > 0 else 1.0


def tolerance_scale(model: Model, horizon: int, dt: float, action_bound: float = 1.0) -> float:
    """Factor ``max(1, horizon * dt / (REFERENCE_STEPS * default_dt))`` on rollout tolerances."""
    try:
        ref = REFERENCE_STEPS * default_dt(model, action_bound)
    except Exception:  # unusable operators are reported by the structure check
        return 1.0
    return max(1.0, horizon * dt / ref)


def _finite(x: float) -> float:
    """JSON-safe error value: non-finite (NaN, inf) is reported as 1.0."""
    return x if math.isfinite(x) else 1.0


class _Context:
    """Shared state of one :func:`model_checks` run."""

    def __init__(
        self,
        model: Model,
        horizon: int,
        dt: float,
        bound: float,
        seed: int,
        method: Method,
    ) -> None:
        self.model, self.horizon, self.dt, self.bound = model, horizon, dt, bound
        self.seed = seed
        self.method = method
        self.name = type(model).__name__
        self.n = int(model.dim)
        self.scale = tolerance_scale(model, horizon, dt, bound)
        key = jax.random.PRNGKey(seed)
        self.actions = jax.random.uniform(
            key, (horizon, model.n_controls), minval=-bound, maxval=bound
        )
        # the environment casts actions to float32 and clips: feed references the same numbers
        applied = np.asarray(jnp.asarray(self.actions, dtype=jnp.float32), dtype=np.float64)
        self.applied = np.clip(applied, -bound, bound)

    def res(self, name: str, category: str, error: float, tol: float, detail: str) -> CheckResult:
        return _result(name, category, _finite(float(error)), tol, detail, model=self.name)

    def tol(self, base: float) -> float:
        """Rollout tolerance: ``base`` times the integrated-time factor."""
        return base * self.scale

    def scaled(
        self, name: str, category: str, error: float, base_tol: float, detail: str
    ) -> CheckResult:
        """Result of a rollout-based check with the tolerance scaled by the episode length."""
        if self.scale > 1.0:
            detail += (
                f"; tolerance {base_tol:.0e} x {self.scale:.3g} for the integrated time "
                f"(horizon*dt = {self.horizon * self.dt:.4g}, rule in the module doc)"
            )
        return self.res(name, category, error, self.tol(base_tol), detail)

    def fail(self, name: str, category: str, tol: float, exc: Exception) -> CheckResult:
        detail = f"could not run: {type(exc).__name__}: {exc}"
        return _result(name, category, 1.0, tol, detail, model=self.name)

    def rollout(self, method: Method) -> np.ndarray:
        """States ``(horizon + 1, n, 1 or n)`` of the random rollout through ``StateTransfer``."""
        env = StateTransfer(
            model=self.model,
            target=dq.basis(self.n, 0),
            horizon=self.horizon,
            dt=self.dt,
            action_bound=self.bound,
            method=method,
        )
        step = eqx.filter_jit(env.step)
        state, _ = env.reset(jax.random.PRNGKey(0))
        traj = [state.qstate.to_jax()]
        for a in self.actions:
            state, _ = step(state, a)
            traj.append(state.qstate.to_jax())
        return np.asarray(jnp.stack(traj), dtype=np.complex128)

    def exact_rollout(self) -> np.ndarray:
        """Same rollout with ``Expm`` (exact for piecewise-constant H), without the environment.

        dynamiqs 0.3.6 returns composite density matrices integrated with ``Expm`` with
        flattened dims, which no environment can step again, so the dims are restored by
        hand here.
        """
        m = self.model
        dims = tuple(m.drift().dims)

        def advance(state: object, a: jax.Array) -> object:
            h = m.hamiltonian(jnp.asarray(a, dtype=jnp.float32).clip(-self.bound, self.bound))
            return dq.asqarray(m.evolve(state, h, self.dt, method=Expm()).to_jax(), dims=dims)

        if self.horizon > JIT_MIN_HORIZON:  # compiling pays off only for long rollouts
            advance = eqx.filter_jit(advance)

        state = m.initial_state()
        traj = [state.to_jax()]
        for a in self.actions:
            state = advance(state, a)
            traj.append(state.to_jax())
        return np.asarray(jnp.stack(traj), dtype=np.complex128)


def _guarded(
    ctx: _Context, name: str, category: str, tol: float, fn: Callable[[], CheckResult]
) -> CheckResult:
    try:
        return fn()
    except Exception as e:  # report instead of aborting the whole validation
        return ctx.fail(name, category, tol, e)


def _asymmetry(op: object) -> float:
    a = np.asarray(op.to_jax(), dtype=np.complex128)  # type: ignore[attr-defined]
    scale = max(float(np.max(np.abs(a), initial=0.0)), 1e-300)
    return float(np.max(np.abs(a - a.conj().T), initial=0.0)) / scale


def _density(states: np.ndarray) -> np.ndarray:
    """Density matrices from a stack of kets ``(T, n, 1)`` or density matrices ``(T, n, n)``."""
    if states.shape[-1] == 1:
        return states @ np.conj(np.swapaxes(states, -1, -2))
    return states


# ------------------------------------------------------------------- checks
def _check_structure(ctx: _Context) -> CheckResult:
    m, n = ctx.model, ctx.n
    ops: dict[str, object] = {"drift": m.drift()}
    ops.update({f"controls[{k}]": c for k, c in enumerate(m.controls())})
    ops.update({f"jump_ops[{k}]": j for k, j in enumerate(m.jump_ops())})
    ops.update({f"observables[{k!r}]": o for k, o in m.observables().items()})
    with contextlib.suppress(NotImplementedError):
        ops["battery_hamiltonian"] = m.battery_hamiltonian()
    problems = []
    if not m.controls():
        problems.append("no control operators")
    dims0 = tuple(m.drift().dims)
    for name, op in ops.items():
        a = np.asarray(op.to_jax())  # type: ignore[attr-defined]
        if a.shape != (n, n):
            problems.append(f"{name} has shape {a.shape}, expected {(n, n)}")
        if not np.all(np.isfinite(a)):
            problems.append(f"{name} contains NaN or inf")
        d = tuple(op.dims)  # type: ignore[attr-defined]
        if math.prod(d) != n or (len(dims0) > 1 and d != dims0):
            problems.append(f"{name} has dims {d}, inconsistent with the drift dims {dims0}")
    detail = (
        f"{len(ops)} operators, dimension {n}, dims {dims0}, {m.n_controls} controls, "
        f"{len(m.jump_ops())} jump operators"
    )
    zero = [name for name in ops if name.startswith("controls[") and not np.any(ops[name].to_jax())]  # type: ignore[attr-defined]
    if zero:
        detail += f"; warning: {', '.join(zero)} is all zero (actions on it have no effect)"
    if problems:
        detail += "; " + "; ".join(problems)
    return ctx.res("structure", "constraints", float(len(problems)), 0.0, detail)


def _check_hermiticity(ctx: _Context) -> CheckResult:
    m = ctx.model
    ops: dict[str, object] = {"drift": m.drift()}
    ops.update({f"controls[{k}]": c for k, c in enumerate(m.controls())})
    ops.update({f"observables[{k!r}]": o for k, o in m.observables().items()})
    with contextlib.suppress(NotImplementedError):
        ops["battery_hamiltonian"] = m.battery_hamiltonian()
    asym = {name: _asymmetry(op) for name, op in ops.items()}
    worst = max(asym, key=lambda k: asym[k])
    return ctx.res(
        "hermiticity",
        "constraints",
        asym[worst],
        TOL_HERMITIAN,
        f"max |A - A^dagger| / max|A| over {len(ops)} operators (drift, controls, observables, "
        f"battery Hamiltonian); worst: {worst} = {asym[worst]:.2e}",
    )


def _check_initial_state(ctx: _Context) -> CheckResult:
    m = ctx.model
    q = m.initial_state()
    a = np.asarray(q.to_jax(), dtype=np.complex128)
    n = ctx.n
    if m.is_open:
        if a.shape != (n, n):
            return ctx.res(
                "initial_state_valid",
                "constraints",
                1.0,
                TOL_STATE,
                f"open model must carry a density matrix {(n, n)}, initial_state() has "
                f"shape {a.shape}",
            )
        herm = float(np.max(np.abs(a - a.conj().T)))
        tr = abs(float(np.trace(a).real) - 1.0)
        neg = max(0.0, -float(np.linalg.eigvalsh(0.5 * (a + a.conj().T))[0]))
        err = max(herm, tr, neg)
        detail = f"density matrix: hermiticity={herm:.3e}, |tr-1|={tr:.3e}, negativity={neg:.3e}"
    else:
        if a.shape != (n, 1):
            return ctx.res(
                "initial_state_valid",
                "constraints",
                1.0,
                TOL_STATE,
                f"closed model must carry a ket {(n, 1)}, initial_state() has shape {a.shape}",
            )
        err = abs(float(np.vdot(a, a).real) - 1.0)
        detail = f"ket: |<psi|psi> - 1| = {err:.3e}"
    return ctx.res("initial_state_valid", "constraints", err, TOL_STATE, detail)


def _check_conservation(ctx: _Context, states: np.ndarray) -> CheckResult:
    if not ctx.model.is_open:
        norm = np.sqrt(np.sum(np.abs(states) ** 2, axis=(-2, -1)))
        err = float(np.max(np.abs(norm - 1.0)))
        return ctx.scaled(
            "norm_conservation",
            "constraints",
            err,
            TOL_CONSERVATION,
            f"max_t | ||psi(t)|| - 1 | over {ctx.horizon} random piecewise-constant steps "
            f"(dt={ctx.dt:.4g}, |u| <= {ctx.bound:g}), loss and gain both counted",
        )
    rho = states
    herm = float(np.max(np.abs(rho - np.conj(np.swapaxes(rho, -1, -2)))))
    tr = float(np.max(np.abs(np.trace(rho, axis1=-2, axis2=-1).real - 1.0)))
    lam = np.linalg.eigvalsh(0.5 * (rho + np.conj(np.swapaxes(rho, -1, -2))))
    neg = max(0.0, -float(np.min(lam)))
    return ctx.scaled(
        "density_matrix_valid",
        "constraints",
        max(herm, tr, neg),
        TOL_CONSERVATION,
        f"along {ctx.horizon} random steps (dt={ctx.dt:.4g}): hermiticity={herm:.3e}, "
        f"|tr-1|={tr:.3e}, negativity={neg:.3e}",
    )


def _tight_method() -> Method:
    tol = 1e-10 if _x64() else 3e-7
    return Tsit5(rtol=tol, atol=tol)


def _check_solver(ctx: _Context, states: np.ndarray) -> CheckResult:
    tight = ctx.rollout(_tight_method())
    err_tight = float(np.max(np.abs(states - tight)))
    detail = f"max |state(method) - state(tight Tsit5)| = {err_tight:.2e}"
    err = err_tight
    if not ctx.model.is_open or ctx.n <= EXPM_MAX_DIM_OPEN:
        exact = ctx.exact_rollout()
        err_expm = float(np.max(np.abs(states - exact)))
        detail += f"; vs Expm (exact for piecewise-constant H) = {err_expm:.2e}"
        err = max(err, err_expm)
    else:
        detail += f"; Expm skipped (open model, dimension {ctx.n} > {EXPM_MAX_DIM_OPEN})"
    return ctx.scaled(
        "solver_convergence",
        "numerics",
        err,
        TOL_SOLVER,
        f"{detail}; method under test: {ctx.method!r}",
    )


def _check_qutip(ctx: _Context, states: np.ndarray, reference: Model | None) -> CheckResult:
    name = "qutip_cross_check" if reference is not None else "qutip_solver_cross_check"
    try:
        import qutip as qt
    except ImportError:
        return skipped(
            name,
            "cross_check",
            TOL_QUTIP,
            "not run: qutip is not installed (pip install 'qenvs[dev]')",
            model=ctx.name,
        )
    ref = ctx.model if reference is None else reference
    if int(ref.dim) != ctx.n or ref.n_controls != ctx.model.n_controls:
        msg = (
            f"reference has dimension {ref.dim} and {ref.n_controls} controls, the model "
            f"{ctx.n} and {ctx.model.n_controls}"
        )
        raise ValueError(msg)

    def qobj(op: object) -> object:
        return qt.Qobj(np.asarray(op.to_jax(), dtype=np.complex128))  # type: ignore[attr-defined]

    h0 = qobj(ref.drift())
    hc = [qobj(c) for c in ref.controls()]
    c_ops = [qobj(j) for j in ref.jump_ops()]
    a0 = np.asarray(ref.initial_state().to_jax(), dtype=np.complex128)
    mixed = ref.is_open or ctx.model.is_open
    opts = {"atol": 1e-12, "rtol": 1e-10, "nsteps": 100_000, "progress_bar": ""}
    x = qt.Qobj(a0 @ a0.conj().T if (mixed and a0.shape[1] == 1) else a0)
    traj = [x.full()]
    for u in ctx.applied:
        h = h0
        for ui, hk in zip(u, hc, strict=True):
            h = h + float(ui) * hk
        if mixed or x.isoper:
            x = qt.mesolve(h, x, [0.0, ctx.dt], c_ops=c_ops, options=opts).states[-1]
        else:
            x = qt.sesolve(h, x, [0.0, ctx.dt], options=opts).states[-1]
        traj.append(x.full())
    ref_states = np.stack(traj)
    err = float(np.max(np.abs(_density(states) - _density(ref_states))))
    if reference is not None:
        what = "QuTiP integrates the independent reference model, so a wrong operator is detected"
    else:
        what = (
            "QuTiP integrates the model's own matrices: this checks the solver stack only and "
            "does NOT verify that the matrices are the physics you meant (pass reference=)"
        )
    return ctx.scaled(
        name,
        "cross_check",
        err,
        TOL_QUTIP,
        f"max |state_dynamiqs - state_qutip| along {ctx.horizon} random steps (qutip "
        f"{qt.__version__}); {what}",
    )


def _battery_arrays(ctx: _Context, states: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple]:
    """``battery_reduced`` of each rollout state: rho_B, H_B as ``(T, d, d)`` arrays, H_B dims."""
    dims = tuple(ctx.model.drift().dims)
    seen: dict[str, tuple[int, ...]] = {}

    def one(s: jax.Array) -> tuple[jax.Array, jax.Array]:
        rho_b, hb = ctx.model.battery_reduced(dq.asqarray(s, dims=dims))
        seen["dims"] = tuple(hb.dims)
        return rho_b.todm().to_jax(), hb.to_jax()

    rho, hb = jax.vmap(one)(jnp.asarray(states))  # eager vmap: one batched dispatch
    return np.asarray(rho, dtype=np.complex128), np.asarray(hb, dtype=np.complex128), seen["dims"]


def _check_battery_consistency(ctx: _Context, states: np.ndarray) -> CheckResult:
    name = "battery_consistency"
    r, h, _ = _battery_arrays(ctx, states)
    if r.shape[-1] != r.shape[-2] or r.shape != h.shape:
        detail = (
            f"step 0: reduced battery state {r.shape[1:]} and battery Hamiltonian {h.shape[1:]} "
            "do not conform"
        )
        return ctx.res(name, "thermodynamics", 1.0, ctx.tol(TOL_CONSERVATION), detail)
    rh = np.conj(np.swapaxes(r, -1, -2))
    herm = float(np.max(np.abs(r - rh)))
    tr = float(np.max(np.abs(np.trace(r, axis1=-2, axis2=-1).real - 1.0)))
    neg = max(0.0, -float(np.min(np.linalg.eigvalsh(0.5 * (r + rh)))))
    worst = max(herm, tr, neg)
    return ctx.scaled(
        name,
        "thermodynamics",
        worst,
        TOL_CONSERVATION,
        f"battery_reduced() gives a valid density matrix conforming with H_B at all "
        f"{len(r)} points (max hermiticity/trace/negativity error {worst:.3e})",
    )


def _random_density_matrices(d: int, count: int, seed: int) -> list[np.ndarray]:
    """``count`` random density matrices of dimension ``d`` with ranks cycling from 1 to ``d``."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(count):
        g = rng.normal(size=(d, 1 + k % d)) + 1j * rng.normal(size=(d, 1 + k % d))
        rho = g @ g.conj().T
        out.append(rho / np.trace(rho).real)
    return out


def _check_ergotropy(ctx: _Context, states: np.ndarray) -> CheckResult:
    r, h, dims = _battery_arrays(ctx, states)
    n_rollout = len(r)
    rand = np.stack(_random_density_matrices(h.shape[-1], N_RANDOM_STATES, ctx.seed))
    rho = jnp.asarray(np.concatenate([r, rand]))
    ham = jnp.asarray(np.concatenate([h, np.broadcast_to(h[-1], rand.shape)]))

    def both(rho_i: jax.Array, h_i: jax.Array) -> tuple[jax.Array, jax.Array]:
        qr, qh = dq.asqarray(rho_i, dims=dims), dq.asqarray(h_i, dims=dims)
        return energy(qr, qh), ergotropy(qr, qh)

    e_all, w_all = (np.asarray(x, dtype=float) for x in jax.vmap(both)(rho, ham))
    lam = np.linalg.eigvalsh(np.asarray(ham, dtype=np.complex128))
    e_min, spread = lam[:, 0], float(np.max(lam[:, -1] - lam[:, 0], initial=1e-300))
    slack = e_all - e_min
    viol = float(np.max(np.maximum(-w_all, w_all - slack)))
    return ctx.scaled(
        "ergotropy_bounds",
        "thermodynamics",
        max(viol, 0.0) / max(spread, 1e-300),
        TOL_THERMO,
        f"0 <= ergotropy <= energy - E_min along {n_rollout} rollout points and "
        f"{len(rand)} random density matrices of the battery space; min ergotropy "
        f"{w_all.min():.3g}, min slack (energy - E_min - ergotropy) "
        f"{(slack - w_all).min():.3g}; violation relative to the width of H_B",
    )


def model_checks(
    model: Model,
    *,
    horizon: int = 12,
    dt: float | None = None,
    action_bound: float = 1.0,
    seed: int = 0,
    method: Method | None = None,
    reference: Model | None = None,
) -> list[CheckResult]:
    """Run the generic physics and numerics checks on ``model``.

    Args:
        model: any :class:`~qenvs.models.Model`.
        horizon: number of piecewise-constant control steps of the random rollout.
        dt: control period; ``None`` chooses it from the operator norms (see the module
            docstring for the rule, also available as :func:`default_dt`).
        action_bound: random amplitudes are drawn uniformly from ``[-b, b]``.
        seed: seed of the random amplitudes.
        method: dynamiqs solver under test (default: the library default, adaptive
            ``Tsit5``). The convergence check compares it with a tight Tsit5 and, where
            cheap, with ``Expm``.
        reference: optional independently built model of the *intended* physics. When
            given, QuTiP integrates the reference's matrices and the result is compared
            with the dynamiqs rollout of ``model`` (check name ``qutip_cross_check``); this
            is the only generic check that can detect a wrong operator (it needs you to
            have built the operator twice). Without it the check is named
            ``qutip_solver_cross_check`` and validates the solver only. An error is caught
            only if it moves the state by more than the tolerance over the episode, roughly
            ``rate * T * relative_error > tol`` (see the guide).

    Returns:
        One :class:`~qenvs.validation.core.CheckResult` per check: ``structure``,
        ``hermiticity``, ``initial_state_valid``, ``norm_conservation`` (closed) or
        ``density_matrix_valid`` (open), ``solver_convergence``, the QuTiP check
        (``qutip_solver_cross_check`` without ``reference``, ``qutip_cross_check`` with it;
        ``skipped`` without QuTiP) and, when the model defines a battery Hamiltonian,
        ``battery_consistency`` and ``ergotropy_bounds``. The rollout-based tolerances are
        scaled with the integrated time (see the module docstring).
    """
    if horizon < 1:
        msg = f"horizon must be >= 1, got {horizon}"
        raise ValueError(msg)
    if action_bound <= 0:
        msg = f"action_bound must be positive, got {action_bound}"
        raise ValueError(msg)
    if dt is not None and dt <= 0:
        msg = f"dt must be positive, got {dt}"
        raise ValueError(msg)
    method = DEFAULT_METHOD if method is None else method
    if dt is None:
        try:
            dt = default_dt(model, action_bound)
        except Exception:  # unusable operators (NaN) are reported by the structure check
            dt = 1.0
    ctx = _Context(model, horizon, float(dt), float(action_bound), seed, method)
    results: list[CheckResult] = []

    static = (
        ("structure", "constraints", 0.0, _check_structure),
        ("hermiticity", "constraints", TOL_HERMITIAN, _check_hermiticity),
        ("initial_state_valid", "constraints", TOL_STATE, _check_initial_state),
    )
    for name, cat, tol, fn in static:
        results.append(_guarded(ctx, name, cat, tol, lambda fn=fn: fn(ctx)))

    cons_name = "density_matrix_valid" if model.is_open else "norm_conservation"
    qutip_name = "qutip_cross_check" if reference is not None else "qutip_solver_cross_check"
    dynamic = [
        (cons_name, "constraints", TOL_CONSERVATION, _check_conservation),
        ("solver_convergence", "numerics", TOL_SOLVER, _check_solver),
        (qutip_name, "cross_check", TOL_QUTIP, lambda c, s: _check_qutip(c, s, reference)),
    ]
    try:
        model.battery_hamiltonian()
        dynamic += [
            ("battery_consistency", "thermodynamics", TOL_CONSERVATION, _check_battery_consistency),
            ("ergotropy_bounds", "thermodynamics", TOL_THERMO, _check_ergotropy),
        ]
    except NotImplementedError:
        pass
    try:
        states = ctx.rollout(method)
    except Exception as e:  # a failing rollout fails every dependent check
        return results + [ctx.fail(name, cat, ctx.tol(tol), e) for name, cat, tol, _ in dynamic]
    for name, cat, tol, fn in dynamic:
        results.append(_guarded(ctx, name, cat, ctx.tol(tol), lambda fn=fn: fn(ctx, states)))
    return results
