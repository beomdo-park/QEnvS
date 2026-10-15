"""G4: analytic validation of control errors against closed-form pi-pulse infidelities.

Assumptions of the closed forms (spec section 6, G4): relative amplitude error epsilon with
mean zero, fixed for the episode, two-level limit, resonant pi pulse of constant amplitude
a = pi / T on the I channel only (Q = 0). The rotation angle is pi (1 + epsilon), so

    1 - F_avg   = (2/3) sin^2(pi epsilon / 2)       (gate fidelity, transmon/XGate-v0, levels=2)
    1 - F_state = sin^2(pi epsilon / 2)             (|0> -> |1>, both tasks)

and the expectations are exact: Normal(0, sigma^2): E[1 - F_avg] = (1 - exp(-pi^2 sigma^2 / 2)) / 3,
Uniform(-a, a): E[1 - F_avg] = (1 - sin(pi a) / (pi a)) / 3 (and 1/2 instead of 1/3 for the
state). Per-step noise of a constant pulse split into H equal steps rotates by
pi (1 + mean(xi)), so the Normal formula holds with sigma = s / sqrt(H) (this restricted case only).

Order of the checks (the statistical stage runs only after the deterministic preconditions):
P1 configuration, P2 epsilon = 0 gives 1 - F <= 1e-12, P3 fixed epsilon (and fixed step-noise
vectors) match the closed form to 1e-10, P4 no clipping in any sample of the statistical stage.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from dynamiqs.method import Expm

import qenvs
from qenvs import keys
from qenvs.distributions import Normal, Uniform
from qenvs.wrappers.control import CLIP_FRACTION, ControlErrors, apply_control_errors

N_STAT = 4096
RUN_KEY = jax.random.PRNGKey(20261017)
EPS_FIXED = (0.01, 0.1, 0.3)
NORMAL_SIGMAS = (0.01, 0.05, 0.2)
UNIFORM_HALF_WIDTHS = (0.02, 0.1, 0.3)
STEP_NOISE_STD = (0.04, 0.2)  # per-step std s; sigma_eff = s / sqrt(H)


# ---------------------------------------------------------------------- the two tasks
class Task:
    """``transmon/XGate-v0`` (levels=2) or ``qubit/StateTransfer-v0`` with a pi pulse."""

    def __init__(self, name: str) -> None:
        self.name = name
        if name == "gate":
            self.env = qenvs.make("transmon/XGate-v0", levels=2)
            amp = math.pi / (self.env.horizon * self.env.dt)  # T = 32 ns
            self.pulse = jnp.zeros((self.env.horizon, 2)).at[:, 0].set(amp)
            self.metrics = ("avg", "state")
        else:
            # Two settings differ from the task defaults (reported in the final deviations):
            # the default bound 2.0 is only 1.27 a, so it is widened for P4 (no clipping at
            # sigma = 0.2); the default Tsit5 solver has a 1e-6 tolerance, which alone costs
            # about 1e-11 in 1 - F and would fail P2, so the exact piecewise-constant
            # propagator Expm() is used (the step Hamiltonian is constant).
            self.env = qenvs.make("qubit/StateTransfer-v0", action_bound=4.0, method=Expm())
            amp = math.pi / (self.env.horizon * self.env.dt)  # T = 2
            self.pulse = jnp.full((self.env.horizon, 1), amp)
            self.metrics = ("state",)
        self.amplitude = amp

    def effective_epsilon(self, u) -> float:
        """Relative rotation error ``theta / pi - 1`` of the amplitudes ``u`` (channel I, (H,)).

        ``theta = dt * sum(u_k)`` is the rotation angle actually delivered. ``StateTransfer``
        casts the action to float32 before stepping (released behaviour, kept), which changes
        an amplitude by up to 6e-8 relative; that quantisation is part of the delivered angle,
        so the closed form is evaluated at it. ``GateControl`` keeps float64: no quantisation.
        """
        u = np.asarray(u, dtype=np.float64)
        if self.name != "gate":
            u = u.astype(np.float32).astype(np.float64)
        return float(u.sum() * self.env.dt / math.pi - 1.0)

    def infidelities(self, info) -> dict[str, jax.Array]:
        if self.name == "gate":
            return {"avg": 1.0 - info["fidelity"], "state": 1.0 - info["p1"]}
        return {"state": 1.0 - info["fidelity"]}


@pytest.fixture(scope="module", params=["gate", "state_transfer"])
def task(request) -> Task:
    return Task("gate" if request.param == "gate" else "qubit")


def closed_form(metric: str, eps: float) -> float:
    s2 = math.sin(math.pi * eps / 2.0) ** 2
    return (2.0 / 3.0) * s2 if metric == "avg" else s2


PREFACTOR = {"avg": 1.0 / 3.0, "state": 0.5}


def expect_normal(metric: str, sigma: float) -> float:
    return PREFACTOR[metric] * (1.0 - math.exp(-(math.pi**2) * sigma**2 / 2.0))


def expect_uniform(metric: str, half_width: float) -> float:
    if half_width == 0.0:
        return 0.0
    x = math.pi * half_width
    return PREFACTOR[metric] * (1.0 - math.sin(x) / x)


def final_state(wrapper, task: Task, key, gain=None, noise=None):
    """Run the pulse; ``gain`` fixes epsilon (episode data), ``noise`` fixes xi_k (H, A)."""
    state, _ = wrapper.reset(key)
    if gain is not None:
        state = eqx.tree_at(lambda s: s.data["gain"], state, jnp.full(task.pulse.shape[1], gain))
    inner = wrapper.env
    for k, a in enumerate(task.pulse):
        if noise is None:
            state, _ = wrapper.step(state, a)
        else:  # fixed xi vector: apply the documented law and step the inner environment
            u, _ = apply_control_errors(a, None, None, noise[k], inner.action_bound)
            inner_state, _ = inner.step(state.inner, u)
            state = eqx.tree_at(lambda s: s.inner, state, inner_state)
    return wrapper.info(state)


# ====================================================================== deterministic reference
class TestClosedFormReference:
    """The closed forms against numerical integration of the per-epsilon formula."""

    @pytest.mark.parametrize("metric", ["avg", "state"])
    @pytest.mark.parametrize("sigma", NORMAL_SIGMAS)
    def test_normal_expectation_equals_quadrature(self, metric, sigma):
        x, w = np.polynomial.hermite_e.hermegauss(120)  # weight exp(-x^2/2)
        quad = float((w * np.array([closed_form(metric, sigma * t) for t in x])).sum())
        quad /= math.sqrt(2.0 * math.pi)
        assert expect_normal(metric, sigma) == pytest.approx(quad, rel=1e-12)

    @pytest.mark.parametrize("metric", ["avg", "state"])
    @pytest.mark.parametrize("half_width", UNIFORM_HALF_WIDTHS)
    def test_uniform_expectation_equals_quadrature(self, metric, half_width):
        x, w = np.polynomial.legendre.leggauss(120)
        quad = 0.5 * float((w * np.array([closed_form(metric, half_width * t) for t in x])).sum())
        assert expect_uniform(metric, half_width) == pytest.approx(quad, rel=1e-12)

    def test_limits_and_known_values(self):
        assert expect_uniform("avg", 0.0) == 0.0
        assert expect_uniform("avg", 1e-9) == pytest.approx(0.0, abs=1e-15)
        assert closed_form("state", 1.0) == pytest.approx(1.0)  # angle 2 pi: back to |0>
        assert closed_form("avg", 1.0) == pytest.approx(2.0 / 3.0)
        assert closed_form("avg", 0.0) == 0.0
        # the identity gate has F_avg = 1/3 against X: rotation angle 0 means epsilon = -1
        assert closed_form("avg", -1.0) == pytest.approx(2.0 / 3.0)

    def test_per_step_noise_scales_as_one_over_root_h(self):
        """Mean of H independent xi (std s) has std s / sqrt(H) (exact), the claim used below."""
        rng = np.random.default_rng(0)
        h, s = 16, 0.2
        xi = rng.normal(0.0, s, size=(200_000, h)).mean(axis=1)
        assert xi.std() == pytest.approx(s / math.sqrt(h), rel=5e-3)


# ====================================================================== P1 to P4
class TestPreconditions:
    def test_p1_configuration(self, task):
        env, pulse = task.env, task.pulse
        model = env.model
        assert not model.is_open and model.jump_ops() == []  # closed (ket or propagator) path
        if task.name == "gate":
            assert model.levels == 2
            assert float(model.detuning_ghz) == 0.0
            assert model.t1_us is None and model.t2_us is None
            assert np.all(np.asarray(pulse[:, 1]) == 0.0)  # Q channel off
        else:
            assert float(model.detuning) == 0.0
            assert model.gamma1 == 0.0 and model.gamma_phi == 0.0
        # constant amplitude pi / T on the (I) channel: area equals pi
        assert float(pulse[:, 0].sum()) * env.dt == pytest.approx(math.pi, rel=1e-14)
        # the wrapper under test has only the source under test switched on
        cenv = ControlErrors(env, amplitude_scale=Normal(0.05))
        assert cenv.offset is None and cenv.step_noise is None
        assert cenv.env is env  # no parameter randomisation or other wrapper in the loop
        assert float(jnp.max(jnp.abs(pulse))) < env.action_bound  # base pulse is not clipped

    @pytest.mark.parametrize("zero_source", ["gain", "gain_zero_width", "noise_zero_width"])
    def test_p2_epsilon_zero_gives_unit_fidelity(self, task, zero_source):
        key = jax.random.PRNGKey(0)
        if zero_source == "gain":
            cenv = ControlErrors(task.env, amplitude_scale=Normal(0.05))
            info = final_state(cenv, task, key, gain=0.0)
        elif zero_source == "gain_zero_width":
            info = final_state(ControlErrors(task.env, Normal(0.0)), task, key)
        else:
            info = final_state(ControlErrors(task.env, step_noise=Normal(0.0)), task, key)
        for metric, value in task.infidelities(info).items():
            assert abs(float(value)) <= 1e-12, metric

    @pytest.mark.parametrize("eps", EPS_FIXED)
    def test_p3_fixed_epsilon_matches_the_closed_form(self, task, eps):
        cenv = ControlErrors(task.env, amplitude_scale=Normal(0.05))
        info = final_state(cenv, task, jax.random.PRNGKey(1), gain=eps)
        got = task.infidelities(info)
        assert set(got) == set(task.metrics)
        u = np.full(task.env.horizon, task.amplitude * (1.0 + eps))
        eps_eff = task.effective_epsilon(u)
        assert abs(eps_eff - eps) <= 1e-7  # the quantisation is tiny, but not below 1e-10 in 1 - F
        for metric, value in got.items():
            assert float(value) == pytest.approx(closed_form(metric, eps_eff), abs=1e-10, rel=0)
        # the closed form is not trivially satisfied: the error is far above the tolerance
        assert closed_form(task.metrics[0], eps) > 1e-4

    @pytest.mark.parametrize("seed", [0, 1])
    def test_p3_fixed_step_noise_vector_matches_the_closed_form(self, task, seed):
        h = task.env.horizon
        rng = np.random.default_rng(seed)
        xi_c = rng.normal(0.0, 0.2, size=h)  # channel I; the Q channel stays 0 whatever xi is
        noise = jnp.asarray(np.stack([xi_c] * task.pulse.shape[1], axis=1))
        cenv = ControlErrors(task.env, step_noise=Normal(0.2))
        info = final_state(cenv, task, jax.random.PRNGKey(2), noise=noise)
        # equal steps, same rotation axis: angle pi (1 + mean xi) (up to float32 quantisation)
        eps_eff = task.effective_epsilon(task.amplitude * (1.0 + xi_c))
        assert eps_eff == pytest.approx(float(np.mean(xi_c)), abs=1e-7)
        for metric, value in task.infidelities(info).items():
            assert float(value) == pytest.approx(closed_form(metric, eps_eff), abs=1e-10, rel=0)
        assert abs(eps_eff) > 0.005  # a nontrivial error

    def test_p3_wrapper_step_noise_equals_the_closed_form_of_its_own_draw(self, task):
        """Integration: the wrapper's xi_k (documented key leaf) reproduce pi (1 + mean xi)."""
        s = 0.2
        key = jax.random.PRNGKey(3)
        cenv = ControlErrors(task.env, step_noise=Normal(s))
        info = final_state(cenv, task, key)
        a_dim = task.pulse.shape[1]
        xi = np.stack(
            [
                np.asarray(
                    Normal(s).sample(keys.step_noise_key(key, k), 1.0, (a_dim,), jnp.float64)
                )
                for k in range(task.env.horizon)
            ]
        )
        eps_eff = task.effective_epsilon(task.amplitude * (1.0 + xi[:, 0]))
        assert eps_eff == pytest.approx(float(xi[:, 0].mean()), abs=1e-7)
        for metric, value in task.infidelities(info).items():
            assert float(value) == pytest.approx(closed_form(metric, eps_eff), abs=1e-10, rel=0)

    def test_p3_wrapper_gain_equals_the_closed_form_of_its_own_draw(self, task):
        key = jax.random.PRNGKey(4)
        cenv = ControlErrors(task.env, amplitude_scale=Normal(0.05))
        info = final_state(cenv, task, key)
        a_dim = task.pulse.shape[1]
        gain = float(Normal(0.05).sample(keys.calibration_key(key, keys.GAIN), 1.0, (a_dim,))[0])
        u = task.amplitude * (1.0 + gain) * np.ones(task.env.horizon)
        eps = task.effective_epsilon(u)
        assert eps == pytest.approx(gain, abs=1e-7)
        for metric, value in task.infidelities(info).items():
            assert float(value) == pytest.approx(closed_form(metric, eps), abs=1e-10, rel=0)


# ====================================================================== statistical stage
def batch_infidelities(cenv: ControlErrors, task: Task, n: int = N_STAT):
    """Infidelities and the maximum clip fraction of ``n`` episodes ``fold_in(RUN_KEY, i)``."""

    @jax.jit
    def run(ks):
        def episode(k):
            state, _ = cenv.reset(k)

            def body(s, a):
                s, ts = cenv.step(s, a)
                return s, ts.info[CLIP_FRACTION]

            state, fractions = jax.lax.scan(body, state, task.pulse)
            return task.infidelities(cenv.info(state)), jnp.max(fractions)

        return jax.vmap(episode)(ks)

    ks = jax.vmap(lambda i: keys.episode_key(RUN_KEY, i))(jnp.arange(n))
    infid, max_fraction = run(ks)
    return {m: np.asarray(v) for m, v in infid.items()}, float(jnp.max(max_fraction))


def assert_mean_within_4se(values: np.ndarray, expected: float, label: str) -> None:
    se = values.std(ddof=1) / math.sqrt(values.size)
    assert abs(values.mean() - expected) <= 4.0 * se, (
        label,
        values.mean(),
        expected,
        se,
        (values.mean() - expected) / se,
    )


class TestStatisticalStage:
    """4096 samples: the sample mean is within 4 standard errors of the exact expectation."""

    @pytest.mark.parametrize("sigma", NORMAL_SIGMAS)
    def test_normal_gain(self, task, sigma):
        cenv = ControlErrors(task.env, amplitude_scale=Normal(sigma))
        infid, max_fraction = batch_infidelities(cenv, task)
        assert max_fraction == 0.0  # P4: clipping never breaks the formula
        for metric in task.metrics:
            assert_mean_within_4se(infid[metric], expect_normal(metric, sigma), (metric, sigma))

    @pytest.mark.parametrize("half_width", UNIFORM_HALF_WIDTHS)
    def test_uniform_gain(self, task, half_width):
        cenv = ControlErrors(task.env, amplitude_scale=Uniform(half_width))
        infid, max_fraction = batch_infidelities(cenv, task)
        assert max_fraction == 0.0
        for metric in task.metrics:
            expected = expect_uniform(metric, half_width)
            assert_mean_within_4se(infid[metric], expected, (metric, half_width))

    @pytest.mark.parametrize("s", STEP_NOISE_STD)
    def test_step_noise_of_a_constant_pulse(self, task, s):
        """Constant pi pulse in H equal steps: sigma_eff = s / sqrt(H) (this case only)."""
        cenv = ControlErrors(task.env, step_noise=Normal(s))
        infid, max_fraction = batch_infidelities(cenv, task)
        assert max_fraction == 0.0
        sigma_eff = s / math.sqrt(task.env.horizon)
        for metric in task.metrics:
            assert_mean_within_4se(infid[metric], expect_normal(metric, sigma_eff), (metric, s))
