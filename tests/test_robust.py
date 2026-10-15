"""Robust evaluation (spec 4.5, gate G6) and the policy leaf of the key tree (G2).

Deterministic reference tests come first; the statistical tests use fixed keys, so their
verdicts are deterministic, and the exact criteria of the design document.
"""

from __future__ import annotations

import dataclasses
import math
import statistics
import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from dynamiqs.method import Expm

import qenvs
from qenvs import keys, robust
from qenvs.envs.base import Environment
from qenvs.wrappers import Wrapper
from qenvs.wrappers.observation import ObservationModel

HORIZON, DT = 4, 0.5
U_PI = math.pi / (HORIZON * DT)  # constant pulse of area pi: |0> -> |1>
KEY = jax.random.PRNGKey(2026)


# ----------------------------------------------------------------------------- helpers
def qubit_env():
    env = qenvs.make("qubit/StateTransfer-v0", horizon=HORIZON, dt=DT)
    return dataclasses.replace(env, method=Expm())


class RandomGain(Wrapper):
    """Test-only randomiser: the action is scaled by ``1 + eps``, ``eps ~ N(0, std^2)``."""

    std: float = eqx.field(static=True)

    def __init__(self, env, std):
        self.env = env
        self.std = float(std)

    def _episode_data(self, key):
        return {"eps": self.std * jax.random.normal(keys.calibration_key(key, keys.GAIN))}

    def _transform_action(self, state, action):
        return action * (1.0 + state.data["eps"])


def pi_pulse(scale=1.0):
    return jnp.full((HORIZON, 1), U_PI * scale)


def episode_keys(key, n):
    root = jax.random.fold_in(key, 0)
    return jax.vmap(lambda i: keys.episode_key(root, i))(jnp.arange(n))


# ------------------------------------------------------------------ estimators (G6a, exact)
class TestEstimators:
    @pytest.mark.parametrize(
        ("n", "k"), [(1, 1), (9, 1), (10, 1), (11, 2), (256, 26), (1000, 100), (1024, 103)]
    )
    def test_tail_count_is_an_integer_ceil(self, n, k):
        assert robust.tail_count(n) == k  # ceil(0.1 * 1000) in floats would be 101

    def test_cvar_hand_computed_with_ties(self):
        # n = 11, k = 2: the two lowest are 0.5 and 0.5
        v = np.array([0.9, 0.5, 0.5, 0.7, 0.5, 1.0, 0.8, 0.6, 0.5, 0.9, 0.95])
        assert robust.cvar(v) == 0.5
        # n = 20, k = 2: lowest two are 0.1 and one of the tied 0.2
        v = np.array([0.2, 0.9, 0.2, 0.1, 0.2, *np.linspace(0.3, 0.8, 15)])
        assert robust.cvar(v) == pytest.approx(0.15, abs=1e-15)
        # k = 1 is the minimum
        assert robust.cvar(np.array([3.0, 1.0, 2.0])) == 1.0
        # n = 25, k = 3: 0.1, 0.1, 0.4 (a tie crossing the tail boundary)
        v = np.concatenate([[0.1, 0.1, 0.4, 0.4, 0.4], np.linspace(0.5, 1.0, 20)])
        assert robust.cvar(v) == pytest.approx((0.1 + 0.1 + 0.4) / 3, abs=1e-15)

    def test_cvar_along_an_axis(self):
        m = np.arange(40, dtype=float).reshape(2, 20)
        np.testing.assert_allclose(robust.cvar(m, axis=1), [0.5, 20.5])  # k = 2

    def test_cvar_estimator_mean_matches_finite_sample_value(self):
        # U(0, 1): E[CVaR_hat] = (k + 1) / (2 (n + 1)); 2000 repetitions, 4 SE
        rng = np.random.default_rng(1)
        for n in (256, 1024, 4096):
            k = robust.tail_count(n)
            est = robust.cvar(rng.uniform(size=(2000, n)), axis=1)
            exact = (k + 1) / (2 * (n + 1))
            se = est.std(ddof=1) / math.sqrt(est.size)
            assert abs(est.mean() - exact) <= 4 * se, (n, est.mean(), exact)

    def test_bootstrap_is_a_function_of_values_and_key(self):
        v = np.random.default_rng(0).uniform(size=300)
        a = robust.bootstrap_intervals(v, KEY)
        b = robust.bootstrap_intervals(v, KEY)
        c = robust.bootstrap_intervals(v, jax.random.PRNGKey(1))
        assert a == b and a != c
        assert a["mean"][0] < v.mean() < a["mean"][1]
        assert a["cvar"][0] < robust.cvar(v) < a["cvar"][1]
        only = robust.bootstrap_intervals(v, KEY, tail=False)
        assert set(only) == {"mean"} and only["mean"] == a["mean"]

    def test_bootstrap_of_constant_values_has_zero_width(self):
        out = robust.bootstrap_intervals(np.full(50, 0.75), KEY)
        assert out["mean"] == (0.75, 0.75) and out["cvar"] == (0.75, 0.75)

    def test_bootstrap_mean_interval_matches_the_normal_approximation(self):
        v = np.random.default_rng(3).normal(size=4000)
        lo, hi = robust.bootstrap_intervals(v, KEY, tail=False)["mean"]
        half = 1.96 * v.std(ddof=1) / math.sqrt(v.size)
        assert lo == pytest.approx(v.mean() - half, rel=0.0, abs=0.15 * half)
        assert hi == pytest.approx(v.mean() + half, rel=0.0, abs=0.15 * half)


# -------------------------------------------------------------------------- metric choice
class TestPerformanceMetric:
    @pytest.mark.parametrize(
        ("task", "expected"),
        [
            ("transmon/XGate-v0", "fidelity"),
            ("transmon/SXGate-v0", "fidelity"),
            ("transmon/CRGate-v0", "fidelity"),
            ("qubit/StateTransfer-v0", "fidelity"),
            ("transfer/LambdaSTIRAP-v0", "fidelity"),
            ("battery/TwoLevel-v0", "energy"),
            ("battery/Dicke-v0", "energy"),
            ("battery/DickeLocal-v0", "energy"),
        ],
    )
    def test_default_per_task(self, task, expected):
        env = qenvs.make(task)
        assert robust.performance_metric(env) == expected
        assert robust.performance_metric(ObservationModel(env, "time")) == expected

    @pytest.mark.parametrize("task", ["battery/TwoLevel-v0", "battery/Dicke-v0"])
    def test_battery_follows_the_reward_kwarg(self, task):
        env = qenvs.make(task, reward="ergotropy")
        assert robust.performance_metric(env) == "ergotropy"
        assert robust.performance_metric(RandomGain(env, 0.1)) == "ergotropy"

    def test_unknown_environment_needs_an_explicit_metric(self):
        class Odd(Environment):
            action_dim = 1
            observation_dim = 1
            action_bound = 1.0
            horizon = 1

            def reset(self, key):  # pragma: no cover
                raise NotImplementedError

            def step(self, state, action):  # pragma: no cover
                raise NotImplementedError

        with pytest.raises(ValueError, match="no default performance metric"):
            robust.performance_metric(Odd())


# --------------------------------------------------------------------------- evaluate
class TestEvaluateDeterministic:
    def test_metric_equals_final_info_and_return_is_reported_apart(self):
        env = qubit_env()
        for scale in (1.0, 0.8):
            res = robust.evaluate(env, pi_pulse(scale), KEY, n=4, metric=None)
            final, rewards = env.rollout(pi_pulse(scale))
            f_final = float(env.info(final)["fidelity"])
            assert res.metric == "fidelity"
            np.testing.assert_allclose(res.values, f_final, atol=1e-12)
            assert res.mean == pytest.approx(f_final, abs=1e-12)
            assert res.min == pytest.approx(f_final, abs=1e-12)
            assert res.cvar == pytest.approx(f_final, abs=1e-12)
            np.testing.assert_allclose(res.returns, float(rewards.sum()), atol=1e-12)
            assert res.return_mean == pytest.approx(float(rewards.sum()), abs=1e-12)
            # the return is F_final - F_initial, a gain, not the metric
            f_initial = float(env.reset(KEY)[1].info["fidelity"])
            assert res.return_mean == pytest.approx(f_final - f_initial, abs=1e-6)
        full = robust.evaluate(env, pi_pulse(), KEY, n=2)
        assert full.mean == pytest.approx(1.0, abs=1e-5)

    def test_return_and_metric_differ_when_the_initial_value_is_not_zero(self):
        # TwoLevel battery: F is not used, energy gain: metric = final energy, return = gain
        env = qenvs.make("battery/TwoLevel-v0", horizon=4, dt=0.5)
        env = dataclasses.replace(env, method=Expm())
        actions = jnp.full((4, 1), 0.7)
        res = robust.evaluate(env, actions, KEY, n=2)
        final, rewards = env.rollout(actions)
        assert res.metric == "energy"
        assert res.mean == pytest.approx(float(env.info(final)["energy"]), abs=1e-12)
        assert res.return_mean == pytest.approx(float(rewards.sum()), abs=1e-12)

    def test_custom_metrics(self):
        env = qubit_env()
        res = robust.evaluate(env, pi_pulse(0.9), KEY, n=2, metric="p1")
        final, _ = env.rollout(pi_pulse(0.9))
        assert res.metric == "p1"
        assert res.mean == pytest.approx(float(env.info(final)["p1"]), abs=1e-12)
        fn = lambda state: 2.0 * env.info(state)["fidelity"]  # noqa: E731
        res2 = robust.evaluate(env, pi_pulse(0.9), KEY, n=2, metric=fn)
        assert res2.mean == pytest.approx(2.0 * float(env.info(final)["fidelity"]), abs=1e-12)
        with pytest.raises(KeyError, match="not a key"):
            robust.evaluate(env, pi_pulse(), KEY, n=2, metric="nope")

    def test_evaluation_key_is_required(self):
        env = qubit_env()
        with pytest.raises(TypeError):
            robust.evaluate(env, pi_pulse())
        for bad in (None, 3, np.int64(3)):
            with pytest.raises(ValueError, match="key"):
                robust.evaluate(env, pi_pulse(), bad)
        with pytest.raises(ValueError, match="positive integer"):
            robust.evaluate(env, pi_pulse(), KEY, n=0)

    def test_open_loop_shape_is_checked(self):
        with pytest.raises(ValueError, match="open-loop policy"):
            robust.evaluate(qubit_env(), jnp.zeros((HORIZON + 1, 1)), KEY, n=2)

    def test_typed_keys_work(self):
        res = robust.evaluate(qubit_env(), pi_pulse(), jax.random.key(0), n=2)
        assert res.mean == pytest.approx(1.0, abs=1e-5)

    def test_nominal_env_is_reported_and_stochastic_nominal_warns(self):
        env = qubit_env()
        noisy = RandomGain(env, 0.1)
        res = robust.evaluate(noisy, pi_pulse(), KEY, n=8, nominal_env=env)
        assert res.nominal == pytest.approx(1.0, abs=1e-5)
        assert res.nominal_return == pytest.approx(1.0, abs=1e-5)
        assert res.mean < res.nominal
        with pytest.warns(UserWarning, match="not deterministic"):
            robust.evaluate(env, pi_pulse(), KEY, n=2, nominal_env=noisy)


# ------------------------------------------------------------- randomised evaluation
class TestEvaluateRandomised:
    @pytest.fixture(scope="class")
    def noisy(self):
        return RandomGain(qubit_env(), 0.1)

    def test_interval_rules_by_sample_size(self, noisy):
        with pytest.warns(UserWarning, match="no confidence intervals"):
            small = robust.evaluate(noisy, pi_pulse(), KEY, n=100)
        assert small.mean_interval is None and small.cvar_interval is None
        assert small.n == 100 and np.isfinite(small.cvar)
        with pytest.warns(UserWarning, match="CVaR interval is only checked"):
            mid = robust.evaluate(noisy, pi_pulse(), KEY, n=256)
        assert mid.mean_interval is not None and mid.cvar_interval is None
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            full = robust.evaluate(noisy, pi_pulse(), KEY, n=1024)
        lo, hi = full.mean_interval
        assert lo < full.mean < hi
        clo, chi = full.cvar_interval
        assert clo < full.cvar < chi
        assert full.cvar < full.mean and full.min <= full.cvar
        assert full.cvar == pytest.approx(robust.cvar(full.values), abs=1e-15)
        assert full.n == 1024 == full.values.size == full.returns.size
        d = full.as_dict()
        assert d["cvar_interval"] == list(full.cvar_interval) and "values" not in d

    def test_same_key_same_result_and_other_key_differs(self, noisy):
        a = robust.evaluate(noisy, pi_pulse(), KEY, n=300)
        b = robust.evaluate(noisy, pi_pulse(), KEY, n=300)
        c = robust.evaluate(noisy, pi_pulse(), jax.random.PRNGKey(5), n=300)
        assert np.array_equal(a.values, b.values) and a.mean == b.mean
        assert not np.array_equal(a.values, c.values)

    def test_any_episode_can_be_regenerated_from_the_key_and_its_index(self, noisy):
        n = 40
        res = robust.evaluate(noisy, pi_pulse(), KEY, n=n)
        ep = episode_keys(KEY, n)
        for i in (0, 7, 39):
            assert np.array_equal(ep[i], keys.episode_key(jax.random.fold_in(KEY, 0), i))
            final, _ = noisy.rollout(pi_pulse(), ep[i])
            assert res.values[i] == pytest.approx(float(noisy.info(final)["fidelity"]), abs=1e-12)
        # and the data of the episode is the one of that key
        state, _ = noisy.reset(ep[7])
        expected = 0.1 * jax.random.normal(keys.calibration_key(ep[7], keys.GAIN))
        assert float(state.data["eps"]) == float(expected)

    def test_batch_size_does_not_change_the_result(self, noisy):
        full = robust.evaluate(noisy, pi_pulse(), KEY, n=70)
        for bs in (1, 16, 70, 500):
            chunked = robust.evaluate(noisy, pi_pulse(), KEY, n=70, batch_size=bs)
            np.testing.assert_allclose(chunked.values, full.values, atol=1e-12)

    def test_stability_reports_nested_sizes(self, noisy):
        rep = robust.stability(noisy, pi_pulse(), KEY, sizes=(256, 1024))
        assert set(rep.results) == {256, 1024}
        r256, r1024 = rep.results[256], rep.results[1024]
        assert r256.n == 256 and r1024.n == 1024
        assert r256.mean_interval is not None and r256.cvar_interval is None
        assert r1024.cvar_interval is not None
        # same episode keys as evaluate(n=256)
        direct = robust.evaluate(noisy, pi_pulse(), KEY, n=256)
        np.testing.assert_allclose(r256.values, direct.values, atol=1e-10)
        np.testing.assert_array_equal(r1024.values[:256], r256.values)
        assert rep.mean_spread == pytest.approx(abs(r256.mean - r1024.mean))
        assert rep.cvar_spread == pytest.approx(abs(r256.cvar - r1024.cvar))
        assert rep.mean_spread < 0.02
        with pytest.raises(ValueError, match="key"):
            robust.stability(noisy, pi_pulse(), None)


# ------------------------------------------------------------------ policy leaf (G2)
class TestPolicyKeys:
    def test_policy_keys_are_distinct_in_time_and_across_leaves(self):
        ep = jax.random.fold_in(KEY, 17)
        data = lambda k: tuple(np.asarray(jax.random.key_data(k)).tolist())  # noqa: E731
        pol = [data(keys.policy_key(ep, t)) for t in range(64)]
        assert len(set(pol)) == 64  # a different key at every decision of an episode
        others = {data(keys.measurement_key(ep, t, j)) for t in range(64) for j in range(8)}
        others |= {data(keys.parameter_key(ep, f"p{i}")) for i in range(8)}
        others |= {data(keys.step_noise_key(ep, k)) for k in range(64)}
        others |= {data(keys.calibration_key(ep, w)) for w in (keys.GAIN, keys.OFFSET)}
        assert not set(pol) & others
        # and across episodes
        ep2 = jax.random.fold_in(KEY, 18)
        assert not set(pol) & {data(keys.policy_key(ep2, t)) for t in range(64)}

    def test_stochastic_policy_receives_policy_keys_and_is_reproducible(self):
        env = qubit_env()
        n = 8
        c = 2.0 * U_PI

        def policy(obs, key):
            return jnp.full((1,), c * jax.random.uniform(key))

        res = robust.evaluate(env, policy, KEY, n=n)
        again = robust.evaluate(env, policy, KEY, n=n)
        assert np.array_equal(res.values, again.values)  # same key, same action sequence
        ep = episode_keys(KEY, n)
        for i in range(n):
            acts = jnp.stack(
                [
                    jnp.full((1,), c * jax.random.uniform(keys.policy_key(ep[i], t)))
                    for t in range(HORIZON)
                ]
            )
            final, _ = env.rollout(acts)
            assert res.values[i] == pytest.approx(float(env.info(final)["fidelity"]), abs=1e-10)
        assert res.values.std() > 0  # different episodes drew different action sequences

    def test_policy_randomness_does_not_change_the_other_leaves(self):
        # measurement noise (leaf 2) and the gain error (leaf 1) are the same whether or not the
        # policy draws from its own leaf (3): the observation-dependent actions stay identical
        env = RandomGain(
            ObservationModel(qubit_env(), "expectations", shots=20, observables=("p1",)), 0.05
        )

        def plain(obs, key):
            return jnp.full((1,), U_PI * (1.0 + 0.3 * obs[0]))

        def with_policy_noise(obs, key):
            return plain(obs, key) + 0.0 * jax.random.normal(key, (1,))

        a = robust.evaluate(env, plain, KEY, n=64)
        b = robust.evaluate(env, with_policy_noise, KEY, n=64)
        assert np.array_equal(a.values, b.values)
        assert a.values.std() > 0
        # sanity: the measurement leaf really matters for this policy
        other_shots = RandomGain(
            ObservationModel(qubit_env(), "expectations", shots=21, observables=("p1",)), 0.05
        )
        c = robust.evaluate(other_shots, plain, KEY, n=64)
        assert not np.array_equal(a.values, c.values)


# ------------------------------------------------------------------- coverage (G6b)
def fidelity_samples(rng, shape, sigma=0.05):
    eps = rng.normal(0.0, sigma, size=shape)
    return 1.0 - (2.0 / 3.0) * np.sin(math.pi * eps / 2.0) ** 2


def fidelity_truth(sigma=0.05, level=0.1):
    """Population mean and lower-tail CVaR of F = 1 - (2/3) sin^2(pi eps / 2), eps ~ N(0, s^2)."""
    mean = 1.0 - (1.0 / 3.0) * (1.0 - math.exp(-(math.pi**2) * sigma**2 / 2.0))
    # F <= q_level  <=>  |eps| >= e*, with P(|eps| >= e*) = level
    e_star = sigma * statistics.NormalDist().inv_cdf(1.0 - level / 2.0)
    x = np.linspace(e_star, 12 * sigma, 400_001)
    pdf = np.exp(-(x**2) / (2 * sigma**2)) / (sigma * math.sqrt(2 * math.pi))
    f = 1.0 - (2.0 / 3.0) * np.sin(math.pi * x / 2.0) ** 2
    y = f * pdf
    tail = 2.0 * float(
        np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
    )  # both tails of |eps|, trapezoid
    return mean, tail / level


def test_fidelity_truth_reference_values():
    # deterministic reference before the coverage experiment
    mean, cvar = fidelity_truth()
    assert mean == pytest.approx(1.0 - (1 / 3) * (1 - math.exp(-(math.pi**2) * 0.0025 / 2)))
    big = fidelity_samples(np.random.default_rng(0), 4_000_000)
    assert big.mean() == pytest.approx(mean, abs=2e-6)
    assert robust.cvar(big) == pytest.approx(cvar, abs=2e-5)


@pytest.mark.slow
@pytest.mark.parametrize("dist", ["uniform", "fidelity"])
def test_bootstrap_interval_coverage(dist):
    """G6b: n = 1024, B = 2000, 1000 independent repetitions, coverage inside [0.92, 0.98]."""
    n, reps = 1024, 1000
    rng = np.random.default_rng(20261015 if dist == "uniform" else 20261016)
    if dist == "uniform":
        truth_mean, truth_cvar = 0.5, 0.05
        draw = lambda: rng.uniform(size=n)  # noqa: E731
    else:
        truth_mean, truth_cvar = fidelity_truth()
        draw = lambda: fidelity_samples(rng, n)  # noqa: E731
    root = jax.random.PRNGKey(7 if dist == "uniform" else 8)
    hit_mean = hit_cvar = 0
    for r in range(reps):
        out = robust.bootstrap_intervals(draw(), jax.random.fold_in(root, r), n_boot=robust.N_BOOT)
        hit_mean += out["mean"][0] <= truth_mean <= out["mean"][1]
        hit_cvar += out["cvar"][0] <= truth_cvar <= out["cvar"][1]
    cov_mean, cov_cvar = hit_mean / reps, hit_cvar / reps
    print(f"coverage {dist}: mean {cov_mean:.3f} cvar {cov_cvar:.3f}")
    assert 0.92 <= cov_mean <= 0.98, cov_mean
    assert 0.92 <= cov_cvar <= 0.98, cov_cvar


# ------------------------------------------------- gaps found by mutation (review round 6)
def test_tail_count_is_exact_at_round_n():
    from qenvs.robust import tail_count

    # pins k. math.ceil(0.1 * n) agreed for every n we checked (1 to 2e6), which is evidence, not
    # a proof, that a float version is equivalent; the code keeps integer arithmetic to be safe
    assert [tail_count(n) for n in (1, 9, 10, 11, 1000, 1024, 4096)] == [1, 1, 1, 2, 100, 103, 410]


def test_bootstrap_cvar_uses_the_cvar_definition():
    from qenvs.robust import _numpy_rng, bootstrap_intervals, cvar

    values = np.random.default_rng(3).normal(size=257)
    key = jax.random.PRNGKey(5)
    got = bootstrap_intervals(values, key, n_boot=400)
    # replay the same resamples and apply the point estimators
    rng = _numpy_rng(key)
    sample = values[rng.integers(0, values.size, size=(400, values.size), dtype=np.int32)]
    ref_cvar = np.percentile(cvar(sample, axis=1), [2.5, 97.5])
    ref_mean = np.percentile(sample.mean(axis=1), [2.5, 97.5])
    np.testing.assert_array_equal(np.asarray(got["cvar"]), ref_cvar)
    np.testing.assert_array_equal(np.asarray(got["mean"]), ref_mean)
