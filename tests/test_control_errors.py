"""Control errors: u = clip((a (1 + eps) + o)(1 + xi_k), -B, B), keys, clip fraction, gates.

Gates (docs/superpowers/specs/2026-10-15-realistic-rl-conditions-proposal.md): G1 (zero
widths reproduce the released tasks), G2 (reproducibility and leaf separation of the control
leaves), and the empirical 4 SE checks. The closed-form physics check (G4) is in
tests/test_robust_analytic.py.
"""

from __future__ import annotations

import math
import pathlib
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import qenvs
from qenvs import keys
from qenvs.distributions import Normal, TruncatedNormal, Uniform
from qenvs.wrappers.control import CLIP_FRACTION, ControlErrors, apply_control_errors
from qenvs.wrappers.randomization import ParameterRandomization

GOLDEN = pathlib.Path(__file__).with_name("golden")
sys.path.insert(0, str(GOLDEN))
import make_task_snapshot  # noqa: E402

RUN_KEY = jax.random.PRNGKey(20261016)
UNIT = 0.1  # a probe amplitude well inside the action bound of the gate task (0.314)
TASKS = sorted(qenvs.registry)
HEAVY_TASKS = {"battery/DickeLocal-v0"}


# ---------------------------------------------------------------------- helpers
def x32():
    """Context with jax x64 off (the library default; conftest turns it on)."""
    if hasattr(jax, "enable_x64"):
        return jax.enable_x64(False)
    from jax.experimental import disable_x64

    return disable_x64()


def corr(a, b):
    return float(np.corrcoef(np.asarray(a), np.asarray(b))[0, 1])


def assert_moments_within_4se(x, mean, var, mu4):
    x = np.asarray(x, dtype=float)
    n = x.size
    se_mean = math.sqrt(var / n)
    se_var = math.sqrt((mu4 - (n - 3) / (n - 1) * var**2) / n)
    assert abs(x.mean() - mean) <= 4 * se_mean, (x.mean(), mean, se_mean)
    assert abs(x.var(ddof=1) - var) <= 4 * se_var, (x.var(ddof=1), var, se_var)


def normal_moments(sigma):
    return 0.0, sigma**2, 3 * sigma**4


def uniform_moments(h):
    return 0.0, h**2 / 3, h**4 / 5


def gate_env(**kw):
    return qenvs.make("transmon/XGate-v0", **kw)


def run(env, pulse, key, steps):
    state, ts = env.reset(key)
    step = eqx.filter_jit(env.step)
    obs, rewards, fractions = [np.asarray(ts.observation)], [], []
    for a in pulse[:steps]:
        state, ts = step(state, a)
        obs.append(np.asarray(ts.observation))
        rewards.append(np.asarray(ts.reward))
        fractions.append(float(ts.info.get(CLIP_FRACTION, 0.0)))
    return np.stack(obs), np.stack(rewards), np.asarray(state.qstate.to_jax()), fractions


# ====================================================================== the law
class TestControlLaw:
    """Deterministic reference tests of the application formula."""

    def test_formula_term_by_term(self):
        a = jnp.asarray([0.2, -0.1, 0.0])
        gain = jnp.asarray([0.1, -0.5, 0.3])
        off = jnp.asarray([0.01, 0.02, -0.03])
        xi = jnp.asarray([0.05, -0.1, 0.2])
        u, frac = apply_control_errors(a, gain, off, xi, 10.0)
        expect = (np.asarray(a) * (1 + np.asarray(gain)) + np.asarray(off)) * (1 + np.asarray(xi))
        assert np.allclose(np.asarray(u), expect, rtol=0, atol=1e-15)
        assert float(frac) == 0.0
        # each term alone
        assert np.allclose(apply_control_errors(a, gain, None, None, 10.0)[0], a * (1 + gain))
        assert np.allclose(apply_control_errors(a, None, off, None, 10.0)[0], a + off)
        assert np.allclose(apply_control_errors(a, None, None, xi, 10.0)[0], a * (1 + xi))
        # order: the noise multiplies the sum of the offset and the gain-scaled action
        assert np.allclose(
            apply_control_errors(a, None, off, xi, 10.0)[0], (np.asarray(a) + off) * (1 + xi)
        )
        # a gain below -1 inverts the pulse (support is the real line)
        assert (
            float(
                apply_control_errors(jnp.asarray([0.5]), jnp.asarray([-1.5]), None, None, 9)[0][0]
            )
            == -0.25
        )

    def test_everything_off_is_clip_bit_for_bit(self):
        a = jnp.asarray([0.3, -2.7, 1.1, -0.0, 5.0])
        u, frac = apply_control_errors(a, None, None, None, 2.0)
        assert np.array_equal(np.asarray(u), np.asarray(jnp.clip(a, -2.0, 2.0)))
        assert float(frac) == 2 / 5  # -2.7 and 5.0 exceed the bound
        # zero-valued terms (all widths zero) also leave the action unchanged
        z = jnp.zeros(5)
        u0, _ = apply_control_errors(a, z, z, z, 2.0)
        assert np.array_equal(np.asarray(u0), np.asarray(jnp.clip(a, -2.0, 2.0)))

    def test_clip_fraction_counts_channels_beyond_the_bound(self):
        a = jnp.asarray([1.0, 1.0, 1.0, 1.0])
        gain = jnp.asarray([0.5, 0.0, -2.5, 0.99])  # u = 1.5, 1, -1.5, 1.99; bound 1.2
        u, frac = apply_control_errors(a, gain, None, None, 1.2)
        assert float(frac) == 0.75
        assert np.allclose(np.asarray(u), [1.2, 1.0, -1.2, 1.2])
        # exactly at the bound is not clipped
        assert float(apply_control_errors(jnp.asarray([1.2]), None, None, None, 1.2)[1]) == 0.0

    def test_jit_and_vmap(self):
        f = jax.jit(jax.vmap(lambda a, g: apply_control_errors(a, g, None, None, 1.0)))
        u, frac = f(jnp.ones((3, 2)) * 0.9, jnp.asarray([[0.0, 0.2], [0.0, 0.0], [0.5, 0.5]]))
        assert np.allclose(np.asarray(frac), [0.5, 0.0, 1.0])
        assert u.shape == (3, 2)


class TestConstruction:
    def test_types(self):
        env = gate_env()
        for kw in ({"amplitude_scale": 0.1}, {"offset": "x"}, {"step_noise": 1}):
            with pytest.raises(TypeError, match="Distribution"):
                ControlErrors(env, **kw)
        ControlErrors(env, Normal(0.1), Uniform(0.01), TruncatedNormal(0.01, -0.05, 0.05))
        ControlErrors(env)  # nothing enabled: only clip_fraction

    def test_attributes_are_delegated_and_dims_unchanged(self):
        env = gate_env()
        cenv = ControlErrors(env, Normal(0.1))
        assert cenv.action_dim == env.action_dim and cenv.observation_dim == env.observation_dim
        assert cenv.action_bound == env.action_bound and cenv.horizon == env.horizon
        assert cenv.model is env.model and cenv.unwrapped is env

    def test_relative_errors_may_be_negative_or_below_minus_one(self):
        # no support check on epsilon / xi: a Normal is accepted and may flip the pulse sign
        cenv = ControlErrors(gate_env(), Normal(2.0))
        state, _ = cenv.reset(jax.random.PRNGKey(0))
        assert state.data["gain"].shape == (2,)


# ====================================================================== wrapper == formula
class TestWrapperEqualsFormula:
    def manual(self, env, cenv, key, actions):
        """Run the base env with u computed from the documented key leaves and formula."""
        a_dim = env.action_dim
        shape = () if cenv.correlated else (a_dim,)
        dt = jnp.float64
        gain = off = None
        if cenv.amplitude_scale is not None:
            gain = cenv.amplitude_scale.sample(keys.calibration_key(key, keys.GAIN), 1.0, shape, dt)
        if cenv.offset is not None:
            off = cenv.offset.sample(keys.calibration_key(key, keys.OFFSET), 1.0, shape, dt)
        state, ts = env.reset(key)
        obs = [ts.observation]
        rewards, fractions = [], []
        for k, a in enumerate(actions):
            xi = None
            if cenv.step_noise is not None:
                xi = cenv.step_noise.sample(keys.step_noise_key(key, k), 1.0, shape, dt)
            u, frac = apply_control_errors(a, gain, off, xi, env.action_bound)
            state, ts = env.step(state, u)
            obs.append(ts.observation)
            rewards.append(ts.reward)
            fractions.append(frac)
        return np.stack(obs), np.stack(rewards), np.asarray(state.qstate.to_jax()), fractions

    @pytest.mark.parametrize("correlated", [False, True])
    def test_gate_task(self, correlated):
        env = gate_env()
        cenv = ControlErrors(
            env,
            amplitude_scale=Normal(0.1),
            offset=Uniform(0.01),
            step_noise=TruncatedNormal(0.05, -0.1, 0.12),
            correlated=correlated,
        )
        key = jax.random.PRNGKey(8)
        actions = make_task_snapshot.fixed_pulse(env)[:6]
        o1, r1, q1, f1 = run(cenv, actions, key, 6)
        o2, r2, q2, f2 = self.manual(env, cenv, key, actions)
        # the manual run is not jitted: simulation results agree to 1e-12 (XLA fusion)
        for x, y in ((o1, o2), (r1, r2), (q1, q2)):
            assert np.max(np.abs(x - y)) <= 1e-12
        assert np.allclose(f1, [float(f) for f in f2], atol=0)
        # and the errors are not a no-op
        o0, *_ = run(env, actions, key, 6)
        assert not np.array_equal(o0, o1)

    def test_battery_task_float32_actions(self):
        env = qenvs.make("battery/TwoLevel-v0")
        cenv = ControlErrors(env, Normal(0.05), step_noise=Normal(0.02))
        key = jax.random.PRNGKey(2)
        actions = make_task_snapshot.fixed_pulse(env)[:5]
        o1, r1, _, _ = run(cenv, actions, key, 5)
        o2, r2, _, _ = self.manual(env, cenv, key, actions)
        assert np.max(np.abs(o1 - o2)) <= 1e-12 and np.max(np.abs(r1 - r2)) <= 1e-12

    def test_correlated_shares_one_value_across_channels(self):
        env = gate_env()
        ind = ControlErrors(env, Normal(0.1), Normal(0.01))
        cor = ControlErrors(env, Normal(0.1), Normal(0.01), correlated=True)
        si, _ = ind.reset(jax.random.PRNGKey(3))
        sc, _ = cor.reset(jax.random.PRNGKey(3))
        assert si.data["gain"].shape == (2,) and sc.data["gain"].shape == ()
        assert float(si.data["gain"][0]) != float(si.data["gain"][1])
        # one draw applied to both channels: equal relative scaling
        u, _ = cor.applied_action(sc, jnp.asarray([0.05, 0.05]))
        assert float(u[0]) == float(u[1])
        u, _ = ind.applied_action(si, jnp.asarray([0.05, 0.05]))
        assert float(u[0]) != float(u[1])
        cn = ControlErrors(env, step_noise=Normal(0.1), correlated=True)
        u, _ = cn.applied_action(cn.reset(jax.random.PRNGKey(4))[0], jnp.asarray([0.05, 0.05]))
        assert float(u[0]) == float(u[1]) != 0.05

    def test_clip_fraction_in_info(self):
        env = gate_env()
        bound = env.action_bound
        # a gain of +50% on a drive at 80% of the bound clips channel 0 only
        cenv = ControlErrors(env, Uniform(0.0))  # zero width: no error, only reporting
        state, ts = cenv.reset(jax.random.PRNGKey(0))
        assert CLIP_FRACTION in ts.info and float(ts.info[CLIP_FRACTION]) == 0.0
        assert set(ts.info) == set(env.reset(jax.random.PRNGKey(0))[1].info) | {CLIP_FRACTION}
        # command beyond the bound clips even without any error source
        _, ts = cenv.step(state, jnp.asarray([2 * bound, 0.1 * bound]))
        assert float(ts.info[CLIP_FRACTION]) == 0.5
        # with a constant gain error of exactly +50% (Uniform has no fixed value: use data)
        state2 = eqx.tree_at(lambda s: s.data["gain"], state, jnp.asarray([0.5, 0.5]))
        cenv2 = ControlErrors(env, Uniform(0.1))
        _, ts = cenv2.step(state2, jnp.asarray([0.8 * bound, 0.5 * bound]))
        assert float(ts.info[CLIP_FRACTION]) == 0.5  # 1.2 B clipped, 0.75 B not
        # the inner environment receives the clipped amplitude
        applied, _ = cenv2.applied_action(state2, jnp.asarray([0.8 * bound, 0.5 * bound]))
        assert float(applied[0]) == bound and float(applied[1]) == pytest.approx(0.75 * bound)

    def test_info_and_state_structure_for_vector_envs(self):
        """reset and step infos have the same structure (needed by the vector env)."""
        cenv = ControlErrors(gate_env(), Normal(0.05), step_noise=Normal(0.01))
        state, ts0 = cenv.reset(jax.random.PRNGKey(0))
        state1, ts1 = cenv.step(state, jnp.zeros(2))
        assert jax.tree.structure(ts0.info) == jax.tree.structure(ts1.info)
        assert jax.tree.structure(state) == jax.tree.structure(state1)
        assert ts0.info[CLIP_FRACTION].dtype == ts1.info[CLIP_FRACTION].dtype
        # exact diagnostics are still available from the wrapper
        assert "fidelity" in cenv.info(state1) and "fidelity" in cenv.diagnostics(state1)


# ====================================================================== G1
@pytest.mark.parametrize("task", TASKS)
def test_g1_zero_widths_reproduce_the_released_task(task):
    """With every width 0 (or no error source) transitions, rewards, observations are equal."""
    env = qenvs.make(task)
    pulse = make_task_snapshot.fixed_pulse(env)
    key = jax.random.PRNGKey(0)
    steps = 2 if task in HEAVY_TASKS else 5
    obs, rew, qs, _ = run(env, pulse, key, steps)
    variants = {
        "zero widths": ControlErrors(env, Normal(0.0), Uniform(0.0), Normal(0.0)),
        "truncated zero": ControlErrors(env, TruncatedNormal(0.0, -0.1, 0.1), step_noise=Normal(0)),
        "correlated zero": ControlErrors(env, Normal(0.0), step_noise=Normal(0.0), correlated=True),
        "none": ControlErrors(env),
    }
    if task in HEAVY_TASKS:
        variants = {"zero widths": variants["zero widths"]}
    for what, w in variants.items():
        o, r, q, fractions = run(w, pulse, key, steps)
        assert np.array_equal(obs, o), (what, "observation")
        assert np.array_equal(rew, r), (what, "reward")
        assert np.array_equal(qs, q), (what, "state")
        assert fractions == [0.0] * steps  # the fixed pulse stays inside the bound


# ====================================================================== G2
class TestG2ControlLeaves:
    N = 1024
    SIGMA_G, H_O, SIGMA_X = 0.07, 0.02, 0.03

    def cenv(self, **kw):
        base = {
            "amplitude_scale": Normal(self.SIGMA_G),
            "offset": Uniform(self.H_O),
            "step_noise": Normal(self.SIGMA_X),
        }
        base.update(kw)
        return ControlErrors(gate_env(), **base)

    def keys_n(self, n, run_key=RUN_KEY):
        return jax.vmap(lambda i: keys.episode_key(run_key, i))(jnp.arange(n))

    def test_same_key_same_outputs(self):
        cenv = self.cenv()
        pulse = make_task_snapshot.fixed_pulse(cenv)
        key = jax.random.PRNGKey(12)
        a = run(cenv, pulse, key, 8)
        b = run(cenv, pulse, key, 8)
        for x, y in zip(a[:3], b[:3], strict=True):
            assert np.array_equal(x, y)
        assert a[3] == b[3]
        # a different key gives a different episode
        c = run(cenv, pulse, jax.random.PRNGKey(13), 8)
        assert not np.array_equal(a[0], c[0])

    def test_same_key_same_outputs_float32(self):
        with x32():
            cenv = self.cenv()
            pulse = make_task_snapshot.fixed_pulse(cenv)
            key = jax.random.PRNGKey(12)
            a = run(cenv, pulse, key, 6)
            b = run(cenv, pulse, key, 6)
            assert a[0].dtype == np.float32
            for x, y in zip(a[:3], b[:3], strict=True):
                assert np.array_equal(x, y)

    def test_batch_position_independence(self):
        """Single vs 1024-batch: random samples equal in value, simulation within 1e-12."""
        cenv = self.cenv()
        n = self.N
        ks = self.keys_n(n, jax.random.PRNGKey(5))
        pulse = make_task_snapshot.fixed_pulse(cenv)[:5]

        def episode(k):
            state, ts = cenv.reset(k)
            obs, rew, applied = [ts.observation], [], []
            for a in pulse:
                u, _ = cenv.applied_action(state, a)  # includes the step noise of this step
                state, ts = cenv.step(state, a)
                applied.append(u)
                obs.append(ts.observation)
                rew.append(ts.reward)
            return state.data, jnp.stack(applied), jnp.stack(obs), jnp.stack(rew)

        data_b, u_b, obs_b, rew_b = jax.jit(jax.vmap(episode))(ks)
        single = jax.jit(episode)
        for i in (0, 1, 333, n - 1):
            data_s, u_s, obs_s, rew_s = single(ks[i])
            for k in data_s:
                assert np.array_equal(np.asarray(data_s[k]), np.asarray(data_b[k][i])), k
            assert np.array_equal(np.asarray(u_s), np.asarray(u_b[i]))  # includes step noise
            assert np.max(np.abs(np.asarray(obs_s) - np.asarray(obs_b[i]))) <= 1e-12
            assert np.max(np.abs(np.asarray(rew_s) - np.asarray(rew_b[i]))) <= 1e-12

    def test_arbitrary_episode_from_the_run_key(self):
        cenv = self.cenv()
        n = 32
        batch, _ = jax.vmap(cenv.reset)(self.keys_n(n))
        for i in (0, 9, 31):
            state, _ = cenv.reset(jax.random.fold_in(RUN_KEY, i))
            for k in state.data:
                assert np.array_equal(np.asarray(state.data[k]), np.asarray(batch.data[k][i]))

    def noise_of(self, cenv, key, step):
        """Relative noise of ``step`` as seen through the wrapper (no other error source)."""
        state, _ = cenv.reset(key)
        state = eqx.tree_at(lambda s: s.step, state, jnp.asarray(step))
        u, _ = cenv.applied_action(state, UNIT * jnp.ones(2))
        return np.asarray(u) / UNIT - 1.0

    def test_leaf_separation(self):
        """Switching or reconfiguring one source never changes the draws of another."""
        key = jax.random.PRNGKey(21)
        full = self.cenv()
        s_full, _ = full.reset(key)
        configs = {
            "no gain": self.cenv(amplitude_scale=None),
            "no offset": self.cenv(offset=None),
            "no noise": self.cenv(step_noise=None),
            "other gain": self.cenv(amplitude_scale=TruncatedNormal(0.3, -0.5, 0.5)),
            "other noise": self.cenv(step_noise=Uniform(0.5)),
            "other offset": self.cenv(offset=Normal(1.0)),
        }
        for what, cenv in configs.items():
            state, _ = cenv.reset(key)
            if what not in ("no gain", "other gain"):
                assert np.array_equal(
                    np.asarray(state.data["gain"]), np.asarray(s_full.data["gain"])
                )
            if what not in ("no offset", "other offset"):
                assert np.array_equal(
                    np.asarray(state.data["offset"]), np.asarray(s_full.data["offset"])
                )
        assert "gain" not in configs["no gain"].reset(key)[0].data
        assert "offset" not in configs["no offset"].reset(key)[0].data
        # step noise: same value whether or not the calibration errors are on
        only_noise = ControlErrors(gate_env(), step_noise=Normal(self.SIGMA_X))
        noise_only = [self.noise_of(only_noise, key, k) for k in range(4)]
        for k in range(4):
            ref = Normal(self.SIGMA_X).sample(keys.step_noise_key(key, k), 1.0, (2,), jnp.float64)
            assert np.allclose(noise_only[k], np.asarray(ref), rtol=0, atol=1e-15)
        # ... and with the calibration errors on (solve u = (a (1 + g) + o)(1 + xi) for xi)
        for k in range(4):
            st = eqx.tree_at(lambda x: x.step, s_full, jnp.asarray(k))
            u, _ = full.applied_action(st, UNIT * jnp.ones(2))
            lin = UNIT * (1.0 + np.asarray(s_full.data["gain"])) + np.asarray(s_full.data["offset"])
            assert np.allclose(np.asarray(u) / lin - 1.0, noise_only[k], rtol=0, atol=1e-12)
        # correlated only changes the shape of the draw, not the other leaves
        corr_cfg = self.cenv(correlated=True)
        sc, _ = corr_cfg.reset(key)
        assert sc.data["gain"].shape == () and sc.data["offset"].shape == ()
        # the three sources use three different key leaves
        k_g = keys.calibration_key(key, keys.GAIN)
        k_o = keys.calibration_key(key, keys.OFFSET)
        k_n = keys.step_noise_key(key, 0)
        assert len({tuple(np.asarray(k).tolist()) for k in (k_g, k_o, k_n)}) == 3
        # and other wrappers do not change them either
        pr = ParameterRandomization(gate_env(), {"detuning_ghz": Normal(0.002)})
        outer = ControlErrors(
            pr,
            Normal(self.SIGMA_G),
            Uniform(self.H_O),
            Normal(self.SIGMA_X),
        )
        so, _ = outer.reset(key)
        assert np.array_equal(np.asarray(so.data["gain"]), np.asarray(s_full.data["gain"]))
        assert np.array_equal(np.asarray(so.data["offset"]), np.asarray(s_full.data["offset"]))

    def test_step_noise_is_a_function_of_the_step_index_only(self):
        """Step k uses fold_in(step-noise leaf, k): independent of how the run got there."""
        cenv = self.cenv(amplitude_scale=None, offset=None)
        key = jax.random.PRNGKey(33)
        state, _ = cenv.reset(key)
        by_stepping = []
        for _ in range(5):
            u, _ = cenv.applied_action(state, UNIT * jnp.ones(2))
            by_stepping.append(np.asarray(u) / UNIT - 1.0)
            state, _ = cenv.step(state, UNIT * jnp.ones(2))
        by_index = [self.noise_of(cenv, key, k) for k in range(5)]
        for a, b in zip(by_stepping, by_index, strict=True):
            assert np.allclose(a, b, rtol=0, atol=1e-15)
        assert not np.array_equal(by_index[0], by_index[1])  # different steps, different noise

    def test_empirical_distribution_and_correlations(self):
        """1024 episodes: moments within 4 SE; adjacent-episode and leaf correlations small."""
        n = self.N
        cenv = self.cenv()
        thr = 4 / math.sqrt(n)
        ks = self.keys_n(n)
        state, _ = jax.vmap(cenv.reset)(ks)
        gain = np.asarray(state.data["gain"])  # (n, 2)
        offset = np.asarray(state.data["offset"])
        only_noise = self.cenv(amplitude_scale=None, offset=None)

        def noise(step):
            def one(k):
                s, _ = only_noise.reset(k)
                s = eqx.tree_at(lambda x: x.step, s, jnp.asarray(step))
                return only_noise.applied_action(s, UNIT * jnp.ones(2))[0] / UNIT - 1.0

            return np.asarray(jax.vmap(one)(ks))

        xi0, xi1 = noise(0), noise(1)
        for x, moments in (
            (gain, normal_moments(self.SIGMA_G)),
            (offset, uniform_moments(self.H_O)),
            (xi0, normal_moments(self.SIGMA_X)),
        ):
            for c in range(2):
                assert_moments_within_4se(x[:, c], *moments)
                assert abs(corr(x[:-1, c], x[1:, c])) <= thr  # adjacent episodes
        leaves = {"gain": gain, "offset": offset, "xi0": xi0, "xi1": xi1}
        names = sorted(leaves)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                for c in range(2):
                    assert abs(corr(leaves[a][:, c], leaves[b][:, c])) <= thr, (a, b, c)
        # the two channels are independent unless correlated=True
        assert abs(corr(gain[:, 0], gain[:, 1])) <= thr
        # steps of one episode are independent
        for c in range(2):
            assert abs(corr(xi0[:, c], xi1[:, c])) <= thr


# ====================================================================== integration
def test_vector_env_and_gymnasium_smoke():
    pytest.importorskip("gymnasium")
    from qenvs.wrappers import GymnasiumEnv, GymnasiumVectorEnv

    env = ControlErrors(
        ParameterRandomization(gate_env(), {"detuning_ghz": Normal(0.002)}),
        Normal(0.05),
        step_noise=Normal(0.01),
    )
    genv = GymnasiumEnv(env, seed=1)
    obs, _ = genv.reset(seed=1)
    obs, _reward, _done, _, info = genv.step(np.zeros(2, dtype=np.float32))
    assert obs.shape == (env.observation_dim,) and CLIP_FRACTION in info
    venv = GymnasiumVectorEnv(env, 4, seed=1)
    obs, _ = venv.reset(seed=1)
    for _ in range(env.horizon + 1):  # through one autoreset
        obs, _reward, _term, _, info = venv.step(np.zeros((4, 2), dtype=np.float32))
    assert obs.shape == (4, env.observation_dim)
    assert info[CLIP_FRACTION].shape == (4,)
