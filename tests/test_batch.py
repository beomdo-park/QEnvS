"""Batch execution (spec G2 batch position, G7), the episode-key rule of the Gymnasium adapters.

* the key tree gives the same samples to an episode in a batch of 1024 and alone, and its
  leaves, episodes and neighbouring episodes are empirically uncorrelated (n = 1024, 4 SE);
* simulation results (observation, reward) of an episode in a batch agree with the single run to
  1e-12 in x64 (XLA changes the operation order, so the last bits may differ);
* ``GymnasiumEnv`` and ``GymnasiumVectorEnv`` follow ``keys.episode_key(run_key, i)``.
"""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from dynamiqs.method import Expm

import qenvs
from qenvs import keys
from qenvs.envs.base import Environment, EnvState, TimeStep
from qenvs.wrappers.observation import ObservationModel

try:
    from qenvs.distributions import Normal
    from qenvs.wrappers.control import ControlErrors
    from qenvs.wrappers.randomization import ParameterRandomization

    HAVE_RANDOMISERS = True
except ImportError:  # pragma: no cover (the randomisers are a separate module of the release)
    HAVE_RANDOMISERS = False

N = 1024
RUN_KEY = jax.random.PRNGKey(20261016)
HORIZON, DT = 4, 0.5


def episode_keys(n=N, run_key=RUN_KEY):
    return jax.vmap(lambda i: keys.episode_key(run_key, i))(jnp.arange(n))


def qubit_env():
    env = qenvs.make("qubit/StateTransfer-v0", horizon=HORIZON, dt=DT)
    return dataclasses.replace(env, method=Expm())


# --------------------------------------------------------------- key tree, 1024 episodes
LEAVES = {
    "parameter/detuning": lambda e: keys.parameter_key(e, "detuning"),
    "parameter/anharmonicity_ghz": lambda e: keys.parameter_key(e, "anharmonicity_ghz"),
    "calibration/gain": lambda e: keys.calibration_key(e, keys.GAIN),
    "calibration/offset": lambda e: keys.calibration_key(e, keys.OFFSET),
    "step_noise/0": lambda e: keys.step_noise_key(e, 0),
    "step_noise/1": lambda e: keys.step_noise_key(e, 1),
    "measurement/t0_j0": lambda e: keys.measurement_key(e, 0, 0),
    "measurement/t0_j1": lambda e: keys.measurement_key(e, 0, 1),
    "measurement/t1_j0": lambda e: keys.measurement_key(e, 1, 0),
    "policy/t0": lambda e: keys.policy_key(e, 0),
    "policy/t1": lambda e: keys.policy_key(e, 1),
}


def leaf_draws(ep):
    """One standard normal per episode and leaf: array ``(n_leaves, n_episodes)``."""
    return jnp.stack(
        [jax.vmap(lambda e, f=f: jax.random.normal(f(e)))(ep) for f in LEAVES.values()]
    )


class TestKeyTreeBatch:
    def test_samples_do_not_depend_on_the_batch_position(self):
        ep = episode_keys()
        batch = np.asarray(jax.jit(leaf_draws)(ep))
        for i in (0, 1, 2, 511, 1022, 1023):
            single = np.asarray(jnp.stack([jax.random.normal(f(ep[i])) for f in LEAVES.values()]))
            assert np.array_equal(batch[:, i], single)
        # the episode key is the rule fold_in(run_key, i)
        for i in (0, 5, 1023):
            assert np.array_equal(ep[i], jax.random.fold_in(RUN_KEY, i))
        # reversing the batch only reverses the columns
        rev = np.asarray(jax.jit(leaf_draws)(ep[::-1]))
        assert np.array_equal(rev[:, ::-1], batch)

    def test_leaf_marginals_adjacent_and_cross_correlations(self):
        z = np.asarray(jax.jit(leaf_draws)(episode_keys()), dtype=float)
        n = z.shape[1]
        names = list(LEAVES)
        tol_r = 4 / math.sqrt(n)
        for name, row in zip(names, z, strict=True):
            assert abs(row.mean()) <= 4 / math.sqrt(n), name  # N(0, 1): SE of the mean
            assert abs(row.var(ddof=1) - 1.0) <= 4 * math.sqrt(2 / (n - 1)), name
            r_adjacent = np.corrcoef(row[:-1], row[1:])[0, 1]
            assert abs(r_adjacent) <= tol_r, (name, r_adjacent)
        corr = np.corrcoef(z)
        off = corr[~np.eye(len(names), dtype=bool)]
        assert np.max(np.abs(off)) <= tol_r, np.max(np.abs(off))


# ---------------------------------------------------- simulation results in a batch (1e-12)
def stacked_envs():
    envs = {
        "observation": ObservationModel(qubit_env(), "expectations", shots=30, observables=("p1",))
    }
    if HAVE_RANDOMISERS:
        base = qubit_env()
        envs["stack"] = ObservationModel(
            ControlErrors(
                ParameterRandomization(base, {"detuning": Normal(std=0.05)}),
                amplitude_scale=Normal(std=0.02),
                step_noise=Normal(std=0.01),
            ),
            "expectations",
            shots=30,
            observables=("p1",),
        )
    return envs


@pytest.mark.parametrize("which", ["observation", "stack"])
def test_batch_and_single_runs_agree(which):
    env = stacked_envs().get(which)
    if env is None:
        pytest.skip("randomisation wrappers not available")
    ep = episode_keys()
    actions = jnp.asarray([[0.8], [1.9], [-0.4], [1.2]])

    def run(key):
        state, ts = env.reset(key)
        obs, rew = [ts.observation], []
        for a in actions:
            state, ts = env.step(state, a)
            obs.append(ts.observation)
            rew.append(ts.reward)
        return state.data if hasattr(state, "data") else {}, jnp.stack(obs), jnp.stack(rew)

    batch_data, batch_obs, batch_rew = jax.jit(jax.vmap(run))(ep)
    single = jax.jit(run)
    for i in (0, 1, 333, 1023):
        data, obs, rew = single(ep[i])
        np.testing.assert_allclose(batch_obs[i], obs, atol=1e-12, rtol=0)
        np.testing.assert_allclose(batch_rew[i], rew, atol=1e-12, rtol=0)
        # random samples (parameters, calibration errors) are values, not simulation results
        for k, v in data.items():
            assert np.array_equal(batch_data[k][i], v), k


@pytest.mark.skipif(not HAVE_RANDOMISERS, reason="randomisation wrappers not available")
class TestStackedLeafSeparation:
    """Changing one wrapper's configuration does not change another leaf's samples."""

    def sample(self, env, i=3):
        state, _ = env.reset(episode_keys(8)[i])
        out = {}
        s = state
        while hasattr(s, "data"):
            out.update({k: np.asarray(v) for k, v in s.data.items()})
            s = s.inner
        return out

    def test_observation_settings_do_not_touch_other_samples(self):
        base = ParameterRandomization(qubit_env(), {"detuning": Normal(std=0.05)})
        env_a = ControlErrors(base, amplitude_scale=Normal(std=0.02))
        ref = self.sample(env_a)
        for kw in (
            {"observation": "time"},
            {"observation": "expectations", "shots": 10, "observables": ("p1",)},
            {"observation": "expectations", "observables": ("energy",)},
        ):
            got = self.sample(ObservationModel(env_a, **kw))
            for k, v in ref.items():
                assert np.array_equal(got[k], v), (kw, k)

    def test_other_wrappers_do_not_touch_the_parameter_samples(self):
        d = {"detuning": Normal(std=0.05)}
        plain = self.sample(ParameterRandomization(qubit_env(), d))
        with_control = self.sample(
            ControlErrors(
                ParameterRandomization(qubit_env(), d),
                amplitude_scale=Normal(std=0.02),
                step_noise=Normal(std=0.01),
            )
        )
        with_obs = self.sample(
            ObservationModel(
                ParameterRandomization(qubit_env(), d), "expectations", shots=5, observables=("p1",)
            )
        )
        for k, v in plain.items():
            assert np.array_equal(with_control[k], v) and np.array_equal(with_obs[k], v)

    def test_oracle_entries_survive_the_observation_modes(self):
        base = ParameterRandomization(
            qubit_env(), {"detuning": Normal(std=0.05)}, oracle=True
        )  # appends one entry to the observation
        for mode, k in (("expectations", 1), ("time", 0), ("state", None)):
            kw = {"observables": ("p1",)} if mode == "expectations" else {}
            inner = ObservationModel(base, mode, **kw)  # oracle inside, measured outside
            _, ts = inner.reset(episode_keys(2)[1])
            assert ts.observation.shape == (inner.observation_dim,)
            if k is not None:
                assert ts.observation.shape == (k + 1 + 1,)
            outer = ParameterRandomization(
                ObservationModel(qubit_env(), mode, **kw),
                {"detuning": Normal(std=0.05)},
                oracle=True,
            )  # oracle outside
            _, ts2 = outer.reset(episode_keys(2)[1])
            assert ts2.observation.shape == (outer.observation_dim,)
            np.testing.assert_allclose(ts2.observation[-1], ts.observation[-1], atol=1e-12)


# --------------------------------------------------- wrapper-level noise: 1024 episodes (G7)
def test_measurement_noise_is_uncorrelated_across_episodes_observables_and_steps():
    env = qenvs.make("battery/Dicke-v0", n_cells=3, horizon=4, dt=0.2)
    env = dataclasses.replace(env, method=Expm())
    shots = 50
    noisy = ObservationModel(env, "expectations", shots=shots)
    exact = ObservationModel(env, "expectations")
    ep = episode_keys()
    acts = jnp.asarray([[0.3], [0.4], [0.2], [0.1]])

    def run(model):
        def one(key):
            state, ts = model.reset(key)
            out = [ts.observation[:-1]]
            for a in acts:
                state, ts = model.step(state, a)
                out.append(ts.observation[:-1])
            return jnp.stack(out)  # (steps + 1, K)

        return np.asarray(jax.jit(jax.vmap(one))(ep), dtype=float)

    noise = run(noisy) - run(exact)  # (episodes, steps + 1, K)
    n = noise.shape[0]
    tol = 4 / math.sqrt(n)
    # the noise is large enough to be measured (the Dicke initial state is an eigenstate: t = 0 is
    # exactly noiseless, later steps are not)
    assert np.all(np.abs(noise[:, 0, :]) <= 1e-12)
    flat = noise[:, 1:, :].reshape(n, -1)
    for col in range(flat.shape[1]):
        if flat[:, col].std() < 1e-9:
            continue
        assert abs(np.corrcoef(flat[:-1, col], flat[1:, col])[0, 1]) <= tol  # neighbouring episodes
    keep = flat.std(axis=0) > 1e-9
    corr = np.corrcoef(flat[:, keep], rowvar=False)  # steps and observables against each other
    assert np.max(np.abs(corr[~np.eye(corr.shape[0], dtype=bool)])) <= tol


# ---------------------------------------------------------- Gymnasium adapters (episode rule)
try:
    from qenvs.wrappers import GymnasiumEnv, GymnasiumVectorEnv

    HAVE_GYM = True
except ImportError:  # pragma: no cover (the gym extra is optional)
    HAVE_GYM = False

needs_gym = pytest.mark.skipif(not HAVE_GYM, reason="gymnasium not installed")


class ToyRandomReset(Environment):
    """Cheap env whose reset is random, to expose the key each episode gets."""

    @property
    def action_dim(self):
        return 1

    @property
    def observation_dim(self):
        return 2

    @property
    def action_bound(self):
        return 1.0

    @property
    def horizon(self):
        return 3

    def _ts(self, x, t, done):
        return TimeStep(jnp.stack([x, t]), jnp.asarray(0.0), done, {"x": x})

    def reset(self, key):
        x = jax.random.normal(key)
        t = jnp.asarray(0.0)
        state = EnvState(qstate=x, t=t, step=jnp.asarray(0), key=key)
        return state, self._ts(x, t, jnp.asarray(False))

    def step(self, state, action):
        x = state.qstate + 0.0 * action[0]
        step = state.step + 1
        new = EnvState(qstate=x, t=state.t + 1.0, step=step, key=state.key)
        return new, self._ts(x, new.t, step >= self.horizon)


def expected_first_coordinate(seed, i):
    key = keys.episode_key(jax.random.PRNGKey(seed), i)
    return np.float32(jax.random.normal(key))


@needs_gym
class TestEpisodeKeyRule:
    def test_single_env_episode_i_uses_fold_in_run_key_i(self):
        env = GymnasiumEnv(ToyRandomReset(), seed=11)
        obs, _ = env.reset()
        assert obs[0] == expected_first_coordinate(11, 0)
        for i in range(1, 4):
            for _ in range(3):
                env.step(np.zeros(1, np.float32))
            obs, _ = env.reset()
            assert obs[0] == expected_first_coordinate(11, i)
        # reset(seed=...) restarts the run at episode 0
        obs, _ = env.reset(seed=11)
        assert obs[0] == expected_first_coordinate(11, 0)
        obs, _ = env.reset(seed=4)
        assert obs[0] == expected_first_coordinate(4, 0)

    @pytest.mark.parametrize("num_envs", [1, 3])
    def test_vector_env_episode_index_is_j_times_num_envs_plus_b(self, num_envs):
        seed = 5
        venv = GymnasiumVectorEnv(ToyRandomReset(), num_envs, seed=seed)
        obs, _ = venv.reset()
        for b in range(num_envs):
            assert obs[b, 0] == expected_first_coordinate(seed, b)
        a = np.zeros((num_envs, 1), np.float32)
        for j in range(1, 4):
            for _ in range(3):  # horizon steps, the last one terminates
                venv.step(a)
            obs, *_ = venv.step(a)  # NEXT_STEP autoreset: the next call returns the new episode
            for b in range(num_envs):
                assert obs[b, 0] == expected_first_coordinate(seed, j * num_envs + b), (j, b)

    def test_vector_env_replays_the_single_env_run_in_interleaved_order(self):
        n, seed = 2, 9
        single = GymnasiumEnv(ToyRandomReset(), seed=seed)
        firsts = []
        for _ in range(2 * n):
            obs, _ = single.reset()
            firsts.append(obs[0])
        venv = GymnasiumVectorEnv(ToyRandomReset(), n, seed=seed)
        obs, _ = venv.reset()
        a = np.zeros((n, 1), np.float32)
        for _ in range(4):
            second, *_ = venv.step(a)
        np.testing.assert_array_equal(obs[:, 0], firsts[:n])
        np.testing.assert_array_equal(second[:, 0], firsts[n : 2 * n])

    def test_vector_env_reset_semantics(self):
        venv = GymnasiumVectorEnv(ToyRandomReset(), 2, seed=3)
        a, _ = venv.reset(seed=8)
        b, _ = venv.reset(seed=8)
        np.testing.assert_array_equal(a, b)  # seeding restarts the run
        c, _ = venv.reset()  # without a seed the run continues with fresh episodes
        assert not np.array_equal(a[:, 0], c[:, 0])
        np.testing.assert_array_equal(
            c[:, 0], [expected_first_coordinate(8, 2), expected_first_coordinate(8, 3)]
        )

    def test_deterministic_tasks_are_unchanged(self):
        # a deterministic task ignores the key, so the adapters behave as before
        raw = qenvs.make("qubit/StateTransfer-v0", horizon=4, dt=0.5)
        single = GymnasiumEnv(raw, seed=0)
        venv = GymnasiumVectorEnv(raw, 2, seed=0)
        so, _ = single.reset()
        vo, _ = venv.reset()
        np.testing.assert_allclose(vo[0], so, atol=1e-6)
        np.testing.assert_allclose(vo[1], so, atol=1e-6)


@needs_gym
class TestNestedDiagnosticsInfo:
    def test_single_and_vector_info_hold_nested_diagnostics(self):
        raw = ObservationModel(qubit_env(), "time", diagnostics=True)
        single = GymnasiumEnv(raw, seed=0)
        _, info = single.reset()
        assert isinstance(info["diagnostics"], dict)
        assert isinstance(info["diagnostics"]["fidelity"], np.ndarray)
        _, _, _, _, info = single.step(np.array([1.0], np.float32))
        assert set(info) == {"shots_used", "diagnostics"}
        venv = GymnasiumVectorEnv(raw, 3, seed=0)
        _, vinfo = venv.reset()
        assert vinfo["diagnostics"]["fidelity"].shape == (3,)
        assert vinfo["diagnostics"]["_fidelity"].shape == (3,) and vinfo["_diagnostics"].all()
        _, _, _, _, vinfo = venv.step(np.ones((3, 1), np.float32))
        assert vinfo["shots_used"].shape == (3,)
        assert vinfo["diagnostics"]["fidelity"].shape == (3,)

    def test_partial_observation_dims_reach_the_spaces(self):
        raw = ObservationModel(qubit_env(), "expectations", shots=10, observables=("p1",))
        assert GymnasiumEnv(raw).observation_space.shape == (2,)
        assert GymnasiumVectorEnv(raw, 2).single_observation_space.shape == (2,)
