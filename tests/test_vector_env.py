"""GymnasiumVectorEnv: API, equivalence with single envs, autoreset, throughput smoke."""

import time
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

import qenvs  # noqa: E402
from qenvs.envs.base import Environment, EnvState, TimeStep  # noqa: E402
from qenvs.wrappers import GymnasiumEnv  # noqa: E402
from qenvs.wrappers.vector import GymnasiumVectorEnv  # noqa: E402

TID = "qubit/StateTransfer-v0"
H = 4


@pytest.fixture
def x32():
    """Run with jax x64 off, the library default. conftest.py enables x64 globally, which
    makes float32 conversion copy (writable) and would hide read-only-view bugs."""
    if hasattr(jax, "enable_x64"):
        ctx = jax.enable_x64(False)
    else:  # pragma: no cover (older jax)
        from jax.experimental import disable_x64

        ctx = disable_x64()
    with ctx:
        assert not jax.config.jax_enable_x64
        yield


def _make(horizon=H):
    return qenvs.make(TID, horizon=horizon, dt=0.2)


class ToyRandomReset(Environment):
    """Cheap env with a random reset, to test per-env PRNG streams."""

    horizon_: int = 3

    @property
    def action_dim(self):
        return 1

    @property
    def observation_dim(self):
        return 2

    @property
    def action_bound(self):
        return 1.5

    @property
    def horizon(self):
        return self.horizon_

    def _ts(self, x, t, r, done):
        return TimeStep(jnp.stack([x, t]), r, done, {"x": x})

    def reset(self, key):
        x = jax.random.normal(key)
        t = jnp.asarray(0.0)
        s = EnvState(qstate=x, t=t, step=jnp.asarray(0), key=key)
        return s, self._ts(x, t, jnp.asarray(0.0), jnp.asarray(False))

    def step(self, state, action):
        a = jnp.clip(action[0], -1.5, 1.5)
        x = state.qstate + a
        step = state.step + 1
        s = EnvState(qstate=x, t=state.t + 1.0, step=step, key=state.key)
        return s, self._ts(x, s.t, a, step >= self.horizon_)


def _actions(rng, n, d, bound, steps):
    return rng.uniform(-bound, bound, size=(steps, n, d)).astype(np.float32)


def test_spaces_shapes_dtypes_and_metadata():
    n = 5
    venv = GymnasiumVectorEnv(_make(), n)
    assert isinstance(venv, gym.vector.VectorEnv)
    assert venv.num_envs == n
    d = venv.single_observation_space.shape[0]
    assert venv.observation_space.shape == (n, d)
    assert venv.action_space.shape == (n, 1)
    assert venv.single_action_space.high[0] == 2.0
    if hasattr(gym.vector, "AutoresetMode"):
        assert venv.metadata["autoreset_mode"] == gym.vector.AutoresetMode.NEXT_STEP
    obs, info = venv.reset(seed=0)
    assert obs.shape == (n, d) and obs.dtype == np.float32
    assert venv.observation_space.contains(obs)
    assert info["fidelity"].shape == (n,) and info["_fidelity"].all()
    obs, r, term, trunc, info = venv.step(venv.action_space.sample())
    assert obs.dtype == np.float32 and r.dtype == np.float32
    assert r.shape == term.shape == trunc.shape == (n,)
    assert term.dtype == np.bool_ and trunc.dtype == np.bool_ and not trunc.any()
    assert info["fidelity"].shape == (n,)


def test_normalized_space_and_exact_scaling():
    raw = _make()
    b = float(raw.action_bound)
    v1 = GymnasiumVectorEnv(raw, 3, normalize_actions=True)
    v2 = GymnasiumVectorEnv(raw, 3)
    assert v1.single_action_space.low[0] == -1.0 and v1.single_action_space.high[0] == 1.0
    assert v2.single_action_space.high[0] == b
    v1.reset(seed=2)
    v2.reset(seed=2)
    a = np.array([[0.25], [-1.0], [0.6]], dtype=np.float32)
    o1, r1, *_ = v1.step(a)
    o2, r2, *_ = v2.step(a * b)
    np.testing.assert_array_equal(o1, o2)
    np.testing.assert_array_equal(r1, r2)


@pytest.mark.parametrize("normalize", [False, True])
def test_equivalent_to_independent_single_envs(normalize):
    n, steps = 3, 2 * H + 2  # crosses the horizon and one autoreset
    raw = _make()
    venv = GymnasiumVectorEnv(raw, n, seed=7, normalize_actions=normalize)
    singles = [GymnasiumEnv(raw, seed=7 + i, normalize_actions=normalize) for i in range(n)]
    bound = 1.0 if normalize else float(raw.action_bound)
    acts = _actions(np.random.default_rng(0), n, 1, bound, steps)
    vobs, _ = venv.reset()
    sobs = [e.reset()[0] for e in singles]
    np.testing.assert_allclose(vobs, np.stack(sobs), atol=1e-6)
    pending = [False] * n
    for k in range(steps):
        vo, vr, vt, _, _ = venv.step(acts[k])
        for i, e in enumerate(singles):
            if pending[i]:  # NEXT_STEP: this call only resets the single env
                so, _ = e.reset()
                sr, st = 0.0, False
            else:
                so, sr, st, _, _ = e.step(acts[k, i])
            np.testing.assert_allclose(vo[i], so, atol=1e-6)
            assert vr[i] == pytest.approx(sr, abs=1e-6)
            assert bool(vt[i]) == st
            pending[i] = st


def test_autoreset_next_step_after_horizon():
    n = 2
    venv = GymnasiumVectorEnv(_make(), n)
    obs0, info0 = venv.reset(seed=0)
    a = np.full((n, 1), 0.7, dtype=np.float32)
    for k in range(H):
        obs, r, term, trunc, info = venv.step(a)
        assert term.all() == (k == H - 1)
        assert not trunc.any()
    final_obs, final_f = obs, info["fidelity"]
    assert final_obs[0, -1] == pytest.approx(1.0, abs=1e-6)  # normalised time at horizon
    # next call: action ignored, reset obs, zero reward, not terminated
    obs, r, term, trunc, info = venv.step(np.full((n, 1), -2.0, dtype=np.float32))
    np.testing.assert_allclose(obs, obs0, atol=1e-6)
    assert (r == 0).all() and not term.any() and not trunc.any()
    np.testing.assert_allclose(info["fidelity"], info0["fidelity"], atol=1e-6)
    assert (final_f > info["fidelity"]).all()
    # and the episode then runs a full fresh horizon
    for k in range(H):
        _, _, term, _, _ = venv.step(a)
        assert term.all() == (k == H - 1)


def test_reset_seed_reproducible_and_errors():
    venv = GymnasiumVectorEnv(ToyRandomReset(), 4)
    with pytest.raises(RuntimeError):
        venv.step(np.zeros((4, 1), np.float32))
    a, _ = venv.reset(seed=5)
    b, _ = venv.reset(seed=5)
    np.testing.assert_array_equal(a, b)
    assert len({float(x) for x in a[:, 0]}) == 4  # distinct per sub-env
    with pytest.raises(ValueError, match="actions must have shape"):
        venv.step(np.zeros((3, 1), np.float32))
    # a 1-D action of shape (num_envs,) is rejected too: no implicit reshape
    with pytest.raises(ValueError, match="actions must have shape"):
        venv.step(np.zeros((4,), np.float32))
    with pytest.raises(ValueError, match="num_envs must be"):
        GymnasiumVectorEnv(ToyRandomReset(), 0)


def test_reset_clears_pending_autoreset(x32):
    n = 2
    venv = GymnasiumVectorEnv(_make(), n)
    venv.reset(seed=0)
    a = np.full((n, 1), 0.7, dtype=np.float32)
    for _ in range(H):
        _, _, term, _, _ = venv.step(a)
    assert term.all()  # autoreset is now pending
    venv.reset()  # explicit reset must discard the pending autoreset
    obs, r, term, trunc, _ = venv.step(a)
    # the action is used (not ignored): nonzero reward, one step of time elapsed
    assert (np.abs(r) > 1e-6).all()
    assert not term.any() and not trunc.any()
    np.testing.assert_allclose(obs[:, -1], 1.0 / H, atol=1e-6)


def _all_writeable(arrays_and_infos):
    obs, r, term, trunc, info = arrays_and_infos
    arrs = [obs, r, term, trunc, *info.values()]
    return all(x.flags.writeable for x in arrs)


def test_returned_arrays_are_writable(x32):
    n = 3
    venv = GymnasiumVectorEnv(_make(), n)
    obs, info = venv.reset(seed=0)
    assert obs.flags.writeable and all(v.flags.writeable for v in info.values())
    out = venv.step(np.zeros((n, 1), np.float32))
    assert _all_writeable(out)
    # in-place edits work and do not corrupt the env (next step is unchanged)
    out[0][0] = 0.0
    out[1][:] = 5.0
    out[2][:] = True
    out[4]["fidelity"][0] = -1.0
    ref = GymnasiumVectorEnv(_make(), n)
    ref.reset(seed=0)
    ref.step(np.zeros((n, 1), np.float32))
    a = np.full((n, 1), 0.3, np.float32)
    np.testing.assert_array_equal(venv.step(a)[0], ref.step(a)[0])


def test_toy_env_arrays_are_writable(x32):
    venv = GymnasiumVectorEnv(ToyRandomReset(), 2)
    venv.reset(seed=0)
    assert _all_writeable(venv.step(np.zeros((2, 1), np.float32)))


def test_clip_reward_wrapper(x32):
    from gymnasium.wrappers.vector import ClipReward

    n = 3
    venv = ClipReward(GymnasiumVectorEnv(_make(), n), max_reward=0.05)
    venv.reset(seed=0)
    a = np.full((n, 1), 0.7, np.float32)
    for _ in range(H + 1):
        _, r, *_ = venv.step(a)
        assert (r <= 0.05 + 1e-7).all()


def test_vectorize_transform_reward_wrapper(x32):
    from gymnasium.wrappers import TransformReward
    from gymnasium.wrappers.vector import VectorizeTransformReward

    n = 3
    base = GymnasiumVectorEnv(_make(), n)
    venv = VectorizeTransformReward(base, TransformReward, func=lambda r: r * 2.0)
    ref = GymnasiumVectorEnv(_make(), n)
    venv.reset(seed=0)
    ref.reset(seed=0)
    a = np.full((n, 1), 0.7, np.float32)
    for _ in range(H):
        r2 = venv.step(a)[1]
        r1 = ref.step(a)[1]
        np.testing.assert_allclose(r2, 2.0 * r1, atol=1e-6)


def test_vectorize_transform_observation_wrapper(x32):
    from gymnasium.wrappers import TransformObservation
    from gymnasium.wrappers.vector import VectorizeTransformObservation

    n = 3
    d = GymnasiumVectorEnv(_make(), 1).single_observation_space.shape[0]

    def zero_first(o):
        o = np.array(o)
        o[0] = 0.0
        return o

    single_space = gym.spaces.Box(-np.inf, np.inf, shape=(d,), dtype=np.float32)
    venv = VectorizeTransformObservation(
        GymnasiumVectorEnv(_make(), n),
        TransformObservation,
        func=zero_first,
        observation_space=single_space,
    )
    obs, _ = venv.reset(seed=0)
    assert (obs[:, 0] == 0.0).all()
    for _ in range(H + 1):
        obs, *_ = venv.step(np.full((n, 1), 0.3, np.float32))
        assert (obs[:, 0] == 0.0).all()


@pytest.mark.parametrize("use_normalized", [False, True])
def test_in_place_obs_edit_does_not_raise(x32, use_normalized):
    venv = GymnasiumVectorEnv(_make(), 2, normalize_actions=use_normalized)
    obs, _ = venv.reset(seed=0)
    obs[0] = 0.0
    obs, *_ = venv.step(np.zeros((2, 1), np.float32))
    obs[0] = 0.0
    assert (obs[0] == 0.0).all()


def test_record_episode_statistics():
    from gymnasium.wrappers.vector import RecordEpisodeStatistics

    n = 3
    venv = RecordEpisodeStatistics(GymnasiumVectorEnv(_make(), n))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        venv.reset(seed=0)
        a = np.full((n, 1), 0.5, dtype=np.float32)
        seen = None
        for _ in range(H + 1):
            _, _, _, _, info = venv.step(a)
            if "episode" in info:
                seen = info
    assert seen is not None
    assert seen["_episode"].all()
    assert (seen["episode"]["l"] == H).all()
    # return equals final fidelity (reward = fidelity gain, initial fidelity is 0)
    assert np.all(seen["episode"]["r"] >= 0.0) and np.all(seen["episode"]["r"] <= 1.0 + 1e-5)


def test_cleanrl_style_loop_three_episodes():
    n = 4
    envs = GymnasiumVectorEnv(_make(), n, normalize_actions=True)
    obs, _ = envs.reset(seed=1)
    done_count = 0
    returns = np.zeros(n)
    finished = []
    for _ in range(3 * (H + 1)):
        actions = envs.action_space.sample()
        obs, r, term, trunc, _ = envs.step(actions)
        returns += r
        for i in np.flatnonzero(term | trunc):
            finished.append(returns[i])
            returns[i] = 0.0
            done_count += 1
        assert np.isfinite(obs).all()
    assert done_count == 3 * n
    assert all(-1e-5 <= f <= 1.0 + 1e-5 for f in finished)


def test_other_task_works_in_vector():
    venv = GymnasiumVectorEnv(qenvs.make("battery/TwoLevel-v0"), 2, normalize_actions=True)
    obs, _ = venv.reset(seed=0)
    obs, r, _, _, _ = venv.step(np.zeros((2, venv.single_action_space.shape[0]), np.float32))
    assert obs.shape[0] == 2 and np.isfinite(obs).all() and np.isfinite(r).all()


def test_throughput_smoke(capsys):
    rows = []
    for n in (1, 64, 1024):
        venv = GymnasiumVectorEnv(_make(horizon=20), n, normalize_actions=True)
        venv.reset(seed=0)
        a = np.zeros((n, 1), np.float32)
        venv.step(a)  # compile
        k = 20 if n < 1024 else 5
        t0 = time.perf_counter()
        for _ in range(k):
            venv.step(a)
        dt = time.perf_counter() - t0
        rows.append((n, k * n / dt))
    with capsys.disabled():
        for n, sps in rows:
            print(f"\nTHROUGHPUT num_envs={n} steps_per_s={sps:.0f}")
    assert all(s > 0 for _, s in rows)
