"""Observation modes and finite-shot measurement (spec 4.4, gates G2 measurement leaf and G5).

Each statistical test is preceded by deterministic reference tests; the statistical tests use
fixed keys (deterministic verdicts) and the exact 4 SE criteria of the design document.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

import dynamiqs as dq
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from dynamiqs.method import Expm

import qenvs
from qenvs import keys
from qenvs.envs import StateTransfer
from qenvs.models.custom import HamiltonianModel
from qenvs.wrappers import Wrapper
from qenvs.wrappers.observation import ObservationModel, observation_kind, outcome_counts

R = 10_000  # repetitions of the statistical checks
RUN_KEY = jax.random.PRNGKey(20261015)

Z = np.diag([1.0, -1.0])
X = np.array([[0.0, 1.0], [1.0, 0.0]])
I2 = np.eye(2)


# ----------------------------------------------------------------------------- helpers
class TwoQubit(HamiltonianModel):
    """Small model with second-order correlators, to test ``correlation_observables``."""

    def correlation_observables(self):
        return {
            "zz": dq.asqarray(np.kron(Z, Z), dims=self.dims),
            "xx": dq.asqarray(np.kron(X, X), dims=self.dims),
        }


def two_qubit_env(model_cls=TwoQubit, **overrides):
    kw = {
        "drift": 0.5 * (np.kron(Z, I2) + 0.7 * np.kron(I2, Z)) + 0.1 * np.kron(X, X),
        "controls": [np.kron(X, I2), np.kron(I2, X)],
        "observables": {"z1": np.kron(Z, I2), "z2": np.kron(I2, Z)},
        "dims": (2, 2),
    }
    kw.update(overrides)
    model = model_cls(**kw)
    target = dq.basis(4, 3)
    return StateTransfer(model=model, target=target, horizon=4, dt=0.5, method=Expm())


def single_observable_env(op, dim=None):
    """StateTransfer on a model whose only observable is ``op`` (Hermitian ndarray)."""
    n = op.shape[0]
    drift = np.diag(np.arange(n, dtype=float))
    model = HamiltonianModel(
        drift=drift, controls=[X if n == 2 else np.eye(n)], observables={"o": op}
    )
    return StateTransfer(model=model, target=dq.basis(n, n - 1), horizon=2, dt=0.1, method=Expm())


def ket(*amps):
    v = np.asarray(amps, dtype=complex)[:, None]
    return dq.asqarray(v / np.linalg.norm(v))


def run_observations(obs, key, actions):
    """Jitted episode: stacked observations and the quantum states after every step."""

    @eqx.filter_jit
    def run(k):
        state, ts = obs.reset(k)
        outs, qs = [ts.observation], [state.qstate.to_jax()]
        for a in actions:
            state, ts = obs.step(state, jnp.asarray(a))
            outs.append(ts.observation)
            qs.append(state.qstate.to_jax())
        return jnp.stack(outs), jnp.stack(qs)

    return run(key)


def reference_distribution(op, qstate):
    """Independent numpy outcome distribution: distinct eigenvalues, probabilities, <O>."""
    lam, vec = np.linalg.eigh(np.asarray(op, dtype=complex))
    x = np.asarray(qstate.to_jax())
    pop = (
        np.abs(vec.conj().T @ x[:, 0]) ** 2
        if x.shape[1] == 1
        else np.real(np.einsum("ij,ik,kj->j", vec.conj(), x, vec))
    )
    distinct = np.unique(np.round(lam, 8))
    p = np.array([pop[np.abs(lam - v) < 1e-7].sum() for v in distinct])
    return distinct, p / p.sum()


def moments(lam, p):
    mean = float(np.sum(p * lam))
    sigma2 = float(np.sum(p * (lam - mean) ** 2))
    mu4 = float(np.sum(p * (lam - mean) ** 4))
    return mean, sigma2, mu4


def episode_measure_keys(r=R, t=0, j=0):
    eps = jax.vmap(lambda i: keys.episode_key(RUN_KEY, i))(jnp.arange(r))
    return eps, jax.vmap(lambda e: keys.measurement_key(e, t, j))(eps)


def check_statistics(counts, est, lam, p, n):
    """The spec's G5 criteria: frequencies, mean, variance (exact 4th moment), all at 4 SE."""
    r = est.shape[0]
    mean, sigma2, mu4 = moments(lam, p)
    freq = np.asarray(counts).mean(axis=0) / n
    se_f = np.sqrt(p * (1 - p) / (n * r))
    assert np.all(np.abs(freq - p) <= 4 * se_f + 1e-12), (freq, p)
    v = sigma2 / n
    est = np.asarray(est)
    assert abs(est.mean() - mean) <= 4 * math.sqrt(v / r) + 1e-12
    m4 = mu4 / n**3 + 3 * (n - 1) * sigma2**2 / n**3
    se_v = math.sqrt((m4 - (r - 3) / (r - 1) * v**2) / r)
    assert abs(est.var(ddof=1) - v) <= 4 * se_v + 1e-15


# ------------------------------------------------------------ configuration (G5 setup checks)
class TestConfiguration:
    def test_unknown_mode(self):
        env = qenvs.make("qubit/StateTransfer-v0")
        with pytest.raises(ValueError, match="observation must be one of"):
            ObservationModel(env, "tomography")

    @pytest.mark.parametrize("mode", ["state", "time"])
    def test_shots_rejected_outside_expectation_modes(self, mode):
        env = qenvs.make("qubit/StateTransfer-v0")
        with pytest.raises(ValueError, match="shots is only defined"):
            ObservationModel(env, mode, shots=100)

    @pytest.mark.parametrize("shots", [0, -5, 1.5, True, "100", 2**24 + 1])
    def test_shots_must_be_a_positive_integer(self, shots):
        env = qenvs.make("qubit/StateTransfer-v0")
        with pytest.raises(ValueError, match="shots must"):
            ObservationModel(env, "expectations", shots=shots, observables=("p1",))

    def test_non_hermitian_observable_rejected(self):
        # HamiltonianModel itself rejects it; bypass with a model that returns the raw operator
        class Bad(TwoQubit):
            def observables(self):
                return {"sm": dq.asqarray(np.kron(np.array([[0, 1], [0, 0]], dtype=complex), I2))}

        env = two_qubit_env(Bad)
        with pytest.raises(ValueError, match="not Hermitian"):
            ObservationModel(env, "expectations")

    def test_duplicate_operator_rejected_and_selection_resolves_it(self):
        env = qenvs.make("qubit/StateTransfer-v0")  # energy == p1 for omega0 = 1
        with pytest.raises(ValueError, match="same operator"):
            ObservationModel(env, "expectations")
        obs = ObservationModel(env, "expectations", observables=("energy",))
        assert obs.observable_names == ("energy",)

    def test_duplicate_name_between_observables_and_correlations(self):
        class Clash(TwoQubit):
            def correlation_observables(self):
                return {"z1": dq.asqarray(np.kron(Z, Z), dims=self.dims)}

        with pytest.raises(ValueError, match="unique"):
            ObservationModel(two_qubit_env(Clash), "correlations")

    def test_unknown_observable_name(self):
        with pytest.raises(ValueError, match="unknown observables"):
            ObservationModel(two_qubit_env(), "expectations", observables=("nope",))

    def test_observables_need_a_measuring_mode(self):
        with pytest.raises(ValueError, match="needs observation"):
            ObservationModel(two_qubit_env(), "time", observables=("z1",))

    def test_model_without_observables(self):
        env = two_qubit_env(observables=None)
        with pytest.raises(ValueError, match="no observables"):
            ObservationModel(env, "expectations")
        # correlations alone are enough
        obs = ObservationModel(env, "correlations")
        assert obs.observable_names == ("zz", "xx")

    def test_diagnostics_must_be_bool(self):
        with pytest.raises(ValueError, match="diagnostics"):
            ObservationModel(two_qubit_env(), "time", diagnostics="yes")

    def test_gate_tasks_allow_state_and_time_only(self):
        gate = qenvs.make("transmon/XGate-v0")
        for mode in ("expectations", "correlations"):
            with pytest.raises(ValueError, match="gate tasks support"):
                ObservationModel(gate, mode)
        for mode in ("state", "time"):
            assert ObservationModel(gate, mode).observation == mode

    def test_observation_kind(self):
        assert observation_kind(qenvs.make("transmon/XGate-v0")) == "propagator"
        open_gate = qenvs.make("transmon/XGate-v0", t1_us=20.0)
        assert observation_kind(open_gate) == "superoperator"
        assert ObservationModel(open_gate).observation_kind == "superoperator"
        assert ObservationModel(qenvs.make("battery/Dicke-v0")).observation_kind == "state"
        assert observation_kind(qenvs.make("qubit/StateTransfer-v0")) == "state"

    def test_make_wraps_only_when_asked(self):
        plain = qenvs.make("qubit/StateTransfer-v0")
        assert not isinstance(plain, Wrapper)
        wrapped = qenvs.make("qubit/StateTransfer-v0", observation="time")
        assert isinstance(wrapped, ObservationModel) and wrapped.observation == "time"
        shot = qenvs.make(
            "battery/Dicke-v0", n_cells=2, observation="expectations", shots=10, diagnostics=True
        )
        assert shot.shots == 10 and shot.with_diagnostics
        with pytest.raises(ValueError, match="shots is only defined"):  # state mode, not ignored
            qenvs.make("qubit/StateTransfer-v0", shots=10)
        # task kwargs still reach the factory
        assert qenvs.make("qubit/StateTransfer-v0", horizon=3, observation="time").horizon == 3


# ----------------------------------------------------------- deterministic reference tests
class TestDeterministic:
    @pytest.mark.parametrize(
        "task", ["qubit/StateTransfer-v0", "battery/Dicke-v0", "transmon/XGate-v0"]
    )
    def test_state_mode_is_the_identity(self, task):
        raw = qenvs.make(task)
        wrapped = ObservationModel(raw, "state")
        key = jax.random.PRNGKey(3)
        a = jnp.full((raw.action_dim,), 0.05)
        (s0, t0), (s1, t1) = raw.reset(key), wrapped.reset(key)
        for _ in range(2):
            assert np.array_equal(t0.observation, t1.observation)
            assert sorted(t0.info) == sorted(t1.info)
            for k in t0.info:
                assert np.array_equal(t0.info[k], t1.info[k])
            s0, t0 = raw.step(s0, a)
            s1, t1 = wrapped.step(s1, a)
            assert np.array_equal(t0.reward, t1.reward)
            assert np.array_equal(t0.done, t1.done)
        assert wrapped.observation_dim == raw.observation_dim

    def test_exact_expectations_equal_info_and_time_fraction(self):
        env = two_qubit_env()
        obs = ObservationModel(env, "correlations")
        names = obs.observable_names
        assert names == ("z1", "z2", "zz", "xx")
        key = jax.random.PRNGKey(0)
        state, ts = obs.reset(key)
        a = jnp.array([0.4, -0.3])
        for _ in range(3):
            exact = [float(dq.expect(m.operator, state.qstate).real) for m in obs.measured]
            np.testing.assert_allclose(ts.observation[:-1], exact, atol=1e-12)
            assert float(ts.observation[-1]) == pytest.approx(float(state.t) / env.duration)
            assert ts.observation.shape == (obs.observation_dim,) == (5,)
            state, ts = obs.step(state, a)

    def test_info_values_match_when_shots_none(self):
        # StateTransfer reports energy and p1 in info: the exact observation is the same number
        raw = qenvs.make("battery/Dicke-v0", n_cells=3)
        obs = ObservationModel(raw, "expectations")
        key = jax.random.PRNGKey(1)
        s, ts = obs.reset(key)
        s, ts = obs.step(s, jnp.array([0.3]))
        info = obs.info(s)
        np.testing.assert_allclose(
            ts.observation[:-1], [info[n] for n in obs.observable_names], atol=1e-12
        )

    def test_measurement_does_not_change_the_state(self):
        raw = two_qubit_env()
        obs = ObservationModel(raw, "expectations", shots=50)
        key = jax.random.PRNGKey(5)
        sr, _ = raw.reset(key)
        so, _ = obs.reset(key)
        a = jnp.array([0.2, 0.1])
        for _ in range(3):
            sr, _ = raw.step(sr, a)
            so, _ = obs.step(so, a)
            assert np.array_equal(sr.qstate.to_jax(), so.qstate.to_jax())
            before = so.qstate.to_jax()
            obs.measure(so.qstate, key, 7)
            assert np.array_equal(before, so.qstate.to_jax())

    def test_probabilities_match_an_independent_eigendecomposition(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        herm = a + a.conj().T
        w, v = np.linalg.eigh(herm)
        degenerate = v @ np.diag([w[0], w[0], w[2], w[3]]) @ v.conj().T  # one 2-fold eigenvalue
        for op in (herm, degenerate):
            env = single_observable_env(op)
            obs = ObservationModel(env, "expectations", shots=10)
            psi = rng.normal(size=4) + 1j * rng.normal(size=4)
            kets = dq.asqarray((psi / np.linalg.norm(psi))[:, None])
            b = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
            rho = b @ b.conj().T
            dm = dq.asqarray(rho / np.trace(rho).real)
            for q in (kets, dm):
                lam, p_ref = reference_distribution(op, q)
                p = np.asarray(obs.outcome_probabilities(q)[0])
                lam_obs = np.asarray(obs.measured[0].eigenvalues)
                np.testing.assert_allclose(lam_obs, lam, atol=1e-8)
                np.testing.assert_allclose(p, p_ref, atol=1e-12)
                assert float(np.sum(p * lam_obs)) == pytest.approx(
                    float(dq.expect(obs.measured[0].operator, q).real), abs=1e-12
                )
        assert len(obs.measured[0].eigenvalues) == 3  # the degenerate pair is one outcome

    def test_counts_sum_to_shots_and_are_integers(self):
        p = jnp.array([0.2, 0.0, 0.5, 0.3])
        ks = jax.random.split(jax.random.PRNGKey(1), 500)
        counts = np.asarray(jax.vmap(lambda k: outcome_counts(k, p, 77))(ks))
        assert np.all(counts.sum(axis=1) == 77)
        assert np.all(counts == np.round(counts)) and np.all(counts >= 0)
        assert np.all(counts[:, 1] == 0)

    @pytest.mark.parametrize("n", [1, 100, 12345])
    def test_probability_zero_and_one_are_exact(self, n):
        ks = jax.random.split(jax.random.PRNGKey(2), 300)
        for p in ([1.0, 0.0], [0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]):
            p = jnp.asarray(p)
            counts = np.asarray(jax.vmap(lambda k, p=p: outcome_counts(k, p, n))(ks))
            expected = np.where(np.asarray(p) == 1.0, n, 0)
            assert np.all(counts == expected[None, :])

    def test_eigenstate_has_zero_variance(self):
        # the Dicke initial state is an exact eigenstate of every measured observable
        raw = qenvs.make("battery/Dicke-v0", n_cells=4)
        obs = ObservationModel(raw, "expectations", shots=100)
        state, _ = raw.reset(jax.random.PRNGKey(0))
        exact = np.array([float(dq.expect(m.operator, state.qstate).real) for m in obs.measured])
        eps, _ = episode_measure_keys(2000)
        est = np.asarray(jax.vmap(lambda e: obs.measure(state.qstate, e, 0))(eps))
        np.testing.assert_allclose(est, np.broadcast_to(exact, est.shape), atol=1e-12)
        # counts: all shots in the eigen-outcome
        for m in obs.measured:
            p = np.asarray(m.probabilities(state.qstate))
            assert p.max() == 1.0 and (p > 0).sum() == 1

    def test_qubit_eigenstates_exact_counts(self):
        env = single_observable_env(np.diag([0.0, 1.0]))
        obs = ObservationModel(env, "expectations", shots=1000)
        eps, mk = episode_measure_keys(R)
        for amps, value in (((0, 1), 1.0), ((1, 0), 0.0)):
            q = ket(*amps)
            counts = np.asarray(
                jax.vmap(lambda k, q=q: outcome_counts(k, obs.measured[0].probabilities(q), 1000))(
                    mk
                )
            )
            assert np.all(counts.max(axis=1) == 1000)
            est = np.asarray(jax.vmap(lambda e, q=q: obs.measure(q, e, 0))(eps))
            np.testing.assert_allclose(est, value, atol=1e-12)

    def test_same_key_same_output_and_jit_vmap(self):
        obs = ObservationModel(two_qubit_env(), "correlations", shots=40, diagnostics=True)
        key = jax.random.PRNGKey(9)

        def run(k):
            s, ts = obs.reset(k)
            outs = [ts.observation]
            for a in (0.3, -0.2, 0.5):
                s, ts = obs.step(s, jnp.array([a, 0.1]))
                outs.append(ts.observation)
            return jnp.stack(outs)

        jitted = eqx.filter_jit(run)
        a = jitted(key)
        assert np.array_equal(a, jitted(key))
        batch = jax.jit(jax.vmap(run))(jnp.stack([key, jax.random.PRNGKey(10)]))
        np.testing.assert_allclose(batch[0], a, atol=1e-12)
        assert not np.allclose(batch[0], batch[1])

    def test_measurement_uses_the_measurement_leaf_and_nothing_else(self):
        # observable j at observation t draws only from keys.measurement_key(ep, t, j)
        n = 30
        obs = ObservationModel(two_qubit_env(), "correlations", shots=n)
        key = jax.random.PRNGKey(11)
        acts = [[0.4, 0.2]] * 3
        seen, qs = run_observations(obs, key, acts)

        @jax.jit
        def reference(qstack):
            rows = []
            for t in range(qstack.shape[0]):
                row = []
                for j, m in enumerate(obs.measured):
                    p = m.probabilities(dq.asqarray(qstack[t]))
                    counts = outcome_counts(keys.measurement_key(key, t, j), p, n)
                    row.append(jnp.sum(counts * m.eigenvalues) / n)
                rows.append(jnp.stack(row))
            return jnp.stack(rows)

        np.testing.assert_allclose(seen[:, :-1], reference(qs), atol=1e-12)

    def test_batch_position_independence_of_counts(self):
        p = jnp.array([0.1, 0.25, 0.65])
        _, mk = episode_measure_keys(1024)
        batch = np.asarray(jax.vmap(lambda k: outcome_counts(k, p, 500))(mk))
        for i in (0, 1, 17, 511, 1023):
            assert np.array_equal(batch[i], np.asarray(outcome_counts(mk[i], p, 500)))
        half = np.asarray(jax.vmap(lambda k: outcome_counts(k, p, 500))(mk[::-1]))
        assert np.array_equal(half[::-1], batch)  # position in the batch does not matter

    def test_common_observables_share_noise_between_modes(self):
        env = two_qubit_env()
        e = ObservationModel(env, "expectations", shots=25)
        c = ObservationModel(env, "correlations", shots=25)
        acts = [[0.3, 0.3]] * 3
        for seed in range(2):
            key = jax.random.PRNGKey(seed)
            oe, _ = run_observations(e, key, acts)
            oc, _ = run_observations(c, key, acts)
            assert np.array_equal(oe[:, :2], oc[:, :2])
            assert np.array_equal(oe[:, 2], oc[:, 4])  # time fraction
        # selecting a prefix of the observables does not change the draws of the kept ones
        one = ObservationModel(env, "expectations", shots=25, observables=("z1",))
        o1, _ = run_observations(one, jax.random.PRNGKey(0), acts)
        o2, _ = run_observations(e, jax.random.PRNGKey(0), acts)
        assert np.array_equal(o1[:, 0], o2[:, 0])


# ------------------------------------------------------------------------ information boundary
class TestInformationBoundary:
    EXACT_KEYS = {"fidelity", "energy", "ergotropy", "power", "purity", "p1", "n_photons"}

    @pytest.mark.parametrize("mode", ["expectations", "correlations", "time"])
    @pytest.mark.parametrize("shots", [None, 20])
    def test_partial_modes_hide_exact_values(self, mode, shots):
        if mode == "time" and shots is not None:
            pytest.skip("shots is not defined for time")
        raw = two_qubit_env()
        obs = ObservationModel(raw, mode, shots=shots)
        state, ts = obs.reset(jax.random.PRNGKey(0))
        for _ in range(2):
            assert set(ts.info) <= {"shots_used", "clip_fraction"}
            assert "fidelity" not in ts.info and "diagnostics" not in ts.info
            if shots is not None:
                assert int(ts.info["shots_used"]) == shots * len(obs.observable_names)
                assert obs.shots_per_step == shots * len(obs.observable_names)
            else:
                assert int(ts.info["shots_used"]) == 0
            state, ts = obs.step(state, jnp.array([0.3, 0.1]))

    def test_diagnostics_hold_the_exact_values(self):
        raw = two_qubit_env()
        key = jax.random.PRNGKey(4)
        for mode in ("state", "expectations", "time"):
            obs = ObservationModel(raw, mode, shots=None, diagnostics=True)
            s, ts = obs.reset(key)
            s, ts = obs.step(s, jnp.array([0.3, 0.1]))
            exact = obs.info(s)
            assert set(ts.info["diagnostics"]) == set(exact)
            for k, v in exact.items():
                assert np.array_equal(ts.info["diagnostics"][k], v)
            if mode != "state":
                assert set(ts.info) == {"shots_used", "diagnostics"}

    def test_functional_info_and_diagnostics_are_always_exact(self):
        raw = qenvs.make("battery/Dicke-v0", n_cells=3)
        key = jax.random.PRNGKey(2)
        s_raw, _ = raw.reset(key)
        s_raw, _ = raw.step(s_raw, jnp.array([0.3]))
        for mode, shots in (("state", None), ("expectations", 10), ("time", None)):
            obs = ObservationModel(raw, mode, shots=shots)
            s, _ = obs.reset(key)
            s, _ = obs.step(s, jnp.array([0.3]))
            for fn in (obs.info, obs.diagnostics):
                got = fn(s)
                assert set(got) == set(raw.info(s_raw))
                for k, v in raw.info(s_raw).items():
                    assert np.array_equal(got[k], v)

    def test_state_mode_info_is_the_released_info(self):
        for task in ("qubit/StateTransfer-v0", "battery/Dicke-v0", "transmon/XGate-v0"):
            raw = qenvs.make(task)
            obs = ObservationModel(raw, "state")
            assert sorted(obs.reset(jax.random.PRNGKey(0))[1].info) == sorted(
                raw.reset(jax.random.PRNGKey(0))[1].info
            )

    def test_noisy_observation_differs_from_exact_and_reward_is_exact(self):
        raw = qenvs.make("battery/Dicke-v0", n_cells=3)
        key = jax.random.PRNGKey(0)
        a = jnp.array([0.3])
        exact = ObservationModel(raw, "expectations")
        noisy = ObservationModel(raw, "expectations", shots=30)
        s_exact, ts_exact = exact.reset(key)
        s_noisy, ts_noisy = noisy.reset(key)
        for _ in range(3):
            s_exact, ts_exact = exact.step(s_exact, a)
            s_noisy, ts_noisy = noisy.step(s_noisy, a)
            assert not np.array_equal(ts_exact.observation[:-1], ts_noisy.observation[:-1])
            # the reward is not measured (spec 4.4, item 5)
            assert np.array_equal(ts_exact.reward, ts_noisy.reward)

    def test_extra_observation_entries_stay_at_the_end(self):
        class Oracle(Wrapper):
            @property
            def observation_dim(self):
                return self.env.observation_dim + 2

            def _transform_timestep(self, state, ts):
                extra = jnp.array([7.0, -3.0], dtype=ts.observation.dtype)
                return eqx.tree_at(
                    lambda t: t.observation, ts, jnp.concatenate([ts.observation, extra])
                )

        raw = two_qubit_env()
        for mode, k in (("time", 0), ("expectations", 2), ("correlations", 4)):
            obs = ObservationModel(Oracle(raw), mode)
            s, ts = obs.reset(jax.random.PRNGKey(0))
            assert ts.observation.shape == (obs.observation_dim,) == (k + 1 + 2,)
            np.testing.assert_array_equal(ts.observation[-2:], [7.0, -3.0])
            assert float(ts.observation[k]) == 0.0  # time fraction at reset
            s, ts = obs.step(s, jnp.array([0.1, 0.1]))
            assert float(ts.observation[k]) == pytest.approx(1 / raw.horizon)


# ------------------------------------------------------------------- statistics (G5)
def sample_case(op, q, n, ref_op=None):
    """Run R repetitions of the measurement of ``op`` on ``q`` with n shots."""
    env = single_observable_env(op)
    obs = ObservationModel(env, "expectations", shots=n)
    eps, mk = episode_measure_keys(R, t=0, j=0)
    p = obs.outcome_probabilities(q)[0]
    counts = np.asarray(jax.vmap(lambda k: outcome_counts(k, p, n))(mk))
    est = np.asarray(jax.vmap(lambda e: obs.measure(q, e, 0)[0])(eps))
    est1 = np.asarray(jax.vmap(lambda e: obs.measure(q, e, 1)[0])(eps))
    lam, p_ref = reference_distribution(op if ref_op is None else ref_op, q)
    # deterministic link: the estimate is the count-weighted mean of the outcome eigenvalues
    np.testing.assert_allclose(
        est, counts @ np.asarray(obs.measured[0].eigenvalues) / n, atol=1e-12
    )
    np.testing.assert_allclose(np.asarray(p), p_ref, atol=1e-12)
    return counts, est, est1, lam, p_ref


def dicke_state():
    raw = qenvs.make("battery/Dicke-v0", n_cells=4)
    final, _ = raw.rollout(jnp.full((raw.horizon, 1), 0.3), jax.random.PRNGKey(0))
    return raw, final.qstate


def cases():
    p3 = ket(np.sqrt(0.7), np.sqrt(0.3))  # p(1) = 0.3
    rare = ket(np.sqrt(1 - 1e-3), np.sqrt(1e-3))  # p(1) = 1e-3
    proj = np.diag([0.0, 1.0])
    rng = np.random.default_rng(7)
    q4, _ = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
    deg = q4 @ np.diag([0.0, 0.0, 1.0, 2.5]) @ q4.conj().T  # degenerate eigenvalue
    psi = rng.normal(size=4) + 1j * rng.normal(size=4)
    return {
        "projector_p03": (proj, p3),
        "rare_p1e-3": (proj, rare),
        "degenerate": (deg, ket(*psi)),
    }


class TestStatistics:
    @pytest.mark.parametrize("n", [100, 1000])
    @pytest.mark.parametrize("case", ["projector_p03", "rare_p1e-3", "degenerate"])
    def test_measurement_statistics(self, case, n):
        op, q = cases()[case]
        counts, est, est1, lam, p = sample_case(op, q, n)
        check_statistics(counts, est, lam, p, n)
        # steps are independent: same episode, observation 0 against observation 1
        r = np.corrcoef(est, est1)[0, 1]
        assert abs(r) <= 4 / math.sqrt(R)

    @pytest.mark.parametrize("n", [100, 1000])
    def test_dicke_n4_energy_statistics(self, n):
        raw, q = dicke_state()
        env = raw
        obs = ObservationModel(env, "expectations", shots=n, observables=("energy",))
        p = obs.outcome_probabilities(q)[0]
        eps, mk = episode_measure_keys(R)
        counts = np.asarray(jax.vmap(lambda k: outcome_counts(k, p, n))(mk))
        est = np.asarray(jax.vmap(lambda e: obs.measure(q, e, 0)[0])(eps))
        est1 = np.asarray(jax.vmap(lambda e: obs.measure(q, e, 1)[0])(eps))
        op = np.asarray(obs.measured[0].operator.to_jax())
        lam, p_ref = reference_distribution(op, q)
        np.testing.assert_allclose(np.asarray(p), p_ref, atol=1e-10)
        assert len(lam) == 5  # energy eigenvalues 0 ... 4, each degenerate over the cavity
        check_statistics(counts, est, lam, p_ref, n)
        assert abs(np.corrcoef(est, est1)[0, 1]) <= 4 / math.sqrt(R)


# ------------------------------------------------------------- float32 (subprocess)
def _float32_checks():
    """Run under float32 (x64 off): determinism and the main statistics; used by the test below."""
    assert not jax.config.jax_enable_x64
    op = np.diag([0.0, 1.0])
    for amps, n in (((np.sqrt(0.7), np.sqrt(0.3)), 100), ((np.sqrt(0.7), np.sqrt(0.3)), 1000)):
        q = ket(*amps)
        env = single_observable_env(op)
        obs = ObservationModel(env, "expectations", shots=n)
        eps = jax.vmap(lambda i: keys.episode_key(RUN_KEY, i))(jnp.arange(R))
        p = obs.outcome_probabilities(q)[0]
        draw = jax.vmap(lambda e, obs=obs, q=q: obs.measure(q, e, 0)[0])
        est = np.asarray(draw(eps), dtype=float)
        assert np.array_equal(est, np.asarray(draw(eps), dtype=float))
        lam = np.array([0.0, 1.0])
        p_ref = np.array([0.7, 0.3])
        np.testing.assert_allclose(np.asarray(p, dtype=float), p_ref, atol=1e-6)
        mean, sigma2, mu4 = moments(lam, p_ref)
        v = sigma2 / n
        assert abs(est.mean() - mean) <= 4 * math.sqrt(v / R) + 1e-6
        m4 = mu4 / n**3 + 3 * (n - 1) * sigma2**2 / n**3
        se_v = math.sqrt((m4 - (R - 3) / (R - 1) * v**2) / R)
        assert abs(est.var(ddof=1) - v) <= 4 * se_v
    # exact eigenstates stay exact in float32 too
    q = ket(0, 1)
    obs = ObservationModel(single_observable_env(op), "expectations", shots=1000)
    est = np.asarray(jax.vmap(lambda e: obs.measure(q, e, 0)[0])(eps), dtype=float)
    assert np.all(np.abs(est - 1.0) <= 1e-6)
    # the whole evaluation path is reproducible in float32 as well
    from qenvs import robust

    wrapped = ObservationModel(single_observable_env(np.diag([0.0, 1.0])), "expectations", shots=20)
    policy = lambda obs, key: jnp.full((1,), 0.4 + 0.3 * obs[0])  # noqa: E731
    r1 = robust.evaluate(wrapped, policy, RUN_KEY, n=16, metric="o")
    r2 = robust.evaluate(wrapped, policy, RUN_KEY, n=16, metric="o")
    assert np.array_equal(r1.values, r2.values) and r1.values.std() > 0
    print("float32 ok")


def test_float32_statistics_and_determinism():
    here = Path(__file__).resolve().parent
    code = (
        f"import sys; sys.path.insert(0, {str(here)!r}); "
        "import test_observation as t; t._float32_checks()"
    )
    env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    env.pop("JAX_ENABLE_X64", None)
    res = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr[-3000:]
    assert "float32 ok" in res.stdout
