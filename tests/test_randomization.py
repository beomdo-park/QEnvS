"""Parameter randomisation: distributions, model parameter contract, ParameterRandomization.

Gates (docs/superpowers/specs/2026-10-15-realistic-rl-conditions-proposal.md, section 6):
G1 (defaults reproduce today's behaviour), G2 (reproducibility and leaf separation of the
parameter leaves), G3 (physical constraints and distribution rules), G3b (model contract).
Statistical tests use fixed keys (deterministic verdicts), a deterministic reference test
first, and the 4 SE criterion of the spec.
"""

from __future__ import annotations

import math
import pathlib
import sys

import dynamiqs as dq
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import qenvs
from qenvs import keys
from qenvs.distributions import Normal, TruncatedNormal, Uniform, check_strength
from qenvs.envs import StateTransfer
from qenvs.models import (
    DickeBattery,
    DickeBatteryLocal,
    LambdaSystem,
    QubitBattery,
    Transmon,
    TwoLevel,
    TwoTransmon,
)
from qenvs.models.base import Model, ParameterSpec
from qenvs.models.custom import HamiltonianModel
from qenvs.wrappers.control import ControlErrors
from qenvs.wrappers.randomization import ParameterRandomization

GOLDEN = pathlib.Path(__file__).with_name("golden")
sys.path.insert(0, str(GOLDEN))
import make_task_snapshot  # noqa: E402

RUN_KEY = jax.random.PRNGKey(20261015)
TASKS = sorted(qenvs.registry)
INF = math.inf

SX = np.array([[0, 1], [1, 0]], dtype=complex)
SZ = np.diag([1.0, -1.0]).astype(complex)


# ---------------------------------------------------------------------- helpers
def reference_moments(dist, strength=1.0):
    """Mean, variance and 4th central moment of the offset by Gauss-Legendre quadrature.

    Independent of the closed forms in ``qenvs.distributions``: it integrates the density.
    """
    if isinstance(dist, Normal):
        s = dist.std * strength
        lo, hi = -12 * s, 12 * s
        dens = lambda x: np.exp(-0.5 * (x / s) ** 2)  # noqa: E731
    elif isinstance(dist, Uniform):
        h = dist.half_width * strength
        lo, hi = -h, h
        dens = lambda x: np.ones_like(x)  # noqa: E731
    else:
        s = dist.std * strength
        lo, hi = dist.low, dist.high
        dens = lambda x: np.exp(-0.5 * (x / s) ** 2)  # noqa: E731
    x, w = np.polynomial.legendre.leggauss(600)
    x = 0.5 * (hi - lo) * x + 0.5 * (hi + lo)
    w = w * dens(x)
    w = w / w.sum()
    mean = float((w * x).sum())
    var = float((w * (x - mean) ** 2).sum())
    mu4 = float((w * (x - mean) ** 4).sum())
    return mean, var, mu4


def assert_moments_within_4se(x, mean, var, mu4):
    """Sample mean and variance within 4 standard errors (exact m4 for the variance)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    se_mean = math.sqrt(var / n)
    se_var = math.sqrt((mu4 - (n - 3) / (n - 1) * var**2) / n)
    assert abs(x.mean() - mean) <= 4 * se_mean, (x.mean(), mean, se_mean)
    assert abs(x.var(ddof=1) - var) <= 4 * se_var, (x.var(ddof=1), var, se_var)


def sample_offsets(dist, n, strength=1.0, dtype=None, key=RUN_KEY):
    ks = jax.vmap(lambda i: keys.episode_key(key, i))(jnp.arange(n))
    return np.asarray(jax.vmap(lambda k: dist.sample(k, strength, (), dtype))(ks))


def episode_data(env, n, run_key=RUN_KEY):
    """Episode data of ``n`` episodes ``fold_in(run_key, i)`` (batched reset)."""
    ks = jax.vmap(lambda i: keys.episode_key(run_key, i))(jnp.arange(n))
    state, _ = jax.vmap(env.reset)(ks)
    return param_data(state)


def corr(a, b):
    return float(np.corrcoef(np.asarray(a), np.asarray(b))[0, 1])


def x32():
    """Context with jax x64 off (the library default; conftest turns it on)."""
    if hasattr(jax, "enable_x64"):
        return jax.enable_x64(False)
    from jax.experimental import disable_x64

    return disable_x64()


def param_data(state):
    """Merged episode data of a (possibly nested) wrapped state."""
    out = {}
    while hasattr(state, "data"):
        out.update(state.data)
        state = state.inner
    return out


def gate_env(**kw):
    return qenvs.make("transmon/XGate-v0", **kw)


# ====================================================================== distributions
class TestDistributionReference:
    """Deterministic reference tests that precede every statistical test (spec section 6)."""

    @pytest.mark.parametrize(
        "dist",
        [
            Normal(0.003),
            Uniform(0.02),
            TruncatedNormal(0.01, -0.02, 0.02),
            TruncatedNormal(0.01, -0.005, 0.03),  # asymmetric
            TruncatedNormal(0.01, 0.0, 0.03),  # one-sided
            TruncatedNormal(0.02, -0.001, 0.002),  # narrow (quadrature branch)
            TruncatedNormal(0.002, -0.05, 0.05),  # wide in units of sigma
        ],
    )
    @pytest.mark.parametrize("strength", [0.5, 1.0, 3.0])
    def test_analytic_moments_equal_quadrature(self, dist, strength):
        mean, var, _ = reference_moments(dist, strength)
        assert dist.mean(strength) == pytest.approx(mean, abs=1e-12, rel=1e-9)
        assert dist.stddev(strength) == pytest.approx(math.sqrt(var), rel=1e-8)

    def test_known_values(self):
        assert Normal(0.5).stddev(2.0) == 1.0
        assert Uniform(math.sqrt(3.0)).stddev() == pytest.approx(1.0)
        assert Normal(1.0).mean() == 0.0 == Uniform(1.0).mean()
        # symmetric: zero mean; var of N(0,1) on [-1, 1] is 1 - 2 phi(1) / erf(1/sqrt 2)
        tn = TruncatedNormal(1.0, -1.0, 1.0)
        phi1 = math.exp(-0.5) / math.sqrt(2 * math.pi)
        assert tn.mean() == pytest.approx(0.0, abs=1e-15)
        assert tn.stddev() == pytest.approx(math.sqrt(1 - 2 * phi1 / math.erf(math.sqrt(0.5))))
        # one-sided: mean of the half normal is sqrt(2/pi)
        assert TruncatedNormal(1.0, 0.0, 40.0).mean() == pytest.approx(math.sqrt(2 / math.pi))

    def test_truncated_normal_strength_scales_std_not_bounds(self):
        tn = TruncatedNormal(0.01, -0.02, 0.03)
        assert tn.offset_range(0.1) == tn.offset_range(10.0) == (-0.02, 0.03)
        assert Uniform(0.02).offset_range(2.0) == (-0.04, 0.04)
        assert Normal(0.02).stddev(2.0) == pytest.approx(0.04)
        # bounds fixed: a huge strength approaches the uniform on [low, high]
        wide = TruncatedNormal(0.01, -0.02, 0.03)
        assert wide.mean(1e4) == pytest.approx(0.005, rel=1e-3)
        assert wide.stddev(1e4) == pytest.approx(0.05 / math.sqrt(12.0), rel=1e-3)

    def test_unit_field_and_frozen(self):
        n = Normal(0.002, unit="GHz")
        assert n.unit == "GHz"
        assert Uniform(1.0).unit == ""
        with pytest.raises(Exception, match=r"cannot assign|frozen"):
            n.std = 1.0  # type: ignore[misc]
        assert hash(n) == hash(Normal(0.002, unit="GHz"))

    def test_sample_equals_documented_transform_of_the_key(self):
        """Deterministic: the draw is a fixed function of the key, std and strength."""
        k = jax.random.PRNGKey(3)
        got = Normal(0.25).sample(k, 2.0, (4,), jnp.float64)
        ref = 0.5 * jax.random.normal(k, (4,), jnp.float64)
        assert np.array_equal(np.asarray(got), np.asarray(ref))
        got = Uniform(0.25).sample(k, 2.0, (4,), jnp.float64)
        ref = 0.5 * jax.random.uniform(k, (4,), jnp.float64, minval=-1.0, maxval=1.0)
        assert np.array_equal(np.asarray(got), np.asarray(ref))

    def test_sample_is_jittable_and_vmappable_and_batch_independent(self):
        dist = TruncatedNormal(0.01, -0.02, 0.03)
        ks = jax.random.split(RUN_KEY, 64)
        batch = jax.jit(jax.vmap(lambda k: dist.sample(k, 1.0)))(ks)
        single = np.array([float(dist.sample(k, 1.0)) for k in ks[:5]])
        assert np.array_equal(np.asarray(batch[:5]), single)


class TestDistributionConstruction:
    @pytest.mark.parametrize("bad", [-1.0, math.inf, math.nan])
    def test_width_must_be_finite_nonnegative(self, bad):
        with pytest.raises(ValueError, match="std"):
            Normal(bad)
        with pytest.raises(ValueError, match="half_width"):
            Uniform(bad)
        with pytest.raises(ValueError, match="std"):
            TruncatedNormal(bad, -1.0, 1.0)

    @pytest.mark.parametrize(
        ("low", "high"), [(0.1, 0.2), (-0.2, -0.1), (0.0, 0.0), (0.1, -0.1), (-math.inf, 1.0)]
    )
    def test_truncated_normal_bounds(self, low, high):
        with pytest.raises(ValueError, match=r"low|finite"):
            TruncatedNormal(0.1, low, high)

    @pytest.mark.parametrize("bad", [-0.1, math.inf, -math.inf, math.nan, "a"])
    def test_strength_must_be_finite_nonnegative(self, bad):
        with pytest.raises(ValueError, match="strength"):
            check_strength(bad)
        with pytest.raises(ValueError, match="strength"):
            ParameterRandomization(
                gate_env(),
                {"detuning_ghz": Normal(0.001)},
                strength=bad,  # type: ignore[arg-type]
            )
        assert check_strength(0) == 0.0
        assert check_strength(2) == 2.0


# ====================================================================== G3 rules at the wrapper
class TestG3ConstructionErrors:
    """Errors at construction: support, unit, names, strength."""

    def test_normal_on_bounded_support(self):
        with pytest.raises(ValueError, match=r"Normal needs.*whole real line.*TruncatedNormal"):
            ParameterRandomization(gate_env(), {"anharmonicity_ghz": Normal(0.005)})
        # a coupling in [0, inf) is bounded below
        env = qenvs.make("transmon/CRGate-v0", levels=2)
        with pytest.raises(ValueError, match="Normal"):
            ParameterRandomization(env, {"coupling_ghz": Normal(1e-4)})
        # even with zero width: the check is on the declaration, not on the draw
        with pytest.raises(ValueError, match="Normal"):
            ParameterRandomization(gate_env(), {"anharmonicity_ghz": Normal(0.0)})
        # the real line is fine
        ParameterRandomization(gate_env(), {"detuning_ghz": Normal(0.002)})

    def test_uniform_beyond_support(self):
        # anharmonicity -0.34 with support (-inf, 0]: half-width 0.34 reaches 0 exactly (ok)
        ParameterRandomization(gate_env(), {"anharmonicity_ghz": Uniform(0.34)})
        with pytest.raises(ValueError, match="support"):
            ParameterRandomization(gate_env(), {"anharmonicity_ghz": Uniform(0.35)})
        # strength widens the effective range: 0.2 * 2 = 0.4 > 0.34
        with pytest.raises(ValueError, match="strength 2"):
            ParameterRandomization(gate_env(), {"anharmonicity_ghz": Uniform(0.2)}, strength=2.0)
        env = qenvs.make("transmon/CRGate-v0", levels=2)  # coupling 0.003 in [0, inf)
        with pytest.raises(ValueError, match="support"):
            ParameterRandomization(env, {"coupling_ghz": Uniform(0.004)})

    def test_truncated_normal_range_beyond_support(self):
        ParameterRandomization(
            gate_env(), {"anharmonicity_ghz": TruncatedNormal(0.005, -0.02, 0.34)}
        )
        with pytest.raises(ValueError, match="support"):
            ParameterRandomization(
                gate_env(), {"anharmonicity_ghz": TruncatedNormal(0.005, -0.02, 0.35)}
            )
        # strength does not widen low and high, so a large strength is accepted
        ParameterRandomization(
            gate_env(), {"anharmonicity_ghz": TruncatedNormal(0.005, -0.02, 0.02)}, strength=50.0
        )

    def test_names_units_and_types(self):
        env = gate_env()
        with pytest.raises(ValueError, match=r"levels.*structural"):
            ParameterRandomization(env, {"levels": Uniform(1.0)})
        with pytest.raises(ValueError, match=r"t1_us.*dissipation"):
            ParameterRandomization(env, {"t1_us": Uniform(1.0)})
        with pytest.raises(ValueError, match="not a randomisable parameter"):
            ParameterRandomization(env, {"typo": Uniform(1.0)})
        with pytest.raises(ValueError, match="unit"):
            ParameterRandomization(env, {"detuning_ghz": Normal(0.002, unit="rad/ns")})
        ParameterRandomization(env, {"detuning_ghz": Normal(0.002, unit="GHz")})
        ParameterRandomization(env, {"detuning_ghz": Normal(0.002)})  # unit not stated: accepted
        with pytest.raises(TypeError, match="Distribution"):
            ParameterRandomization(env, {"detuning_ghz": 0.002})  # type: ignore[dict-item]
        with pytest.raises(TypeError, match="Model"):
            ParameterRandomization(object(), {})  # type: ignore[arg-type]

    def test_crc32_collision_is_an_error(self):
        # two names with the same CRC32 tag (found by a birthday search over random names)
        pair = ("lyrxjlqk", "erbiqqka")
        assert keys.name_tag(pair[0]) == keys.name_tag(pair[1])
        with pytest.raises(ValueError, match="share a key tag"):
            keys.check_name_tags(pair)
        sx = np.array([[0, 1], [1, 0]])
        spec = (0.1, sx, "u", (-INF, INF), 0.1)
        model = HamiltonianModel.from_operators(
            0.1 * SZ.real, [sx / 2], parameters={pair[0]: spec, pair[1]: spec}
        )
        env = StateTransfer(model=model, target=dq.basis(2, 1))
        with pytest.raises(ValueError, match="share a key tag"):
            ParameterRandomization(env, {pair[0]: Normal(0.1), pair[1]: Normal(0.1)})


# ====================================================================== G3 samples
class TestG3Samples:
    N = 4096

    def _values(self, name, dist, strength=1.0, env=None):
        env = gate_env() if env is None else env
        renv = ParameterRandomization(env, {name: dist}, strength=strength)
        data = episode_data(renv, self.N)
        return np.asarray(data[f"value/{name}"]), np.asarray(data[f"offset/{name}"])

    @pytest.mark.parametrize("strength", [1.0, 4.0])
    def test_all_samples_in_range_and_support(self, strength):
        nominal = -0.34
        for dist, lo, hi in [
            (Uniform(0.05), -0.05 * strength, 0.05 * strength),
            (TruncatedNormal(0.01, -0.02, 0.03), -0.02, 0.03),
            (TruncatedNormal(0.01, 0.0, 0.03), 0.0, 0.03),
        ]:
            values, offsets = self._values("anharmonicity_ghz", dist, strength)
            assert values.shape == (self.N,)
            lo_v, hi_v = nominal + lo, nominal + hi  # float64, same arithmetic as the draw
            assert np.all(values >= lo_v) and np.all(values <= hi_v)
            assert np.all(values <= 0.0)  # support (-inf, 0]
            assert np.all(offsets >= lo) and np.all(offsets <= hi)
            assert len(np.unique(values)) > self.N // 2  # not degenerate

    def test_moments_in_float32(self):
        """The statistical checks hold in float32 too (x64 off), 4096 draws."""
        with x32():
            for dist in (Normal(0.003), TruncatedNormal(0.01, -0.005, 0.03)):
                offsets = sample_offsets(dist, self.N, dtype=jnp.float32)
                assert offsets.dtype == np.float32
                assert_moments_within_4se(offsets, *reference_moments(dist))

    def test_samples_inside_range_in_float32(self):
        with x32():
            env = gate_env()
            dist = TruncatedNormal(0.01, -0.02, 0.03)
            renv = ParameterRandomization(env, {"anharmonicity_ghz": dist})
            data = episode_data(renv, 1024)
            values = data["value/anharmonicity_ghz"]
            assert values.dtype == jnp.float32
            lo = jnp.asarray(-0.34, jnp.float32) + jnp.asarray(-0.02, jnp.float32)
            hi = jnp.asarray(-0.34, jnp.float32) + jnp.asarray(0.03, jnp.float32)
            assert bool(jnp.all(values >= lo)) and bool(jnp.all(values <= hi))

    def test_zero_effective_width_gives_exactly_zero_offset(self):
        for dist, strength in [
            (Normal(0.002), 0.0),
            (Uniform(0.002), 0.0),
            (TruncatedNormal(0.002, -0.01, 0.01), 0.0),
            (Normal(0.0), 1.0),
            (Uniform(0.0), 1.0),
            (TruncatedNormal(0.0, -0.01, 0.01), 1.0),
        ]:
            name = "detuning_ghz" if isinstance(dist, Normal) else "anharmonicity_ghz"
            values, offsets = self._values(name, dist, strength)
            assert np.all(offsets == 0.0) and not np.any(np.signbit(offsets))
            nominal = {"detuning_ghz": 0.0, "anharmonicity_ghz": -0.34}[name]
            assert np.all(values == nominal)

    def test_asymmetric_truncated_normal_moments(self):
        dist = TruncatedNormal(0.01, -0.005, 0.03)
        mean, var, mu4 = reference_moments(dist)
        assert mean > 1e-3  # clearly not zero-mean
        _, offsets = self._values("anharmonicity_ghz", dist)
        assert_moments_within_4se(offsets, mean, var, mu4)
        assert_moments_within_4se(offsets, dist.mean(), dist.stddev() ** 2, mu4)

    def test_normal_and_uniform_moments(self):
        for name, dist in [
            ("detuning_ghz", Normal(0.003)),
            ("anharmonicity_ghz", Uniform(0.05)),
        ]:
            _, offsets = self._values(name, dist)
            mean, var, mu4 = reference_moments(dist)
            assert_moments_within_4se(offsets, mean, var, mu4)

    @pytest.mark.parametrize("task", TASKS)
    def test_every_sampled_hamiltonian_is_hermitian(self, task):
        env = qenvs.make(task)
        env_model = env.model
        specs = env_model.parameter_specs()
        dists = {n: _wide_dist(s, float(env_model.parameter_values()[n])) for n, s in specs.items()}
        renv = ParameterRandomization(env, dists, strength=2.0)
        ks = jax.vmap(lambda i: keys.episode_key(RUN_KEY, i))(jnp.arange(8))
        data = jax.vmap(renv._episode_data)(ks)
        for i in range(8):
            one = jax.tree_util.tree_map(lambda x, i=i: x[i], data)
            h = np.asarray(renv._model(one).drift().to_jax())
            assert np.max(np.abs(h - h.conj().T)) <= 1e-12 * max(1.0, np.max(np.abs(h)))
            # the sampled values really are inside the declared supports
            for n, sp in specs.items():
                assert sp.contains(float(one[f"value/{n}"])), n


def _wide_dist(spec, nominal):
    """A distribution of the reference width that is valid for ``spec``."""
    lo, hi = spec.support
    if math.isinf(lo) and math.isinf(hi):
        return Normal(spec.w_ref)
    return TruncatedNormal(
        spec.w_ref, max(-2 * spec.w_ref, lo - nominal), min(2 * spec.w_ref, hi - nominal)
    )


# ====================================================================== G3b model contract
BUILTIN_MODELS = {
    "TwoLevel": TwoLevel(),
    "QubitBattery": QubitBattery(gamma1=0.1, n_th=0.2),
    "LambdaSystem": LambdaSystem(gamma=0.1),
    "Transmon": Transmon(levels=3, t1_us=50.0, t2_us=70.0),
    "TwoTransmon": TwoTransmon(levels=2),
    "DickeBattery": DickeBattery(n_cells=2, kappa=0.1),
    "DickeBatteryLocal": DickeBatteryLocal(n_cells=2),
}


def custom_model(support=(-INF, INF), nominal=0.3, w_ref=0.05):
    drift = 0.1 * SZ + 0.0 * SX
    return HamiltonianModel.from_operators(
        drift,
        [SX / 2],
        parameters={"J": (nominal, SX, "a.u.", support, w_ref), "d": (0.2, SZ, "a.u.", (0, 1), 0)},
    )


class TestG3bModelContract:
    @pytest.mark.parametrize("name", sorted(BUILTIN_MODELS))
    def test_with_parameters_under_jit_and_vmap(self, name):
        model = BUILTIN_MODELS[name]
        specs = model.parameter_specs()
        assert specs, f"{name} declares no parameter"
        nominal = {k: float(v) for k, v in model.parameter_values().items()}
        probe = {k: nominal[k] + 0.5 * specs[k].w_ref for k in specs}
        assert all(specs[k].contains(v) for k, v in probe.items())
        eager = np.asarray(model.with_parameters(**probe).drift().to_jax())
        jitted = np.asarray(jax.jit(lambda kw: model.with_parameters(**kw).drift().to_jax())(probe))
        assert np.allclose(eager, jitted, atol=1e-14, rtol=0)
        # vmap over a batch of values of every parameter at once
        batch = {k: v + jnp.linspace(-1.0, 1.0, 5) * 0.1 * specs[k].w_ref for k, v in probe.items()}
        out = jax.vmap(lambda kw: model.with_parameters(**kw).drift().to_jax())(batch)
        assert out.shape[0] == 5
        for i in range(5):
            ref = model.with_parameters(**{k: float(v[i]) for k, v in batch.items()}).drift()
            assert np.allclose(np.asarray(out[i]), np.asarray(ref.to_jax()), atol=1e-14, rtol=0)
        # the drift really depends on every declared parameter
        for k in specs:
            moved = model.with_parameters(**{k: nominal[k] + 3 * specs[k].w_ref})
            if name == "Transmon" and k == "anharmonicity_ghz":
                continue
            if name == "TwoTransmon" and k.startswith("anharmonicity"):
                continue  # levels=2: n(n-1) = 0, no dependence (documented)
            assert not np.array_equal(
                np.asarray(moved.drift().to_jax()), np.asarray(model.drift().to_jax())
            ), k

    def test_with_parameters_is_pure_and_empty_is_identity(self):
        model = Transmon()
        new = model.with_parameters(detuning_ghz=0.01)
        assert float(model.detuning_ghz) == 0.0 and float(new.detuning_ghz) == 0.01
        assert model.with_parameters() is model
        assert type(new) is Transmon and new.levels == model.levels
        # equal nominal fields (omega0 == omega_c == 1.0): only the named one changes
        d = DickeBattery().with_parameters(omega_c=1.1)
        assert float(d.omega_c) == 1.1 and float(d.omega0) == 1.0

    @pytest.mark.parametrize("name", sorted(BUILTIN_MODELS))
    def test_check_parameters_rejects_outside_support_and_junk(self, name):
        model = BUILTIN_MODELS[name]
        for k, spec in model.parameter_specs().items():
            nominal = float(model.parameter_values()[k])
            model.check_parameters(**{k: nominal})
            lo, hi = spec.support
            if math.isfinite(hi):
                with pytest.raises(ValueError, match="outside its support"):
                    model.check_parameters(**{k: hi + 1e-3})
            if math.isfinite(lo):
                with pytest.raises(ValueError, match="outside its support"):
                    model.check_parameters(**{k: lo - 1e-3})
            for bad in (math.nan, math.inf, 1.0 + 1.0j):
                with pytest.raises(ValueError, match="finite real"):
                    model.check_parameters(**{k: bad})
        with pytest.raises(ValueError, match="not a randomisable"):
            model.check_parameters(not_a_parameter=1.0)

    def test_check_parameters_needs_concrete_values(self):
        model = Transmon()
        with pytest.raises(ValueError, match="concrete"):
            jax.jit(lambda x: model.check_parameters(detuning_ghz=x))(0.1)

    def test_check_parameters_rejects_non_hermitian_result(self):
        class Broken(Model):
            """Drift ``H_0 + p A`` with a non-Hermitian ``A`` (bypasses all validation)."""

            p: float = 0.0

            @property
            def dim(self):
                return 2

            def drift(self):
                import dynamiqs as dq

                return dq.asqarray(jnp.asarray([[0.0, 1.0], [0.0, 0.0]]) * self.p + 0j)

            def controls(self):
                import dynamiqs as dq

                return [dq.asqarray(jnp.asarray(SX))]

            def initial_state(self):
                import dynamiqs as dq

                return dq.basis(2, 0)

            def parameter_specs(self):
                return {"p": ParameterSpec("a.u.", (-INF, INF), 1.0)}

        m = Broken()
        m.check_parameters(p=0.0)  # H = 0 is Hermitian
        with pytest.raises(ValueError, match="not Hermitian"):
            m.check_parameters(p=0.5)
        # the same check catches a complex parameter value on a Hermitian model
        with pytest.raises(ValueError, match="finite real"):
            custom_model().check_parameters(J=0.2 + 0.1j)

    def test_declaring_structure_or_dissipation_is_an_error_with_reason(self):
        cases = {
            Transmon(): ("levels", "initial", "t1_us", "t2_us"),
            TwoTransmon(): ("levels", "t1_c_us", "t2_t_us", "detuning_ghz"),
            TwoLevel(): ("gamma1", "gamma_phi", "omega0"),
            QubitBattery(): ("gamma1", "n_th", "gamma_phi", "omega0"),
            LambdaSystem(): ("gamma",),
            DickeBattery(): ("n_cells", "n_max", "rwa", "kappa", "gamma", "gamma_phi", "omega0"),
            DickeBatteryLocal(n_cells=2): ("gamma_local", "pumping_local", "n_cells", "omega0"),
        }
        for model, names in cases.items():
            for n in names:
                assert n not in model.parameter_specs()
                with pytest.raises(ValueError, match=f"{n!r} is not a randomisable parameter"):
                    model.with_parameters(**{n: 0.1})
        with pytest.raises(ValueError, match=r"rate|dissipation"):
            Transmon().with_parameters(t1_us=1.0)
        with pytest.raises(ValueError, match="shapes"):
            Transmon().with_parameters(levels=4)
        with pytest.raises(ValueError, match="energy unit"):
            TwoLevel().with_parameters(omega0=2.0)

    @pytest.mark.parametrize(
        ("task", "name", "reason"),
        [
            ("transmon/CRGate-v0", "detuning_ghz", "frame"),
            ("battery/Dicke-v0", "omega0", "energy unit"),
            ("battery/DickeLocal-v0", "omega0", "energy unit"),
        ],
    )
    def test_unsupported_parameters_are_refused_with_reason(self, task, name, reason):
        # 0.3 does not randomise these two (review round 6): the CR target stays in the
        # nominal frame, and omega0 of the Dicke models is also the energy unit of H_B
        env = qenvs.make(task)
        assert name not in env.model.parameter_specs()
        with pytest.raises(ValueError, match=rf"{name!r} is not a randomisable.*{reason}"):
            ParameterRandomization(env, {name: Normal(0.001)})
        with pytest.raises(ValueError, match=reason):
            env.model.with_parameters(**{name: 1.0})

    def test_neighbours_of_the_refused_parameters_stay_supported(self):
        # refusing CR detuning and Dicke omega0 must not take these with them
        assert "detuning_ghz" in Transmon().parameter_specs()
        assert "omega_c" in DickeBattery().parameter_specs()
        assert "omega_c" in DickeBatteryLocal(n_cells=2).parameter_specs()
        cases = [
            ("transmon/XGate-v0", "detuning_ghz", Normal(0.002)),
            ("battery/Dicke-v0", "omega_c", TruncatedNormal(0.05, -0.2, 0.2)),
            ("battery/DickeLocal-v0", "omega_c", TruncatedNormal(0.05, -0.2, 0.2)),
        ]
        for task, name, dist in cases:
            env = ParameterRandomization(qenvs.make(task), {name: dist})
            state, _ = env.reset(jax.random.PRNGKey(1))
            assert float(state.data[f"offset/{name}"]) != 0.0, (task, name)

    def test_parameter_spec_validation(self):
        assert ParameterSpec("GHz", (-INF, 0.0), 0.01).contains(-1.0)
        assert not ParameterSpec("GHz", (-INF, 0.0), 0.01).contains(0.1)
        for support in [(1.0, 0.0), (math.nan, 1.0)]:
            with pytest.raises(ValueError, match="support"):
                ParameterSpec("x", support)
        for w in (-1.0, math.inf, math.nan):
            with pytest.raises(ValueError, match="w_ref"):
                ParameterSpec("x", (0.0, 1.0), w)

    def test_correlation_observables_default_is_empty(self):
        for name in ("TwoLevel", "QubitBattery", "LambdaSystem", "Transmon", "TwoTransmon"):
            assert BUILTIN_MODELS[name].correlation_observables() == {}

    def test_declared_values_are_inside_their_support_and_scale(self):
        for name, model in BUILTIN_MODELS.items():
            for k, spec in model.parameter_specs().items():
                v = float(model.parameter_values()[k])
                assert spec.contains(v), (name, k)
                assert spec.w_ref > 0
                assert isinstance(spec.unit, str) and spec.unit


class TestHamiltonianModelParameters:
    def test_no_parameters_is_unchanged_bit_for_bit(self):
        drift = 0.05 * SZ
        model = HamiltonianModel.from_operators(drift, [SX / 2])
        assert model.parameter_specs() == {} and model.parameter_values() == {}
        assert model.parameter_matrices == {} and model.parameter_nominals == {}
        assert np.array_equal(np.asarray(model.drift().to_jax()), np.asarray(model.drift_matrix))
        assert np.array_equal(np.asarray(model.drift_matrix), drift)
        assert model.with_parameters() is model
        with pytest.raises(ValueError, match="not a randomisable"):
            model.with_parameters(J=1.0)

    def test_drift_is_h0_plus_parameter_terms(self):
        model = custom_model()
        expect = 0.1 * SZ + 0.3 * SX + 0.2 * SZ
        assert np.allclose(np.asarray(model.drift().to_jax()), expect, atol=1e-15)
        new = model.with_parameters(J=-0.4)
        assert np.allclose(np.asarray(new.drift().to_jax()), 0.1 * SZ - 0.4 * SX + 0.2 * SZ)
        assert model.parameter_specs()["J"] == ParameterSpec("a.u.", (-INF, INF), 0.05)
        assert float(model.parameter_values()["d"]) == 0.2

    def test_with_parameters_jit_vmap_and_support(self):
        model = custom_model()
        f = jax.jit(lambda j: model.with_parameters(J=j).drift().to_jax())
        assert np.allclose(np.asarray(f(0.7))[0, 1], 0.7)
        out = jax.vmap(f)(jnp.asarray([0.0, 0.5, 1.0]))
        assert np.allclose(np.asarray(out)[:, 0, 1], [0.0, 0.5, 1.0])
        bounded = custom_model(support=(0.0, 1.0))
        bounded.check_parameters(J=1.0)
        with pytest.raises(ValueError, match="outside its support"):
            bounded.check_parameters(J=1.0001)
        with pytest.raises(ValueError, match="outside its support"):
            bounded.check_parameters(d=-0.1)

    def test_construction_errors(self):
        good = (0.3, SX, "u", (-INF, INF), 0.1)
        bad = {
            "non-Hermitian operator": (0.3, np.array([[0, 1], [0, 0]]), "u", (-INF, INF), 0.1),
            "wrong size": (0.3, np.eye(3), "u", (-INF, INF), 0.1),
            "nominal outside support": (2.0, SX, "u", (0.0, 1.0), 0.1),
            "support order": (0.3, SX, "u", (1.0, 0.0), 0.1),
            "negative w_ref": (0.3, SX, "u", (-INF, INF), -0.1),
            "nan nominal": (math.nan, SX, "u", (-INF, INF), 0.1),
            "wrong arity": (0.3, SX, "u"),
        }
        for what, entry in bad.items():
            with pytest.raises(ValueError):
                HamiltonianModel.from_operators(SZ, [SX], parameters={"J": entry})
            del what
        with pytest.raises(ValueError, match="mapping"):
            HamiltonianModel.from_operators(SZ, [SX], parameters=[good])  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="non-empty string"):
            HamiltonianModel.from_operators(SZ, [SX], parameters={"": good})

    def test_randomised_custom_model_end_to_end(self):
        model = custom_model()
        env = StateTransfer(model=model, target=dq.basis(2, 1), horizon=4, dt=0.2)
        renv = ParameterRandomization(
            env, {"J": Normal(0.1, "a.u."), "d": TruncatedNormal(0.05, -0.1, 0.1)}, oracle=True
        )
        # "d" has w_ref 0: not in the oracle observation
        assert renv.oracle_names == ("J",)
        assert renv.observation_dim == env.observation_dim + 1
        state, ts = renv.reset(jax.random.PRNGKey(2))
        assert ts.observation.shape == (renv.observation_dim,)
        assert float(ts.observation[-1]) == pytest.approx(float(state.data["offset/J"]) / 0.05)
        m = renv.episode_model(state)
        assert float(m.parameter_values()["J"]) == pytest.approx(
            0.3 + float(state.data["offset/J"])
        )
        assert float(m.parameter_values()["d"]) == pytest.approx(
            0.2 + float(state.data["offset/d"])
        )


# ====================================================================== oracle observation
class TestOracle:
    def test_observation_and_dim(self):
        env = gate_env()
        d = {
            "detuning_ghz": Normal(0.002),
            "anharmonicity_ghz": TruncatedNormal(0.005, -0.02, 0.02),
        }
        plain = ParameterRandomization(env, d)
        orc = ParameterRandomization(env, d, oracle=True)
        assert plain.observation_dim == env.observation_dim
        assert orc.observation_dim == env.observation_dim + 2
        key = jax.random.PRNGKey(5)
        _, ts0 = plain.reset(key)
        state, ts = orc.reset(key)
        assert np.array_equal(np.asarray(ts.observation[:-2]), np.asarray(ts0.observation))
        w = env.model.parameter_specs()
        for j, n in enumerate(("detuning_ghz", "anharmonicity_ghz")):
            nominal = float(env.model.parameter_values()[n])
            expect = (float(state.data[f"value/{n}"]) - nominal) / w[n].w_ref
            assert float(ts.observation[-2 + j]) == pytest.approx(expect, rel=1e-12, abs=1e-12)
        _, ts1 = orc.step(state, jnp.zeros(2))
        assert ts1.observation.shape == (orc.observation_dim,)
        assert np.array_equal(np.asarray(ts1.observation[-2:]), np.asarray(ts.observation[-2:]))

    def test_scale_is_w_ref_not_the_configured_width_or_strength(self):
        env = gate_env()
        key = jax.random.PRNGKey(6)
        a = ParameterRandomization(env, {"detuning_ghz": Normal(0.002)}, oracle=True)
        b = ParameterRandomization(env, {"detuning_ghz": Normal(0.004)}, strength=0.5, oracle=True)
        _, ta = a.reset(key)
        _, tb = b.reset(key)
        # same effective width -> same sample -> same oracle value (normalised by w_ref only)
        assert float(ta.observation[-1]) == pytest.approx(float(tb.observation[-1]), rel=1e-12)
        # strength 0: finite zero, no division by the effective width
        z = ParameterRandomization(env, {"detuning_ghz": Normal(0.002)}, strength=0.0, oracle=True)
        _, tz = z.reset(key)
        assert float(tz.observation[-1]) == 0.0

    def test_oracle_in_vmap_and_scan(self):
        renv = ParameterRandomization(gate_env(), {"detuning_ghz": Normal(0.002)}, oracle=True)
        ks = jax.random.split(jax.random.PRNGKey(0), 4)
        _, ts = jax.vmap(renv.reset)(ks)
        assert ts.observation.shape == (4, renv.observation_dim)
        _, rewards = renv.rollout(jnp.zeros((renv.horizon, 2)), ks[0])
        assert rewards.shape == (renv.horizon,)


# ====================================================================== G1
HEAVY_TASKS = {"battery/DickeLocal-v0"}


def _run(env, pulse, key, steps):
    state, ts = env.reset(key)
    step = eqx.filter_jit(env.step)
    obs, rewards = [np.asarray(ts.observation)], []
    for a in pulse[:steps]:
        state, ts = step(state, a)
        obs.append(np.asarray(ts.observation))
        rewards.append(np.asarray(ts.reward))
    return np.stack(obs), np.stack(rewards), np.asarray(state.qstate.to_jax())


def _zero_width_dists(model):
    out = {}
    for n, s in model.parameter_specs().items():
        unbounded = math.isinf(s.support[0]) and math.isinf(s.support[1])
        out[n] = Normal(0.0) if unbounded else TruncatedNormal(0.0, -s.w_ref, s.w_ref)
    return out


@pytest.mark.parametrize("task", TASKS)
def test_g1_zero_widths_reproduce_the_released_task(task):
    """G1: with every width 0 the wrapped environment equals the original (x64, IEEE equal)."""
    env = qenvs.make(task)
    pulse = make_task_snapshot.fixed_pulse(env)
    key = jax.random.PRNGKey(0)
    steps = 2 if task in HEAVY_TASKS else 5
    ref = _run(env, pulse, key, steps)
    dists = _zero_width_dists(env.model)
    wrapped = {"zero width": ParameterRandomization(env, dists)}
    if task not in HEAVY_TASKS:
        wrapped["strength 0"] = ParameterRandomization(
            env, _real_width_dists(env.model), strength=0.0
        )
    for what, w in wrapped.items():
        got = _run(w, pulse, key, steps)
        for a, b, label in zip(ref, got, ("observation", "reward", "state"), strict=True):
            assert np.array_equal(a, b), (what, label)


def _real_width_dists(model):
    return {
        n: _wide_dist(s, float(model.parameter_values()[n]))
        for n, s in model.parameter_specs().items()
    }


def test_g1_strength_zero_with_real_widths():
    env = qenvs.make("transmon/XGate-v0")
    pulse = make_task_snapshot.fixed_pulse(env)
    d = {"detuning_ghz": Normal(0.002), "anharmonicity_ghz": TruncatedNormal(0.005, -0.02, 0.02)}
    ref = _run(env, pulse, jax.random.PRNGKey(1), 8)
    got = _run(ParameterRandomization(env, d, strength=0.0), pulse, jax.random.PRNGKey(1), 8)
    for a, b in zip(ref, got, strict=True):
        assert np.array_equal(a, b)


def test_g1_empty_distributions_and_nesting_order():
    env = qenvs.make("qubit/StateTransfer-v0")
    pulse = make_task_snapshot.fixed_pulse(env)
    key = jax.random.PRNGKey(4)
    ref = _run(env, pulse, key, 6)
    inner = ParameterRandomization(env, {"detuning": Normal(0.0)})
    for w in (
        ParameterRandomization(env, {}),
        ControlErrors(inner, Normal(0.0), Normal(0.0), Normal(0.0)),
        ParameterRandomization(
            ControlErrors(env, Normal(0.0), step_noise=Normal(0.0)), {"detuning": Normal(0.0)}
        ),
    ):
        got = _run(w, pulse, key, 6)
        for a, b in zip(ref, got, strict=True):
            assert np.array_equal(a, b)


# ====================================================================== G2
class TestG2ParameterLeaves:
    N = 1024

    def renv(self, **kw):
        d = kw.pop(
            "dists",
            {
                "detuning_ghz": Normal(0.002, "GHz"),
                "anharmonicity_ghz": TruncatedNormal(0.005, -0.02, 0.02, "GHz"),
            },
        )
        return ParameterRandomization(gate_env(), d, **kw)

    def test_same_key_same_outputs(self):
        renv = self.renv(oracle=True)
        key = jax.random.PRNGKey(11)
        pulse = make_task_snapshot.fixed_pulse(renv)
        a = _run(renv, pulse, key, 6)
        b = _run(renv, pulse, key, 6)
        for x, y in zip(a, b, strict=True):
            assert np.array_equal(x, y)
        s1, _ = renv.reset(key)
        s2, _ = renv.reset(key)
        for k in s1.data:
            assert np.array_equal(np.asarray(s1.data[k]), np.asarray(s2.data[k]))

    def test_same_key_same_outputs_float32(self):
        with x32():
            renv = self.renv(oracle=True)
            key = jax.random.PRNGKey(11)
            pulse = make_task_snapshot.fixed_pulse(renv)
            a = _run(renv, pulse, key, 4)
            b = _run(renv, pulse, key, 4)
            assert a[0].dtype == np.float32
            for x, y in zip(a, b, strict=True):
                assert np.array_equal(x, y)

    def test_sample_is_the_documented_function_of_the_key(self):
        """Deterministic reference: value = nominal + dist.sample(parameter_key(key, name))."""
        renv = self.renv()
        key = keys.episode_key(RUN_KEY, 7)
        state, _ = renv.reset(key)
        for name, dist in renv.distributions:
            nominal = float(renv.env.model.parameter_values()[name])
            delta = dist.sample(keys.parameter_key(key, name), 1.0, (), jnp.float64)
            assert np.array_equal(np.asarray(state.data[f"offset/{name}"]), np.asarray(delta))
            assert float(state.data[f"value/{name}"]) == nominal + float(delta)

    def test_key_is_passed_unchanged_to_the_inner_env(self):
        renv = self.renv()
        key = jax.random.PRNGKey(99)
        state, _ = renv.reset(key)
        assert np.array_equal(np.asarray(state.key), np.asarray(key))
        assert np.array_equal(np.asarray(state.inner.key), np.asarray(key))

    def test_batch_position_independence(self):
        """Single run vs 1024-batch: samples equal in value, simulation results to 1e-12."""
        renv = self.renv(oracle=True)
        n = self.N
        run_key = jax.random.PRNGKey(31)
        ks = jax.vmap(lambda i: keys.episode_key(run_key, i))(jnp.arange(n))
        pulse = make_task_snapshot.fixed_pulse(renv)[:4]

        def episode(k):
            state, ts = renv.reset(k)
            obs = [ts.observation]
            rew = []
            for a in pulse:
                state, ts = renv.step(state, a)
                obs.append(ts.observation)
                rew.append(ts.reward)
            return state.data, jnp.stack(obs), jnp.stack(rew)

        # compared in the same execution mode (both jitted): eager and jitted programs may
        # differ in the last bit of erf_inv-based draws because XLA fuses the scaling
        data_b, obs_b, rew_b = jax.jit(jax.vmap(episode))(ks)
        single = jax.jit(episode)
        for i in (0, 1, 500, n - 1):
            data_s, obs_s, rew_s = single(ks[i])
            for k in data_s:
                assert np.array_equal(np.asarray(data_s[k]), np.asarray(data_b[k][i])), k
            assert np.max(np.abs(np.asarray(obs_s) - np.asarray(obs_b[i]))) <= 1e-12
            assert np.max(np.abs(np.asarray(rew_s) - np.asarray(rew_b[i]))) <= 1e-12

    def test_arbitrary_episode_can_be_regenerated_from_the_run_key(self):
        renv = self.renv()
        data = episode_data(renv, 64)
        for i in (0, 13, 63):
            state, _ = renv.reset(jax.random.fold_in(RUN_KEY, i))
            for k in state.data:
                assert np.array_equal(np.asarray(state.data[k]), np.asarray(data[k][i]))

    def test_leaf_separation(self):
        n = 256
        base = {"detuning_ghz": Normal(0.002), "anharmonicity_ghz": Uniform(0.01)}
        ref = episode_data(ParameterRandomization(gate_env(), base), n)

        def same(a_data, name):
            assert np.array_equal(
                np.asarray(a_data[f"offset/{name}"]), np.asarray(ref[f"offset/{name}"])
            )

        # reconfigure one parameter: the other is unchanged
        d1 = ParameterRandomization(
            gate_env(), {**base, "anharmonicity_ghz": TruncatedNormal(0.003, -0.01, 0.01)}
        )
        same(episode_data(d1, n), "detuning_ghz")
        d2 = ParameterRandomization(gate_env(), {**base, "detuning_ghz": Normal(0.5)})
        same(episode_data(d2, n), "anharmonicity_ghz")
        # remove one: the other is unchanged
        d3 = ParameterRandomization(gate_env(), {"detuning_ghz": Normal(0.002)})
        same(episode_data(d3, n), "detuning_ghz")
        # adding a parameter (here on a model with more of them) keeps the existing samples
        two = qenvs.make("transmon/CRGate-v0", levels=2)
        a = {"anharmonicity_c_ghz": TruncatedNormal(0.005, -0.02, 0.02)}
        b = {**a, "coupling_ghz": TruncatedNormal(1e-4, -1e-3, 1e-3)}
        da = episode_data(ParameterRandomization(two, a), n)
        db = episode_data(ParameterRandomization(two, b), n)
        assert np.array_equal(
            np.asarray(da["offset/anharmonicity_c_ghz"]),
            np.asarray(db["offset/anharmonicity_c_ghz"]),
        )
        # the order of the mapping does not matter
        d4 = ParameterRandomization(gate_env(), dict(reversed(list(base.items()))))
        same(episode_data(d4, n), "detuning_ghz")
        same(episode_data(d4, n), "anharmonicity_ghz")
        # other wrappers (any order) do not change the parameter samples
        inner = ParameterRandomization(gate_env(), base)
        ce_out = ControlErrors(inner, Normal(0.05), Normal(0.01), Normal(0.02))
        ce_in = ParameterRandomization(ControlErrors(gate_env(), Normal(0.05)), base)
        ce_corr = ControlErrors(inner, Normal(0.05), step_noise=Normal(0.01), correlated=True)
        for w in (ce_out, ce_in, ce_corr):
            data = episode_data(w, n)
            for name in base:
                assert np.array_equal(
                    np.asarray(data[f"offset/{name}"]), np.asarray(ref[f"offset/{name}"])
                )

    def test_empirical_distribution_and_correlations(self):
        """1024 episodes: moments within 4 SE of the declaration, |r| <= 4/sqrt(n)."""
        n = self.N
        dists = {
            "detuning_ghz": Normal(0.002),
            "anharmonicity_ghz": TruncatedNormal(0.005, -0.004, 0.02),
        }
        renv = ParameterRandomization(gate_env(), dists)
        data = episode_data(renv, n)
        thr = 4 / math.sqrt(n)
        x = {name: np.asarray(data[f"offset/{name}"]) for name in dists}
        for name, dist in dists.items():
            mean, var, mu4 = reference_moments(dist)
            assert_moments_within_4se(x[name], mean, var, mu4)
            assert abs(corr(x[name][:-1], x[name][1:])) <= thr  # adjacent episodes
        assert abs(corr(x["detuning_ghz"], x["anharmonicity_ghz"])) <= thr  # leaf to leaf


# ====================================================================== the guide
def test_uncertainty_guide_examples_run():
    """Every python block of docs/guides/uncertainty.md runs (the guide is the user contract)."""
    import re

    text = (pathlib.Path(__file__).resolve().parents[1] / "docs/guides/uncertainty.md").read_text()
    blocks = re.findall(r"```python\n(.*?)```", text, flags=re.S)
    assert len(blocks) >= 2
    ns: dict = {}
    for code in blocks:
        exec(compile(code, "uncertainty.md", "exec"), ns)
    # the full example evaluated 256 random devices of the X gate: fidelities are sane
    assert ns["fidelity"].shape == (256,)
    assert 0.9 < float(ns["fidelity"].min()) <= float(ns["fidelity"].max()) <= 1.0
    assert float(ns["clipped"].max()) == 0.0
    assert ns["model"].parameter_specs()["J"].w_ref == 0.05


# ====================================================================== correlation observables
class TestCorrelationObservables:
    """The six symmetrised second moments of the collective spin (Dicke models)."""

    NAMES = ("Jx2", "Jy2", "Jz2", "JxJy", "JyJz", "JzJx")

    @pytest.fixture(params=["dicke", "dicke_local"])
    def model(self, request):
        if request.param == "dicke":
            return DickeBattery(n_cells=3, kappa=0.1)
        return DickeBatteryLocal(n_cells=3)

    def test_names_dims_and_hermiticity(self, model):
        obs = model.correlation_observables()
        assert tuple(obs) == self.NAMES
        for name, op in obs.items():
            m = np.asarray(op.to_jax())
            assert m.shape == (model.dim, model.dim), name
            assert np.max(np.abs(m - m.conj().T)) == 0.0, name  # Hermitian exactly
            assert tuple(op.dims) == tuple(model.jz().dims), name

    def test_equal_to_symmetrised_products_of_the_spin_operators(self, model):
        jp, jm, jz = (np.asarray(o.to_jax()) for o in (model.jplus(), model.jminus(), model.jz()))
        jx, jy = (jp + jm) / 2, (jp - jm) / 2j
        expect = {
            "Jx2": jx @ jx,
            "Jy2": jy @ jy,
            "Jz2": jz @ jz,
            "JxJy": (jx @ jy + jy @ jx) / 2,
            "JyJz": (jy @ jz + jz @ jy) / 2,
            "JzJx": (jz @ jx + jx @ jz) / 2,
        }
        for name, op in model.correlation_observables().items():
            assert np.allclose(np.asarray(op.to_jax()), expect[name], atol=1e-13, rtol=0), name

    def test_casimir_and_known_expectations(self, model):
        """Jx2 + Jy2 + Jz2 = j(j+1) on the symmetric sector; <J_a^2> of |m = -j> are known."""
        j = model.n_cells / 2
        obs = {k: np.asarray(v.to_jax()) for k, v in model.correlation_observables().items()}
        total = obs["Jx2"] + obs["Jy2"] + obs["Jz2"]
        psi = np.asarray(model.initial_state().to_jax())
        rho = psi if psi.shape[-1] == psi.shape[0] else psi @ psi.conj().T
        # the initial state (all cells ground, m = -j) lives in the j = N/2 sector
        assert np.trace(total @ rho).real == pytest.approx(j * (j + 1), abs=1e-12)
        expect = {"Jx2": j / 2, "Jy2": j / 2, "Jz2": j * j}
        for name in self.NAMES:
            val = np.trace(obs[name] @ rho)
            assert abs(val.imag) <= 1e-12, name
            assert val.real == pytest.approx(expect.get(name, 0.0), abs=1e-12), name
        # the Casimir has the eigenvalues j'(j'+1) of the sectors present in the space
        allowed = {sj * (sj + 1) for sj in np.arange(model.n_cells % 2 / 2, j + 1)}
        eig = np.linalg.eigvalsh(total)
        assert all(min(abs(e - a) for a in allowed) < 1e-9 for e in eig)

    def test_non_hermitian_input_is_rejected(self):
        from qenvs.models.dicke import collective_spin_matrices, spin_second_moments

        jp, jm, jz = collective_spin_matrices(2)
        spin_second_moments(jp, jm, jz)  # consistent: fine
        with pytest.raises(ValueError, match="not Hermitian"):
            spin_second_moments(jp, 2.0 * jm, jz)  # J- is no longer the adjoint of J+

    def test_models_without_a_spin_have_none(self):
        assert TwoLevel().correlation_observables() == {}


# ------------------------------------------------- gaps found by mutation (review round 6)
def test_public_exports_of_qenvs_wrappers():
    import qenvs.wrappers as w

    for name in (
        "ObservationModel",
        "ParameterRandomization",
        "ControlErrors",
        "Normal",
        "Uniform",
        "TruncatedNormal",
        "Wrapper",
        "WrappedState",
    ):
        assert name in w.__all__ and hasattr(w, name), name


@pytest.mark.parametrize("dist", [Normal(0.1), Uniform(0.1), TruncatedNormal(0.1, -0.2, 0.3)])
def test_samples_have_the_requested_dtype(dist):
    key = jax.random.PRNGKey(0)
    for dtype in (jnp.float32, jnp.float64):
        x = dist.sample(key, 1.0, (8,), dtype=dtype)
        assert x.dtype == dtype, (type(dist).__name__, dtype, x.dtype)
