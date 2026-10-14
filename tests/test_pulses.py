"""Pulse I/O (qenvs.pulses) and Transmon.from_properties.

Gates: JSON and NPZ round trips are exact; a pulse refined to dt/k gives the same map as the
original on the finer grid; MHz conversions match hand values; device properties convert
from SI to GHz and us and build the same task as explicit kwargs.
"""

from __future__ import annotations

import io
import json
import math
import os
import pathlib
import pickle
import subprocess
import sys
import warnings
from datetime import datetime

import dynamiqs as dq
import numpy as np
import pytest

import qenvs
from qenvs.envs.gate import GateControl
from qenvs.gates import gate
from qenvs.models.transmon import Transmon, gaussian_amplitude
from qenvs.pulses import (
    SCHEMA,
    DeviceProperties,
    Pulse,
    mhz_to_rad_per_ns,
    parse_properties,
    rad_per_ns_to_mhz,
)
from qenvs.tasks.transmon import reference_actions

XGATE = "transmon/XGate-v0"
QUBIT = "qubit/StateTransfer-v0"
TWO_PI = 2.0 * math.pi
KS = (2, 9, 16)


def _random_actions(seed, n, c, bound):
    return np.random.default_rng(seed).uniform(-bound, bound, size=(n, c))


# ------------------------------------------------------------ resampling gate (ii)
def _final_map(env, actions):
    state, _ = env.rollout(np.asarray(actions))
    return np.asarray(state.qstate.to_jax())


def _resample_cases():
    """(label, task, task kwargs): the closed and open transmon and the qubit state transfer."""
    expm = dq.method.Expm()
    return [
        ("transmon closed", XGATE, {}),
        ("transmon open", XGATE, {"t1_us": 30.0, "t2_us": 25.0}),
        ("qubit closed", QUBIT, {"method": expm}),
        ("qubit open", QUBIT, {"method": expm, "gamma1": 0.3, "gamma_phi": 0.2}),
    ]


def measure_resample_errors():
    """Max abs difference of the final state/map, original grid vs refined grid, per case and k.

    The random pulse spans the action bound so a wrong hold order, wrong ``dt`` or wrong
    repeat count would show up at O(1). Returns ``{(label, k): error}``.
    """
    out = {}
    for label, task, kw in _resample_cases():
        env = qenvs.make(task, **kw)
        bound = float(env.action_bound)
        actions = _random_actions(7, env.horizon, env.action_dim, bound)
        pulse = Pulse.from_actions(actions, env.dt, task_id=task)
        ref = _final_map(env, actions)  # raw actions, not pulse.to_actions()
        for k in KS:
            fine = pulse.resample(env.dt / k)
            env_k = qenvs.make(task, horizon=env.horizon * k, dt=env.dt / k, **kw)
            assert fine.n_samples == env_k.horizon
            assert math.isclose(fine.dt_ns, env_k.dt, rel_tol=1e-15)
            out[(label, k)] = float(np.max(np.abs(_final_map(env_k, fine.to_actions()) - ref)))
    return out


def test_resampled_pulse_gives_the_same_final_state():
    errs = measure_resample_errors()
    assert len(errs) == 4 * len(KS)
    for key, err in errs.items():
        assert err < 1e-12, (key, err)


def test_resample_check_is_discriminating():
    """A tiled (instead of held) refinement, and a wrong dt, are far from the original."""
    env = qenvs.make(XGATE)
    actions = _random_actions(7, env.horizon, 2, float(env.action_bound))
    ref = _final_map(env, actions)
    env9 = qenvs.make(XGATE, horizon=env.horizon * 9, dt=env.dt / 9)
    tiled = np.tile(actions, (9, 1))
    assert np.max(np.abs(_final_map(env9, tiled) - ref)) > 1e-2
    env9_wrong_dt = qenvs.make(XGATE, horizon=env.horizon * 9, dt=env.dt)
    held = np.repeat(actions, 9, axis=0)
    assert np.max(np.abs(_final_map(env9_wrong_dt, held) - ref)) > 1e-2
    # the correct construction
    assert np.max(np.abs(_final_map(env9, held) - ref)) < 1e-12


def test_resample_float32_in_subprocess():
    """Same gate in single precision (tolerance 1e-4: float32 expm error accumulates over steps)."""
    code = (
        "import importlib.util, json, sys\n"
        "import jax; jax.config.update('jax_enable_x64', False)\n"
        f"spec = importlib.util.spec_from_file_location('tp', {str(pathlib.Path(__file__))!r})\n"
        "m = importlib.util.module_from_spec(spec); sys.modules['tp'] = m\n"
        "spec.loader.exec_module(m)\n"
        "import jax.numpy as jnp; assert jnp.asarray(1.0).dtype == jnp.float32\n"
        "errs = m.measure_resample_errors()\n"
        "print(json.dumps({f'{a}|{k}': e for (a, k), e in errs.items()}))\n"
        "assert max(errs.values()) < 1e-4, errs\n"
    )
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "JAX_ENABLE_X64": "0"}
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=180
    )
    assert out.returncode == 0, out.stderr[-2000:] + out.stdout[-500:]


def test_resample_tolerance_and_errors():
    p = Pulse(np.ones((4, 1)), 2.0, units="rad/ns")
    assert p.resample(2.0) is p
    assert p.resample(2.0 / 9).n_samples == 36
    assert p.resample(2.0 * (1 + 1e-10) / 9).n_samples == 36  # within 1e-9 relative
    with pytest.raises(ValueError, match="not a positive integer"):
        p.resample(2.0 * (1 + 1e-6) / 9)
    with pytest.raises(ValueError, match="not a positive integer"):
        p.resample(0.7)
    with pytest.raises(ValueError, match="not a positive integer"):
        p.resample(4.0)  # coarsening would change the pulse
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="sample_dt_ns"):
            p.resample(bad)


def test_resample_metadata_composes():
    p = Pulse(np.ones((3, 2)), 2.0, metadata={"a": 1})
    q = p.resample(1.0).resample(0.25)
    assert q.n_samples == 24
    assert q.dt_ns == 0.25
    assert q.metadata["resampled"] == {"factor": 8, "from_dt_ns": 2.0}
    assert q.metadata["a"] == 1
    assert np.array_equal(q.samples, np.repeat(p.samples, 8, axis=0))
    assert q.area() == pytest.approx(p.area(), rel=1e-15)  # held area is unchanged


# ------------------------------------------------------------ round trips gate (i)
def _example_pulse():
    rng = np.random.default_rng(3)
    samples = rng.normal(size=(23, 2)) * 1e-2
    samples[0] = (-0.0, 5e-324)  # signed zero and a subnormal survive too
    samples[1] = (np.nextafter(1.0, 2.0), 1.0 / 3.0)
    return Pulse.from_actions(
        samples,
        2.0 / 3.0,
        task_id=XGATE,
        task_kwargs={"levels": 3, "t1_us": 100.0, "anharmonicity_ghz": -0.34},
        fidelity=np.float64(0.9999876543210123),
        note="x",
        nested={"a": (1, 2.5), "b": [True, None]},
        arr=np.arange(3),
    )


def test_example_metadata_is_stored_without_loss():
    m = _example_pulse().metadata
    assert m["fidelity"] == 0.9999876543210123  # exactly what was passed in
    assert m["nested"] == {"a": [1, 2.5], "b": [True, None]}  # tuples become lists
    assert m["arr"] == [0, 1, 2]
    assert m["task_kwargs"] == {"levels": 3, "t1_us": 100.0, "anharmonicity_ghz": -0.34}


def _assert_identical(a, b):
    assert isinstance(b, Pulse)
    assert a.samples.dtype == b.samples.dtype == np.float64
    assert a.samples.shape == b.samples.shape
    assert a.samples.tobytes() == b.samples.tobytes()  # bit-exact, signed zeros included
    assert a.dt_ns == b.dt_ns
    assert a.channels == b.channels
    assert a.units == b.units
    assert a.metadata == b.metadata
    assert a == b


def test_json_round_trip_exact(tmp_path):
    p = _example_pulse()
    text = p.to_json()
    assert json.loads(text)["schema"] == SCHEMA == "qenvs.pulse/v1"
    _assert_identical(p, Pulse.from_json(text))
    _assert_identical(p, Pulse.from_json(p.to_json(indent=2)))
    path = tmp_path / "p.json"
    assert p.to_json(path) == text
    _assert_identical(p, Pulse.from_json(path))
    _assert_identical(p, Pulse.from_json(str(path)))
    _assert_identical(p, Pulse.from_json(json.loads(text)))
    _assert_identical(p, Pulse.from_json(text.encode()))
    assert text.isascii()


def test_npz_round_trip_exact(tmp_path):
    p = _example_pulse()
    for name, compressed in (("a.npz", False), ("b.npz", True), ("noext", False)):
        path = tmp_path / name
        p.to_npz(path, compressed=compressed)
        assert path.exists()  # written exactly where asked, no suffix appended
        _assert_identical(p, Pulse.from_npz(path))
        _assert_identical(p, Pulse.from_npz(str(path)))
    buf = io.BytesIO()
    p.to_npz(buf)
    buf.seek(0)
    _assert_identical(p, Pulse.from_npz(buf))
    with np.load(tmp_path / "a.npz", allow_pickle=False) as z:  # no pickled objects needed
        assert str(z["schema"]) == SCHEMA


def test_json_npz_agree_and_pickle_copy():
    import copy

    p = _example_pulse()
    buf = io.BytesIO()
    p.to_npz(buf)
    buf.seek(0)
    assert Pulse.from_npz(buf) == Pulse.from_json(p.to_json())
    assert pickle.loads(pickle.dumps(p)) == p
    assert copy.deepcopy(p) == p


def test_single_channel_and_generic_channel_round_trip():
    p = Pulse(np.linspace(0, 1, 5).reshape(5, 1), 0.1, units="dimensionless")
    assert p.channels == ("u0",)
    assert Pulse.from_json(p.to_json()) == p
    buf = io.BytesIO()
    p.to_npz(buf)
    buf.seek(0)
    assert Pulse.from_npz(buf) == p


def test_bad_records_are_rejected(tmp_path):
    d = _example_pulse().to_dict()
    with pytest.raises(ValueError, match="schema"):
        Pulse.from_dict({**d, "schema": "qenvs.pulse/v2"})
    with pytest.raises(ValueError, match="schema"):
        Pulse.from_dict({k: v for k, v in d.items() if k != "schema"})
    with pytest.raises(ValueError, match="missing fields"):
        Pulse.from_dict({k: v for k, v in d.items() if k != "units"})
    with pytest.raises(ValueError, match="channels"):
        Pulse.from_dict({**d, "channels": ["I"]})
    np.savez(tmp_path / "x.npz", foo=np.zeros(2))
    with pytest.raises(ValueError, match="not a qenvs pulse archive"):
        Pulse.from_npz(tmp_path / "x.npz")


def test_metadata_must_be_json_and_is_copied():
    with pytest.raises(TypeError, match=r"metadata\['task_kwargs'\]\['method'\]"):
        Pulse.from_actions(np.zeros((2, 2)), 1.0, task_kwargs={"method": dq.method.Expm()})
    with pytest.raises(ValueError, match="not finite"):
        Pulse(np.zeros((2, 1)), 1.0, metadata={"f": float("nan")})
    with pytest.raises(TypeError, match="non-string key"):
        Pulse(np.zeros((2, 1)), 1.0, metadata={"a": {1: 2}})
    meta = {"a": [1, 2]}
    p = Pulse(np.zeros((2, 1)), 1.0, metadata=meta)
    meta["a"].append(3)
    assert p.metadata == {"a": [1, 2]}
    p.metadata["a"].append(9)  # the accessor returns a copy
    assert p.metadata == {"a": [1, 2]}


# -------------------------------------------------------------- construction
def test_pulse_is_immutable_and_validates():
    src = np.arange(6.0).reshape(3, 2)
    p = Pulse(src, 2.0)
    src[0, 0] = 99.0  # the constructor copied
    assert p.samples[0, 0] == 0.0
    with pytest.raises(ValueError, match="read-only"):
        p.samples[0, 0] = 1.0
    for attr in ("dt_ns", "samples", "units", "extra"):
        with pytest.raises(AttributeError, match="immutable"):
            setattr(p, attr, 1)
    with pytest.raises(AttributeError, match="immutable"):
        del p.units
    w = p.to_actions()
    w[0, 0] = 5.0  # the exported copy is writable and detached
    assert p.samples[0, 0] == 0.0
    assert p.channels == ("u0", "u1")
    assert (p.n_samples, p.n_channels, p.duration_ns) == (3, 2, 6.0)
    asym = np.array([[1.0, -2.0], [3.0, 5.0], [7.0, 11.0], [-13.0, 17.0]])  # asymmetric in time
    assert np.array_equal(Pulse(asym, 0.5).to_actions(), asym)  # catches a time reversal
    for bad in (np.zeros(4), np.zeros((0, 2)), np.zeros((2, 0)), np.zeros((2, 2, 2))):
        with pytest.raises(ValueError, match="samples"):
            Pulse(bad, 1.0)
    with pytest.raises(ValueError, match="finite"):
        Pulse(np.array([[np.nan]]), 1.0)
    with pytest.raises(TypeError, match="real"):
        Pulse(np.array([[1.0 + 0.5j], [0.0j]]), 1.0)
    with pytest.raises(TypeError, match="real"):
        Pulse([[1 + 2j]], 1.0)
    for dt in (0.0, -1.0, float("inf")):
        with pytest.raises(ValueError, match="dt_ns"):
            Pulse(np.zeros((2, 1)), dt)
    with pytest.raises(ValueError, match="unique"):
        Pulse(np.zeros((2, 2)), 1.0, channels=("a", "a"))
    with pytest.raises(ValueError, match="unique"):
        Pulse(np.zeros((2, 2)), 1.0, channels=("a",))
    with pytest.raises(TypeError, match="sequence"):
        Pulse(np.zeros((2, 1)), 1.0, channels="I")
    with pytest.raises(ValueError, match="units"):
        Pulse(np.zeros((2, 1)), 1.0, units="")
    assert Pulse(src, 2.0) != Pulse(src, 2.5)
    assert Pulse(src, 2.0) != "pulse"


def test_from_actions_task_units_channels_and_provenance():
    import jax.numpy as jnp

    a32 = jnp.asarray([[0.1, 0.2], [0.3, -0.4]], dtype=jnp.float32)
    p = Pulse.from_actions(a32, 2.0, task_id=XGATE, task_kwargs={"horizon": 2}, fidelity=0.5)
    assert p.units == "rad/ns"
    assert p.channels == ("I", "Q")
    assert p.samples.dtype == np.float64
    assert np.array_equal(p.samples, np.asarray(a32, dtype=np.float64))  # widening is exact
    m = p.metadata
    assert m["task_id"] == XGATE
    assert m["task_kwargs"] == {"horizon": 2}
    assert m["fidelity"] == 0.5
    assert m["qenvs_version"] == qenvs.__version__
    # no task id: hardware units assumed, generic channel names; explicit overrides win
    q = Pulse.from_actions(np.zeros((2, 2)), 1.0)
    assert (q.units, q.channels) == ("rad/ns", ("u0", "u1"))
    assert "task_id" not in q.metadata
    r = Pulse.from_actions(np.zeros((2, 2)), 1.0, channels=("x", "y"), units="dimensionless")
    assert (r.units, r.channels) == ("dimensionless", ("x", "y"))
    with pytest.raises(ValueError, match="contradicts"):
        Pulse.from_actions(np.zeros((2, 2)), 1.0, task_id=XGATE, units="dimensionless")
    with pytest.raises(KeyError, match="unknown task id"):
        Pulse.from_actions(np.zeros((2, 2)), 1.0, task_id="nope/Nothing-v0")
    with pytest.raises(ValueError, match="n_samples, n_channels"):
        Pulse.from_actions(np.zeros(4), 1.0)


def test_units_follow_the_registry():
    released = {
        "transmon/XGate-v0": "rad/ns",
        "transmon/SXGate-v0": "rad/ns",
        "qubit/StateTransfer-v0": "dimensionless",
        "battery/TwoLevel-v0": "dimensionless",
        "battery/Dicke-v0": "dimensionless",
        "transfer/LambdaSTIRAP-v0": "dimensionless",
    }
    for task_id, units in released.items():
        env = qenvs.make(task_id)
        p = Pulse.from_actions(np.zeros((env.horizon, env.action_dim)), env.dt, task_id=task_id)
        assert p.units == units, task_id
        if units == "dimensionless":
            assert p.channels == tuple(f"u{k}" for k in range(env.action_dim))
            with pytest.raises(ValueError, match="MHz"):
                p.rabi_mhz()
        else:
            assert p.channels == ("I", "Q")


# ------------------------------------------------------------- MHz gate (iii)
def test_mhz_hand_values():
    # pi pulse of 20 ns: Omega T = pi, Omega = pi/20 rad/ns, Rabi = Omega/2pi = 1/(2 T) = 25 MHz
    omega = math.pi / 20.0
    assert float(rad_per_ns_to_mhz(omega)) == pytest.approx(25.0, rel=1e-14)
    assert float(mhz_to_rad_per_ns(25.0)) == pytest.approx(omega, rel=1e-14)
    # 50 MHz is the default action bound 2 pi x 0.05 rad/ns = 0.31415926...
    assert float(mhz_to_rad_per_ns(50.0)) == pytest.approx(0.1 * math.pi, rel=1e-14)
    assert float(mhz_to_rad_per_ns(1.0)) == pytest.approx(0.006283185307179586, rel=1e-14)
    env = qenvs.make(XGATE)
    assert float(rad_per_ns_to_mhz(env.action_bound)) == pytest.approx(50.0, rel=1e-14)
    p = Pulse(np.array([[omega, -2 * omega], [0.0, 1.0]]), 10.0, units="rad/ns")
    np.testing.assert_allclose(
        p.rabi_mhz(), [[25.0, -50.0], [0.0, 1.0 / TWO_PI * 1e3]], rtol=1e-14, atol=0
    )
    back = Pulse.from_rabi_mhz(p.rabi_mhz(), 10.0, channels=None)
    np.testing.assert_allclose(back.samples, p.samples, rtol=1e-15, atol=0)
    assert back.units == "rad/ns"
    assert Pulse.from_rabi_mhz(np.zeros((2, 2)), 1.0).channels == ("I", "Q")
    with pytest.raises(ValueError, match="always produces"):
        Pulse.from_rabi_mhz(np.zeros((2, 2)), 1.0, units="dimensionless")
    with pytest.raises(ValueError, match="MHz"):
        Pulse(np.zeros((2, 1)), 1.0, units="dimensionless").rabi_mhz()


def test_25_mhz_for_20_ns_is_an_x_gate_in_the_simulation():
    """The MHz formula tied to the physics: constant 25 MHz on I for 20 ns flips the qubit."""
    env = qenvs.make(XGATE, levels=2, horizon=10, dt=2.0)
    rabi = np.zeros((10, 2))
    rabi[:, 0] = 25.0
    p = Pulse.from_rabi_mhz(rabi, 2.0)
    assert p.duration_ns == 20.0
    state, _ = env.rollout(p.to_actions())
    assert float(env.fidelity(state.qstate)) == pytest.approx(1.0, abs=1e-12)
    # 20 MHz for 20 ns is a 0.8 pi rotation, clearly not X
    rabi[:, 0] = 20.0
    state, _ = env.rollout(Pulse.from_rabi_mhz(rabi, 2.0).to_actions())
    assert float(env.fidelity(state.qstate)) < 0.95


def test_area_hand_values():
    p = Pulse(np.array([[1.0, 2.0], [3.0, -2.0]]), 0.5)
    np.testing.assert_allclose(p.area(), [2.0, 0.0], atol=1e-15)
    env = qenvs.make(XGATE)
    sigma = env.horizon * env.dt / 4
    amp = gaussian_amplitude(math.pi, env.horizon, env.dt, sigma)
    from qenvs.models.transmon import gaussian_drag

    pi = Pulse.from_actions(gaussian_drag(env.horizon, env.dt, amp, sigma), env.dt, task_id=XGATE)
    assert pi.area()[0] == pytest.approx(math.pi, rel=1e-12)
    assert abs(pi.area()[1]) < 1e-15
    # area survives refinement
    assert pi.resample(2.0 / 9).area()[0] == pytest.approx(math.pi, rel=1e-12)


# ----------------------------------------------------------------------- pad
def test_pad_total_multiple_and_errors():
    p = Pulse(np.ones((5, 2)), 0.25, metadata={"k": 1})
    q = p.pad(8)
    assert q.n_samples == 8
    assert np.array_equal(q.samples[:5], p.samples)
    assert not q.samples[5:].any()
    assert q.dt_ns == p.dt_ns
    assert q.metadata["unpadded_n_samples"] == 5
    assert q.pad(16).metadata["unpadded_n_samples"] == 5  # remembers the first length
    assert p.pad(multiple=16).n_samples == 16
    assert Pulse(np.ones((16, 1)), 1.0).pad(multiple=16).n_samples == 16
    assert Pulse(np.ones((17, 1)), 1.0).pad(multiple=16).n_samples == 32
    assert p.pad(5) is p
    np.testing.assert_allclose(q.area(), p.area(), atol=1e-15)
    with pytest.raises(ValueError, match="already has 5"):
        p.pad(4)
    with pytest.raises(ValueError, match=r"pad\(multiple=N\)"):
        p.pad(4)
    with pytest.raises(ValueError, match="exactly one"):
        p.pad()
    with pytest.raises(ValueError, match="exactly one"):
        p.pad(8, multiple=4)
    with pytest.raises(ValueError, match="positive integer"):
        p.pad(multiple=0)
    with pytest.raises(ValueError, match="integer"):
        p.pad(7.5)


def test_pad_keeps_the_computational_block_for_resonant_drive():
    """Zero drive on resonance leaves {|0>,|1>} untouched; |2> only picks up a phase."""
    env = qenvs.make(XGATE)
    p = Pulse.from_actions(reference_actions(XGATE, env), env.dt, task_id=XGATE).resample(2.0 / 9)
    assert p.n_samples == 144  # already a multiple of 16: pad to 160 to force real padding
    padded = p.pad(160)
    base_env = qenvs.make(XGATE, horizon=p.n_samples, dt=p.dt_ns)
    pad_env = qenvs.make(XGATE, horizon=padded.n_samples, dt=padded.dt_ns)
    u0 = _final_map(base_env, p.to_actions())
    u1 = _final_map(pad_env, padded.to_actions())
    np.testing.assert_allclose(u1[:2, :2], u0[:2, :2], atol=1e-12)
    assert abs(float(base_env.fidelity(u0)) - float(pad_env.fidelity(u1))) < 1e-12
    assert abs(float(base_env.leakage(u0)) - float(pad_env.leakage(u1))) < 1e-12
    # with a detuning the idle drift does act: the padded map is no longer the same
    kw = {"detuning_ghz": 0.01}
    u0d = _final_map(qenvs.make(XGATE, horizon=p.n_samples, dt=p.dt_ns, **kw), p.to_actions())
    u1d = _final_map(
        qenvs.make(XGATE, horizon=padded.n_samples, dt=padded.dt_ns, **kw), padded.to_actions()
    )
    assert np.max(np.abs(u1d[:2, :2] - u0d[:2, :2])) > 1e-2


def test_resample_scales_unpadded_n_samples():
    p = Pulse(np.ones((5, 1)), 0.9)
    padded = p.pad(8)
    assert padded.metadata["unpadded_n_samples"] == 5
    fine = padded.resample(0.1)  # k = 9
    assert fine.n_samples == 72
    assert fine.metadata["unpadded_n_samples"] == 45  # still counts the samples before the zeros
    assert not fine.samples[45:].any() and fine.samples[:45].all()
    assert fine.resample(0.05).metadata["unpadded_n_samples"] == 90
    # the recommended order (resample, then pad) gives the same record
    other = p.resample(0.1).pad(72)
    assert other.metadata["unpadded_n_samples"] == 45
    assert np.array_equal(other.samples, fine.samples)
    assert "unpadded_n_samples" not in p.resample(0.1).metadata


def test_pad_in_an_open_system_lets_t1_t2_act_on_the_padding():
    kw = {"t1_us": 30.0, "t2_us": 25.0}
    env = qenvs.make(XGATE, **kw)
    p = Pulse.from_actions(reference_actions(XGATE, env), env.dt, task_id=XGATE).resample(2.0 / 9)
    padded = p.pad(160)
    maps = [
        _final_map(qenvs.make(XGATE, horizon=q.n_samples, dt=q.dt_ns, **kw), q.to_actions())
        for q in (p, padded)
    ]
    assert maps[0].shape == (9, 9)
    diff = float(np.max(np.abs(maps[1][:2, :2] - maps[0][:2, :2])))
    assert 1e-5 < diff < 1e-3, diff  # measured 1.2e-4: decay during the 16 idle samples


def test_rollout_clips_actions_to_the_bound():
    env = qenvs.make(XGATE)
    bound = float(env.action_bound)
    assert float(rad_per_ns_to_mhz(bound)) == pytest.approx(50.0, rel=1e-14)
    big = np.full((env.horizon, 2), 10.0 * bound)
    at_bound = np.full((env.horizon, 2), bound)
    np.testing.assert_array_equal(_final_map(env, big), _final_map(env, at_bound))
    assert Pulse.from_actions(big, env.dt, task_id=XGATE).rabi_mhz().max() > 50.0  # not clipped


# -------------------------------------------------- device properties gate (iv)
STAMP = datetime(2026, 10, 1, 12, 0, 0)


def _qiskit_like():
    """Shape of BackendProperties.qubit_property(q): {name: (SI value, datetime)} plus extras."""
    return {
        "T1": (120e-6, STAMP),  # seconds
        "T2": (85e-6, STAMP),
        "frequency": (4.9e9, STAMP),  # Hz
        "anharmonicity": (-0.33e9, STAMP),
        "readout_error": (0.012, STAMP),  # unrelated entries are ignored
        "prob_meas0_prep1": (0.02, STAMP),
    }


def test_from_properties_si_hand_values():
    p = parse_properties(_qiskit_like())
    assert p == DeviceProperties(
        frequency_ghz=pytest.approx(4.9, rel=1e-14),
        anharmonicity_ghz=pytest.approx(-0.33, rel=1e-14),
        t1_us=pytest.approx(120.0, rel=1e-14),
        t2_us=pytest.approx(85.0, rel=1e-14),
    )
    m = Transmon.from_properties(_qiskit_like())
    assert m.t1_us == pytest.approx(120.0, rel=1e-14)
    assert m.t2_us == pytest.approx(85.0, rel=1e-14)
    assert m.anharmonicity_ghz == pytest.approx(-0.33, rel=1e-14)
    assert m.levels == 3
    assert m.detuning_ghz == 0.0
    # odd values and bare numbers; a tuple with extra entries uses the first
    m2 = Transmon.from_properties(
        {"T1": 87.3e-6, "T2": [61.9e-6, "ignored"], "anharmonicity": -3.4712e8}, levels=4
    )
    assert m2.levels == 4
    assert m2.t1_us == pytest.approx(87.3, rel=1e-14)
    assert m2.t2_us == pytest.approx(61.9, rel=1e-14)
    assert m2.anharmonicity_ghz == pytest.approx(-0.34712, rel=1e-14)
    # derived rates in the model's own units (1/ns): 120 us -> 1/120000 per ns
    assert m.gamma1 == pytest.approx(1.0 / 120_000.0, rel=1e-14)
    assert m.gamma_phi == pytest.approx(1.0 / 85_000.0 - 0.5 / 120_000.0, rel=1e-12)
    assert m.alpha == pytest.approx(TWO_PI * -0.33, rel=1e-14)


def test_from_properties_explicit_units_equal_si():
    explicit = Transmon.from_properties(
        {"frequency_ghz": 4.9, "anharmonicity_ghz": -0.33, "t1_us": 120.0, "t2_us": 85.0},
        detuning_ghz=0.004,
    )
    si = Transmon.from_properties(_qiskit_like(), detuning_ghz=0.004)
    assert explicit.detuning_ghz == si.detuning_ghz == 0.004
    for f in ("anharmonicity_ghz", "t1_us", "t2_us"):
        assert getattr(explicit, f) == pytest.approx(getattr(si, f), rel=1e-14)
    assert parse_properties({"frequency_ghz": 4.9}).frequency_ghz == 4.9  # recorded, not modelled
    # partial information: no T2 means no pure dephasing, no T1 means T2 is all dephasing
    only_t1 = Transmon.from_properties({"anharmonicity_ghz": -0.3, "t1_us": 50.0})
    assert only_t1.t2_us is None
    assert only_t1.gamma_phi == 0.0
    only_t2 = Transmon.from_properties({"anharmonicity_ghz": -0.3, "T2": 40e-6})
    assert only_t2.t1_us is None
    assert only_t2.gamma_phi == pytest.approx(1.0 / 40_000.0, rel=1e-12)
    closed = Transmon.from_properties({"anharmonicity_ghz": -0.3})
    assert not closed.is_open


def test_from_properties_t2_above_two_t1_raises():
    with pytest.raises(ValueError, match=r"T2 = 250 us exceeds 2 T1 = 200 us"):
        Transmon.from_properties({"T1": 100e-6, "T2": 250e-6, "anharmonicity": -0.3e9})
    with pytest.raises(ValueError, match="clip it yourself"):
        Transmon.from_properties({"t1_us": 100.0, "t2_us": 200.1, "anharmonicity_ghz": -0.3})
    # exactly 2 T1 is allowed (pure relaxation)
    m = Transmon.from_properties({"t1_us": 100.0, "t2_us": 200.0, "anharmonicity_ghz": -0.3})
    assert m.gamma_phi == 0.0


def test_from_properties_missing_anharmonicity_warns_and_keeps_default():
    with pytest.warns(UserWarning, match=r"no anharmonicity.*-0\.34"):
        m = Transmon.from_properties({"t1_us": 80.0})
    assert m.anharmonicity_ghz == Transmon().anharmonicity_ghz == -0.34
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a complete dict does not warn
        Transmon.from_properties({"anharmonicity_ghz": -0.3, "t1_us": 80.0})
    with pytest.warns(UserWarning, match="positive"):
        m = Transmon.from_properties({"anharmonicity_ghz": 0.3})
    assert m.anharmonicity_ghz == 0.3


def test_from_properties_rejects_ambiguous_or_bad_input():
    with pytest.raises(ValueError, match="not both"):
        parse_properties({"t1_us": 100.0, "T1": 100e-6})
    for key in ("t1_ms", "T1_us", "frequency_mhz", "anharmonicity_hz", "t2_ns"):
        with pytest.raises(ValueError, match="unsupported property name"):
            parse_properties({key: 1.0})
    for bad in (-1e-6, 0.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match=r"positive|finite"):
            parse_properties({"T1": bad})
    with pytest.raises(TypeError, match="number"):
        parse_properties({"T1": "fast"})
    with pytest.raises(TypeError, match="number"):
        parse_properties({"T1": True})
    with pytest.raises(ValueError, match="empty"):
        parse_properties({"T1": ()})
    with pytest.raises(TypeError, match="mapping"):
        parse_properties([("T1", 1e-4)])
    assert parse_properties({}) == DeviceProperties()


def test_from_properties_accepts_value_attribute_objects():
    class Nduv:  # duck type of qiskit's Nduv(name, date, unit, value)
        def __init__(self, value):
            self.value = value

    p = parse_properties({"T1": Nduv(1e-4), "frequency": Nduv(5e9)})
    assert p.t1_us == pytest.approx(100.0, rel=1e-14)
    assert p.frequency_ghz == pytest.approx(5.0, rel=1e-14)


def test_task_from_properties_equals_task_from_explicit_kwargs():
    kwargs = {"anharmonicity_ghz": -0.33, "t1_us": 120.0, "t2_us": 85.0, "detuning_ghz": 0.002}
    env_a = qenvs.make(XGATE, **kwargs)
    model = Transmon.from_properties(_qiskit_like(), detuning_ghz=0.002)
    ref = qenvs.make(XGATE)
    env_b = GateControl(
        model=model,
        target=gate("X"),
        horizon=ref.horizon,
        dt=ref.dt,
        action_bound=ref.action_bound,
    )
    actions = reference_actions(XGATE, env_b)
    for name in ("anharmonicity_ghz", "t1_us", "t2_us", "detuning_ghz"):
        assert getattr(env_a.model, name) == pytest.approx(getattr(env_b.model, name), rel=1e-14)
    ua, ub = _final_map(env_a, actions), _final_map(env_b, actions)
    assert ua.shape == ub.shape == (9, 9)  # open: superoperator
    np.testing.assert_allclose(ub, ua, rtol=0, atol=1e-13)
    assert float(env_a.fidelity(ua)) == pytest.approx(float(env_b.fidelity(ub)), abs=1e-13)
    # the drift and jump operators (the physics) agree too
    np.testing.assert_allclose(
        np.asarray(env_b.model.drift().to_jax()),
        np.asarray(env_a.model.drift().to_jax()),
        atol=1e-14,
    )
    for ja, jb in zip(env_a.model.jump_ops(), env_b.model.jump_ops(), strict=True):
        np.testing.assert_allclose(np.asarray(jb.to_jax()), np.asarray(ja.to_jax()), atol=1e-15)
    # a model whose properties differ is detectably different (the equality is not vacuous)
    other = Transmon.from_properties({**_qiskit_like(), "T1": 60e-6})
    assert abs(other.gamma1 - env_a.model.gamma1) > 1e-6
