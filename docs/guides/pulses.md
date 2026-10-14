# Pulses for hardware toolchains

`qenvs.pulses` turns the actions of a rollout (an optimised or hand-made open-loop pulse)
into a record you can store, hand to a colleague and feed to a hardware or pulse-level
toolchain: samples, sample period, channel names, units and provenance in one immutable
object, with lossless JSON and NPZ files, exact refinement onto a finer sample grid, zero
padding to the granularity of an arbitrary waveform generator (AWG), and the conversion
between rad/ns and MHz. `Transmon.from_properties` builds a transmon from device
parameters. Nothing here imports Qiskit or talks to hardware.

!!! warning "Units"
    For the transmon tasks the actions are **angular frequencies in rad/ns**, time is in
    **ns**, and the Rabi frequency is $\Omega/2\pi$. A constant $\Omega = \pi/20$ rad/ns for
    20 ns is a pi pulse and a **25 MHz** Rabi frequency. Tasks in units of the transition
    frequency (qubit, battery, lambda system) give `'dimensionless'` pulses whose `dt_ns`
    field holds the period in units of $1/\omega_0$; the MHz helpers refuse them.

## Export a GRAPE pulse in MHz on a 2/9 ns grid

The example optimises a 40 ns X gate on a 2 ns control grid, refines it to a 2/9 ns
hardware sample period, pads it to a multiple of 16 samples and prints the amplitudes in
MHz (float64, CPU; `examples/grape_jax.py` on the path):

```python
import jax

jax.config.update("jax_enable_x64", True)
import numpy as np
import qenvs
from grape_jax import run_grape
from qenvs.pulses import Pulse

task, kw = "transmon/XGate-v0", {"horizon": 20, "dt": 2.0}  # 40 ns on a 2 ns grid
g = run_grape(task, steps=300, lr=0.1, seed=0, **kw)
env = qenvs.make(task, **kw)
state, _ = env.rollout(np.asarray(g["best_pulse"]))
fidelity = float(env.fidelity(state.qstate))  # 1 - F about 6e-16

pulse = Pulse.from_actions(
    g["best_pulse"], env.dt, task_id=task, task_kwargs=kw, fidelity=fidelity
)
hw = pulse.resample(2 / 9)  # 20 x 9 = 180 samples of 2/9 ns, exact (sample-and-hold)
awg = hw.pad(multiple=16)  # 192 samples; metadata['unpadded_n_samples'] = 180

rabi_mhz = awg.rabi_mhz()  # (192, 2) array, columns I and Q, in MHz
print(awg.channels, awg.dt_ns, np.abs(rabi_mhz).max())  # ('I', 'Q') 0.2222... 49.02

pulse.to_json("xgate.json")  # lossless; Pulse.from_json("xgate.json") reads it back
awg.to_npz("xgate_awg.npz")  # NumPy archive, no pickle
```

* **Channels and units** come from the task: `Pulse.from_actions(..., task_id=...)` reads
  `registry[task_id].units` (a task whose time unit is ns is in rad/ns, every other task is
  dimensionless) and names the two controls of a `Transmon` task `('I', 'Q')`. Other tasks
  get `'u0'`, `'u1'`, ... Without a `task_id` the units default to `'rad/ns'` (assumed, not
  checked) unless you pass `units=`.
* **Provenance** is kept in `pulse.metadata`: `task_id`, `task_kwargs`, `qenvs_version` and
  anything you add (`fidelity=...`). Metadata must be JSON-compatible; numpy scalars and
  arrays are converted, other objects (such as a dynamiqs `method`) raise `TypeError`
  naming the offending key. After `resample` the `task_kwargs` still describe the original
  task; `metadata['resampled']` records the factor.
* **Files.** The JSON document has `"schema": "qenvs.pulse/v1"`, `dt_ns`, `channels`,
  `units`, `samples` (a list of rows) and `metadata`. Floats are written with their shortest
  round-trip representation, so JSON and NPZ round trips reproduce the samples bit for bit
  (checked, including signed zero and subnormals). Other schema versions are rejected.
* **Back into QEnvS.** `pulse.to_actions()` returns the `(n, channels)` float64 array for
  `env.rollout`. Samples must be real: complex input raises `TypeError` (pass I and Q as two channels). `Pulse.from_rabi_mhz(table_mhz, dt_ns)` builds a rad/ns pulse from a table
  of Rabi frequencies in MHz.

### Resampling is exact, within a stated condition

A piecewise-constant pulse held for `dt` is the same function of time as the same samples
repeated `k` times and held for `dt/k`. `resample(sample_dt_ns)` does exactly that, so the
refined pulse gives the same propagator on the finer grid. It requires
`dt / sample_dt` to be an integer within $10^{-9}$ relative (2 ns on 2/9 ns is 9). A
non-integer ratio or a coarser grid raises `ValueError`: no interpolation, no filtering, no
silent change of the waveform. The new period is `dt / k`, which can differ from the number
you typed by one rounding error.

Measured (`tests/test_pulses.py`; random full-range pulse, final map of the environment on
the original grid against the grid refined by $k$ with `horizon * k` steps and `dt / k`):

| task | $k = 2$ | $k = 9$ | $k = 16$ | float32, worst of the three |
|---|---|---|---|---|
| `transmon/XGate-v0`, closed | 1.9e-15 | 5.9e-15 | 7.6e-15 | 9.6e-6 |
| `transmon/XGate-v0`, `t1_us=30, t2_us=25` | 1.7e-15 | 6.8e-15 | 1.3e-14 | 1.2e-5 |
| `qubit/StateTransfer-v0`, closed, `Expm()` | 1.3e-15 | 1.7e-15 | 4.3e-15 | 8.3e-7 |
| `qubit/StateTransfer-v0`, `gamma1=0.3, gamma_phi=0.2`, `Expm()` | 3.3e-16 | 5.0e-15 | 8.1e-15 | 1.1e-6 |

The test requires $10^{-12}$ in float64 and $10^{-4}$ in float32 (single-precision matrix
exponentials accumulate rounding over the steps). Tiling instead of holding, or forgetting
to divide `dt`, give errors above $10^{-2}$ and fail the same test.

!!! note "Use an exact solver for the comparison"
    `qubit/StateTransfer-v0` integrates each step with adaptive `Tsit5()` by default, whose
    tolerance is about $10^{-6}$, so two grids of the same pulse differ at that level even
    though the pulse is identical. Pass `method=dq.method.Expm()` to the task (as above) to
    see the exact agreement. The transmon gate tasks use `Expm()` already.

### Padding

`pad(n_samples)` appends zeros up to a total length, `pad(multiple=16)` up to the next
multiple (AWG granularity); the original length stays in `metadata['unpadded_n_samples']`.
The area is unchanged. Resample first and pad second: `resample` multiplies an existing
`unpadded_n_samples` by the refinement factor, so the record keeps counting samples of the
current grid in either order, but the padding length you choose is always in samples of the
grid it is applied to.

The padded pulse has the same map in QEnvS only if idle time is harmless. With zero
detuning in a closed system the computational block of the propagator and the gate fidelity
and leakage are unchanged to $10^{-12}$ (3 levels, checked; the block is identical). Two
things make the idle samples act:

* **Drift.** With a detuning of 10 MHz the drift acts during the 16 padded samples and the
  2x2 block changes by 0.2 (largest matrix element).
* **Decay.** In an open system T1 and T2 decay also acts during the padded samples. For
  `transmon/XGate-v0` with `t1_us=30, t2_us=25` the padded 2x2 block of the 9x9 map differs
  from the unpadded one by about $1.2\times10^{-4}$ (reference pulse refined to 2/9 ns and
  padded from 144 to 160 samples; `test_pad_in_an_open_system_lets_t1_t2_act_on_the_padding`).

On hardware the idle samples do not advance a frame the way a simulation does, so decide
whether your toolchain wants the padding in front of or behind the gate and how the frame
is tracked.

### Amplitude bound

`env.rollout` clips every action to $\pm$`action_bound` (50 MHz for the transmon tasks).
A `Pulse` does not know that bound: samples above it are stored as given and the simulation
would silently use the clipped value. Before simulating or exporting, compare
`pulse.rabi_mhz().max()` (or `np.abs(...)`) with the bound; the test
`test_rollout_clips_actions_to_the_bound` pins the behaviour.

## Device parameters

```python
from datetime import datetime

import qenvs
from qenvs.models.transmon import Transmon
from qenvs.pulses import parse_properties

stamp = datetime.now()  # the timestamp slot of Qiskit's (value, timestamp) tuples

# (a) explicit units in the key names
m = Transmon.from_properties(
    {"frequency_ghz": 4.9, "anharmonicity_ghz": -0.33, "t1_us": 120.0, "t2_us": 85.0}
)

# (b) SI values shaped like Qiskit's BackendProperties.qubit_property(q): seconds and hertz,
#     each a bare number or a (value, timestamp) tuple
props = {"T1": (120e-6, stamp), "T2": (85e-6, stamp),
         "frequency": (4.9e9, stamp), "anharmonicity": (-0.33e9, stamp)}
m = Transmon.from_properties(props, levels=3, detuning_ghz=0.0)
# Transmon(anharmonicity_ghz=-0.33, t1_us=120.0, t2_us=85.0)

freq_ghz = parse_properties(props).frequency_ghz  # 4.9: not part of the model, keep it as metadata

# a task from the result: the registered factories take kwargs, so pass the fields on
env = qenvs.make(
    "transmon/XGate-v0",
    anharmonicity_ghz=m.anharmonicity_ghz, t1_us=m.t1_us, t2_us=m.t2_us,
)
```

Conversions are seconds to microseconds ($\times 10^6$) and hertz to gigahertz
($\times 10^{-9}$); `tests/test_pulses.py` checks them against hand values (120 us, 85 us,
4.9 GHz, -0.33 GHz from the SI numbers above) and that the environment built from the
properties has the same drift, jump operators, superoperator and fidelity as one built from
the explicit kwargs (agreement 1e-13).

* **Frequency** is accepted and not used: the model is in the rotating frame of the drive,
  so only the detuning matters. Pass it as `detuning_ghz` (qubit minus drive frequency).
* **T2 greater than 2 T1** raises `ValueError`. Measured T2 can exceed $2 T_1$ within its
  error bar on real calibration data; the message says so and suggests clipping
  (`min(t2, 2 * t1)`) rather than doing it silently.
* **Missing anharmonicity** keeps the model default (-0.34 GHz) and emits a `UserWarning`.
  A positive anharmonicity also warns (the sign is kept). A missing `t1_us` or `t2_us`
  means no dissipation from that source: only T1 gives $T_2 = 2T_1$, only T2 gives
  $T_1 = \infty$ (see the [transmon page](../models/transmon.md)).
* **Ambiguity is an error.** The same quantity in both forms (`t1_us` and `T1`), or an
  unrecognised unit suffix (`t1_ms`, `frequency_mhz`), raises instead of guessing units.
  Unrelated keys, such as readout errors, are ignored.
* **Qiskit.** `BackendProperties.qubit_property` stores SI values (the library converts the
  calibration units with SI prefixes; verified in the Qiskit 1.4 source,
  `qiskit/providers/models/backendproperties.py` and `qiskit/utils/units.py`). That class
  belongs to `BackendV1`, which Qiskit 2.0 removed (2.0 release notes), so with a current
  Qiskit read the numbers from the backend target and put them into a dict yourself;
  nothing here depends on the Qiskit version.

## What QEnvS does not do

* **No AWG transfer function.** The pulse is the ideal envelope of the model. Bandwidth,
  rise time, sample interpolation of the DAC, mixer imbalance, skew and distortion of the
  line are not modelled and not corrected.
* **No amplitude calibration.** Rad/ns and MHz are the model's units; the mapping from a
  Rabi frequency to a DAC amplitude (volts, or a fraction of full scale) is a measurement
  on your device. QEnvS never produces a calibrated amplitude.
* **No hardware transfer and no Qiskit dependency.** The library writes JSON and NPZ files
  and imports no Qiskit, Qiskit Pulse or vendor package. The I/Q convention is the one of
  the transmon page: lab-frame drive $\Omega_I\cos\omega_d t + \Omega_Q\sin\omega_d t$
  (our own derivation there, not compared with a vendor stack).
  Vendor tools may define Q with the opposite sign, or use a different frame; check one
  known pulse (a pi pulse on I) before trusting a long sequence.
* **No virtual Z, no frame tracking, no readout.** Fidelities are against the stated gate
  without frame updates.
* **Resampling only refines.** It does not coarsen, interpolate or filter.
* **Metadata is JSON.** Anything else has to be converted by the caller; NaN and infinity
  are rejected because JSON cannot represent them.

## Validation

All tests are in `tests/test_pulses.py` (30 tests, about 50 s on CPU), including a float32
subprocess run of the resampling gate. Each formula above is exercised numerically: the
MHz conversion by hand values (25 MHz for a 20 ns pi pulse, 50 MHz for the default action
bound) and by simulation (25 MHz held for 20 ns gives $F = 1$ on a two-level transmon,
20 MHz does not), the SI conversions by hand values, the resampling by the table above and
the padding statement by the 3-level and detuned runs.
