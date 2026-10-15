# Uncertainty: randomised parameters and control errors

A real device is never exactly the model you trained on. Its frequencies drift between
calibrations, its couplings differ from the design values, and its control electronics
miscalibrate and jitter the amplitudes you ask for. QEnvS lets you put these effects around any
task, so that a policy is trained and judged on a distribution of devices instead of one ideal
model. Two wrappers do it:

| Wrapper | What varies | When it is drawn |
|---|---|---|
| `ParameterRandomization` | Hamiltonian parameters of the model (detunings, anharmonicities, couplings, ...) | once per episode, fixed during the episode |
| `ControlErrors` | gain error and offset of the control amplitudes (fixed per episode), per-step noise (every step) | at `reset` and at every `step` |

Both are ordinary environments: they have the same `reset(key)` and `step(state, action)` API,
work under `jit` and `vmap`, and nest in any order. The defaults reproduce the released task
exactly: with every width at zero the wrapped environment gives bit-identical observations,
rewards and states (`np.array_equal`, x64) to the environment it wraps.

## A complete example

```python
import jax
import jax.numpy as jnp

import qenvs
from qenvs.distributions import Normal, TruncatedNormal
from qenvs.wrappers.control import ControlErrors
from qenvs.wrappers.randomization import ParameterRandomization

env = qenvs.make("transmon/XGate-v0")

# a different device in every episode: qubit frequency drift and anharmonicity spread
renv = ParameterRandomization(
    env,
    {
        "detuning_ghz": Normal(std=0.002, unit="GHz"),
        "anharmonicity_ghz": TruncatedNormal(std=0.005, low=-0.02, high=0.02, unit="GHz"),
    },
    strength=1.0,
    oracle=False,
)

# a miscalibrated, noisy drive on top of it
cenv = ControlErrors(renv, amplitude_scale=Normal(0.02), step_noise=Normal(0.01))

# the analytic baseline pulse of the task, evaluated on 256 random devices
actions = jnp.asarray(qenvs.registry["transmon/XGate-v0"].reference(env))
run_key = jax.random.PRNGKey(0)
episode_keys = jax.vmap(lambda i: jax.random.fold_in(run_key, i))(jnp.arange(256))


def episode(key):
    state, ts = cenv.reset(key)
    clipped = jnp.zeros(())
    for a in actions:  # 16 control steps
        state, ts = cenv.step(state, a)
        clipped = jnp.maximum(clipped, ts.info["clip_fraction"])
    return cenv.info(state)["fidelity"], clipped  # exact simulator value, evaluation only


fidelity, clipped = jax.jit(jax.vmap(episode))(episode_keys)
print(f"mean F = {fidelity.mean():.6f}, worst F = {fidelity.min():.6f}")
print(f"largest clip fraction = {float(clipped.max()):.2f}")
```

`cenv.info(state)` returns the exact simulator quantities (fidelity, leakage, ...). It is meant
for evaluation and debugging; do not feed it to a policy, whose input is the observation only.

## Parameter randomisation

```text
ParameterRandomization(env, distributions, strength=1.0, oracle=False)
```

`distributions` maps a parameter name to an offset distribution. The value used in the episode
is `nominal + offset`, in the unit of the parameter. The model declares which parameters can be
randomised, with a unit, a support (the closed interval of physically valid values) and a
reference width `w_ref` (a typical spread, used only to normalise the oracle observation):

| Model | Parameter | Unit | Support | `w_ref` |
|---|---|---|---|---|
| `Transmon` (X, SX gates) | `detuning_ghz` | GHz | whole line | 0.002 |
| | `anharmonicity_ghz` | GHz | `(-inf, 0]` | 0.005 |
| `TwoTransmon` (CR gate) | `anharmonicity_c_ghz`, `anharmonicity_t_ghz` | GHz | `(-inf, 0]` | 0.005 |
| | `coupling_ghz` | GHz | `[0, inf)` | 0.0003 |
| `TwoLevel`, `QubitBattery` | `detuning` | omega0 | whole line | 0.05 |
| `LambdaSystem` | `delta_one_photon` | Rabi frequency | whole line | 0.05 |
| | `delta_two_photon` | Rabi frequency | whole line | 0.01 |
| `DickeBattery`, `DickeBatteryLocal` | `omega_c` | omega0 | `[0, inf)` | 0.05 |

`model.parameter_specs()` returns the same information in code. Dissipation rates (`t1_us`,
`gamma1`, `kappa`, ...) and structural sizes (`levels`, `n_cells`, ...) are not randomisable: the
closed or open path and the array shapes depend on them. Naming one raises an error that says
why.

Three distributions are available. `strength` multiplies the configured width (so you can scale
a whole experiment, and `strength=0` gives the nominal model exactly):

| Distribution | Configured width | Scaled by `strength` | Checked at construction |
|---|---|---|---|
| `Normal(std)` | `std` | `std` | the parameter's support must be the whole real line |
| `Uniform(half_width)` | `half_width` | `half_width` | `[nominal - h, nominal + h]` inside the support |
| `TruncatedNormal(std, low, high)` | `std` | `std` only | `low <= 0 <= high`, `low < high`, `[nominal + low, nominal + high]` inside the support |

`low` and `high` of a `TruncatedNormal` are offsets from the nominal value and are not scaled by
`strength`. When they are not symmetric the mean offset is not zero; `dist.mean()` and
`dist.stddev()` give the exact analytic moments. A transmon anharmonicity is negative, so a
`Normal` on it is an error and you use a `TruncatedNormal` (or a `Uniform`). Samples are never
clipped into the support, because that would change the distribution you declared; instead a
distribution that could leave the support is rejected when you construct the wrapper.

A distribution may state its `unit`; it must then equal the unit the model declares, which
catches a standard deviation given in rad/ns for a parameter in GHz.

### The oracle observation

With `oracle=True` the observation gets `(value - nominal) / w_ref` appended for every randomised
parameter whose `w_ref` is positive, in the order of your mapping. This tells the policy the
sampled parameters. A real device would have to estimate them with a separate calibration, so
read results with `oracle=True` as an upper bound on what a policy can do (an oracle policy), not
as a model of a measurement. The scale `w_ref` belongs to the model and does not depend on
`strength` or on the configured widths, so training and evaluation distributions are read on the
same scale and `strength=0` does not divide by zero.

### Your own Hamiltonian

`HamiltonianModel` has no named parameters, so you declare them as parameter terms:

```python
import numpy as np
from qenvs.models.custom import HamiltonianModel

sx = np.array([[0, 1], [1, 0]])
sz = np.diag([1.0, -1.0])
model = HamiltonianModel.from_operators(
    drift=0.05 * sz,
    controls=[sx / 2],
    parameters={
        "J": (0.3, sx, "a.u.", (-np.inf, np.inf), 0.05)
    },  # nominal, operator, unit, support, w_ref
)
```

The drift is then `H_0 + J * sx`, with `J` the randomisable parameter. The operator must be
Hermitian (checked at construction). Without `parameters` the drift is exactly `H_0`.

A model class of your own joins the same contract by implementing `parameter_specs()` (name to
`ParameterSpec(unit, support, w_ref)`) and, when a parameter is not a field of the model,
`with_parameters(**values)` (a pure function that works under `jit` and `vmap`). The default
`with_parameters` replaces the fields of the same name.

### Checking a model contract

```python
transmon = qenvs.make("transmon/XGate-v0").model
transmon.with_parameters(detuning_ghz=0.001)  # pure; works inside jit and vmap
transmon.check_parameters(detuning_ghz=0.001)  # concrete values: support and Hermiticity
```

`check_parameters` raises `ValueError` for a value outside the support, a non-finite or complex
value, an unknown name, or a drift that is not Hermitian.

### Things to keep in mind

* Outside the parameters `ParameterRandomization` supports in 0.3 (a limit of the current API and
  objective, not a claim that the parameter cannot vary physically); naming one raises an error
  with the reason:
  * `omega0` of `DickeBattery` and `DickeBatteryLocal`. It is also the energy unit of the battery
    Hamiltonian, so a sampled `omega0` rescales the reward and the evaluation metric of its episode
    (a change of units, several times larger than its effect on the charging dynamics), which a
    robustness score would mistake for a physical effect. Randomise `omega_c` for a cell-cavity
    detuning at a fixed energy scale. Supporting `omega0` needs a scale-normalised objective (for
    example energy over `N omega0`) applied the same way to every battery model. `TwoLevel` and
    `QubitBattery` refuse `omega0` for the same reason.
  * `detuning_ghz` of `TwoTransmon` (`transmon/CRGate-v0`). The target is the ZX gate pulled back
    with the nominal detuning, so a sampled detuning adds a control Z phase that no pulse can
    correct. Supporting it needs the target, and any virtual-Z frame update, recomputed from the
    episode model; that is planned as separate work. The single-transmon `detuning_ghz` (X, SX
    gates) is supported.
* For `levels = 2` the transmon anharmonicity term vanishes, so randomising it changes nothing.

## Control errors

```text
ControlErrors(
    env, amplitude_scale=Normal(0.02), offset=None, step_noise=Normal(0.01), correlated=False
)
```

At step `k` and channel `c` the amplitude sent to the environment is

$$
u = \mathrm{clip}\big((a\,(1+\epsilon) + o)\,(1+\xi_k),\ -B,\ B\big),
$$

with the commanded amplitude `a`, a relative gain error `epsilon` and an absolute offset `o`
(both fixed for the episode), a relative noise `xi_k` redrawn every step, and the bound
`B = env.action_bound`. A source that you leave as `None` is switched off. This is an explicitly
defined simulation model, not a claim about the order in which every device produces its errors.

* The relative errors `epsilon` and `xi` have the whole real line as support: a gain below -1
  inverts the pulse, and the simulation is defined for it. Use a `TruncatedNormal` to force a
  positive gain.
* By default every channel draws its own value; `correlated=True` draws one value per source and
  shares it between channels (for example a common gain error of the signal generator). It
  applies to every source that is switched on.
* Noise followed by clipping piles probability up at the bounds, so every `reset` and `step`
  reports `info["clip_fraction"]`, the share of channels clipped at `+-B` after the errors were
  applied (0 at `reset`). Report it with your results.

## Reproducibility and the key tree

All random numbers of an episode derive from the episode key alone, through a fixed tree
(`qenvs.keys`): one leaf per randomised parameter (named by the CRC32 of its name), one for the
gain error, one for the offset, and one per step for the noise. Changing how one leaf is used
(another distribution, switching it on or off, `correlated`, adding or removing a parameter)
never changes the samples of another leaf. The wrappers pass the episode key unchanged to the
wrapped environment, so nesting them in any order gives the same samples, and the episode with
index `i` of a run is always `jax.random.fold_in(run_key, i)`, so any episode can be regenerated
from the run key and its number.

Within the same execution mode (both eager or both jitted) a key gives identical results.
Between a single run and a batched run the random samples are identical; the simulated
observations and rewards agree to about 1e-12 in float64, because XLA may order floating-point
operations differently. An eager and a jitted program can differ in the last bit of draws that
use `erf_inv` (Gaussian samples) because XLA fuses the scaling.

## Validation

The control-error wrapper is checked against exact closed forms. For a resonant pi pulse of
constant amplitude in the two-level limit, a relative amplitude error `epsilon` rotates by
`pi (1 + epsilon)`, so `1 - F_state = sin^2(pi epsilon / 2)` and, for the gate task,
`1 - F_avg = (2/3) sin^2(pi epsilon / 2)`. For a Gaussian `epsilon` of standard deviation `sigma`
the expectation is `E[1 - F_avg] = (1 - exp(-pi^2 sigma^2 / 2)) / 3`. The tests reproduce these
values on `transmon/XGate-v0` (`levels=2`) and `qubit/StateTransfer-v0` to 1e-10 at fixed
errors, and the sample means over 4096 episodes lie within four standard errors of them.
Per-step noise of a constant pulse split into `H` equal steps is covered by the same formula
with `sigma = s / sqrt(H)`; no such formula is claimed for pulses whose amplitude varies from
step to step.
