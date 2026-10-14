# Tutorial: an X gate on a transmon

This tutorial walks through the daily task of a pulse engineer on the
`transmon/XGate-v0` task: start from a Gaussian pi pulse, add DRAG, optimise with GRAPE,
then look at the same pulses with decoherence and export the result. All numbers on this
page come from `examples/tutorial_transmon_x_gate.py` (float64, CPU, about 15 s), which
`tests/test_tutorials.py` also runs in a reduced form.

## The task

```python
import jax

jax.config.update("jax_enable_x64", True)  # errors of interest are 1e-6 to 1e-4
import jax.numpy as jnp
import numpy as np
import qenvs

env = qenvs.make("transmon/XGate-v0")
```

| setting | value |
|---|---|
| model | three-level transmon, anharmonicity $\alpha/2\pi = -0.34$ GHz, rotating frame of the drive, RWA |
| controls | $[\Omega_I, \Omega_Q]$ in rad/ns, $|\Omega|/2\pi \le 50$ MHz per quadrature |
| time grid | 16 steps of 2 ns, a 32 ns gate |
| objective | gain of the leakage-aware average gate fidelity to X on $\{|0\rangle, |1\rangle\}$ |

See the [transmon model page](../models/transmon.md) for the Hamiltonian, the sign
conventions and the validation table.

## 1. A Gaussian pi pulse

```python
from qenvs.models.transmon import gaussian_amplitude, gaussian_drag

T = env.horizon * env.dt
sigma = T / 4
amp = gaussian_amplitude(jnp.pi, env.horizon, env.dt, sigma)  # area pi
pulse = gaussian_drag(env.horizon, env.dt, amp, sigma, beta=0.0)
state, rewards = env.rollout(pulse)
info = env.info(state)  # fidelity, leakage, populations p0, p1, p2
```

The plain Gaussian reaches $1 - F = 9.07\times10^{-4}$ with leakage
$2.4\times10^{-6}$: on a 32 ns gate the dominant error is not leakage but the phase error
that the second excited level induces on the 0-1 transition.

## 2. DRAG

The quadrature $\Omega_Q = \beta\,\dot\Omega_I$ cancels that error to first order. In this
library's sign convention the coefficients are

```python
from qenvs.models.transmon import drag_beta

drag_beta(-0.34, "phase")  # 0.234 ns  = -1/(2 alpha)
drag_beta(-0.34, "leakage")  # 0.468 ns  = -1/alpha
```

With the phase-error value the infidelity drops by a factor 187:

| $\beta$ (ns) | 0 | 0.059 | 0.117 | 0.176 | **0.234** | 0.293 | 0.351 | 0.410 | 0.468 |
|---|---|---|---|---|---|---|---|---|---|
| $1 - F$ | 9.07e-4 | 5.10e-4 | 2.28e-4 | 5.94e-5 | **4.85e-6** | 6.43e-5 | 2.38e-4 | 5.25e-4 | 9.25e-4 |

The minimum of the sweep sits exactly at the first-order value. The leakage-optimal value
($-1/\alpha$) is not the best choice for gate fidelity here, because at this gate length
leakage is already small.

## 3. GRAPE

The environment is differentiable, so a gradient-based optimiser can shape the pulse
directly. `examples/grape_jax.py` parametrises the 32 control amplitudes and runs Adam on
the return of `env.rollout`:

```python
from grape_jax import run_grape  # examples/ on sys.path

g = run_grape("transmon/XGate-v0", steps=300, lr=0.1, seed=0)
pulse = jnp.asarray(g["best_pulse"])  # (16, 2) in rad/ns
```

From a random start it passes $F > 0.999$ within 30 Adam steps and ends at
$1 - F = 3.7\times10^{-15}$ with leakage $3.6\times10^{-15}$, about 2 s on CPU. A closed
three-level model has no fundamental limit at this gate length, so this is a check of the
simulator and the optimiser rather than a statement about hardware.

## 4. Decoherence

Passing coherence times switches the environment to the superoperator of the Lindblad
evolution:

```python
env_open = qenvs.make("transmon/XGate-v0", t1_us=100.0, t2_us=100.0)
```

| pulse | $1 - F$ with $T_1 = T_2 = 100\ \mu$s |
|---|---|
| idle for 32 ns, measured against the identity | 1.600e-4 |
| DRAG ($\beta = 0.234$ ns) | 1.652e-4 |
| GRAPE pulse optimised without decoherence | 1.674e-4 |
| GRAPE optimised on the open task (300 steps) | 1.673e-4 |

The idle value is a decoherence reference, not a strict bound: a driven gate spends time
in $|1\rangle$, where it relaxes. The table shows the practical conclusion for these
parameters: the gate error is dominated by coherence, and pulse shaping changes it only at
the $10^{-6}$ level. GRAPE from a random start does not beat the analytic DRAG pulse
here within 300 steps; it leaves more population in $|0\rangle$ ($2.0\times10^{-4}$
against $1.6\times10^{-4}$) and slightly more in $|2\rangle$.

## 5. Export

The pulse is a list of piecewise-constant Rabi frequencies on a 2 ns grid. In MHz:

```python
t_ns = (np.arange(env.horizon) + 0.5) * env.dt
omega_mhz = np.asarray(pulse) / (2 * np.pi) * 1e3  # columns I, Q
```

`python examples/tutorial_transmon_x_gate.py --out pulse.npz` writes `t_ns`,
`omega_i_mhz` and `omega_q_mhz`. The GRAPE pulse peaks at 48.8 MHz. Mapping Rabi
frequencies to AWG amplitudes needs a calibration of your hardware, which QEnvS does not
model.

## 6. Reinforcement learning

The same task is a Gymnasium environment:

```python
from qenvs.wrappers import GymnasiumEnv

gym_env = GymnasiumEnv(env, normalize_actions=True)  # actions in [-1, 1]
```

The observation has 19 entries (real and imaginary parts of the $3\times3$ propagator and
the elapsed time fraction); an episode ends after 16 steps. A random policy reaches
$F \approx 0.6$. See the [RL libraries guide](../guides/rl-libraries.md) for vectorised
environments and training loops.

## What this model leaves out

Single qubit only (no couplers or cross-resonance), rotating-wave approximation, a
harmonic $T_1$ ladder, piecewise-constant controls without an AWG transfer function, and
no virtual-Z frame updates (Z errors count as infidelity, whereas hardware calibrations
usually absorb them in software).
