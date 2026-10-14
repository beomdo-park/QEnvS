# Open two-level quantum battery

Task id: `battery/TwoLevel-v0`. Model: `qenvs.models.qubit_battery.QubitBattery`.
Environment: `qenvs.envs.BatteryCharging` (shared with every model that defines a
battery Hamiltonian).

The smallest quantum battery: one qubit whose excited state stores one quantum
of energy $\omega_0$. A classical drive charges it, a thermal bath at finite
temperature discharges (and partially charges) it, and pure dephasing destroys
the coherence that makes the stored energy extractable as work (ergotropy).

## Physics

Rotating-frame Hamiltonian ($\hbar = 1$):

$$
H(t) = \frac{\Delta}{2}\,\sigma_z + \frac{u(t)}{2}\,\sigma_x ,
\qquad
H_B = \omega_0\,|1\rangle\langle 1| .
$$

$H_B$ is the battery Hamiltonian used for the stored energy
$E = \operatorname{tr}(\rho H_B)$ and the ergotropy
$\mathcal{E} = \operatorname{tr}(\rho H_B) - \min_U \operatorname{tr}(U\rho U^\dagger H_B)$
(Allahverdyan, Balian and Nieuwenhuizen 2004). For a qubit with eigenvalues
$r_\pm$ of $\rho$ ($r_+ \ge r_-$) this is $\mathcal{E} = \omega_0 (p_1 - r_-)$.

Lindblad master equation
$\dot\rho = -i[H(t), \rho] + \sum_k \mathcal{D}[L_k]\rho$ with the thermal
qubit dissipator of Breuer and Petruccione (2002, Sec. 3.4) at bath Bose
occupation $n_{\mathrm{th}}$ and coupling rate $\gamma_1$, plus pure dephasing:

$$
L_\downarrow = \sqrt{\gamma_1 (n_{\mathrm{th}} + 1)}\;|0\rangle\langle 1| ,\qquad
L_\uparrow = \sqrt{\gamma_1 n_{\mathrm{th}}}\;|1\rangle\langle 0| ,\qquad
L_\phi = \sqrt{\gamma_\phi / 2}\;\sigma_z .
$$

Without drive the populations obey the rate equation
$\dot p_1 = \gamma_1 n_{\mathrm{th}}\,p_0 - \gamma_1 (n_{\mathrm{th}} + 1)\,p_1$,
so $p_1(t)$ relaxes at rate $\gamma_1 (2 n_{\mathrm{th}} + 1)$ to the Gibbs value

$$
p_1^{\mathrm{th}} = \frac{n_{\mathrm{th}}}{2 n_{\mathrm{th}} + 1},
\qquad
\frac{p_1^{\mathrm{th}}}{p_0^{\mathrm{th}}} = \frac{n_{\mathrm{th}}}{n_{\mathrm{th}} + 1} = e^{-\beta \omega_0}.
$$

Coherences decay at $\gamma_1 (2 n_{\mathrm{th}} + 1)/2 + \gamma_\phi$. With
$n_{\mathrm{th}} = 0$ the model reduces exactly to `TwoLevel` with the same
$\gamma_1$ and $\gamma_\phi$.

Closed, resonant, constant drive $u$ from $|0\rangle$:
$p_1(t) = \sin^2(u t / 2)$, hence $E(t) = \omega_0 \sin^2(u t / 2)$, a
$\pi$-pulse ($\int u\,dt = \pi$) stores $E = \mathcal{E} = \omega_0$, and the
instantaneous charging power is bounded by $|dE/dt| \le \omega_0 u / 2$, with
equality at $u t = \pi/2$. Detuned: $p_1(t) = (u/\Omega)^2 \sin^2(\Omega t/2)$,
$\Omega = \sqrt{u^2 + \Delta^2}$.

## Basis and index conventions

| Index | State | $H_B$ energy |
|---|---|---|
| 0 | ground $\lvert 0\rangle$ | 0 |
| 1 | excited $\lvert 1\rangle$ | $\omega_0$ |

$\sigma_z = \lvert 0\rangle\langle 0\rvert - \lvert 1\rangle\langle 1\rvert = \mathrm{diag}(1, -1)$,
so $\Delta > 0$ lowers the excited level in the rotating frame (same convention
as `TwoLevel`). The emission operator is built explicitly as
`dq.basis(2, 0) @ dq.basis(2, 1).dag()`; `dynamiqs.sigmam()` follows the QuTiP
spin convention (index 0 = up) and would raise instead of lower here.

Closed models (`gamma1 = gamma_phi = 0`) carry kets and are integrated with
`sesolve`; open models carry density matrices and use `mesolve`. The observation
is the flattened real and imaginary parts of the state plus $t / T$: 5 numbers
closed, 9 open.

## Assumptions and validity

* Rotating-wave approximation: $H(t)$ is the rotating-frame Hamiltonian; the drive
  frequency is absorbed into $\Delta$ and the counter-rotating term is dropped.
  Valid for $|u|, |\Delta| \ll \omega_0$. Note that the task default
  `action_bound = 2.0` (in units of $\omega_0$) and the pi-pulse example sit outside
  this regime: in these units the results are the formal RWA solution, and
  Bloch-Siegert shifts and counter-rotating effects are not modelled. Lower
  `action_bound` (or raise `omega0`) when physical fidelity to a lab qubit matters.
* Markovian (Lindblad) bath with flat spectral density around $\omega_0$; the
  dissipator is the one of the undriven qubit and is applied unchanged while the
  drive is on (no dressed-state / Floquet master equation). Valid for
  $\gamma_1, \gamma_\phi \ll \omega_0$ and weak drive.
* Dephasing is modelled as a white-noise $\sigma_z$ channel.
* `initial='thermal'` is the Gibbs state of $H_B$ at the bath occupation, i.e. the
  battery starts in equilibrium with its bath (zero ergotropy).

## Parameters and units

Energies in units of $\omega_0$ (default 1), time in units of $1/\omega_0$, rates
in units of $\omega_0$, $\hbar = 1$.

| Field | Default | Meaning |
|---|---|---|
| `omega0` | 1.0 | battery level splitting, energy unit of $H_B$ |
| `detuning` | 0.0 | $\Delta$, drive-qubit detuning |
| `gamma1` | 0.0 | bath coupling rate; emission $\gamma_1 (n_{\mathrm{th}}+1)$, absorption $\gamma_1 n_{\mathrm{th}}$ |
| `n_th` | 0.0 | bath Bose occupation at $\omega_0$ (0 = zero temperature) |
| `gamma_phi` | 0.0 | pure dephasing rate |
| `initial` | `'ground'` | `'ground'` ($\lvert 0\rangle$) or `'thermal'` (Gibbs state, needs `gamma1 > 0` or `gamma_phi > 0` when `n_th > 0`) |

Negative rates, unknown `initial` values, and a mixed thermal state in a closed
model raise `ValueError` at construction.

## Control semantics and bounds

One control channel: the action $u$ multiplies $\sigma_x / 2$, so $u$ is the
Rabi frequency. Actions are held constant for one control period `dt`
(piecewise-constant pulse) and clipped to $[-\text{action\_bound}, \text{action\_bound}]$
(default 2.0). The ODE integrator (adaptive Tsit5 by default) takes as many
internal steps as it needs inside each `dt`.

## Reward

`reward='energy'` (default): per-step gain of stored energy
$r_k = E(t_{k+1}) - E(t_k)$. `reward='ergotropy'`: per-step gain of ergotropy.
The undiscounted return telescopes to $Q(T) - Q(0)$. `info` reports `energy`,
`ergotropy`, `p1`, average power $E(t)/t$ and purity every step.

```python
import jax.numpy as jnp
import qenvs

env = qenvs.make("battery/TwoLevel-v0", horizon=10, dt=0.2, gamma1=0.05, n_th=0.2)
actions = jnp.full((env.horizon, env.action_dim), jnp.pi / (env.horizon * env.dt))
state, rewards = env.rollout(actions)  # pi-pulse; sum(rewards) = E_final - E_initial
```

## Validation

All checks live in `qenvs.validation.qubit_battery.CHECKS`, run in
`tests/test_qubit_battery.py` and are printed by `qenvs validate`. Measured
errors below are from one run on CPU with x64 enabled (dynamiqs 0.3.6,
jax 0.7.2, qutip 5.3.1); every check was also run in float32 and passed. The
three closed-system analytic checks integrate with `Tsit5(rtol=atol=1e-8)` so
the error reflects the model rather than the integrator; the default
`Tsit5()` settings are validated by `solver_convergence`.

| Check | Reference | Tolerance | Measured error (x64) |
|---|---|---|---|
| `thermal_steady_state` | zero drive from $\lvert 0\rangle$, $\gamma_1 = 0.8$, $n_{\mathrm{th}} = 0.6$, $\gamma_\phi = 0.3$, $t = 30$: $p_1 \to n_{\mathrm{th}}/(2 n_{\mathrm{th}}+1)$ | 1e-5 | 8.0e-7 |
| `thermal_state_stationary` | Gibbs initial state, zero drive, $t \le 10$: $\max_t \lVert \rho(t) - \rho(0) \rVert_{\max}$ | 1e-6 | 7.8e-15 |
| `pi_pulse_charging` | closed, $\omega_0 = 1.3$, $u t = \pi$: $E = \mathcal{E} = \omega_0$ | 1e-6 | 4.7e-9 |
| `ergotropy_le_energy` | 20 random open steps: $0 \le \mathcal{E} \le E$ (max violation) | 1e-7 | 0 |
| `charging_power_bound` | closed, $u = 1.4$, $\omega_0 = 0.9$: $\lvert \max_t \lvert dE/dt \rvert - \omega_0 u/2 \rvert$, 4th-order finite differences on 201 points | 1e-4 | 9.1e-8 |
| `env_return_equals_gain` | random open rollout: $\lvert \sum_k r_k - (Q_T - Q_0) \rvert$ for energy and ergotropy | 1e-6 | 0 |
| `qutip_cross_check` | QuTiP 5 `mesolve`, same $H(t)$ (12 random piecewise-constant steps, $\Delta = 0.3$) and jump operators ($\gamma_1 = 0.4$, $n_{\mathrm{th}} = 0.6$, $\gamma_\phi = 0.15$): $\max \lvert \rho_{\mathrm{dq}} - \rho_{\mathrm{qt}} \rvert$ | 1e-6 | 6.6e-8 |
| `density_matrix_constraints` | random open rollout: hermiticity, $\lvert \operatorname{tr}\rho - 1 \rvert$, negativity | 1e-6 | 1.1e-16 |
| `t1_decay_matches_two_level` | $n_{\mathrm{th}} = 0$, from $\lvert 1\rangle$: $p_1 = e^{-\gamma_1 t}$ and equal to `TwoLevel` | 1e-6 | 3.4e-7 |
| `detuned_rabi` | closed, $u = 1.2$, $\Delta = 0.9$: $p_1 = (u/\Omega)^2 \sin^2(\Omega t/2)$ | 1e-6 | 5.4e-9 |
| `thermal_state_passive` | Gibbs state: trace 1, $E = \omega_0 p_1^{\mathrm{th}}$, $\mathcal{E} = 0$ | 1e-6 | 0 |
| `solver_convergence` | open constant-drive rollout: default `Tsit5()` vs exact `Expm()` | 1e-6 | 2.9e-8 |

Discrimination: the checks were run against deliberately broken models (control
sign flipped, control halved or doubled, emission and absorption swapped,
`dq.sigmam()` used for emission, dephasing rate doubled, detuning sign flipped or
doubled, wrong Gibbs population, absorption channel dropped). Every mutant failed
at least one check; the QuTiP cross-check alone catches the control-sign,
detuning-sign and dephasing-rate errors, which population-only checks cannot see.

## References

1. A. E. Allahverdyan, R. Balian and Th. M. Nieuwenhuizen, "Maximal work
   extraction from finite quantum systems", EPL 67, 565 (2004). Ergotropy.
2. R. Alicki and M. Fannes, "Entanglement boost for extractable work from
   ensembles of quantum batteries", Phys. Rev. E 87, 042123 (2013). Quantum
   batteries, passive states.
3. H.-P. Breuer and F. Petruccione, *The Theory of Open Quantum Systems*
   (Oxford University Press, 2002), Sec. 3.4. Thermal qubit master equation,
   Bose occupation and the $n_{\mathrm{th}} + 1$ / $n_{\mathrm{th}}$ rates.
4. I. I. Rabi, "Space quantization in a gyrating magnetic field", Phys. Rev.
   51, 652 (1937). Rabi formula used for the closed-system checks.

## Known limitations

* The bath dissipator is the undriven one; for strong drives the physical
  dissipator becomes drive-dependent (Floquet or dressed-state master equation),
  which this model does not include.
* `initial='thermal'` with `n_th > 0` requires an open model, since a mixed
  initial state needs the density-matrix integrator; a closed unitary charge
  from a thermal state is therefore not available in this version.
* The environment casts actions to float32 before building $H(t)$; the QuTiP
  cross-check uses the same rounded amplitudes.
* `qutip_cross_check` needs the `qenvs[dev]` extra. Without QuTiP it is reported as
  skipped (status `SKIP`) with a "not run" detail rather than raising, so
  `qenvs validate` still runs; a skipped check never counts as passed.
* Only one qubit: no collective effects. See `battery/Dicke-v0` for $N$-qubit
  charging.
