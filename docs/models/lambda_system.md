# Three-level Lambda system (STIRAP)

Task id: `transfer/LambdaSTIRAP-v0`. Model: `qenvs.models.lambda_system.LambdaSystem`.
Environment: `qenvs.envs.StateTransfer` (shared with `qubit/StateTransfer-v0`), target
$\lvert 3\rangle$. Reference pulses: `qenvs.models.lambda_system.stirap_pulses`.

Two ground states $\lvert 1\rangle$ and $\lvert 3\rangle$ are coupled to a common
excited state $\lvert 2\rangle$ by a pump and a Stokes field. The task is to move the
population from $\lvert 1\rangle$ to $\lvert 3\rangle$ without populating
$\lvert 2\rangle$, which decays by spontaneous emission. The textbook solution is
stimulated Raman adiabatic passage (STIRAP): apply the Stokes pulse *before* the pump
pulse (counter-intuitive order) and change the fields slowly enough that the system
follows the dark state (Bergmann, Theuer and Shore 1998; Vitanov et al. 2017).

## Physics

Rotating-frame Hamiltonian under the rotating-wave approximation ($\hbar = 1$):

$$
H(t) = \Delta\,\lvert 2\rangle\langle 2\rvert + \delta\,\lvert 3\rangle\langle 3\rvert
+ \frac{\Omega_p(t)}{2}\bigl(\lvert 1\rangle\langle 2\rvert + \lvert 2\rangle\langle 1\rvert\bigr)
+ \frac{\Omega_s(t)}{2}\bigl(\lvert 3\rangle\langle 2\rvert + \lvert 2\rangle\langle 3\rvert\bigr),
$$

with $\Delta$ the one-photon (pump) detuning, $\delta$ the two-photon detuning and
$\Omega_p$, $\Omega_s$ the real pump and Stokes Rabi frequencies (the two controls).
In matrix form, in the basis $(\lvert 1\rangle, \lvert 2\rangle, \lvert 3\rangle)$,

$$
H = \frac{1}{2}\begin{pmatrix} 0 & \Omega_p & 0 \\ \Omega_p & 2\Delta & \Omega_s \\ 0 & \Omega_s & 2\delta \end{pmatrix}.
$$

**Adiabatic states.** For $\delta = 0$ and any $\Delta$, with the mixing angle
$\tan\theta = \Omega_p / \Omega_s$, the dark state

$$
\lvert D\rangle = \cos\theta\,\lvert 1\rangle - \sin\theta\,\lvert 3\rangle
\propto \Omega_s\,\lvert 1\rangle - \Omega_p\,\lvert 3\rangle
$$

is an exact eigenstate of $H$ with eigenvalue $0$ and no component on
$\lvert 2\rangle$. The other two eigenvalues are
$\bigl(\Delta \pm \sqrt{\Delta^2 + \Omega_p^2 + \Omega_s^2}\bigr)/2$ (bright states,
which do contain $\lvert 2\rangle$). STIRAP works because $\theta$ rotates from $0$
(Stokes only: $\lvert D\rangle = \lvert 1\rangle$) to $\pi/2$ (pump only:
$\lvert D\rangle = -\lvert 3\rangle$) when the Stokes pulse precedes the pump pulse;
if the change is adiabatic, $\lvert\dot\theta\rvert \ll \Omega_{\mathrm{rms}} / 2$ with
$\Omega_{\mathrm{rms}} = \sqrt{\Omega_p^2 + \Omega_s^2}$, the population is carried
from $\lvert 1\rangle$ to $\lvert 3\rangle$ inside the dark state and the lossy state
is never populated. For Gaussian pulses of peak $\Omega_0$ and $1/e$ half-width $T_w$
delayed by $2\tau$, $\dot\theta$ peaks at $2\tau / T_w^2$ in the pulse overlap, and the
global adiabaticity criterion is $\Omega_0 T_w \gtrsim 10$ (Vitanov et al. 2017).

**Intuitive order** (pump first) starts the system in an equal superposition of the two
bright states instead of the dark state. The state then evolves as
$\cos\varphi(t)\,(\sin\theta\lvert 1\rangle + \cos\theta\lvert 3\rangle) - i\sin\varphi(t)\lvert 2\rangle$
with $\varphi(t) = \tfrac12\int_0^t \Omega_{\mathrm{rms}}\,dt'$: the excited-state
population $P_2 = \sin^2\varphi$ reaches 1 and the final $P_3 = \cos^2\varphi(T)$
depends on the pulse area, not on adiabaticity. Only the counter-intuitive order gives
a robust, dissipation-insensitive transfer.

**Dissipation.** Spontaneous emission from $\lvert 2\rangle$ at total rate $\gamma$
enters the Lindblad master equation
$\dot\rho = -i[H(t), \rho] + \sum_k \bigl(L_k \rho L_k^\dagger - \tfrac12\{L_k^\dagger L_k, \rho\}\bigr)$
through two jump operators with *equal branching* into the two ground states,

$$
L_1 = \sqrt{\gamma / 2}\;\lvert 1\rangle\langle 2\rvert,\qquad
L_3 = \sqrt{\gamma / 2}\;\lvert 3\rangle\langle 2\rvert .
$$

Without drive, $P_2(t) = e^{-\gamma t}$ and $P_1(t) = P_3(t) = (1 - e^{-\gamma t})/2$
from $\lvert 2\rangle$. In a real atom the branching ratio is set by the dipole matrix
elements and part of the decay usually leaves the Lambda manifold; v0 models neither
(a configurable branching ratio and a sink level are the natural v1 extension), so the
trace stays inside the three levels and dissipation shows up as *incoherent* loss of
the transfer rather than as population loss.

**Two-level limit.** With $\Omega_s = 0$ and $\Delta = 0$ the pair
$\lvert 1\rangle, \lvert 2\rangle$ is a resonantly driven qubit:
$P_2(t) = \sin^2(\Omega_p t / 2)$. This pins the factor $1/2$ in the control operators.

## Basis and index conventions

| Index | State | Role | Drift energy |
|---|---|---|---|
| 0 | $\lvert 1\rangle$ | initial ground state | 0 |
| 1 | $\lvert 2\rangle$ | excited, radiatively decaying | $\Delta$ |
| 2 | $\lvert 3\rangle$ | target ground state | $\delta$ |

Controls, in action order: `0` pump $\Omega_p$ with operator
$(\lvert 1\rangle\langle 2\rvert + \mathrm{h.c.})/2$ (indices 0, 1), `1` Stokes
$\Omega_s$ with operator $(\lvert 3\rangle\langle 2\rvert + \mathrm{h.c.})/2$
(indices 2, 1). All operators are built explicitly from `dq.basis(3, i)` outer products;
`dynamiqs.sigmam()` / `sigmap()` are never used (their spin convention, index 0 = up,
is the reverse of this library's).

Closed models (`gamma = 0`) carry kets and are integrated with `sesolve`; open models
carry density matrices and use `mesolve`. The observation is the flattened real and
imaginary parts of the state plus $t / T$: 7 numbers closed, 19 open. `info` reports
`p1`, `p2`, `p3` and `fidelity` (to $\lvert 3\rangle$) every step.

## Assumptions and validity

* Rotating-wave approximation in the rotating frame of both fields: valid when
  $\Omega_{p,s}, \Delta, \delta, \gamma \ll$ the optical transition frequencies, which
  is the regime of every STIRAP experiment. Counter-rotating terms are absent.
* Real Rabi frequencies: the actions set the amplitudes, the laser phases are fixed to
  zero. Negative actions flip the sign of the coupling (allowed, the environment clips to
  $[-\Omega_{\max}, \Omega_{\max}]$).
* The dark state is exact only at two-photon resonance ($\delta = 0$); with
  $\delta \neq 0$ there is no exact dark state and adiabatic transfer degrades
  (Vitanov et al. 2017, Sec. IV). $\Delta \neq 0$ keeps the dark state and only
  shifts the bright states.
* Spontaneous emission with equal branching into $\lvert 1\rangle$ and
  $\lvert 3\rangle$, no decay out of the manifold, no dephasing of the ground-state
  coherence (laser linewidth, magnetic noise). Pure dephasing between
  $\lvert 1\rangle$ and $\lvert 3\rangle$ is the main omitted channel for
  Zeeman/hyperfine Lambda systems.
* Piecewise-constant control: each action holds $(\Omega_p, \Omega_s)$ for one period
  `dt`; the adaptive integrator (`Tsit5`) takes as many internal steps as it needs
  inside each period. For the Gaussian reference pulses below the piecewise-constant
  trajectory reproduces the continuous-time solution to better than $10^{-4}$ in
  $P_3$ (checked with `dq.modulated` while designing the checks).
* No energy unit enters the model: all rates ($\Omega$, $\Delta$, $\delta$, $\gamma$)
  are in units of the peak Rabi frequency that `action_bound` fixes (1 by default),
  and time is in units of its inverse. There is no battery Hamiltonian.

## Parameters

Model fields (`LambdaSystem`):

| Field | Default | Meaning |
|---|---|---|
| `delta_one_photon` | 0.0 | $\Delta$, one-photon (pump) detuning of $\lvert 2\rangle$ |
| `delta_two_photon` | 0.0 | $\delta$, two-photon detuning of $\lvert 3\rangle$ |
| `gamma` | 0.0 | total spontaneous-emission rate of $\lvert 2\rangle$; `> 0` switches to Lindblad dynamics |
| `initial` | `"1"` | initially populated bare state, `"1"`, `"2"` or `"3"` |

Task kwargs (`qenvs.make("transfer/LambdaSTIRAP-v0", ...)`):

| Kwarg | Default | Meaning |
|---|---|---|
| `horizon` | 100 | control steps per episode |
| `dt` | 0.2 | control period, so $T = 20$ |
| `action_bound` | 1.0 | $\lvert\Omega_p\rvert, \lvert\Omega_s\rvert \le$ `action_bound` (maximum Rabi frequency, the unit of all rates) |
| `delta_one_photon`, `delta_two_photon`, `gamma` | 0.0 | forwarded to the model |

## Control semantics and reference pulses

The action is $(\Omega_p, \Omega_s)$. `stirap_pulses(horizon, dt, omega0, tau, width,
order)` returns an array of shape `(horizon, 2)` with Gaussian pulses sampled at the
midpoint of each control interval,

$$
\Omega_{s}(t) = \Omega_0\, e^{-\left((t - T/2 + \tau)/T_w\right)^2},\qquad
\Omega_{p}(t) = \Omega_0\, e^{-\left((t - T/2 - \tau)/T_w\right)^2}
$$

for `order="counterintuitive"` (Stokes centred at $T/2 - \tau$, pump at $T/2 + \tau$);
`order="intuitive"` swaps the two. `width` is the $1/e$ half-width $T_w$, `tau` half
the delay between the pulse centres.

Measured final $P_3$ for $\Omega_0 = 1$, horizon 100 (closed, $\Delta = \delta = 0$):

| $T_w$ ($\Omega_0 T_w$) | $T$ | `dt` | $\tau / T_w = 0.4$ | $0.5$ | $0.6$ | $0.7$ |
|---|---|---|---|---|---|---|
| 8 | 40 | 0.4 | 0.9996 | 0.9962 | 0.9592 | 0.8581 |
| 10 | 50 | 0.5 | 0.9629 | 0.9833 | 0.9727 | 0.9232 |
| 12 | 60 | 0.6 | 0.9941 | 0.9975 | 0.9927 | 0.9671 |

The efficiency at $\Omega_0 T_w \approx 10$ is not monotonic in $T_w$ or $\tau$: the
residual non-adiabatic loss of Gaussian pulses oscillates with the pulse area, so the
threshold $\Omega_0 T_w \ge 10$ guarantees "mostly adiabatic", not a particular
digit. In the default task ($T = 20$, `dt = 0.2`, $\Omega_{\max} = 1$) the pulses are
necessarily shorter and truncated at the episode boundaries; the best Gaussian pairs
found by a coarse scan are $T_w = 6, \tau = 3$ ($P_3 = 0.9964$) and
$T_w = 5, \tau = 2$ ($P_3 = 0.9924$), while $T_w = 4, \tau = 2$ gives $0.80$ and
$T_w = 3, \tau = 2$ gives $0.22$. The default task is therefore moderately adiabatic
($\Omega_{\max} T = 20$): textbook pulses reach $\approx 0.99$ and leave room for
learned or optimised controls.

## Reward

Per step: fidelity gain $F_{k} - F_{k-1}$ with $F = \lvert\langle 3 \vert \psi\rangle\rvert^2$
(closed) or $\langle 3 \vert \rho \vert 3\rangle$ (open, `dq.fidelity` with a pure
target). The undiscounted return is $F(T) - F(0) = P_3(T)$ for the default initial
state $\lvert 1\rangle$. Episodes end after `horizon` steps.

## Validation

All checks live in `qenvs.validation.lambda_system.CHECKS` and run in
`tests/test_lambda_system.py` and `qenvs validate`. Reference STIRAP pulses for the
transfer checks: horizon 100, `dt = 0.6` ($T = 60$), $\Omega_0 = 1$, $T_w = 12$
($\Omega_0 T_w = 12$), $\tau = 6$. Measured errors below are from one run of this
module on CPU with dynamiqs 0.3.6, jax 0.7.2, QuTiP 5.3.1, once with x64 enabled and
once in float32 (every check passes in both).

| Check | Reference | Tolerance | Error (x64) | Error (float32) |
|---|---|---|---|---|
| `dark_state_zero_eigenvector` | $\lVert H \lvert D\rangle \rVert$, $\lvert D\rangle \propto \Omega_s\lvert 1\rangle - \Omega_p\lvert 3\rangle$, 6 random $(\Omega_p, \Omega_s, \Delta)$, $\delta = 0$ | 1e-10 | 1.1e-16 | 1.1e-16 |
| `adiabatic_state_eigenvalues` | $\mathrm{eig}(H) = \{0, (\Delta \pm \sqrt{\Delta^2 + \Omega_p^2 + \Omega_s^2})/2\}$, 6 random sets | 1e-9 | 4.4e-16 | 4.4e-16 |
| `two_level_limit_rabi` | $\Omega_s = 0$, $\Delta = 0$, $\Omega_p = 1.3$: $P_2(t) = \sin^2(\Omega_p t/2)$ over one period | 1e-6 | 2.8e-10 | 3.6e-7 |
| `spontaneous_emission_branching` | from $\lvert 2\rangle$, $\gamma = 0.4$, no drive: $P_2 = e^{-\gamma t}$, $P_1 = P_3 = (1 - e^{-\gamma t})/2$, $t \le 6$ | 1e-6 | 1.1e-10 | 1.5e-7 |
| `stirap_transfer` | closed, counter-intuitive reference pulses via repeated `env.step`: $1 - P_3(T)$ ($P_3 = 0.9975$) | 1e-2 | 2.5e-3 | 2.5e-3 |
| `stirap_intermediate_population` | same trajectory: $\max_t P_2$ | 5e-2 | 4.3e-2 | 4.3e-2 |
| `stirap_intuitive_order` | intuitive order, same pulses: $1 - \max_t P_2$ (passes iff $\max_t P_2 > 0.3$; measured $\max_t P_2 = 0.998$, $P_3(T) = 0.017$) | 0.7 | 1.6e-3 | 1.6e-3 |
| `stirap_emission_robustness` | $\gamma = 0.5\,\Omega_0$: $1 - P_3(T)$ for the counter-intuitive order ($0.985$), forced to 1 unless it beats the intuitive order ($0.024$) | 5e-2 | 1.5e-2 | 1.5e-2 |
| `stirap_adiabaticity_sensitivity` | $\Omega_0 T_w = 1.5$ ($T_w = 1.5$, $\tau = 1$, $T = 10$): $P_3(T)$ must stay below 0.9 (diabatic) | 0.9 | 2.2e-2 | 2.2e-2 |
| `qutip_cross_check` | $\gamma = 0.3$, $\Delta = 0.2$, $\delta = 0.1$, 12 random piecewise-constant pump/Stokes steps in $[-1, 1]$, `dt = 0.3`: $\max\lvert\rho_{\mathrm{qenvs}} - \rho_{\mathrm{qutip}}\rvert$ vs QuTiP 5 `mesolve` | 1e-5 | 3.1e-8 | 9.1e-8 |
| `density_matrix_constraints` | open random driving, 15 steps: hermiticity, $\lvert\mathrm{tr}\rho - 1\rvert$, negativity | 1e-6 | 2.2e-16 | 1.2e-7 |
| `solver_convergence` | open random driving: adaptive `Tsit5` vs exact `Expm`, $\max\lvert\Delta\rho\rvert$ | 1e-6 | 2.5e-8 | 2.4e-7 |
| `env_stirap_return` | closed reference pulses: $\lvert \sum_k r_k - (F(T) - F(0))\rvert$ | 1e-6 | 3.3e-16 | 1.1e-8 |

Why these checks discriminate: a sign error in either coupling or in $\Delta$ breaks the
dark-state and eigenvalue checks; a factor-2 error in a control breaks the two-level
Rabi check; a jump operator pointing the wrong way or an unequal branching breaks the
emission check; swapping the control order would turn `stirap_transfer` into the
intuitive sequence and fail it; and the QuTiP comparison covers the full open
dynamics with detunings. Check (3) uses the intermediate population rather than the
final $P_3$ because, as derived above, the intuitive-order $P_3(T) = \cos^2\varphi(T)$
can take any value depending on the pulse area (it is 0.98 for $T_w = 10$,
$\tau = 7$), whereas $\max_t P_2 \approx 1$ always.

## Usage

```python
import qenvs
from qenvs.models.lambda_system import stirap_pulses

env = qenvs.make("transfer/LambdaSTIRAP-v0")  # horizon 100, dt 0.2, |Omega| <= 1
actions = stirap_pulses(env.horizon, env.dt, omega0=1.0, tau=3.0, width=6.0)
state, rewards = env.rollout(actions)
print(float(rewards.sum()))  # 0.996 = P3(T), closed system

noisy = qenvs.make("transfer/LambdaSTIRAP-v0", gamma=0.5)
state, rewards = noisy.rollout(actions)  # density matrix, same pulses
```

```bash
qenvs info transfer/LambdaSTIRAP-v0 --json
qenvs run transfer/LambdaSTIRAP-v0 --policy random --set gamma=0.2 --json
```

The CLI `pi-pulse` policy drives both channels with the same constant amplitude; it is
not a STIRAP reference for this task, use `stirap_pulses` from Python instead.

## References

* K. Bergmann, H. Theuer and B. W. Shore, "Coherent population transfer among quantum
  states of atoms and molecules", Rev. Mod. Phys. 70, 1003 (1998).
* N. V. Vitanov, A. A. Rangelov, B. W. Shore and K. Bergmann, "Stimulated Raman
  adiabatic passage in physics, chemistry, and beyond", Rev. Mod. Phys. 89, 015006 (2017).
* B. W. Shore, "Picturing stimulated Raman adiabatic passage: a STIRAP tutorial",
  Acta Physica Slovaca 58, 243 (2008).
* H.-P. Breuer and F. Petruccione, The Theory of Open Quantum Systems (Oxford, 2002),
  Sec. 3.4 (Lindblad form of spontaneous emission).

## Known limitations

* Equal branching ratio and no decay out of the manifold (see Dissipation); no
  ground-state dephasing. Both change the open-system numbers quantitatively but not
  the STIRAP mechanism.
* Real amplitudes only: no laser-phase control, so chirped or phase-modulated
  variants of STIRAP cannot be expressed in v0.
* `StateTransfer` casts actions to float32 before building $H$; with x64 enabled the
  state is still propagated in float64, but the control amplitudes carry float32
  rounding (the checks account for this by drawing float32 amplitudes).
* The default horizon ($T = 20$ at $\Omega_{\max} = 1$) is only moderately adiabatic;
  the validation checks use longer episodes (`dt = 0.6`, $T = 60$) to reach the
  $\Omega_0 T_w \ge 10$ regime with Gaussian pulses that start and end near zero.
* No solver-step convergence study beyond the `Tsit5` vs `Expm` comparison on random
  driving; the STIRAP checks rely on the default `Tsit5` tolerances
  (`rtol = atol = 1e-6`), which is why their thresholds are percent-level, not
  $10^{-6}$-level.
