# Transmon single-qubit gates

Task ids: `transmon/XGate-v0`, `transmon/SXGate-v0`. Model:
`qenvs.models.transmon.Transmon`. Environment: `qenvs.envs.gate.GateControl`.
Gate tools: `qenvs.gates`.

A multi-level transmon (a weakly anharmonic Duffing oscillator) driven by two
quadratures, I and Q, in the rotating frame of the drive. The agent or optimiser
builds a gate, not a state: the environment state is the propagator of the whole
pulse so far, and the reward is the average gate fidelity on the computational
subspace $\{|0\rangle, |1\rangle\}$. Population that leaks to $|2\rangle$ counts
as error.

!!! warning "Units"
    **Time is in nanoseconds (ns). Angular frequencies, including the Hamiltonian,
    the actions and `action_bound`, are in rad/ns.** Parameters you type are
    ordinary frequencies in **GHz** (`anharmonicity_ghz`, `detuning_ghz`,
    `max_rabi_ghz`), converted with $\omega = 2\pi f$, and coherence times in
    **microseconds** (`t1_us`, `t2_us`). $\hbar = 1$. An action of
    $\Omega = 2\pi \times 0.05$ rad/ns is a 50 MHz Rabi frequency, and a constant
    $\Omega_I$ for $\pi/\Omega$ ns is an X gate.

```python
import jax.numpy as jnp
import qenvs
from qenvs.tasks.transmon import reference_actions

env = qenvs.make("transmon/XGate-v0")  # closed, 3 levels, 16 x 2 ns
actions = reference_actions("transmon/XGate-v0", env)  # Gaussian DRAG baseline, rad/ns
state, rewards = env.rollout(actions)  # state.qstate = 3 x 3 propagator
print(env.fidelity(state.qstate), env.leakage(state.qstate))  # about 1 - 4.9e-6, 4.3e-6

noisy = qenvs.make("transmon/XGate-v0", t1_us=100.0, t2_us=100.0)  # 9 x 9 superoperator
```

GRAPE through the environment (`examples/grape_jax.py`):

```bash
python examples/grape_jax.py --task transmon/XGate-v0 --steps 300 --json
python examples/grape_jax.py --task transmon/SXGate-v0 --set t1_us=150 --set t2_us=120 --json
```

## Physics

Rotating-frame Hamiltonian (rotating-wave approximation, frame at the drive
frequency $\omega_d$), truncated to `levels` = $d$ states, $n = a^\dagger a$:

$$
H(t) = \Delta\, n + \frac{\alpha}{2}\, n (n - 1)
+ \frac{\Omega_I(t)}{2} \left(a + a^\dagger\right)
+ \frac{\Omega_Q(t)}{2}\, i \left(a^\dagger - a\right),
$$

$$
\Delta = 2\pi\,(f_q - f_d) = 2\pi \cdot \texttt{detuning\_ghz},
\qquad
\alpha = 2\pi \cdot \texttt{anharmonicity\_ghz} .
$$

The level energies in the rotating frame are $E_n = \Delta n + \alpha n (n-1)/2$,
so $E_2 - E_1 = \Delta + \alpha$: the $1 \to 2$ transition sits $\alpha$ below
the qubit transition (Koch et al. 2007; Krantz et al. 2019, Sec. II).

**Controls.** Two channels in the order `[Omega_I, Omega_Q]`. With
$\varepsilon = \Omega_I + i \Omega_Q$ the drive term is
$(\varepsilon a^\dagger + \varepsilon^* a)/2$. For $d = 2$ the control operators
are exactly $\sigma_x/2$ and $\sigma_y/2$ in the computational basis (index 0 =
ground, index 1 = excited, $\sigma_y = [[0, -i], [i, 0]]$), so constant controls
give $R_\phi(\theta) = \exp[-i \tfrac{\theta}{2}(\cos\phi\, \sigma_x + \sin\phi\, \sigma_y)]$
with $\theta = \Omega T$, $\Omega_I = \Omega \cos\phi$, $\Omega_Q = \Omega \sin\phi$.
Before the RWA these controls correspond to a lab-frame drive
$[\Omega_I(t) \cos\omega_d t + \Omega_Q(t) \sin\omega_d t]\,(a + a^\dagger)$
(our own derivation: move to the frame $e^{i\omega_d n t}$ and drop terms at
$2\omega_d$; not cross-checked against a lab-frame simulation).

**Dissipation** (Lindblad, only when `t1_us` and/or `t2_us` is given):

$$
L_1 = \sqrt{1/T_1}\; a ,\qquad
L_\phi = \sqrt{2/T_\phi}\; n ,\qquad
\frac{1}{T_\phi} = \frac{1}{T_2} - \frac{1}{2 T_1} .
$$

* Relaxation uses the harmonic ladder: level $n$ decays to $n - 1$ at rate
  $n / T_1$ (matrix element $\sqrt{n}$). This is an approximation for a
  transmon, whose ladder matrix elements are only approximately harmonic.
* Pure dephasing $L_\phi \propto n$ makes the $m$-$n$ coherence decay at
  $(m - n)^2 / T_\phi + (m + n)/(2 T_1)$; the 0-1 coherence decays at exactly
  $1/T_2$. White (Markovian) noise only: no $1/f$ noise, no echo-dependent $T_2$.
* $T_2 \le 2 T_1$ is enforced (`ValueError` otherwise). `t1_us` alone means
  $T_2 = 2 T_1$; `t2_us` alone means $T_1 = \infty$ and $T_\phi = T_2$.
* The dissipator is that of the undriven transmon, applied unchanged during the
  pulse.

## Gate fidelity and leakage

Let $k = 2$ be the size of the target $U_t$ and $P$ the projector on the first
$k$ levels.

**Unitary (closed system).** Pedersen, Moller and Molmer (2007), with
$M = P U_t^\dagger U P$ ($k \times k$):

$$
F = \frac{\operatorname{Tr}(M M^\dagger) + |\operatorname{Tr} M|^2}{k (k+1)},
\qquad
L = 1 - \frac{1}{k} \sum_{i, j < k} |U_{ji}|^2 = \frac{1}{k} \sum_{i < k,\, j \ge k} |U_{ji}|^2 .
$$

The second form of $L$ (equal for unitary $U$) is what the code evaluates; it
avoids cancellation for tiny leakage. $F$ ignores global phases but counts
relative phases, i.e. Z errors. No virtual-Z frame correction is applied: a pulse
that implements X up to a Z rotation is penalised.

**Channel (open system).** The average over Haar-random pure inputs *inside* the
computational subspace of the overlap with the ideal output (Wood and Gambetta
2018, Eqs. (5) and (6)):

$$
F = \frac{k F_{\mathrm{pro}} + 1 - L}{k + 1},\qquad
L = \operatorname{Tr}\!\left[(1 - P)\, \mathcal{E}(P/k)\right],\qquad
F_{\mathrm{pro}} = \frac{1}{k^2} \sum_{i,j<k} \langle i | U_t^\dagger \mathcal{E}(|i\rangle\langle j|) U_t | j\rangle .
$$

This reduces to the unitary formula when $\mathcal{E}$ is unitary (checked).
`tests/test_gate_env.py` also checks it against a direct average over the six
Pauli eigenstates (a qubit 2-design) for a leaky 3-level channel.

**Vectorisation.** dynamiqs 0.3.6 uses column stacking,
$\mathrm{vec}(\rho)_{i + d j} = \rho_{ij}$, for `dq.vectorize` and
`dq.mepropagator` (tested: $S\,\mathrm{vec}(\rho_0)$ matches `mesolve` to
$10^{-8}$). Hence $\mathrm{vec}(U\rho U^\dagger) = (\bar U \otimes U)\,\mathrm{vec}(\rho)$,
`qenvs.gates.unitary_to_superop`. Feeding a row-stacking superoperator
$U \otimes \bar U$ to the column-stacking formulas is the same as replacing $U$
by $\bar U$. Leakage and the fidelity against real targets (X, Z, H, $R_y$) are
unchanged by that swap, and so is Y (its conjugate is $-Y$, a global phase), but the
fidelity against a target whose conjugate is not proportional to it (SX, $R_z(\theta)$)
is not: for a random 3-level unitary (`np.random.default_rng(2)`) the fidelity
against SX is 0.55355 with column stacking and 0.35396 with the swap
(`tests/test_gate_env.py` pins both numbers). `apply_superop` is not invariant
either. Always use `unitary_to_superop` and `apply_superop`, never raw reshapes.

## Environment

* **State:** `state.qstate` is the accumulated map: the $d \times d$ propagator
  $U$ (closed) or the $d^2 \times d^2$ superoperator $S$ (open), both starting
  at the identity. Each step multiplies by the step propagator of constant
  controls, from `dq.sepropagator` / `dq.mepropagator`.
* **Solver:** `method = Expm()` (default) exponentiates the generator exactly
  for piecewise-constant controls. ODE methods are allowed; note that
  `Tsit5()` with default tolerances differs from `Expm()` by about $4 \times 10^{-5}$
  in the matrix elements of a 16-step map, so tighten `rtol`/`atol` if you use it.
* **Actions:** `[Omega_I, Omega_Q]` in rad/ns, held for `dt` ns, clipped to
  $\pm$`action_bound` per channel (box constraint, not a bound on
  $\sqrt{\Omega_I^2 + \Omega_Q^2}$). Actions are cast to the default float
  dtype (float64 under x64).
* **Observation:** real and imaginary parts of the flattened map plus $t/T$:
  $2 d^2 + 1$ numbers closed (19 for 3 levels), $2 d^4 + 1$ open (163).
* **Done:** `done` is False after steps 1 to `horizon - 1` and True after step
  `horizon`; the last observation entry is then exactly 1.0 (the time fraction).
* **Reward:** per-step gain of $F$; the return is $F_T - F_0$ with
  $F_0 = F(\mathbb{1}, U_t)$ = 1/3 for X and 2/3 for SX.
* **info:** `fidelity`, `leakage` (gate quantities above), and the model
  observables `p0` ... `p{d-1}` and `state_leakage` (population outside
  $\{0, 1\}$) evaluated on the map applied to `model.initial_state()` (|0> by
  default). The model observable named `leakage` is renamed `state_leakage` in
  `info` to keep it apart from the gate leakage.
* `jax.jit(jax.vmap(env.step))`, `jax.grad` through `env.rollout` (closed and
  open; checked against finite differences) and `examples/grape_jax.run_grape` work.

## Parameters and defaults

| kwarg | default | unit | meaning |
|---|---|---|---|
| `horizon` | 16 | steps | control steps; gate time `horizon * dt` |
| `dt` | 2.0 | ns | control period (16 x 2 ns = 32 ns gate) |
| `action_bound` | $2\pi \times 0.05 = 0.31416$ | rad/ns | primary bound on each of $|\Omega_I|$, $|\Omega_Q|$ |
| `max_rabi_ghz` | (unset) | GHz | alternative: `action_bound = 2 pi max_rabi_ghz`; giving both raises |
| `levels` | 3 | | transmon levels kept; 2 = ideal qubit |
| `anharmonicity_ghz` | -0.34 | GHz | $\alpha / 2\pi$ |
| `detuning_ghz` | 0.0 | GHz | $f_q - f_d$ |
| `t1_us` | None | us | $T_1$; any of `t1_us` / `t2_us` switches to Lindblad dynamics |
| `t2_us` | None | us | $T_2 \le 2 T_1$ |
| `method` | `Expm()` | | dynamiqs method per step |

Typical values. Krantz et al. (2019) state that transmon anharmonicities are
negative and usually designed to $|\alpha|/2\pi$ = 100 to 300 MHz (Sec. II;
"typically around 200 to 300 MHz", Sec. IV), and that intrinsic coherence times
have reached above 100 us (Sec. V). The default -0.34 GHz and the examples with
$T_1, T_2$ of 100 to 200 us are representative choices of ours for current fixed-frequency
devices, not values taken from that reference. For a specific IBM backend, use
the per-qubit anharmonicity, $T_1$ and $T_2$ from its published calibration
data. The default 32 ns gate is meant to be of the order of single-qubit gate
lengths on current superconducting hardware (check your backend's reported gate
length); arbitrary waveform generators sample much faster than our 2 ns control
period.

IBM backends calibrate X and SX with one shared pulse duration, whereas the two
tasks here are free to use any waveform. IBM calibrations also use virtual-Z frame updates, which this model
does not (see Known limitations), so fidelities here against targets that differ
by a Z rotation are stricter than on that hardware.

## DRAG

`gaussian_drag(horizon, dt, amplitude, sigma, beta)` returns a `(horizon, 2)`
pulse: a Gaussian centred in the window and lifted to vanish at $t = 0$ and
$t = T$ on I, and $\Omega_Q = \beta\, d\Omega_I/dt$ (analytic derivative) on Q,
both sampled at step midpoints. `gaussian_amplitude(angle, ...)` sets the area.

**Coefficient in this sign convention.** On resonance the $1 \to 2$ transition
is driven by the Fourier component of $\varepsilon(t) = \Omega_I + i\Omega_Q$ at
frequency $\alpha$. For $\varepsilon = \Omega_I + i\beta\dot\Omega_I$,
integration by parts gives $\int \varepsilon\, e^{i\alpha t} dt = (1 + \beta\alpha) \int \Omega_I e^{i\alpha t} dt$,
so first-order leakage vanishes at

$$
\beta_L = -\frac{1}{\alpha}\quad(\text{leakage}),\qquad
\beta_\phi = -\frac{1}{2\alpha}\quad(\text{phase error, measured numerically below}).
$$

`drag_beta(anharmonicity_ghz, "leakage" | "phase")` returns these (ns; positive
for a transmon). This is the DRAG idea of Motzoi et al. (2009), a second
quadrature proportional to the derivative of the first; Gambetta et al. (2011)
develop the general family of such corrections. The magnitudes match Krantz et
al. (2019), Eq. (101), $Q = \lambda\, \dot s / \alpha$ with $\lambda = 1$ for
leakage and $\lambda = 0.5$ for phase errors; our sign is opposite, which we
attribute to a different sign convention for the Q quadrature (we did not trace
it). Lucero et al. (2010) call the phase-optimal choice "half derivative". If
your hardware stack defines the drive as $\mathrm{Re}[(I + iQ) e^{i\omega_d t}] = I\cos - Q\sin$,
then $\Omega_Q = -Q$ relative to ours, and so is the sign of $\beta$. We have not
compared numerically with Qiskit Dynamics' transmon examples or with the
Qiskit `Drag` pulse convention, so check the sign on your setup (flipping it
increases leakage, which `tests/test_transmon.py` asserts).

**Measured sweep** (closed, 3 levels, $\alpha/2\pi = -0.34$ GHz, X gate,
$\sigma = T/4$, $dt = 0.25$ ns, $\beta$ swept on $[0, 2\beta_{\mathrm{theory}}]$
with 41 points then refined to 0.25 %; float64):

| $T$ (ns) | peak $\Omega_I/2\pi$ (MHz) | $L(\beta=0)$ | $\min_\beta L$ | $\beta^\star_L / \beta_L$ | $1-F(\beta=0)$ | $\min_\beta (1-F)$ | $\beta^\star_F / \beta_\phi$ |
|---|---|---|---|---|---|---|---|
| 8 | 117 | 3.69e-04 | 2.22e-06 | 1.037 | 1.47e-02 | 2.26e-04 | 0.980 |
| 10 | 93 | 1.02e-04 | 5.74e-10 | 0.948 | 9.35e-03 | 7.99e-05 | 0.985 |
| 12 | 78 | 5.70e-05 | 8.16e-10 | 0.980 | 6.51e-03 | 4.09e-05 | 0.990 |
| 16 | 58 | 1.84e-05 | 2.74e-10 | 0.980 | 3.66e-03 | 1.30e-05 | 0.995 |
| 20 | 47 | 8.19e-06 | 4.22e-10 | 0.978 | 2.34e-03 | 5.31e-06 | 0.997 |

The leakage-optimal $\beta$ is within 5.2 % of $-1/\alpha$ and the
fidelity-optimal $\beta$ within 2 % of $-1/(2\alpha)$ for every length. At
$\beta_L$ leakage drops by 4 to 5 orders of magnitude (from 10 ns on) but the
fidelity barely improves, because the drive-induced phase error dominates; at
$\beta_\phi$ the infidelity drops 65x to 440x. The remaining infidelity is
higher order (and would need the detuning correction of Motzoi et al., not
implemented here).

**Default task grid** (16 x 2 ns, $\sigma = 8$ ns, closed unless stated, float64):

| pulse | $1 - F$ (closed) | $L$ (closed) | $1 - F$ ($T_1 = T_2 = 100$ us) |
|---|---|---|---|
| Gaussian, $\beta = 0$ | 9.07e-04 | 2.35e-06 | 1.07e-03 |
| DRAG $\beta_\phi$ (`reference_actions`) | 4.85e-06 | 4.31e-06 | 1.65e-04 |
| DRAG $\beta_L$ | 9.25e-04 | 6.84e-06 | 1.09e-03 |
| GRAPE, 300 Adam steps (lr 0.1, seed 0, 1.9 s CPU) | 4.7e-15 | | 1.67e-04 |
| idle-decoherence reference $1 - (2F_{\mathrm{pro}}+1)/3$ for 32 ns | | | 1.60e-04 |

The idle-decoherence reference is the infidelity of an idle qubit against the
identity over the gate duration. It sets the scale of the coherence limit, but it
is not a lower bound on the infidelity of a gate: the DRAG and GRAPE rows (1.65e-04
and 1.67e-04) are 3 to 4 percent above it here, and a driven gate need not have
the same T1 and T2 error as an idle qubit.

With 2 ns sample-and-hold the spectrum of the piecewise-constant pulse is no
longer the smooth Gaussian, so $\beta_L$ does not minimise leakage on this grid.
GRAPE (`run_grape`, closed X, 3 levels) passes $F > 0.999$ after 30 to 60 Adam
steps for seeds 0, 1, 2 at lr 0.05 and 0.1 (history sampled every 10 steps) and
the SX task behaves the same; with $T_1 = T_2 = 100$ us it reaches the
coherence limit of the 32 ns window.

## Validation

All checks live in `qenvs.validation.transmon.CHECKS`, run in
`tests/test_transmon.py` (float64 via the test conftest, and once more in a
float32 subprocess) and are printed by `qenvs validate`. Measured on CPU,
dynamiqs 0.3.6, jax 0.10.2, qutip 5.3.1.

| Check | Reference | Tol | Error x64 | Error float32 |
|---|---|---|---|---|
| `qubit_limit_gates` | levels 2: $H_I = \sigma_x/2$, $H_Q = \sigma_y/2$; $\Omega_I T = \pi$ gives X; $\Omega_Q T = \pi/2$ gives $R_y(\pi/2)$ (sign-sensitive) | 2e-6 | 8.9e-16 | 4.8e-7 |
| `rabi_units` | 25 MHz Rabi flips at $1/(2f) = 20$ ns; 20 MHz detuned: $P_1 = \Omega^2/(\Omega^2 + \Delta^2)$ at $t = \pi/\sqrt{\Omega^2+\Delta^2}$ | 1e-6 | 0 | 1.5e-8 |
| `drift_spectrum` | 4 levels, idle 1.5 ns: $U = \mathrm{diag}(e^{-iE_n t})$, $E_n = 2\pi(0.013 n - 0.155 n(n-1))$ | 1e-5 | 4.7e-16 | 1.5e-7 |
| `fidelity_closed_forms` | $R_z(\theta)$ vs I: $(2 + 4\cos^2(\theta/2))/6$ (unitary and superoperator); $\lvert 1\rangle \leftrightarrow \lvert 2\rangle$ permutation: $F = 1/3$ vs I, $1/6$ vs X, $L = 1/2$; $F(\mathbb{1}, X) = 1/3$, $F(\mathbb{1}, SX) = 2/3$ | 1e-9 (x64), 10 eps (f32) | 2.2e-16 | 4.0e-8 |
| `closed_open_agree` | 16 random I/Q steps: $S$ with $T_1 = T_2 = 10^6$ us vs $\bar U \otimes U$; $F$ against SX, H, $R_z(0.7)$, Y and $L$ from both formulas | 1e-5 | 6.5e-8 | 3.4e-6 |
| `info_observables_orientation` | closed and open (T1 = 50 ns, T2 = 40 ns), 3 levels, 10 MHz detuning, initial $\lvert 1\rangle$, 16 random I/Q steps: `info` $p_0, p_1, p_2$ and `state_leakage` vs $\lvert U_{k1}\rvert^2$ and $\mathrm{diag}\,S\,\mathrm{vec}(\lvert 1\rangle\langle 1\rvert)$ from an independent numpy reference (exponentials built in numpy); fails if the transposed map would not change the result | 1e-9 (x64), 100 eps (f32) | 3.2e-15 | 1.6e-7 |
| `sx_orientation` | levels 2, $\Omega_I T = \pi/2$: $e^{i\pi/4} U = \mathrm{SX}$ elementwise (global phase fixed analytically; SX$^\dagger$ is 1.0 away), so SX is told apart from its adjoint | 2e-6 | 6.3e-16 | 1.1e-7 |
| `idle_coherence_fidelity` | levels 2 and 3, $T_1 = 40$ ns, $T_2 = 50$ ns, 32 ns idle: $F = (2F_{\mathrm{pro}} + 1)/3$, $F_{\mathrm{pro}} = (1 + e^{-t/T_1} + 2e^{-t/T_2})/4$ | 1e-6 | 8.9e-16 | 1.9e-7 |
| `harmonic_t1_ladder` | from $\lvert 2\rangle$: $P_2 = e^{-2t/T_1}$, $P_1 = 2(e^{-t/T_1} - e^{-2t/T_1})$ | 1e-6 | 3.9e-16 | 1.4e-7 |
| `superop_trace_preserving_cp` | open random pulse: $\lvert\operatorname{tr} S(\rho) - 1\rvert$, Hermiticity on 5 random $\rho$, Choi matrix negativity | 1e-6 | 4.4e-16 | 2.4e-7 |
| `drag_coefficient` | 12 ns X: $\lvert\beta^\star/\beta_{\mathrm{th}} - 1\rvert$ for leakage ($-1/\alpha$) and fidelity ($-1/(2\alpha)$); fails unless $L$ drops 100x and $1-F$ 50x vs $\beta = 0$ | 0.05 | 2.0e-2 | 2.0e-2 |
| `qutip_cross_check` | QuTiP 5, independent construction from GHz / us: open `mesolve` from a random $\rho_0$ ($T_1 = 30$ ns, $T_2 = 25$ ns, 12 random I/Q steps) and closed propagator | 1e-5 | 2.6e-11 | 5.9e-7 |
| `solver_convergence` | default `Expm()` vs `Tsit5(rtol=atol=1e-10)`, closed and open 16-step maps | 1e-4 | 2.3e-9 | 8.9e-6 |
| `env_return_equals_gain` | $\sum_k r_k = F_T - 1/3$, closed and open | 1e-6 | 1.1e-16 | 3.6e-8 |

`qubit_limit_gates` has a 2e-6 tolerance (4x its float32 error; the float64 error
is 1e-15) so that single-precision matrix exponentials do not make it flaky.
`solver_convergence` has a looser tolerance because in float32 the Pade matrix
exponential is off by about 6e-7 per 2 ns step and these errors add up over the
16 steps; in float64 the difference is 2e-9.

Discrimination: we ran the checks against deliberately broken versions: Q sign
flipped, I control doubled, detuning without $2\pi$, anharmonicity sign flipped,
$\alpha n^2/2$ instead of $\alpha n(n-1)/2$, relaxation rate doubled, relaxation
via $a^\dagger$, dephasing $\sqrt{1/T_\phi}\,n$, $1/T_\phi = 1/T_2$, $T_1$ in
ns instead of us, half-transposed superoperator indices, Pedersen denominator
$k^2$, $\operatorname{Tr}(MM^\dagger)$ replaced by $k$, leakage without $1/k$
(unitary and channel), conjugated target in the channel fidelity, `drag_beta`
sign flipped, DRAG Q without the amplitude factor, map composition order
reversed, reward scaled by 1.01. Every one fails at least one check.
The `GateControl` mutants that the physics checks cannot see are pinned by
`tests/test_gate_env.py`, each verified to fail: the closed and open `info` map
transposed (`m.T @ psi0`; also caught by `info_observables_orientation`), the
fidelity computed against `target.T`, `done = step > horizon`, and actions cast
to float32 under x64. Replacing SX by its adjoint fails `sx_orientation`.

## Assumptions and validity

* Rotating-wave approximation in the drive frame; valid for
  $|\Omega|, |\Delta|, |\alpha| \ll \omega_q$ (GHz-scale qubit, tens of MHz of
  drive). Counter-rotating terms and Bloch-Siegert shifts are not modelled.
* Duffing truncation: $H$ keeps only the Kerr term $\alpha n(n-1)/2$, which is
  accurate for the lowest levels of a transmon with large $E_J/E_C$ (Koch et
  al. 2007); higher-order corrections to level energies and to the charge matrix
  elements are dropped (the drive couples through $a + a^\dagger$ with harmonic
  matrix elements $\sqrt{n}$).
* Harmonic $T_1$ ladder, white-noise dephasing, undriven dissipator (see above).
* Piecewise-constant controls on a `dt` grid; no transfer function, bandwidth
  limit, rise time or amplitude noise of the control electronics.
* Classical, deterministic controls; no drive-line crosstalk.

## Known limitations

* Single qubit only: no couplers, no two-qubit gates (CR, CZ) yet.
* No readout, no measurement-induced dephasing, no state preparation errors.
* No virtual-Z gates: the fidelity is not maximised over Z frame updates.
* The default 2 ns control period is coarse compared with hardware AWGs;
  smaller `dt` with larger `horizon` is supported but costs more steps.
* `action_bound` is a per-channel box, not an amplitude-circle constraint.
* `toolkit.run_rollout`'s generic `pi-pulse` policy puts $\pi/T$ on both
  channels, which is not an X gate here (it gives $F \approx 0.54$); use
  `qenvs.tasks.transmon.reference_actions` for a physical baseline.
* Coherence parameters must be Python floats (not traced arrays) because the
  presence of jump operators is decided when the model is built.
* `qutip_cross_check` needs the `qenvs[dev]` extra; without QuTiP it is reported
  as skipped.

## References

1. J. Koch, T. M. Yu, J. Gambetta, A. A. Houck, D. I. Schuster, J. Majer,
   A. Blais, M. H. Devoret, S. M. Girvin and R. J. Schoelkopf, "Charge-insensitive
   qubit design derived from the Cooper pair box", Phys. Rev. A 76, 042319 (2007).
   Transmon, negative anharmonicity.
2. P. Krantz, M. Kjaergaard, F. Yan, T. P. Orlando, S. Gustavsson and
   W. D. Oliver, "A quantum engineer's guide to superconducting qubits",
   Appl. Phys. Rev. 6, 021318 (2019). Typical anharmonicity and coherence
   (Secs. II, V), DRAG Eq. (101) with $\lambda = 0.5$ (phase) and 1 (leakage).
3. F. Motzoi, J. M. Gambetta, P. Rebentrost and F. K. Wilhelm, "Simple pulses
   for elimination of leakage in weakly nonlinear qubits", Phys. Rev. Lett.
   103, 110501 (2009). DRAG: a second quadrature proportional to the time
   derivative of the first.
4. J. M. Gambetta, F. Motzoi, S. T. Merkel and F. K. Wilhelm, "Analytic control
   methods for high-fidelity unitary operations in a weakly nonlinear
   oscillator", Phys. Rev. A 83, 012308 (2011). General family of DRAG-type
   solutions.
5. E. Lucero et al., "Reduced phase error through optimized control of a
   superconducting qubit", Phys. Rev. A 82, 042339 (2010). "Half derivative"
   DRAG for phase errors.
6. L. H. Pedersen, N. M. Moller and K. Molmer, "Fidelity of quantum operations",
   Phys. Lett. A 367, 47 (2007). Closed-form average gate fidelity.
7. C. J. Wood and J. M. Gambetta, "Quantification and characterization of
   leakage errors", Phys. Rev. A 97, 032306 (2018). Leakage rate (Eq. (2)) and
   average gate fidelity of leaky channels (Eqs. (5), (6)).
