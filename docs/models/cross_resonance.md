# Cross-resonance gate on two coupled transmons

Task id: `transmon/CRGate-v0`. Model: `qenvs.models.two_transmon.TwoTransmon`.
Environment: `qenvs.envs.gate.GateControl` with `subspace=(0, 1, d, d + 1)`.
Task module: `qenvs.tasks.cross_resonance`.

Two fixed-frequency transmons (Duffing oscillators, `levels` = $d$ states each, 3 by
default) with an exchange coupling $J$. Both are driven at the frequency of the target
transmon: the cross-resonance (CR) drive on the control and, on the target, a drive that
can act as a cancellation tone. The agent or optimiser builds the two-qubit gate
$ZX(\pi/2) = \exp[-i (\pi/4)\, Z \otimes X]$ on the computational states
$|00\rangle, |01\rangle, |10\rangle, |11\rangle$ (control digit first). Population in any
other state of the $d^2$-dimensional space counts as leakage.

!!! warning "Units"
    **Time is in nanoseconds (ns). Angular frequencies, including the Hamiltonian, the
    actions and `action_bound`, are in rad/ns.** Parameters you type are ordinary
    frequencies in **GHz** (`detuning_ghz`, `anharmonicity_c_ghz`, `anharmonicity_t_ghz`,
    `coupling_ghz`, `max_rabi_ghz`), converted with $\omega = 2\pi f$, and coherence
    times in **microseconds**. $\hbar = 1$.

```python
import qenvs
from qenvs.tasks.cross_resonance import reference_actions

env = qenvs.make("transmon/CRGate-v0")  # closed, 3 + 3 levels, 64 x 5 ns = 320 ns
actions = reference_actions(env)  # calibrated flat-top CR + cancellation tone, (64, 4) rad/ns
state, rewards = env.rollout(actions)  # state.qstate = 9 x 9 propagator (common frame)
print(env.fidelity(state.qstate), env.leakage(state.qstate))  # about 0.98432, 1.3e-4

noisy = qenvs.make(
    "transmon/CRGate-v0", t1_c_us=100.0, t2_c_us=100.0, t1_t_us=100.0, t2_t_us=100.0
)  # 81 x 81 superoperator
```

```bash
python examples/grape_jax.py --task transmon/CRGate-v0 --steps 300 --lr 0.05 --json
```

## Physics

**Lab frame** (rotating-wave approximation on the coupling), with $a$ the control and
$b$ the target lowering operator, truncated to $d$ levels each, $n_c = a^\dagger a$,
$n_t = b^\dagger b$:

$$
H_{\mathrm{lab}} = \sum_{i \in \{c, t\}} \Big[\omega_i n_i + \frac{\alpha_i}{2} n_i (n_i - 1)\Big]
+ J\,(a^\dagger b + a b^\dagger) + \text{drives}.
$$

**Common frame.** Both transmons are written in the frame rotating at the drive
frequency $\omega_d = \omega_t$ (the bare target frequency), $\psi_r = e^{i\omega_d (n_c + n_t) t}\psi_{\mathrm{lab}}$.
The exchange term commutes with $n_c + n_t$, so the Hamiltonian is time independent
apart from the control envelopes:

$$
H = \Delta\, n_c + \frac{\alpha_c}{2} n_c (n_c - 1) + \frac{\alpha_t}{2} n_t (n_t - 1)
+ J\,(a^\dagger b + a b^\dagger)
+ \frac{\Omega_{cI}}{2}(a + a^\dagger) + \frac{\Omega_{cQ}}{2}\, i(a^\dagger - a)
+ \frac{\Omega_{tI}}{2}(b + b^\dagger) + \frac{\Omega_{tQ}}{2}\, i(b^\dagger - b),
$$

with $\Delta = 2\pi(f_c - f_t)$ = `2 pi detuning_ghz`, $\alpha_i = 2\pi$ `anharmonicity_*_ghz`
and $J = 2\pi$ `coupling_ghz`. The tensor order is control $\otimes$ target: basis index
$i_c d + i_t$, so the computational states sit at indices $(0, 1, d, d+1)$, which is
`TwoTransmon.computational_indices`.

**Controls.** Four channels in the order `[Omega_cI, Omega_cQ, Omega_tI, Omega_tQ]`,
each with the operator convention of the single `Transmon` (index 1 = excited state,
$\Omega/2\pi$ = Rabi frequency in GHz). Both drives are at $\omega_t$; there is no drive
at the control frequency.

**Dissipation** (Lindblad, enabled per transmon by `t1_c_us`, `t2_c_us`, `t1_t_us`,
`t2_t_us`): $\sqrt{1/T_1}\,a$ and $\sqrt{2/T_\phi}\,n_c$ with
$1/T_\phi = 1/T_2 - 1/(2T_1)$, and the same for the target, exactly as in
[the single transmon](transmon.md). The jump operators act on the bare (uncoupled)
modes.

## Frames and the target

**Qubit frames.** The gate is defined in the frames rotating at each bare qubit frequency,
$\psi_q = e^{i(\omega_c n_c + \omega_t n_t) t}\psi_{\mathrm{lab}}$. Since
$\psi_r = e^{i\omega_t (n_c + n_t) t}\psi_{\mathrm{lab}}$,

$$
\psi_q = e^{+i\Delta n_c t}\,\psi_r ,\qquad U_q(T) = e^{+i\Delta n_c T}\, U_r(T),
$$

(`TwoTransmon.frame_unitary(T)`, `qenvs.models.two_transmon.to_qubit_frame`). Sign test:
with $J = 0$ and no drive, $U_r = e^{-iH_0 T}$ and $H_0 = \Delta n_c$ on the qubit states, so
$U_q = 1$ there; the opposite sign would leave $e^{2i\Delta T}$ on $|10\rangle, |11\rangle$
(check `frame_change_identity` and `tests/test_cross_resonance.py::test_frame_change_sign`).

**Target.** $ZX(\pi/2) = \exp[-i(\pi/4) Z \otimes X] = (1 - i\, Z\otimes X)/\sqrt{2}$, control
$\otimes$ target, $Z = \mathrm{diag}(1, -1)$ (`qenvs.tasks.cross_resonance.zx_gate`). It
rotates the target by $+\pi/2$ about $x$ when the control is in $|0\rangle$ and by $-\pi/2$
when it is in $|1\rangle$.

**What the environment stores.** The environment simulates in the common frame. The frame
change is diagonal and maps the computational subspace to itself, so with
$D = \mathrm{diag}(1, 1, e^{i\Delta T}, e^{i\Delta T})$ on that subspace,
$F(U_q, ZX) = F(U_r, D^\dagger ZX)$ for both the unitary and the channel fidelity.
`env.target` holds $D^\dagger ZX(\pi/2)$ for $T$ = `horizon * dt` (`cr_target`), so the
final fidelity and the return refer to the qubit frames at the end of the episode (check
`env_return_equals_gain` compares with `to_qubit_frame` directly). Intermediate values of
`info["fidelity"]` compare the partial map with that end-of-episode target; they are a
shaping signal, not the qubit-frame fidelity at time $t$. With the defaults
$\Delta T = -64\pi$, so $D = 1$ and $F_0 = F(\mathbb{1}, ZX(\pi/2)) = 3/5$; in general
$F_0 = (2 + \cos\Delta T)/5$.

**Qiskit.** Qiskit's `RZXGate(theta)` documentation defines
$R_{ZX}(\theta)\,q_0, q_1 = \exp(-i\tfrac{\theta}{2} X \otimes Z)$ and prints its matrix in
Qiskit's little-endian basis $|q_1 q_0\rangle$, with $Z$ on $q_0$. Reordering that printed
matrix to $|q_0 q_1\rangle$ gives exactly `zx_gate(theta)` with the control on $q_0$ (checked
for $\theta = \pi/2$ and 0.7 against the matrix transcribed from the documentation in
`test_zx_gate_and_qiskit_rzx_convention`; Qiskit itself is not imported or run). So
$ZX(\pi/2)$ here corresponds to `RZXGate(pi/2)` applied with `q0` = control and `q1` =
target, up to that basis reordering. We have not compared with Qiskit's `ECRGate` or with
the CR pulse schedules of any backend.

## Defaults and why

| kwarg | default | unit | reason |
|---|---|---|---|
| `detuning_ghz` | -0.1 | GHz | $f_c - f_t$. **Not** in the straddling regime: straddling means $\omega_c + \alpha_c < \omega_t < \omega_c$, i.e. $0 < \Delta < -\alpha_c$ (+0.34 GHz at the defaults), where the first-order ZX coefficient is larger ($|\alpha_c/(\Delta(\alpha_c+\Delta))|$ = 14.2 per GHz at $\Delta = +0.1$ GHz against 7.7 per GHz at the default $-0.1$ GHz). We keep $-0.1$ GHz because it is a typical size and sign of the frequency difference of a fixed-frequency pair (we did not take it from a specific device), it gives $\nu_{ZX} > 0$ for a positive CR amplitude (0.0232 per GHz of Rabi frequency), and it stays 100 MHz away from the poles of the perturbative ZX rate at $\Delta = 0, -\alpha_c/2, -\alpha_c, -3\alpha_c/2$ (Magesan and Gambetta 2020, after Eq. (4.26)). The price is a weaker CR rate (the reference needs 47 MHz). $\Delta = +0.1$ GHz, the straddling side, also works, with a negative CR amplitude (see the reference pulse) |
| `anharmonicity_c_ghz`, `anharmonicity_t_ghz` | -0.34 | GHz | same value as the single-transmon tasks (see [transmon](transmon.md) for its justification) |
| `coupling_ghz` | 0.003 | GHz | $J/|\Delta| = 0.03$: dispersive, second-order ZZ accurate to $5\times10^{-4}$ (below); large enough that a 320 ns gate needs about 47 MHz of CR drive. Our choice for a fixed-frequency pair, not taken from a specific device |
| `levels` | 3 | | per transmon; $|2\rangle$ captures leakage and the higher-level contributions to ZX and ZZ |
| `horizon`, `dt` | 64, 5.0 | steps, ns | 320 ns gate. A coarse grid keeps an episode at 64 matrix exponentials of $9 \times 9$ (closed). 5 ns is half a period of $\Delta$; see the sampling note below |
| `action_bound` | $2\pi \times 0.06$ = 0.37699 | rad/ns | per-channel box, 60 MHz Rabi frequency, above the 47 MHz the reference pulse needs |
| `max_rabi_ghz` | (unset) | GHz | alternative: `action_bound = 2 pi max_rabi_ghz`; giving both raises |
| `t1_c_us` ... `t2_t_us` | None | us | closed by default; any of them switches to Lindblad dynamics |
| `method` | `Expm()` | | exact for piecewise-constant controls |

Derived values at the defaults: static ZZ $\zeta/2\pi$ = 115.86 kHz (exact
diagonalisation, second order 115.91 kHz) and a first-order ZX rate
$\nu_{ZX}/\Omega$ = 0.0232 (rate in GHz per GHz of CR Rabi frequency).

**Sampling note.** With piecewise-constant controls the control transmon, detuned by
$\Delta$, does not follow its drive adiabatically: every step of the envelope kicks it. In
exploratory runs during development (not part of the test-suite; $J/2\pi$ = 2.5 MHz, 400 ns
gate, flat-top pulses calibrated as below) a 10 ns grid, one full period of $\Delta$, made
these kicks add up coherently and left a control rotation of about 0.4 rad (residual
$X\otimes X$ and $Y\otimes I$ generators of about 0.19 each; best flat-top infidelity 0.064),
while a 5 ns grid (half a period) made consecutive kicks largely cancel (best flat-top
infidelity 0.012). If you change `dt`, expect the reference pulse to change
quality for this reason.

## ZZ and ZX formulas

**Static ZZ.** $\zeta = E_{11} - E_{10} - E_{01} + E_{00}$ of the dressed states (exact
diagonalisation of the undriven $H$, each dressed state assigned to the bare state with
the largest overlap; `TwoTransmon.static_zz`). With $H_{ZZ} = (\zeta/4) Z\otimes Z$ this is
the shift of $|11\rangle$. Second order in $J$ (our derivation: only $|01\rangle \leftrightarrow |10\rangle$,
matrix element $J$, and $|11\rangle \leftrightarrow |20\rangle, |02\rangle$, matrix element
$\sqrt{2}J$, contribute):

$$
\zeta = 2J^2\left[\frac{1}{\Delta - \alpha_t} - \frac{1}{\Delta + \alpha_c}\right]
= \frac{2J^2(\alpha_c + \alpha_t)}{(\Delta + \alpha_c)(\Delta - \alpha_t)},
\qquad \Delta = \omega_c - \omega_t ,
$$

`static_zz_second_order` (in GHz). This is $2\xi$ with $\xi$ of Magesan and Gambetta
(2020), Eq. (4.5), whose Hamiltonian term is $\xi\, Z\otimes Z/2$. The sign convention: $\zeta > 0$
means the $|11\rangle$ energy lies above the sum of the single excitations; for the defaults
$\zeta > 0$.

Measured (check `static_zz`, $\Delta/2\pi = -100$ MHz, $\alpha_c/2\pi = -340$ MHz,
$\alpha_t/2\pi = -300$ MHz, float64):

| $J/2\pi$ | exact $\zeta/2\pi$ | second order | relative difference |
|---|---|---|---|
| 1 MHz | 14.5446 kHz | 14.5455 kHz | -6.0e-5 |
| 3 MHz | 130.838 kHz | 130.909 kHz | -5.4e-4 |
| 5 MHz | 363.090 kHz | 363.636 kHz | -1.5e-3 |

The remainder scales as $(J/\Delta)^2$ (`test_static_zz_second_order_scaling`).

**ZX rate at weak drive.** For a constant CR drive $\Omega$ on the control I channel, to first
order in $\Omega$ and $J$, with $H_{\mathrm{eff}} = (\nu_{ZX}/2)\, Z\otimes X + (\nu_{IX}/2)\, I\otimes X + \dots$:

$$
\nu_{ZX} = -\frac{J\Omega}{\Delta}\,\frac{\alpha_c}{\alpha_c + \Delta},\qquad
\nu_{IX} = -\frac{J\Omega}{\Delta + \alpha_c}
$$

(`zx_rate_first_order`). These are Eq. (4.26) and the IX entry of Appendix C of Magesan
and Gambetta (2020); their drive $\Omega\cos(\omega_d t)(b^\dagger + b)$ becomes
$(\Omega/2)(b^\dagger + b)$ in the rotating frame, which is our convention, and their Pauli
coefficients multiply $P/2$, as ours do. We also derived them independently (dress the
control by the drive to first order, then project the exchange term onto each control
state).

Measured (check `zx_rate_weak_drive`): constant drive $\Omega/2\pi$ = 5 MHz for 2 us,
$J/2\pi$ = 2 MHz, other parameters as above. The target rotation vector for the control in
$|c\rangle$ is read from the unitary part of the $\{|c0\rangle, |c1\rangle\}$ block of the
qubit-frame map; $\nu_{ZX} = (w_0 - w_1)/2$, $\nu_{IX} = (w_0 + w_1)/2$ from the $x$
components.

| | simulation | first order | relative difference |
|---|---|---|---|
| $\nu_{ZX}/2\pi$ | 77.114 kHz | 77.273 kHz | -2.0e-3 |
| $\nu_{IX}/2\pi$ | 22.719 kHz | 22.727 kHz | -3.8e-4 |

At the strong drive of the reference pulse the first-order rate is no longer accurate: the
calibrated flat-top needs 47.2 MHz where the first-order formula predicts 43.4 MHz for the
same pulse area. Magesan and Gambetta give third-order corrections (their Eq. (4.25));
we did not implement or check them.

## Reference pulse

`qenvs.tasks.cross_resonance.reference_actions(env)` (also `qenvs run ... --policy reference`):

* CR drive on `Omega_cI`: a flat-top with $\sin^2$ edges of 30 ns, starting at $t = 0$, averaged
  over each control step (`flat_top`), amplitude $sA$ with $A \ge 0$ and $s$ the sign of the
  first-order $\nu_{ZX}$ (`cr_amplitude_sign`): $s = +1$ at the default $\Delta = -0.1$ GHz and
  $s = -1$ at $\Delta = +0.1$ GHz (the straddling side, where $\nu_{ZX} < 0$).
* Cancellation tone on `Omega_tI`: $c\,s A f(t)$ with the first-order ratio
  $c = J/(\Delta + \alpha_c)$ for `levels >= 3` (only the control anharmonicity enters, also when
  $\alpha_t \ne \alpha_c$), which cancels $\nu_{IX}$ above. For `levels = 2` the control has no
  $|2\rangle$ level and dressing it by the drive gives $H_{\mathrm{eff}} = -(J\Omega/2\Delta)\,Z\otimes X$
  with no IX term, so $c = 0$ (`cancellation_ratio`); the three-level value would add an IX term
  there (the two-level reference pulse with it reaches F = 0.9644 instead of 0.9948, `test_reference_two_level_model`).
* $A$ is calibrated by bisection so that the conditional rotation is $\pi/2$
  ($(\theta_0 - \theta_1)/2 = \pi/2$ from the target Bloch vector reached from $|00\rangle$ and
  $|10\rangle$, `conditional_angles`), in the closed model, for each of 16 flat-top widths,
  then for 16 widths around the best one; the width with the highest fidelity wins. The width
  matters because the control's Stark shift (a $Z\otimes I$ phase growing with $\int A^2 dt$)
  is not removed by virtual-Z frame updates here.

Measured at the defaults (float64, CPU, dynamiqs 0.3.6, jax 0.10.2):

| pulse | $F$ (closed) | leakage | $F$ ($T_1 = T_2 = 100$ us, both) |
|---|---|---|---|
| reference: $A/2\pi$ = 47.2 MHz, pulse ends at 280 ns, conditional angles $(+1.5715, -1.5701)$ | 0.98432 | 1.3e-4 | 0.98057 |
| GRAPE, `run_grape(steps=300, lr=0.05, seed=0)` from a random start: best | 0.99929 | | |
| same run, last iterate (Adam oscillates) | 0.99004 | | |
| GRAPE, 150 steps (as in `test_grape_improves_on_the_reference`): best, last | 0.99805, 0.98894 | | |

Other settings (closed, float64, same procedure): $\Delta/2\pi = +0.1$ GHz, $A/2\pi$ = $-46.3$ MHz,
pulse ends at 180 ns, conditional angles $(+1.5750, -1.5666)$: $F$ = 0.99626, leakage 5.7e-4;
`levels = 2` (either sign of $\Delta$, no cancellation tone): $F$ = 0.99476 (the three-level
default is 0.98432, so removing $|2\rangle$ helps by 0.010).
For a noisy env the reference is the same as for the closed one (calibration is on the closed model).

GRAPE passes the reference after 90 Adam steps (300 steps take 14 s on CPU, x64; GRAPE was
run closed only). Writing the
reference's qubit-frame map (unitary part, polar decomposition of the $4\times4$ block) as
$ZX(\pi/2)\, e^{-iG}$, the Pauli coefficients of $G$ above 0.01 are $Z\otimes Y$ -0.100,
$I\otimes Z$ -0.081, $I\otimes Y$ 0.038, $Z\otimes Z$ 0.030, $X\otimes I$ 0.017,
$Y\otimes X$ -0.015: mostly phase errors of the target. They come from two sources of
opposite sign. (i) The static dressing of the bare frames: exact diagonalisation of the
undriven $H$ gives a target frequency shift of +89.9 kHz for the control in $|0\rangle$ but
+205.8 kHz for the control in $|1\rangle$ (second order, $J^2/\Delta$ = 90 kHz is only the
first of these; the second adds the $|11\rangle \leftrightarrow |20\rangle, |02\rangle$
terms), and a control shift of -89.9 kHz (target in $|0\rangle$) or +26.0 kHz (target in
$|1\rangle$). Over 320 ns without any drive (qubit frames, same decomposition, $e^{-iG_s}$) this
alone gives $I\otimes Z$ -0.149, $Z\otimes Z$ +0.058 (the 115.9 kHz of the static ZZ) and
$Z\otimes I$ +0.032. (ii) The Stark shifts of the drive, which for the reference partly
cancel these: the measured $I\otimes Z$ (-0.081) and $Z\otimes Z$ (+0.030) are about 0.068 and
-0.028 away from the static values (the decompositions do not simply add, since $G$ is
defined relative to $ZX(\pi/2)$, so this is an estimate). The naive estimate from $J^2/\Delta$ (90 kHz over 320 ns, coefficient 0.09) is only the static
target shift for the control in $|0\rangle$ and matches the measured $I\otimes Z$ (-0.081) by
coincidence of the two contributions, not because it explains it. We did not
separate the drive-induced part further. This task counts all of them because it has no
virtual-Z gates.

## Validation

Checks live in `qenvs.validation.cross_resonance.CHECKS` and run in
`tests/test_cross_resonance.py` (float64 via the test conftest, and once more in a float32
subprocess). Measured on CPU, dynamiqs 0.3.6, jax 0.10.2, qutip 5.3.1.

| Check | Reference | Tol | Error x64 | Error float32 |
|---|---|---|---|---|
| `frame_change_identity` | $J = 0$, idle 37 ns: $e^{+i\Delta n_c T}U_r = 1$ on the qubit states (the opposite sign is 1.90 away); env fidelity with the pulled-back target equals $F(U_q, ZX)$ | 1e-5 | 7.0e-15 | 3.0e-6 |
| `j0_decoupling` | $J = 0$: 16 random 4-channel steps give $U_c \otimes U_t$ from two independent `Transmon` maps ($J/2\pi$ = 3 MHz: 4e-2 away); DRAG X on the target gives $1\otimes V$ and $F(1\otimes X) = (2\,\mathrm{Tr}\,mm^\dagger + 4|\mathrm{Tr}\,m|^2)/20$ | 1e-5 | 1.3e-15 | 1.7e-6 |
| `static_zz` | relative difference exact vs second order at $J/2\pi$ = 3 MHz | 5e-3 | 5.4e-4 | 6.1e-4 |
| `zx_rate_weak_drive` | max relative difference of $\nu_{ZX}$, $\nu_{IX}$ vs first order | 1e-2 | 2.0e-3 | 2.0e-3 |
| `subspace_fidelity_closed_forms` | subspace $(0, 1, 3, 4)$: $F(1, ZX) = 3/5$, embedded $ZX$ perfect, $F(CZ, 1) = 2/5$, $|11\rangle \leftrightarrow |20\rangle$: $F = 3/5$, $L = 1/4$; unitary and superoperator; the default first-4-levels subspace gives 0.689 instead | 1e-9 (x64), 10 eps (f32) | 1.1e-16 | 1.2e-7 |
| `qutip_cross_check` | QuTiP 5 from an independent construction: open `mesolve` from a random $\rho_0$ (control $T_1/T_2$ = 30/25 ns, target 40/20 ns) and closed propagator, 16 random 4-channel steps of 0.5 ns | 1e-5 | 1.0e-9 | 1.7e-6 |
| `superop_cptp` | open random pulse: $|\mathrm{tr}\,S(\rho) - 1|$, Hermiticity, Choi negativity | 1e-5 | 2.0e-15 | 2.1e-6 |
| `solver_convergence` | `Expm()` vs `Tsit5(rtol=atol=1e-10)`, closed and open, 16 x 0.5 ns | 1e-4 | 5.2e-9 | 2.4e-6 |
| `env_return_equals_gain` | $\sum r = F_T - F_0$ and $F_T = F(e^{i\Delta n_c T}U, ZX)$, `horizon = 12`, `dt = 4.3` (nontrivial frame), closed and open | 1e-5 | 2.8e-17 | 6.0e-8 |

Discrimination: each check suite was run against in-process mutants, and every mutant
fails at least one check: frame-change sign flipped; CR operator doubled; control and target
channels swapped; sign of $\Delta$ flipped; $\alpha_c \leftrightarrow \alpha_t$; $J$ doubled;
counter-rotating coupling $a^\dagger b^\dagger + ab$ instead of exchange; control and target
$T_1/T_2$ swapped; dephasing $\sqrt{1/T_\phi}\,n$; computational indices $(0,1,2,3)$; the ZZ
formula without its factor 2 or with $\alpha_c \leftrightarrow \alpha_t$; the ZX formula with
$\alpha_c/(\alpha_c - \Delta)$; the IX formula with $\Delta$ instead of $\Delta + \alpha_c$;
`cr_target` with $D$ instead of $D^\dagger$; `subspace` ignored in `qenvs.gates`; the
superoperator frame change conjugated. Two further mutants are caught by dedicated tests
rather than by `CHECKS`: `GateControl.leakage` dropping `subspace` in the open case (the
reference pulse then reads 0.124 instead of 1.4e-4, `test_open_env_leakage_uses_the_computational_subspace`)
and the operand order of `to_qubit_frame` reversed, unitary or superoperator
(`test_to_qubit_frame_operand_order`).

**Float32 and the matrix exponential.** `jax.scipy.linalg.expm` (jax 0.10, used by
`dynamiqs.method.Expm`) chooses the number of squarings as
$\lfloor\log_2(\|A\|_1/\theta)\rfloor$ ($\theta$ = 3.93 in single precision), so the scaled
matrix can have a norm up to $2\theta$, beyond the range of its degree-7 Pade approximant.
We measured single-precision errors of up to about $10^{-3}$ per step for
$\|H\,dt\|_1 \approx 26$ (in float64 the same effect is about $10^{-9}$). With 3 levels per
transmon $\|H\|_1$ is dominated by the $|22\rangle$ energy $|2\Delta + \alpha_c + \alpha_t|$, so the
largest errors sit in the phases of doubly excited states. The checks that compare whole
$9\times 9$ maps therefore use 0.5 ns steps ($\|H\,dt\|_1 < 3.93$, no scaling). On the default
task the calibrated reference pulse reaches $F$ = 0.984331 in float32 and 0.984316 in float64
(difference 1.5e-5). Use x64 (`jax.config.update("jax_enable_x64", True)`, which the CLI does) for this task.

## Assumptions and validity

* Duffing transmons with harmonic matrix elements $\sqrt{n}$ for drive and coupling; RWA on
  the coupling and on the drives (frame at $\omega_t$; valid for $|\Delta|, |\alpha|, J, \Omega \ll \omega$).
* Exchange coupling $J$ is a constant (no tunable coupler, no bus resonator); $J$ is the
  effective transmon-transmon coupling.
* Bare qubit frames, not dressed ones: the target and control frequencies used for the frames
  are the uncoupled ones. The dressed frequencies differ: at the defaults the target is shifted by
  +90 kHz or +206 kHz and the control by -90 kHz or +26 kHz depending on the state of the other
  qubit (exact diagonalisation), which shows up as static $I\otimes Z$, $Z\otimes I$ and
  $Z\otimes Z$ phases over the gate.
* Lindblad dissipation of the bare modes (harmonic $T_1$ ladder, white dephasing), the same
  during drive and idle; no $1/f$ noise, no measurement, no state preparation errors.
* Piecewise-constant controls on the `dt` grid; no transfer function, no microwave
  crosstalk between drive lines (Magesan and Gambetta model one; we do not).

## Known limitations

* Randomisation (`qenvs.wrappers.ParameterRandomization`): `detuning_ghz` is refused in 0.3
  because the target stays in the nominal frame. The supported parameters (`coupling_ghz` and the
  two anharmonicities) also shift the static dressing phases that the target keeps at their
  nominal values; estimated (not measured) at about 0.006 rad for a 10% coupling spread, of order
  1e-5 in infidelity, growing with the randomisation strength.
* No echo pulses: there is no drive at the control frequency, so the echoed CR sequence
  (control $\pi$ pulses) cannot be built. Single-qubit gates on the control are not available.
* No tunable couplers, no spectator qubits, no classical crosstalk.
* RWA on the coupling; counter-rotating coupling terms and Bloch-Siegert shifts are not modelled.
* Bare, not dressed, qubit frames (above), and no virtual-Z frame updates: Stark shifts and
  static phases count as errors, which makes the task stricter than a calibrated device.
* The reference pulse is a simple calibrated flat-top, not an optimised gate; its quality
  depends on `dt` (sampling note) and on the kwargs, and it raises `ValueError` when no width
  reaches the conditional rotation within `action_bound`. It chooses the sign of the CR
  amplitude from the first-order $\nu_{ZX}$, so it works for $\Delta > 0$ as well as $\Delta < 0$
  (checked at $\Delta/2\pi = \pm 0.1$ GHz and for `levels = 2`); the sign and the cancellation
  ratio are first-order results, so very close to a pole of the perturbative rate
  ($\Delta \approx 0, -\alpha_c$) or with `levels >= 3` far from the regime where $J \ll |\Delta|$
  they can be wrong and the calibration may fail or give a poor pulse.
* `info["fidelity"]` before the last step uses the end-of-episode frame (see Frames).
* Float32 accuracy of the default 5 ns steps is limited by the matrix exponential (above).
* Coherence parameters must be Python floats (the presence of jump operators is decided when
  the model is built).

## References

1. E. Magesan and J. M. Gambetta, "Effective Hamiltonian models of the cross-resonance gate",
   Phys. Rev. A 101, 052308 (2020), arXiv:1804.04073. Drive convention, static ZZ $\xi$
   (Eq. (4.5)), first-order ZX (Eq. (4.26)) and IX (Appendix C).
2. J. Koch, T. M. Yu, J. Gambetta, A. A. Houck, D. I. Schuster, J. Majer, A. Blais,
   M. H. Devoret, S. M. Girvin and R. J. Schoelkopf, "Charge-insensitive qubit design derived
   from the Cooper pair box", Phys. Rev. A 76, 042319 (2007). Transmon, negative anharmonicity.
3. P. Krantz, M. Kjaergaard, F. Yan, T. P. Orlando, S. Gustavsson and W. D. Oliver, "A quantum
   engineer's guide to superconducting qubits", Appl. Phys. Rev. 6, 021318 (2019).
4. L. H. Pedersen, N. M. Moller and K. Molmer, "Fidelity of quantum operations", Phys. Lett. A
   367, 47 (2007). Closed-form average gate fidelity.
5. C. J. Wood and J. M. Gambetta, "Quantification and characterization of leakage errors",
   Phys. Rev. A 97, 032306 (2018). Leakage and the fidelity of leaky channels.
6. Qiskit documentation, `qiskit.circuit.library.RZXGate` (matrix and qubit-order convention),
   https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.RZXGate.
