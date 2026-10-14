# Dicke / Tavis-Cummings battery

`qenvs.models.DickeBattery`, task `battery/Dicke-v0` (environment `BatteryCharging`).

A collective quantum battery after Ferraro et al. (2018): $N$ identical two-level
systems (the cells) are charged by a single cavity mode (the charger) that initially
holds $n$ photons. All cells couple to the cavity with the same, time-dependent
strength $g(t)$, which is the control. Charging $N$ cells collectively is faster than
charging them one by one: the charging time shrinks with $N$ (collective advantage).

## Physics

With $\hbar = 1$, collective spin operators $J_\pm, J_z$ ($j = N/2$) and cavity
operators $a, a^\dagger$:

$$
H_{\mathrm{TC}}(t) = \omega_0 \left(J_z + \tfrac{N}{2}\right) + \omega_c\, a^\dagger a
+ g(t)\, \left(a^\dagger J_- + a J_+\right) \qquad (\texttt{rwa=True})
$$

$$
H_{\mathrm{Dicke}}(t) = \omega_0 \left(J_z + \tfrac{N}{2}\right) + \omega_c\, a^\dagger a
+ g(t)\, \left(a + a^\dagger\right)\left(J_+ + J_-\right) \qquad (\texttt{rwa=False})
$$

The two differ by the counter-rotating terms $a J_-$ and $a^\dagger J_+$. The
Tavis-Cummings (TC) Hamiltonian conserves the excitation number
$\hat N_{\mathrm{exc}} = a^\dagger a + J_z + N/2$; the Dicke Hamiltonian only conserves
its parity and creates photons out of the vacuum. There is no diamagnetic $A^2$ term:
this is the bare Dicke Hamiltonian as used by Ferraro et al. (2018).

The stored energy is measured by the battery Hamiltonian

$$
H_B = \omega_0 \left(J_z + \tfrac{N}{2}\right), \qquad 0 \le \langle H_B \rangle \le N\omega_0 ,
$$

which is $0$ for all cells in the ground state and $N\omega_0$ when fully charged.
Energy and ergotropy are properties of the cells alone: the model's
`battery_reduced(state)` traces out the cavity (`dq.ptrace(state, 0)`) and returns
$H_B$ on the $(N+1)$-dimensional Dicke space, and `BatteryCharging` evaluates both
quantities on that reduced state. For closed dynamics the global state stays pure, so
an ergotropy computed on the full cells+cavity state with $H_B \otimes 1$ would equal
the energy identically; on the reduced cell state it is strictly below the energy
whenever the cells are entangled with the cavity (see Reward).

Optional collective dissipators (Lindblad jump operators):

| rate | jump operator | meaning |
|---|---|---|
| `kappa` | $\sqrt{\kappa}\, a$ | cavity photon loss |
| `gamma` | $\sqrt{\gamma}\, J_-$ | collective (superradiant) emission |
| `gamma_phi` | $\sqrt{\gamma_\phi}\, J_z$ | collective dephasing |

For $N = 1$ the dephasing jump $\sqrt{\gamma_\phi}\,\sigma_z/2$ damps the coherence at
rate $\gamma_\phi / 2$, i.e. `DickeBattery(gamma_phi=x)` corresponds to
`TwoLevel(gamma_phi=x/2)`.

## Basis and index conventions

* Permutation-symmetric (Dicke) subspace $|j, m\rangle$, $j = N/2$, $m = -j, \dots, +j$,
  stored at indices $0, \dots, N$: **index 0 = all cells ground, index $N$ = all cells
  excited** (library rule: higher index = more excited).
* $J_+$ is built from $\langle j, m+1 | J_+ | j, m \rangle = \sqrt{j(j+1) - m(m+1)}$,
  $J_- = J_+^\dagger$, $J_z = \mathrm{diag}(-j, \dots, j)$. For $N = 1$ this gives
  $J_+ = |1\rangle\langle 0|$, $J_z = \sigma_z / 2$ (`collective_spin_matrices(1)`).
* Cavity Fock space $|0\rangle, \dots, |n_{\max}\rangle$ with $a|n\rangle = \sqrt{n}\,|n-1\rangle$
  (`boson_matrices(n_max)`).
* Tensor order: battery (first) $\otimes$ cavity (second), $\dim = (N+1)(n_{\max}+1)$;
  the qarrays carry `dims = (N + 1, n_max + 1)`, so `dq.ptrace(state, 0)` is the cell
  state.
* Initial state: all cells ground, cavity in the Fock state $|n\rangle$,
  $|{-j}\rangle \otimes |n\rangle$ (ket for closed models, density matrix when any rate is
  positive).
* Note that QuTiP's `jmat` orders $m$ from $+j$ down to $-j$, the opposite of this
  library; the QuTiP cross-checks permute the spin index accordingly.

## Assumptions and validity

* **Rotating-wave approximation** (`rwa=True`, the default) is justified for
  $g \ll \omega_0$ near resonance ($\omega_c \approx \omega_0$). The check
  `rwa_weak_coupling` shows that at $g = 0.01\,\omega_0$ the TC and Dicke energies agree
  to $6 \times 10^{-4}$ for $t \in [0, \pi/(g\sqrt{N})]$, while at $g = 0.5\,\omega_0$
  they differ by order 1 (`counter_rotating_terms`). The default task bound
  `action_bound = 0.5` therefore lets an agent leave the RWA regime; set a smaller bound
  if you need the TC model to be physically faithful.
* **Symmetric subspace.** Working in the $j = N/2$ sector is exact as long as the
  Hamiltonian and every dissipator are permutation invariant, which holds for the
  collective operators used here. *Local* (per-cell) decay or dephasing couples the
  symmetric sector to lower-$j$ sectors and is **not** supported in v0; a
  permutation-invariant (PIQS-style, Shammah et al. 2018) solver is planned.
* **Fock cutoff (read this before using `rwa=False`).** For TC the excitation number
  is conserved, so $n_{\max} = n$ is exact (the default). For the Dicke model the
  counter-rotating terms create photons, and the default $n_{\max} = n + 4$ is a
  convenience, not a convergence guarantee. The cutoff that is needed grows with $N$
  and with the coupling. Measured for the Ferraro protocol ($n = N$), comparing the
  stored energy with a reference at a much larger cutoff:

  | coupling | $N$ | default $n_{\max} = n + 4$ | converged return (cutoff needed) |
  |---|---|---|---|
  | $g = 0.1\,\omega_0$, 20 steps of `dt = 0.2` | 2, 3, 4 | max energy error $8.5\times10^{-6}$, $1.1\times10^{-4}$, $5.2\times10^{-4}$ | adequate |
  | $g = 0.1\,\omega_0$ | 6, 8 | $4.2\times10^{-3}$, $4.3\times10^{-2}$ | raise $n_{\max}$ |
  | $g = 0.5\,\omega_0$ (task default bound), 20 steps | 2 | return $1.017$ | $1.0005$ ($n + 10$) |
  | $g = 0.5\,\omega_0$ | 3 | $1.754$ (15 percent off) | $1.530$ ($n + 24$) |
  | $g = 0.5\,\omega_0$ | 6 | $2.608$ | $2.604$ ($n + 56$; $n + 24$ still gives $2.752$) |

  The $N = 6$ row shows that a return close to the converged value at the default
  cutoff can be accidental, and that a fixed "$n + 24$" rule does not generalise. The
  reliable criterion is the `top_fock` observable (population of the highest kept Fock
  level): raise `n_max` until its maximum over the episode is below about $10^{-4}$ and
  the return stops changing. In the table, `top_fock` reaches $0.07$ to $0.10$ at the
  default cutoff and $1.3\times10^{-4}$ for $N = 6$ at $n + 56$. The check
  `cutoff_convergence` covers the weak-coupling case $N = 2$, $g = 0.1\,\omega_0$
  only.
* Energies and rates are in units of $\omega_0$ (= 1 by default), times in $1/\omega_0$.

## Parameters

| field | default | meaning |
|---|---|---|
| `n_cells` | 3 | number of cells $N$ |
| `n_photons` | `None` ($= N$) | initial cavity Fock number $n$ (Ferraro protocol: $n = N$) |
| `n_max` | `None` ($n$ if `rwa` else $n + 4$) | Fock cutoff (see above for `rwa=False`) |
| `omega0` | 1.0 | cell frequency, energy unit of $H_B$ |
| `omega_c` | 1.0 | cavity frequency; detuning $\omega_c - \omega_0$ |
| `rwa` | `True` | Tavis-Cummings (`True`) or Dicke (`False`) |
| `kappa`, `gamma`, `gamma_phi` | 0.0 | collective dissipation rates (any $> 0$ switches to Lindblad dynamics) |

Observables reported in `info`: `energy` ($\langle H_B\rangle$ of the reduced cell
state), `ergotropy` (of the reduced cell state with respect to $H_B$), `n_photons`
($\langle a^\dagger a\rangle$), `excitation` ($\langle \hat N_{\mathrm{exc}}\rangle$),
`top_fock` (population of the Fock level $|n_{\max}\rangle$, a truncation monitor), plus
the generic `power` and `purity` (of the full cells+cavity state) of `BatteryCharging`.

## Control semantics

One control channel, the coupling $g(t)$, piecewise constant over each control period
`dt` and clipped to $[-g_{\max}, g_{\max}]$ with $g_{\max} =$ `action_bound` (task
default 0.5). Physically $g \ge 0$. Replacing $g \to -g$ for the *whole* protocol is a
gauge symmetry: $U = (-1)^{a^\dagger a}$ maps $a \to -a$ and leaves every observable
unchanged (also for the Dicke term, which is linear in $a + a^\dagger$), so negative
actions are allowed. A sign change *within* a protocol is physical and can reverse
charging: for $N = n = 1$ the steps $[+g, -g]$ with $g\,\mathrm{dt} = \pi/4$ return the
cell to the ground state while $[+g, +g]$ charges it fully (tested). Setting $g = 0$
decouples the cells and holds the stored charge (closed system), which makes "charge,
then switch off" the natural optimal strategy.

Task kwargs (`qenvs.make("battery/Dicke-v0", ...)`): `n_cells`, `n_photons`, `n_max`,
`omega0`, `omega_c`, `rwa`, `kappa`, `gamma`, `gamma_phi`, `horizon` (20), `dt` (0.2),
`action_bound` (0.5), `reward` (`"energy"`).

## Reward

Per-step gain of the stored quantity $Q$, so the undiscounted return equals
$Q(\text{final}) - Q(\text{initial})$ with $Q = \langle H_B\rangle$ (`reward="energy"`)
or the ergotropy of the reduced cell state $\rho_B = \mathrm{tr}_{\mathrm{cav}}\,\rho$
with respect to $H_B$ (`reward="ergotropy"`). The return is bounded by $N\omega_0$.
For closed dynamics the ergotropy is strictly below the energy whenever the cells and
the cavity are entangled, so the two rewards are different objectives: for $N = 2$,
constant $g = 0.5$ and $t = 1$ the energy is $0.784\,\omega_0$ but the ergotropy is
only $0.277\,\omega_0$ (while the global state is pure, `purity` = 1). The two
coincide only when the reduced cell state is pure, e.g. at full charge.

For Tavis-Cummings charging from a Fock state the reduced cell state stays diagonal in
the Dicke basis (the excitation number is conserved), so its ergotropy is exactly zero
until the cell populations become inverted. `reward="ergotropy"` is then a sparse
reward with an exactly zero gradient on that plateau (for example $N = 2$, constant
$g = 0.3$ for 6 steps of `dt = 0.2` gives return 0). Use `reward="energy"` for early
training or gradient-based optimisation and evaluate the ergotropy afterwards.

Closed-form landmarks (resonant TC, constant $g$):

* $N = n = 1$ (Jaynes-Cummings): $P_e(t) = \sin^2(g t)$. At $g t = \pi/4$ the reduced
  cell state is $\mathrm{diag}(1/2, 1/2)$: energy $\omega_0/2$, ergotropy $0$. At
  $g t = \pi/2$ the cell is fully excited: ergotropy = energy = $\omega_0$
  (`reduced_ergotropy`).
* $N = n = 2$: in the two-excitation manifold the coupling matrix has eigenfrequencies
  $0$ and $\pm\sqrt{6}\,g$, so
  $E(t)/\omega_0 = \tfrac{2}{3}\sin^2(\sqrt{6}\,g t) + \tfrac{4}{9}\,(1 - \cos\sqrt{6}\,g t)^2$,
  with the first maximum $E_{\max} = \tfrac{8}{9}\,N\omega_0$ at
  $t^* = \pi/(\sqrt{6}\,g)$ (`two_cells_closed_form`).

With the Ferraro protocol ($n = N$ photons) and a constant coupling, the first energy
peak reaches $N\omega_0$ only for $N = 1$; for $N = 2, 4, 8$ it reaches
$0.89, 0.86, 0.88$ of $N\omega_0$, because the cells and the cavity do not fully swap
excitations for $N > 1$. Later peaks were not searched; time-dependent couplings (the
point of the task) can do better.

```python
import jax.numpy as jnp
import qenvs

env = qenvs.make("battery/Dicke-v0", n_cells=3, horizon=20, dt=0.2)
# constant coupling g = 0.5 for the first 10 steps, then decouple and hold
actions = jnp.concatenate([jnp.full((10, 1), 0.5), jnp.zeros((10, 1))])
state, rewards = env.rollout(actions)
print(float(rewards.sum()))  # 2.58: stored energy in units of omega0 (max 3)
```

## Validation

All numbers below are from `qenvs.validation.dicke.CHECKS` on this machine
(dynamiqs 0.3.6, JAX 0.7.2, QuTiP 5.3.1, CPU). The analytic and cross-check
integrations use a tight adaptive Tsit5 (`rtol=1e-8, atol=1e-10`) so that they
measure physics rather than the default solver tolerance; the default solver is covered
by `solver_convergence` (closed) and `open_solver_convergence` (Lindblad). Every check
passes in float64 and in float32.

| check | reference | tolerance | measured (float64) | measured (float32) |
|---|---|---|---|---|
| `operator_algebra` | $[J_+, J_-] = 2J_z$ for $N \le 6$; $[a, a^\dagger] = 1$ below the cutoff | 1e-12 | 3.6e-15 | 3.6e-15 |
| `jaynes_cummings_limit` | $N = 1$, $n \in \{1, 3\}$, resonant TC, $g = 0.5$: $P_e(t) = \sin^2(g\sqrt{n}\,t)$ (Jaynes-Cummings) | 3e-6 | 1.1e-8 | 4.2e-7 |
| `detuned_jaynes_cummings` | $N = 1$, $g = 0.5$, $(\omega_0, \omega_c, n) \in \{(1, 1.7, 1), (1, 0.3, 2), (1.3, 1.0, 1)\}$: $P_e = \frac{4g^2 n}{\Omega^2}\sin^2\frac{\Omega t}{2}$, $\Omega^2 = (\omega_c - \omega_0)^2 + 4g^2 n$, $\langle a^\dagger a\rangle = n - P_e$, and the coupling expectation $\langle V\rangle = (d/g)\,P_e$ with $d = \omega_c - \omega_0$ (odd in $d$, so it fixes the sign of the detuning) | 3e-6 | 1.3e-8 | 7.2e-7 |
| `two_cells_closed_form` | $N = n = 2$, $g = 0.5$: $t^* = \pi/(\sqrt{6}\,g)$ (grid of 601 points, parabolic refinement) and $E(t^*) = \tfrac{8}{9} N\omega_0$; error $= \max(|\Delta t^*|, 100\,|\Delta E_{\max}|)$ so the single tolerance enforces 1e-4 on $t^*$ and 1e-6 on $E_{\max}$ | 1e-4 | 7.5e-7 ($|\Delta t^*| = 1.7\times10^{-8}$, $|\Delta E_{\max}| = 7.4\times10^{-9}$) | 2.0e-5 ($2.0\times10^{-5}$, $1.3\times10^{-8}$) |
| `reduced_ergotropy` | $N = n = 1$: reduced cell state at $g t = \pi/4$ has $E = \omega_0/2$, ergotropy $0$; at $g t = \pi/2$ ergotropy $= E = \omega_0$ | 1e-6 | 3.8e-9 | 1.2e-7 |
| `omega0_scaling` | resonant TC, $N = 2$, same random $g(t)$ with $\omega_0 = 1$ and $1.5$: energy and ergotropy scale exactly by $1.5$ (max abs residual over 10 env steps) | 1e-6 | 1.3e-9 | 4.1e-8 |
| `excitation_conservation` | closed TC, random piecewise-constant $g(t)$, $n_{\max} = 6 > n$: $\langle \hat N_{\mathrm{exc}}\rangle = n_{\mathrm{photons}}$ at every time, including $t = 0$ (renormalised state; norm drift reported in `detail`) | 1e-6 | 1.3e-15 | 1.3e-15 |
| `ergotropy_bounds` | $0 \le \mathcal{E}(\rho_B) \le \langle H_B\rangle \le N\omega_0$ along 15-step random rollouts (closed TC and open Dicke), plus $|\mathrm{tr}\rho - 1|$ for the open rollout; min gap $E - \mathcal{E}$ in `detail` ($1.7\times10^{-3}$ closed, $0.175$ open) | 1e-6 | 6.7e-16 | 2.4e-7 |
| `collective_advantage` | Ferraro et al. 2018: $t^*(N)$ strictly decreasing, log-log slope of $t^*$ vs $N$ within $-0.5 \pm 0.25$ for $N \in \{1, 2, 4, 8\}$, $g = 0.05$ | 0.25 (on the slope) | fitted exponent $-0.401$; $t^* = 31.42, 25.64, 18.37, 13.89$ | $-0.401$ |
| `rwa_weak_coupling` | Dicke ($n_{\max} = n + 6$) vs TC energy, $N = 2$, $g = 0.01\,\omega_0$, $t \in [0, \pi/(g\sqrt{N})]$ | 2e-3 | 5.8e-4 | 5.8e-4 |
| `counter_rotating_terms` | same at $g = 0.5\,\omega_0$: energies must differ by $> 0.05$ (error = shortfall) | 0 | 0 (difference 1.03) | 0 (1.03) |
| `cutoff_convergence` | closed Dicke, $N = 2$, constant $g = 0.1\,\omega_0$, 20 steps of `dt = 0.2`: $\langle H_B\rangle(t)$ with $n_{\max} = 6$ (default) vs $12$; max `top_fock` population at $n_{\max} = 6$ in `detail` ($2.4\times10^{-5}$) | 1e-3 | 8.5e-6 | 8.3e-6 |
| `qutip_tavis_cummings` | open TC, $N = n = 3$, $\kappa = 0.1$, $\gamma = 0.05$, $\gamma_\phi = 0.03$, random $g(t)$: final $\rho$ vs QuTiP 5 `mesolve` (max abs element) | 1e-5 | 5.8e-11 | 6.1e-8 |
| `qutip_dicke` | open Dicke, $N = n = 3$, $n_{\max} = 7$, $\kappa = 0.1$, random $g(t)$: final $\rho$ vs QuTiP 5 `mesolve` | 1e-5 | 5.8e-10 | 6.4e-8 |
| `qutip_detuned` | open TC, $N = n = 2$, $\omega_0 = 0.8$, $\omega_c = 1.3$, $\kappa = 0.05$, $\gamma = 0.02$, random $g(t)$: final $\rho$ vs QuTiP 5 built with explicit $\omega_0$, $\omega_c$ (the coherences fix the sign of the detuning) | 1e-5 | 4.0e-11 | 8.7e-8 |
| `collective_dissipators` | $N = 1$, $g = 0$, no QuTiP: $\langle a^\dagger a\rangle = e^{-\kappa t}$ (cavity loss), $\langle H_B\rangle/\omega_0 = e^{-\gamma t}$ from $|e\rangle$ (emission), $2|\langle J_-\rangle| = e^{-\gamma_\phi t/2}$ from $(|g\rangle + |e\rangle)/\sqrt{2}$ (dephasing), $t \le 4$ | 2e-6 | 4.7e-9 | 1.7e-7 |
| `density_matrix_constraints` | open Dicke after random driving: Hermitian, trace 1, positive semidefinite | 1e-6 | 1.1e-16 | 0 |
| `solver_convergence` | default Tsit5 (env rollout) vs exact `Expm` propagator, closed Dicke, dim 32, 8 random steps | 2e-5 | 4.4e-6 | 5.1e-6 |
| `open_solver_convergence` | default Tsit5 vs `Tsit5(rtol=1e-10, atol=1e-12)`, open Dicke ($\kappa = 0.1$, $\gamma = 0.05$), dim 21, 8 random steps, final $\rho$ | 2e-5 | 5.4e-6 | 6.1e-6 |

Notes on the scaling check: the fitted exponent is $-0.40$ for $N \in \{1, 2, 4, 8\}$
and stays at about $-0.40$ when the fit is extended offline to $N = 16, 32, 48$
($t^* = 10.60, 8.06, 6.87$). This is the finite-$N$ exponent of the TC model with $n = N$
photons at $g = 0.05\,\omega_0$ in this protocol (constant coupling, first energy
peak). Ferraro et al. state an $N^{-1/2}$ law; whether this protocol approaches $-1/2$ at
larger $N$ is not established here. The check therefore accepts a window
$[-0.75, -0.25]$ rather than asserting $-0.5$. It is still discriminating: it fails if
$t^*$ does not decrease with $N$ (no collective advantage).

Which check catches what (verified by mutating the model): a drift that uses
$\omega_0$ for the cavity fails `detuned_jaynes_cummings` and `qutip_detuned`; a flipped
detuning sign leaves every population unchanged and is caught by the $\langle V\rangle$
term of `detuned_jaynes_cummings` and by `qutip_detuned` (coherences); a reversed
cavity-loss or emission operator or a wrong dephasing prefactor fails
`collective_dissipators` (no QuTiP needed); an $H_B$ without $\omega_0$ fails `detuned_jaynes_cummings`,
`omega0_scaling` and `qutip_detuned`; ergotropy evaluated on the full cells+cavity state
fails `reduced_ergotropy`; a missing $+N/2$ offset fails `excitation_conservation` (and
`reduced_ergotropy`, `ergotropy_bounds` if it is in $H_B$); a task factory that drops a
kwarg fails `test_task_factory_forwards_every_model_kwarg`. The QuTiP cross-checks are
the independent confirmation of the operator construction, the basis ordering, the
jump-operator directions and the control sign: a reversed $J_-$, a factor 2 in $g$, or
a mis-signed counter-rotating term changes the final density matrix at the $10^{-2}$
level. Without `qutip` installed the three QuTiP checks are reported as skipped
(status `SKIP` in `qenvs validate`, `skipped: true` in `--json`) with an explanatory
`detail`; they never count as passed.

## Known limitations

* Collective dissipation only. Local decay / dephasing (which breaks permutation
  symmetry) is not available; planned via a PIQS-style solver.
* `n_max` for `rwa=False` must be converged by the user. The default $n + 4$ is
  adequate only for weak coupling and small $N$ ($|g| \lesssim 0.1\,\omega_0$,
  $N \le 4$); at the task default `action_bound = 0.5` the needed cutoff grows with $N$
  (see the table in Assumptions and validity). Raise `n_max` until `top_fock` stays
  below about $10^{-4}$.
* The symmetric subspace is $N + 1$ dimensional, but the open-system Liouvillian is
  $((N+1)(n_{\max}+1))^2$ dimensional, so open-system episodes with large $N$ and large
  cutoffs are slow on CPU.
* Only the coupling $g(t)$ is a control; cavity drive and detuning control are not part
  of `-v0`.
* No thermal cavity or thermal cell bath in `-v0` (initial cavity state is a pure Fock
  state).
* The environment clips actions to $[-g_{\max}, g_{\max}]$ and casts them to float32.

## References

1. D. Ferraro, M. Campisi, G. M. Andolina, V. Pellegrini, M. Polini,
   "High-Power Collective Charging of a Solid-State Quantum Battery",
   Phys. Rev. Lett. 120, 117702 (2018). Dicke battery, collective charging advantage.
2. M. Tavis, F. W. Cummings, "Exact Solution for an N-Molecule-Radiation-Field
   Hamiltonian", Phys. Rev. 170, 379 (1968).
3. R. H. Dicke, "Coherence in Spontaneous Radiation Processes",
   Phys. Rev. 93, 99 (1954).
4. G. M. Andolina, M. Keck, A. Mari, V. Giovannetti, M. Polini, "Quantum versus
   classical many-body batteries", Phys. Rev. B 99, 205437 (2019). Ergotropy and
   collective charging of Dicke batteries.
5. P. A. Erdman, G. M. Andolina, V. Giovannetti, F. Noé, "Reinforcement learning
   optimization of the charging of a Dicke quantum battery", Phys. Rev. Lett. 133,
   243602 (2024), arXiv:2212.12397.
6. N. Shammah, S. Ahmed, N. Lambert, S. De Liberato, F. Nori, "Open quantum systems
   with local and collective incoherent processes: Efficient numerical simulations
   using permutational invariance", Phys. Rev. A 98, 063815 (2018). PIQS.
7. E. T. Jaynes, F. W. Cummings, "Comparison of quantum and semiclassical radiation
   theories with application to the beam maser", Proc. IEEE 51, 89 (1963). Single-cell
   limit used in `jaynes_cummings_limit` and `detuned_jaynes_cummings`.
