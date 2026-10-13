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
its parity and creates photons out of the vacuum.

The stored energy is measured by the battery Hamiltonian

$$
H_B = \omega_0 \left(J_z + \tfrac{N}{2}\right), \qquad 0 \le \langle H_B \rangle \le N\omega_0 ,
$$

which is $0$ for all cells in the ground state and $N\omega_0$ when fully charged.

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
* Tensor order: battery (first) $\otimes$ cavity (second), $\dim = (N+1)(n_{\max}+1)$.
* Initial state: all cells ground, cavity in the Fock state $|n\rangle$,
  $|{-j}\rangle \otimes |n\rangle$ (ket for closed models, density matrix when any rate is
  positive).
* Note that QuTiP's `jmat` orders $m$ from $+j$ down to $-j$, the opposite of this
  library; the QuTiP cross-check permutes the spin index accordingly.

## Assumptions and validity

* **Rotating-wave approximation** (`rwa=True`, the default) is justified for
  $g \ll \omega_0$ near resonance ($\omega_c \approx \omega_0$). The check
  `rwa_weak_coupling` shows that at $g = 0.01\,\omega_0$ the TC and Dicke energies agree
  to $6 \times 10^{-4}$ over a collective Rabi period, while at $g = 0.5\,\omega_0$ they
  differ by order 1 (`counter_rotating_terms`). The default task bound
  `action_bound = 0.5` therefore lets an agent leave the RWA regime; set a smaller bound
  if you need the TC model to be physically faithful.
* **Symmetric subspace.** Working in the $j = N/2$ sector is exact as long as the
  Hamiltonian and every dissipator are permutation invariant, which holds for the
  collective operators used here. *Local* (per-cell) decay or dephasing couples the
  symmetric sector to lower-$j$ sectors and is **not** supported in v0; a
  permutation-invariant (PIQS-style, Shammah et al. 2018) solver is planned.
* **Fock cutoff.** For TC the excitation number is conserved, so
  $n_{\max} = n$ is exact (the default). For the Dicke model the counter-rotating terms
  create photons; the default $n_{\max} = n + 4$ is a convenience, not a convergence
  guarantee. Increase `n_max` until your observables stop changing, especially for
  $g \gtrsim 0.1\,\omega_0$.
* Energies and rates are in units of $\omega_0$ (= 1 by default), times in $1/\omega_0$.

## Parameters

| field | default | meaning |
|---|---|---|
| `n_cells` | 3 | number of cells $N$ |
| `n_photons` | `None` ($= N$) | initial cavity Fock number $n$ (Ferraro protocol: $n = N$) |
| `n_max` | `None` ($n$ if `rwa` else $n + 4$) | Fock cutoff |
| `omega0` | 1.0 | cell frequency, energy unit of $H_B$ |
| `omega_c` | 1.0 | cavity frequency; detuning $\omega_c - \omega_0$ |
| `rwa` | `True` | Tavis-Cummings (`True`) or Dicke (`False`) |
| `kappa`, `gamma`, `gamma_phi` | 0.0 | collective dissipation rates (any $> 0$ switches to Lindblad dynamics) |

Observables reported in `info`: `energy` ($\langle H_B\rangle$), `n_photons`
($\langle a^\dagger a\rangle$), `excitation` ($\langle \hat N_{\mathrm{exc}}\rangle$), plus
the generic `ergotropy`, `power` and `purity` of `BatteryCharging`.

## Control semantics

One control channel, the coupling $g(t)$, piecewise constant over each control period
`dt` and clipped to $[-g_{\max}, g_{\max}]$ with $g_{\max} =$ `action_bound` (task
default 0.5). Physically $g \ge 0$; a negative $g$ is equivalent to $g \to |g|$ with a
phase redefinition of the cavity mode ($a \to -a$), which leaves every observable in this
model unchanged (also for the Dicke term, since it is linear in $a + a^\dagger$), so
negative actions are simply allowed. Setting $g = 0$ decouples the cells and holds the
stored charge (closed system), which makes "charge, then switch off" the natural
optimal strategy.

Task kwargs (`qenvs.make("battery/Dicke-v0", ...)`): `n_cells`, `n_photons`, `n_max`,
`omega_c`, `rwa`, `kappa`, `gamma`, `gamma_phi`, `horizon` (20), `dt` (0.2),
`action_bound` (0.5), `reward` (`"energy"`).

## Reward

Per-step gain of the stored quantity $Q$, so the undiscounted return equals
$Q(\text{final}) - Q(\text{initial})$ with $Q = \langle H_B\rangle$ (`reward="energy"`)
or the ergotropy of the cell state with respect to $H_B$ (`reward="ergotropy"`). The
return is bounded by $N\omega_0$. With the Ferraro protocol ($n = N$ photons) the
maximum energy reachable by a constant coupling is $N\omega_0$ only for $N = 1$; for
$N = 2, 4, 8$ the first energy peak reaches $0.89, 0.86, 0.88$ of $N\omega_0$ (the cells
and the cavity do not fully swap excitations for $N > 1$).

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
by `solver_convergence`. Every check passes in float64 and in float32.

| check | reference | tolerance | measured (float64) | measured (float32) |
|---|---|---|---|---|
| `operator_algebra` | $[J_+, J_-] = 2J_z$ for $N \le 6$; $[a, a^\dagger] = 1$ below the cutoff | 1e-12 | 3.6e-15 | 3.6e-15 |
| `jaynes_cummings_limit` | $N = 1$, $n \in \{1, 3\}$, resonant TC, $g = 0.5$: $P_e(t) = \sin^2(g\sqrt{n}\,t)$ (Jaynes-Cummings) | 1e-6 | 1.1e-8 | 4.2e-7 |
| `excitation_conservation` | closed TC, random piecewise-constant $g(t)$, $n_{\max} = 6 > n$: $\langle \hat N_{\mathrm{exc}}\rangle$ constant (renormalised state; norm drift reported in `detail`) | 1e-6 | 1.8e-15 | 1.3e-15 |
| `energy_bounds` | $0 \le \langle H_B\rangle \le N\omega_0$ along random rollouts (closed TC and open Dicke) | 1e-7 | 0 | 0 |
| `collective_advantage` | Ferraro et al. 2018: $t^*(N)$ strictly decreasing, log-log slope of $t^*$ vs $N$ within $-0.5 \pm 0.25$ for $N \in \{1, 2, 4, 8\}$, $g = 0.05$ | 0.25 (on the slope) | fitted exponent $-0.401$; $t^* = 31.42, 25.64, 18.37, 13.89$ | $-0.401$ |
| `rwa_weak_coupling` | Dicke ($n_{\max} = n + 6$) vs TC energy, $N = 2$, $g = 0.01\,\omega_0$, one collective Rabi period | 2e-3 | 5.8e-4 | 5.8e-4 |
| `counter_rotating_terms` | same at $g = 0.5\,\omega_0$: energies must differ by $> 0.05$ (error = shortfall) | 0 | 0 (difference 1.03) | 0 (1.03) |
| `qutip_tavis_cummings` | open TC, $N = n = 3$, $\kappa = 0.1$, $\gamma = 0.05$, $\gamma_\phi = 0.03$, random $g(t)$: final $\rho$ vs QuTiP 5 `mesolve` (max abs element) | 1e-5 | 5.8e-11 | 6.1e-8 |
| `qutip_dicke` | open Dicke, $N = n = 3$, $n_{\max} = 7$, $\kappa = 0.1$, random $g(t)$: final $\rho$ vs QuTiP 5 `mesolve` | 1e-5 | 5.8e-10 | 6.4e-8 |
| `density_matrix_constraints` | open Dicke after random driving: Hermitian, trace 1, positive semidefinite | 1e-6 | 1.1e-16 | 0 |
| `solver_convergence` | default Tsit5 (env rollout) vs exact `Expm` propagator, closed Dicke, dim 32, 8 random steps | 2e-5 | 4.4e-6 | 5.1e-6 |

Notes on the scaling check: the fitted exponent is $-0.40$ for $N \in \{1, 2, 4, 8\}$
(and also $-0.40$ when extended to $N = 16$, $t^* = 10.60$). This is the finite-$N$
exponent of the TC model with $n = N$ photons at $g = 0.05\,\omega_0$; the $N^{-1/2}$ law
of Ferraro et al. is the large-$N$ statement, so the check accepts a window
$[-0.75, -0.25]$ rather than asserting $-0.5$ exactly. The check is still
discriminating: it fails if $t^*$ does not decrease with $N$ (no collective advantage).

The QuTiP cross-checks (`qutip_tavis_cummings`, `qutip_dicke`) are the independent
confirmation of the operator construction, the basis ordering, the jump-operator
directions and the control sign: a reversed $J_-$, a factor 2 in $g$, or a mis-signed
counter-rotating term changes the final density matrix at the $10^{-2}$ level. Without
`qutip` installed these two checks report `error = nan` (FAIL) with an explanatory
`detail`.

## Known limitations

* Collective dissipation only. Local decay / dephasing (which breaks permutation
  symmetry) is not available; planned via a PIQS-style solver.
* `n_max` for `rwa=False` must be converged by the user; the default $n + 4$ is a
  starting point.
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
   optimization of the charging of a Dicke quantum battery", arXiv:2212.12397 (2022).
6. N. Shammah, S. Ahmed, N. Lambert, S. De Liberato, F. Nori, "Open quantum systems
   with local and collective incoherent processes: Efficient numerical simulations
   using permutational invariance", Phys. Rev. A 98, 063815 (2018). PIQS.
7. E. T. Jaynes, F. W. Cummings, "Comparison of quantum and semiclassical radiation
   theories with application to the beam maser", Proc. IEEE 51, 89 (1963). Single-cell
   limit used in `jaynes_cummings_limit`.
