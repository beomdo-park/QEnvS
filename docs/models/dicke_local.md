# Dicke battery with local dissipation

`qenvs.models.dicke_local.DickeBatteryLocal`, task `battery/DickeLocal-v0` (environment
`BatteryCharging`).

The same collective quantum battery as [`battery/Dicke-v0`](dicke.md) ($N$ two-level
cells charged by one cavity mode through a controlled coupling $g(t)$), with dissipation
that acts on every cell separately. Local emission, dephasing and pumping drive the cells
out of the permutation-symmetric (Dicke) subspace, so the $(N+1)$-dimensional description
of `battery/Dicke-v0` no longer applies. This model uses the permutation-invariant method
of Shammah et al. (2018): it keeps every total-spin sector $j = N/2, N/2 - 1, \dots$ and is
exact. Only the *Liouvillian* dimension grows polynomially in $N$ (instead of as $4^N$).
The stored state, and the env's ergotropy, purity and observation, are still the full
$2^N (n_{\max}+1)$-dimensional density matrix and therefore cost $O(4^N)$ memory and time;
this is why the supported range is $N \le 5$ (see Scaling and limits).
`battery/Dicke-v0` is unchanged.

## Physics

With $\hbar = 1$, collective operators $J_\pm = \sum_i \sigma_\pm^{(i)}$,
$J_z = \sum_i \sigma_z^{(i)}/2$ and cavity operators $a, a^\dagger$, the Hamiltonians are
those of `battery/Dicke-v0`:

$$
H_{\mathrm{TC}}(t) = \omega_0 \left(J_z + \tfrac{N}{2}\right) + \omega_c\, a^\dagger a
+ g(t)\, \left(a^\dagger J_- + a J_+\right) \qquad (\texttt{rwa=True})
$$

$$
H_{\mathrm{Dicke}}(t) = \omega_0 \left(J_z + \tfrac{N}{2}\right) + \omega_c\, a^\dagger a
+ g(t)\, \left(a + a^\dagger\right)\left(J_+ + J_-\right) \qquad (\texttt{rwa=False})
$$

The state obeys the Lindblad equation
$\dot\rho = -i[H, \rho] + \sum_k \mathcal{D}[L_k]\rho$,
$\mathcal{D}[L]\rho = L\rho L^\dagger - \tfrac12\{L^\dagger L, \rho\}$, with

| rate | jump operator(s) | kind |
|---|---|---|
| `kappa` | $\sqrt{\kappa}\, a$ | cavity loss |
| `gamma` | $\sqrt{\gamma}\, J_-$ | collective emission |
| `gamma_phi` | $\sqrt{\gamma_\phi}\, J_z$ | collective dephasing |
| `gamma_local` | $\sqrt{\gamma_{\mathrm{loc}}}\, \sigma_-^{(i)}$, $i = 1..N$ | local emission (each cell decays at $\gamma_{\mathrm{loc}}$) |
| `gamma_phi_local` | $\sqrt{\gamma_{\phi,\mathrm{loc}}}\, \sigma_z^{(i)}/2$ | local dephasing (a single-cell coherence decays at $\gamma_{\phi,\mathrm{loc}}/2$) |
| `pumping_local` | $\sqrt{P_{\mathrm{loc}}}\, \sigma_+^{(i)}$ | local incoherent pumping |

with $\sigma_- = |0\rangle\langle 1|$ (index 1 = excited, built explicitly),
$\sigma_+ = \sigma_-^\dagger$ and $\sigma_z = |1\rangle\langle 1| - |0\rangle\langle 0|$.
The local dephasing convention matches the collective one ($J_z = \sigma_z/2$ for one cell),
and both match QuTiP PIQS (`dephasing`, `collective_dephasing`). For $N = 1$ local and
collective processes coincide (tested).

The stored energy is $\langle H_B\rangle$ with $H_B = \omega_0 (J_z + N/2)$
($0 \le \langle H_B \rangle \le N\omega_0$).

## Method

**Permutation invariance.** Every term of the generator is invariant under permutations
of the cells, so a permutation-invariant initial state (here: all cells ground) stays
permutation invariant. By Schur-Weyl duality the joint cells+cavity state is then block
diagonal in the total-spin (Schur) basis $|j, m, \alpha\rangle \otimes |n\rangle$:

$$
\rho = \bigoplus_{j} R_j \otimes \mathbb{1}_{d_j},
\qquad
d_j = \binom{N}{N/2 - j} - \binom{N}{N/2 - j - 1},
\qquad \sum_j d_j (2j+1) = 2^N ,
$$

where $R_j$ acts on spin $j$ $\otimes$ cavity and $\mathbb{1}_{d_j}$ on the multiplicity
index $\alpha$. The dynamics closes on the blocks $\sigma_j = d_j R_j$ (normalised as
$\sum_j \mathrm{tr}\,\sigma_j = 1$, the convention of Shammah et al. and QuTiP PIQS).

**Liouvillian.** The blocks are vectorised row-major
($\mathrm{vec}(AXB) = (A \otimes B^T)\,\mathrm{vec}(X)$) and concatenated. In that vector:

* the Hamiltonian and the collective dissipators act sector by sector,
  $-i(H_j \otimes 1 - 1 \otimes H_j^T)$ and
  $L \otimes L^* - \tfrac12 L^\dagger L \otimes 1 - \tfrac12\, 1 \otimes (L^\dagger L)^T$
  with the spin-$j$ $\otimes$ cavity matrices (collective operators never change $j$);
* each local process couples $\sigma_{j, m, m'}$ to $\sigma_{j', m \pm 1, m' \pm 1}$
  (emission, pumping) or $\sigma_{j', m, m'}$ (dephasing) with $j' \in \{j-1, j, j+1\}$,
  through the closed-form coefficients of Shammah et al. (2018). They are transcribed from
  the QuTiP PIQS implementation (`qutip/piqs`, BSD-3-Clause; N. Shammah et al., Phys. Rev. A
  98, 063815 (2018)) and evaluated in float64. The cavity indices are spectators of the
  local processes.

The Liouvillian has dimension $\sum_j ((2j+1)(n_{\max}+1))^2$
(`model.liouvillian_dim`), against $(2^N (n_{\max}+1))^2$ for the brute-force
Hilbert-space treatment. Each control step is exact for piecewise-constant $g$:
$\sigma(t + \mathrm{d}t) = \exp(\mathcal{L}(g)\,\mathrm{d}t)\,\sigma(t)$ with a dense
`jax.scipy.linalg.expm`. It is differentiable, `jit` and `vmap` compatible, and the env's
`method` argument is ignored (`model.evolve` rejects time-dependent Hamiltonians).
`model.liouvillian(H)` and `model.dissipator()` expose the generator.

## State representation and degeneracies

`EnvState.qstate` holds the *full* $2^N (n_{\max}+1)$-dimensional density matrix of cells
and cavity, written in the Schur basis (a fixed unitary change of the cells' basis): it is
block diagonal with $d_j$ identical copies of $R_j = \sigma_j / d_j$. The qarray carries
`dims = (2^N, n_max + 1)`. Because it is unitarily equivalent to the computational-basis
state, everything the generic `BatteryCharging` env computes is exact without special
cases: the trace, the purity $\mathrm{tr}\,\rho^2$ (of the full joint state), expectation
values of the collective observables, and the partial trace over the cavity.

Ordering inside the stored matrix: sectors $j = N/2, N/2-1, \dots$; inside a sector the
copies $\alpha = 0, \dots, d_j - 1$; inside a copy $m = -j, \dots, +j$ (index 0 = lowest
$m$, library rule "higher index = more excited") and then the Fock index $n$. The initial
state (all cells ground, cavity in $|n_{\mathrm{photons}}\rangle$) lies in the $j = N/2$
sector. Helpers: `to_blocks` / `from_blocks` (compact vector of the $\sigma_j$),
`dicke_density` (cell state in the Dicke basis, $(\sum_j (2j+1))$-dimensional, unit trace;
the representation of PIQS up to the order of $m$), `sectors` ($(j, d_j)$ pairs).

**Energy and ergotropy.** `battery_reduced(state)` traces out the cavity and returns the
$2^N$-dimensional cell state with $H_B$ on that space (levels $k\omega_0$, multiplicity
$\binom{N}{k}$). The spectrum of the cell state consists of the eigenvalues of every cell
block $\sigma_j$ divided by $d_j$, each repeated $d_j$ times; the ergotropy pairs them in
decreasing order with the $2^N$ levels in increasing order. `ergotropy_from_blocks` computes
the same number from the blocks directly; the check `env_return_equals_gain` asserts that
the two routes agree.

This is the ergotropy over *all* unitaries on the $2^N$ cell space. `battery/Dicke-v0`
evaluates the ergotropy of its $(N+1)$-dimensional reduced state with $H_B$ on the
symmetric subspace, i.e. over unitaries that stay in that subspace, which is smaller or
equal. With collective processes only, the two models agree on the state, energy, photon
number and energy reward to roundoff ($10^{-15}$ in float64) when the reference is
`DickeBattery` stepped with the exact `Expm` propagator, as in the check `collective_limit`.
Through the env with the default adaptive `Tsit5` of `battery/Dicke-v0` and its float32-cast
actions the difference is about $10^{-7}$ (the solver tolerance). Their `ergotropy` values
differ: for the
default $N = 3$, $g = 0.5$ for 10 steps then 0, `battery/Dicke-v0` reports $2.164\,\omega_0$
and this model $2.304\,\omega_0$ for the same state (check `collective_limit` reports the
gap: $0.06\,\omega_0$ in float64 and $0.12\,\omega_0$ in float32, whose random pulses differ).

## Assumptions and validity

* **Permutation invariance** of the generator and of the initial state. Inhomogeneous
  cells (different frequencies, couplings or rates per cell) are outside this method.
* **Rotating-wave approximation** for `rwa=True` and the Fock cutoff for `rwa=False`: the
  caveats of [`battery/Dicke-v0`](dicke.md) apply unchanged.
* **Fock cutoff.** For `rwa=True` without local pumping the default cutoff
  `n_max = n_photons` is exact: the excitation number $a^\dagger a + J_z + N/2$ is conserved
  by the Hamiltonian and the dissipators only lower it. With `pumping_local > 0` this is
  false: a pumped cell can emit into the cavity and be pumped again, so the excitation
  number is unbounded and **no finite cutoff is exact** (at the former default
  $n_{\max} = n_{\mathrm{photons}} + N$ the energies were 30 to 50 percent off, for example
  $N = 1$, $n_{\mathrm{photons}} = 0$, $g = 0.5$ for 40 steps). The model therefore raises
  `ValueError` at construction when `pumping_local > 0` and `n_max` is not given. Choose
  `n_max` and raise it until the population of the top Fock level (`model.top_fock()`,
  `info['top_fock']`, as in `battery/Dicke-v0`) is negligible, for example by comparing
  `n_max` and `n_max + 4`. Measured (float64, $N = 1$, $n_{\mathrm{photons}} = 0$,
  `gamma_local=0.05`, $g = 0.5$ for 40 steps, `pumping_local=0.3`): the energy changes by
  $6.7\times10^{-2}$ from `n_max=2` to 6, $1.9\times10^{-3}$ from 4 to 8 and
  $1.1\times10^{-5}$ from 6 to 10 (energy 0.4823), with `top_fock` $2.8\times10^{-1}$,
  $1.7\times10^{-2}$ and $2.0\times10^{-4}$ at the smaller cutoff of each pair; the error
  stays below the `top_fock` value (`test_pumping_truncation_error_shrinks_and_top_fock_flags_it`).
  For `rwa=False` the default $n_{\mathrm{photons}} + 4$ is not a convergence guarantee
  either; watch `top_fock`.
* **Markovian, zero-temperature** dissipation with constant rates. Rates are static model
  parameters: gradients with respect to them are not supported.
* Energies and rates in units of $\omega_0$ (= 1 by default), times in $1/\omega_0$.

## Scaling and limits

Dense matrices throughout. The constructor raises `ValueError` if the Liouvillian
dimension exceeds `MAX_LIOUVILLIAN_DIM = 2100` or the stored matrix dimension
$2^N(n_{\max}+1)$ exceeds `MAX_HILBERT_DIM = 512`.

| $N$ | $n_{\max}$ | Liouvillian dim | stored dim | jitted env step (CPU, float64) |
|---|---|---|---|---|
| 3 | 3 (task default) | 320 | 32 | 0.07 s |
| 4 | 4 | 875 | 80 | 0.21 s |
| 5 | 5 | 2016 | 192 | 1.1 s |
| 6 | 6 | 4116 | 448 | rejected |

The Ferraro protocol with `n_photons = N` and the exact Tavis-Cummings cutoff therefore
runs up to $N = 5$; larger $N$ is possible only with a smaller cavity (e.g. $N = 6$,
$n_{\max} \le 4$; $N = 7$, $n_{\max} \le 3$). Timings were measured on a shared 64-core
machine under heavy load with `OPENBLAS_NUM_THREADS=8`. Without that limit the same steps
were 10 to 40 times slower on that machine: on a busy many-core host, cap the BLAS threads.
Building the Liouvillian is cheap (below 0.1 s); compilation of the first step takes 0.4 s
to 2 s.

**Gradient memory.** `jit(grad(rollout))` stores the dense propagator of every step. Peak
resident memory, measured here (float64, CPU, `n_max = N`, default `dt`):

| $N$ | horizon | peak RSS |
|---|---|---|
| 3 | 20 | 1.9 GB |
| 4 | 5 | 3.1 GB |
| 4 | 20 | 8.8 GB |
| 5 | 5 | 12.8 GB |
| 5 | 20 (task default) | 39.5 GB |

Gradients at $N = 5$ therefore need tens of GB at the default horizon; reduce the horizon
or use $N \le 4$ on a laptop. Forward rollouts are not affected.

## Parameters

| field | default (model / task) | meaning |
|---|---|---|
| `n_cells` | 3 | number of cells $N$ |
| `n_photons` | `None` ($= N$) | initial cavity Fock number |
| `n_max` | `None` (see above) | Fock cutoff; required when `pumping_local > 0` |
| `omega0`, `omega_c` | 1.0 | cell and cavity frequencies |
| `rwa` | `True` | Tavis-Cummings (`True`) or Dicke (`False`) |
| `kappa`, `gamma`, `gamma_phi` | 0.0 | cavity loss, collective emission, collective dephasing |
| `gamma_local` | 0.0 / **0.05** | local emission rate per cell |
| `gamma_phi_local` | 0.0 / **0.05** | local dephasing rate per cell |
| `pumping_local` | 0.0 | local pumping rate per cell |

The task defaults switch on weak local emission and dephasing (0.05 $\omega_0$ each, so
$\gamma T = 0.2$ over the default episode of $T = 4$). The value 0.05 is a reproducible
benchmark setting, chosen so that local dissipation visibly breaks the symmetric subspace
within the default episode; it is not a standard experimental value. The resulting time
scale is $1/\gamma_{\mathrm{loc}} = 20\,/\omega_0$ against the episode length
`horizon * dt` $= 20 \times 0.2 = 4\,/\omega_0$, so a single cell loses about 18 percent of
its excitation over one episode ($1 - e^{-0.05 \times 4}$). Set the rates to your device
values for other studies; pass `gamma_local=0,
gamma_phi_local=0` to recover the energy dynamics of `battery/Dicke-v0`. The model is
always open (`is_open` is `True`, the state is a density matrix even with every rate at
zero); `jump_ops()` raises `NotImplementedError` because the local jumps have no
representation on a single copy of the multiplicity space, so a generic solver cannot
silently drop them.

Observables in `info`: `energy`, `ergotropy` (full cell state), `n_photons`, `excitation`
($\langle a^\dagger a + J_z + N/2\rangle$), `top_fock` (population of $|n_{\max}\rangle$, the
Fock-cutoff convergence diagnostic),
`symmetric` (weight of the $j = N/2$ sector: 1 for `battery/Dicke-v0` dynamics, below 1
once local processes act), and the generic `power` and `purity`.

Control semantics, reward and task kwargs follow `battery/Dicke-v0`: one channel $g(t)$,
piecewise constant, clipped to `action_bound` (0.5), `horizon` 20, `dt` 0.2; the return is
$Q(\text{final}) - Q(\text{initial})$ with $Q$ the energy or the ergotropy.

```python
import jax.numpy as jnp
import qenvs
import qenvs.tasks.dicke_local  # until the task is imported by qenvs.tasks

env = qenvs.make("battery/DickeLocal-v0")  # N = 3, gamma_local = gamma_phi_local = 0.05
actions = jnp.concatenate([jnp.full((10, 1), 0.5), jnp.zeros((10, 1))])
state, rewards = env.rollout(actions)
info = env.info(state)
print(float(rewards.sum()), float(info["ergotropy"]), float(info["symmetric"]))
# 2.199 1.457 0.791   (battery/Dicke-v0 without local dissipation: energy 2.582)
```

## Validation

All numbers from `qenvs.validation.dicke_local.CHECKS` on this machine (dynamiqs 0.3.6,
JAX 0.10.2, QuTiP 5.3.1, CPU). Every check passes in float64 and in float32. The agreement
checks use a precision-dependent tolerance: $10^{-8}$ in float64 (the release gate) and
$2\times10^{-5}$ in float32; the float32 tolerances of the analytic and constraint checks
are $10^{-5}$ (the measured float32 values are below $5\times10^{-7}$).

| check | reference | tolerance (f64 / f32) | measured (float64) | measured (float32) |
|---|---|---|---|---|
| `structure` | $\sum_j d_j(2j+1) = 2^N$ ($N \le 12$); $[J_+, J_-] = 2J_z$ and the $H_B$ multiplicities $\binom{N}{k}$ in the stored basis; the full Liouvillian (every rate on) annihilates the trace functional | 1e-12 / 1e-5 | 8.9e-16 | 4.8e-7 |
| `single_cell_dissipators` | $N = 1$: local coefficients equal $\mathcal{D}[J_-]$, $\mathcal{D}[J_z]$, $\mathcal{D}[J_+]$ built from matrices | 1e-12 | 0 | 0 |
| `brute_force_agreement` | independent simulation in the full $2^N \otimes$ cavity space with explicit per-cell jumps (dynamiqs `mesolve` + `Expm`), mapped to the Schur basis by a unitary built from highest-weight vectors; $N = 2$ (Dicke, $n_{\max} = 3$), 3 (TC, $n_{\max} = 3$), 4 (TC, $n_{\max} = 1$); all six dissipators, detuned; 6 random steps; max over steps of the joint-state, energy, ergotropy, photon-number, Dicke-density and symmetric-weight deviations (the `symmetric` observable against the weight of the brute-force cell state on the first $N+1$ Schur vectors), plus $\mathrm{tr}$ of the symmetric projector $= (N+1)(n_{\max}+1)$ | 1e-8 / 2e-5 | 6.7e-16 | 7.3e-7 |
| `collective_limit` | collective processes only vs `DickeBattery` (`Expm`): open TC $N = 3$ and closed Dicke $N = 2$, 8 random env steps; energy, full-space ergotropy, photons, reward, $j = N/2$ block, weight outside $j = N/2$ | 1e-8 / 2e-5 | 8.9e-16 | 3.6e-7 |
| `piqs_cross_check` | cells only, $N = 3, 4$, every local and collective cell process, random block initial state with coherences in every sector, vs QuTiP PIQS `Dicke(...).liouvillian()` + `mesolve`; Dicke density at $t = 0.7, 1.5, 3$ | 1e-5 | 7.1e-9 | 1.3e-7 |
| `local_emission_relaxation` | $N = 4$ from all excited, $g = 0$: product of independently decaying cells, $p = e^{-\gamma t}$: $E = N\omega_0 p$, binomial excitation distribution, purity $(p^2 + (1-p)^2)^N$, ergotropy $N\omega_0 \max(0, 2p-1)$ | 2e-6 / 1e-5 | 1.8e-15 | 4.8e-7 |
| `local_dephasing_and_pumping` | $N = 3$, $g = 0$: dephasing from $\lvert +\rangle^{\otimes N}$ ($\lvert\langle J_-\rangle\rvert = \tfrac N2 e^{-\gamma_\phi t/2}$, purity $((1 + e^{-\gamma_\phi t})/2)^N$); pumping 0.5 + emission 0.1 from ground ($p(t) = \tfrac{P}{P+\gamma}(1 - e^{-(P+\gamma)t})$: energy, purity, ergotropy, final ergotropy $1.546\,\omega_0$) | 2e-6 | 8.9e-16 | 3.1e-7 |
| `density_matrix_constraints` | Hermiticity, unit trace and positivity of the stored joint state after every step of random rollouts (Dicke $N = 3$; TC $N = 4$ with pumping, `n_max=6`) | 1e-6 / 1e-5 | 6.7e-16 | 1.9e-7 |
| `expm_step_exactness` | one `expm` step vs 32 substeps of a 30-term Taylor series and vs 8 `expm` substeps ($N = 3$, dim 320, every rate on) | 1e-12 / 2e-6 | 2.1e-16 | 4.5e-8 |
| `env_return_equals_gain` | `BatteryCharging` return = $Q_T - Q_0$ for energy and ergotropy rewards, $Q$ recomputed from the Dicke blocks; ergotropy of the $2^N$ matrix = ergotropy from blocks / $d_j$ along the rollout | 1e-6 / 1e-5 | 2.2e-16 | 2.4e-7 |

The brute-force agreement holds for the full joint state, not only for expectation values.
PIQS evaluates its rate coefficients in single precision (`cdef float` in `_piqs.pyx`),
which explains the $7\times10^{-9}$ difference in float64 and the looser $10^{-5}$ tolerance
of `piqs_cross_check`.

Which check catches what (verified by mutating the model): ignoring the degeneracy weights
$d_j$ fails `brute_force_agreement`, both analytic checks and `env_return_equals_gain`
(but not `piqs_cross_check`, which compares the blocks $\sigma_j$ only); removing the
transfer from lower sectors into $j = N/2$ fails `brute_force_agreement`,
`piqs_cross_check`, `structure` (trace preservation) and both analytic checks; a local dephasing rate four times too large ($\sigma_z$ instead of
$\sigma_z/2$) fails `brute_force_agreement`, `piqs_cross_check` and
`local_dephasing_and_pumping`.

## Known limitations

* The state, ergotropy, purity and observation are $O(4^N)$ (full $2^N$-dimensional
  density matrix); only the Liouvillian is polynomial in $N$. Gradients at $N = 5$ need
  tens of GB at the default horizon (see above).
* Dense Liouvillian and dense `expm` per step: about 1 s per step at the largest supported
  size; $N \ge 6$ with the Ferraro cutoff is rejected. A sparse or Krylov propagator, a
  real (Hermitian-basis) Liouvillian, or the excitation-number symmetry of the
  Tavis-Cummings case would extend the range; none is implemented.
* Identical cells only (permutation symmetry); no inhomogeneous broadening.
* No collective pumping and no thermal baths (zero temperature); the initial cavity state
  is a Fock state and the cells start in the ground state.
* Rates are not differentiable parameters; only the control $g(t)$ is.
* The observation is the flattened stored matrix ($2 \cdot (2^N(n_{\max}+1))^2 + 1$
  numbers), which repeats each block $d_j$ times and contains the zeros between blocks.
* The `ergotropy` differs from the one reported by `battery/Dicke-v0` by construction
  (full cell space vs symmetric subspace, see above).
* The environment clips actions to $[-g_{\max}, g_{\max}]$ and casts them to float32.

## References

1. N. Shammah, S. Ahmed, N. Lambert, S. De Liberato, F. Nori, "Open quantum systems
   with local and collective incoherent processes: Efficient numerical simulations using
   permutational invariance", Phys. Rev. A 98, 063815 (2018), arXiv:1805.05129.
   Permutation-invariant method and the local-dissipation coefficients (PIQS).
2. D. Ferraro, M. Campisi, G. M. Andolina, V. Pellegrini, M. Polini, "High-Power
   Collective Charging of a Solid-State Quantum Battery", Phys. Rev. Lett. 120, 117702
   (2018).
3. M. Tavis, F. W. Cummings, "Exact Solution for an N-Molecule-Radiation-Field
   Hamiltonian", Phys. Rev. 170, 379 (1968).
4. R. H. Dicke, "Coherence in Spontaneous Radiation Processes", Phys. Rev. 93, 99 (1954).
5. A. E. Allahverdyan, R. Balian, Th. M. Nieuwenhuizen, "Maximal work extraction from
   finite quantum systems", EPL 67, 565 (2004). Ergotropy.
