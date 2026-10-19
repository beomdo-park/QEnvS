# Bring your own Hamiltonian

You have a Hamiltonian (and maybe loss channels) from a device model, a paper or a
pulse-level simulator, and you want a QEnvS environment around it: reinforcement learning,
GRAPE-style gradient control, battery charging metrics, and the same `jit`/`vmap`/`grad`
behaviour as the built-in tasks. `qenvs.models.custom.HamiltonianModel` takes the matrices
and `qenvs validate --model` runs automatic consistency checks on them.

The model is

$$
H(t) = H_0 + \sum_k u_k(t)\,H_k ,
\qquad
\dot\rho = -i[H(t),\rho] + \sum_j \Big(L_j \rho L_j^\dagger - \tfrac12\{L_j^\dagger L_j,\rho\}\Big),
\qquad \hbar = 1,
$$

with piecewise-constant amplitudes $u_k$ chosen by the agent every control period `dt`.
The dissipator is the Lindblad form (Lindblad 1976; Gorini, Kossakowski and Sudarshan
1976). Closed models (no $L_j$) carry kets and use the Schrodinger equation; open models
carry density matrices.

## Minimal example from NumPy

```python
import dynamiqs as dq
import jax.numpy as jnp
import numpy as np

from qenvs.envs import StateTransfer
from qenvs.models.custom import HamiltonianModel

sx = np.array([[0, 1], [1, 0]])
sz = np.diag([1.0, -1.0])

model = HamiltonianModel.from_operators(
    drift=0.05 * sz,  # H_0, Hermitian (n, n)
    controls=[sx / 2],  # H_k: one operator per control channel
    initial_state=[1, 0],  # ket |0> (this is also the default)
)
env = StateTransfer(model=model, target=dq.basis(2, 1), horizon=10, dt=0.2)
actions = jnp.full((env.horizon, env.action_dim), jnp.pi / (env.horizon * env.dt))
state, rewards = env.rollout(actions)
print(float(rewards.sum()))  # fidelity gain, about 1 (the small detuning costs a little)
```

Conventions are the library's: $\hbar = 1$, index 1 is the excited level of a qubit
($\sigma_z = \mathrm{diag}(1, -1)$, so $|0\rangle$ has eigenvalue $+1$), and you build
every operator explicitly. Do not use `dq.sigmam()` / `dq.sigmap()` for a lowering operator, their
spin convention is the reverse of the one used here; write `|0><1|` as
`np.array([[0, 1], [0, 0]])`.

What you may pass, and what is checked at construction:

| argument | accepted types | required properties |
|---|---|---|
| `drift` | NumPy / JAX array, `dq.QArray`, `qutip.Qobj`, nested lists | square, finite, Hermitian |
| `controls` | one operator, a sequence, or a `(k, n, n)` stack | at least one, same $n$, each Hermitian |
| `jump_ops` | same as `controls` | square, same $n$ (any matrix; include the rate, $L = \sqrt{\gamma}\,A$) |
| `initial_state` | ket `(n,)` or `(n, 1)`, or density matrix `(n, n)`; default $\lvert 0\rangle$ | norm 1, or Hermitian + unit trace + positive semidefinite |
| `observables` | `{name: operator}` | Hermitian; reported in `info` by the environments |
| `battery_hamiltonian` | operator on the full space | Hermitian; needed by `BatteryCharging` |
| `dims` | e.g. `(2, 2)` | product equals $n$; inferred from `QArray`/`Qobj` inputs |

Violations raise `ValueError` that names the offending operator, for example
`controls[1] is not Hermitian: max|A - A^dagger| = 2.0e-01 ...`. Hermiticity is checked to a
relative $10^{-8}$ (`max|A - A^dagger| / max|A|`); accepted operators are then symmetrised
exactly. If you build an operator in single precision and it is only Hermitian to $10^{-7}$,
symmetrise it yourself with `(A + A.conj().T) / 2`. A ket that is normalised to $10^{-6}$ is
accepted and renormalised (a ket with $\lvert\langle\psi\vert\psi\rangle - 1\rvert = 10^{-4}$
is rejected, $10^{-8}$ accepted; the same tolerance applies to the trace, Hermiticity and
negativity of a density matrix). The messages print the deviation in scientific notation, for
example `|<psi|psi> - 1| = 1.000e-04 (<psi|psi> = 1.0001, tolerance 1e-06)`.

Two input slips get a targeted message. A single matrix written as a nested list is a
sequence of 1-d rows, not an operator: `controls=[[0, 1], [1, 0]]` raises "wrap a single
operator in a list" (`controls=[[[0, 1], [1, 0]]]` or `controls=[sx]` is what you want). An
all-zero control matrix is accepted but warns, because actions on that channel do nothing.

State type: a ket with jump operators becomes a density matrix. A pure density matrix
without jump operators becomes a ket. A mixed density matrix without jump operators is kept
(and integrated with the master equation and an empty dissipator), so `model.is_open` is
true whenever the model carries a density matrix.

## From QuTiP (or dynamiqs)

QuTiP is not imported by QEnvS: a `Qobj` is recognised by its `full()` method, so it is
never a dependency. Composite `dims` are taken from the objects.

```python
import qutip as qt
from qenvs.models.custom import HamiltonianModel

sx, sy, sz, qe = qt.sigmax(), qt.sigmay(), qt.sigmaz(), qt.qeye(2)
model = HamiltonianModel.from_operators(
    0.1 * qt.tensor(sz, qe) + 0.05 * qt.tensor(qe, sz),
    [0.5 * (qt.tensor(sx, sx) + qt.tensor(sy, sy)), qt.tensor(sx, qe) / 2],
    initial_state=qt.tensor(qt.basis(2, 0), qt.basis(2, 1)),
)
assert model.dims == (2, 2)
```

QuTiP's `basis(2, 1)` is the vector $(0, 1)$, the same convention as here. Watch out for
QuTiP's `jmat(j)`: it orders $m$ from $+j$ down to $-j$, the opposite of the Dicke model in
this library, so permute the indices if you mix the two.
`dq.QArray` inputs work the same way (`dq.tensor(...)` carries the dims).

## Composite systems and `dims`

Two qubits with an iSWAP-type XY coupling as the control (a tunable coupler) and a local
drive on each qubit:

```python
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]])
SZ = np.diag([1.0, -1.0]).astype(complex)
I2 = np.eye(2)
LOWER = np.array([[0, 1], [0, 0]], dtype=complex)  # |0><1|

xy = 0.5 * (np.kron(SX, SX) + np.kron(SY, SY))  # |01><10| + |10><01|
model = HamiltonianModel.from_operators(
    drift=0.1 * np.kron(SZ, I2) + 0.05 * np.kron(I2, SZ),
    controls=[xy, np.kron(SX, I2) / 2, np.kron(I2, SX) / 2],
    jump_ops=[0.02 * np.kron(LOWER, I2), 0.03 * np.kron(I2, LOWER)],  # T1 on each qubit
    initial_state=np.array([0, 1, 0, 0]),  # |01>, becomes a density matrix
    battery_hamiltonian=np.diag([0.0, 1.0, 1.0, 2.0]),  # one quantum per excited qubit
    dims=(2, 2),
)
rho_q0 = dq.ptrace(model.initial_state(), 0)  # works because dims are set
```

The tensor ordering is the `np.kron` ordering: `|01>` is index 1 and `|10>` is index 2. For
the XY coupling alone, $H = g\,(XX + YY)/2 = g(|01\rangle\langle 10| + |10\rangle\langle 01|)$
gives $P_{10}(t) = \sin^2(g t)$ from $|01\rangle$, so a constant amplitude with
$g\,T = \pi/2$ is the iSWAP; this is verified in `tests/test_custom_model.py`.

## Using it with the environments

```python
import jax
import qenvs
from qenvs.envs import BatteryCharging, StateTransfer
from qenvs.wrappers import GymnasiumEnv  # needs qenvs[gym]

# 1. state transfer: |01> -> |10> with the XY coupler
env = StateTransfer(model=model, target=dq.basis(4, 2), horizon=10, dt=0.3, action_bound=2.0)

# 2. battery charging: reward = per-step gain of energy tr(rho H_B) or ergotropy
battery = BatteryCharging(model=model, horizon=10, dt=0.3, reward="ergotropy")

# 3. Gymnasium (stable-baselines3, CleanRL, ...)
gym_env = GymnasiumEnv(battery)
obs, info = gym_env.reset(seed=0)

# 4. jit / vmap / grad work as for the built-in tasks
grad = jax.grad(lambda a: env.rollout(a)[1].sum())(jnp.zeros((10, 3)))
```

The model is a pytree, so you can also differentiate with respect to its matrices:
`eqx.tree_at(lambda m: m.drift_matrix, model, new_drift)` returns a model with another drift
(this replacement skips the construction checks, so re-run `model_checks` on a model you
modified that way).

`examples/grape_jax.py::run_grape` takes a registered task id. Register your environment
once and every tool (`qenvs run`, GRAPE, PPO examples) can use it:

```python
def iswap_task(**kw):
    return StateTransfer(model=model, target=dq.basis(4, 2), action_bound=2.0, **kw)


qenvs.register(
    "mylab/ISwap-v0",
    iswap_task,
    description="Two-qubit iSWAP state transfer |01> -> |10> with an XY coupler.",
    model="mylab.models.TwoQubitXY",
)
# from examples/grape_jax.py:  run_grape("mylab/ISwap-v0", steps=100, horizon=8, dt=0.3)
```

A task registered in your own code is only known to processes that import that code, so
`qenvs run mylab/ISwap-v0` works only inside your package's entry point; `qenvs validate
--model` takes the import path instead.

Battery on one subsystem of a composite system: `BatteryCharging` evaluates energy and
ergotropy of whatever `battery_reduced(state)` returns. For `HamiltonianModel` that is the
full state with the full-space $H_B$, which is correct only if the battery is the whole
system or the global state stays a product with a passive ancilla. If the battery is a
subsystem that becomes entangled with the rest, subclass and trace the rest out (this is
what `DickeBattery` does):

```python
HB_LOCAL = dq.asqarray(np.diag([0.0, 1.0]))  # H_B on qubit 0 alone


class FirstQubitBattery(HamiltonianModel):
    def battery_reduced(self, qstate):
        return dq.ptrace(qstate, 0), HB_LOCAL
```

Build it with the same arguments as `HamiltonianModel` (a `battery_hamiltonian` is still
needed so that the model reports one).

## Validating your model

```bash
qenvs validate --model my_model:MODEL            # table, exit code 1 if a check fails
qenvs validate --model my_model:build --json     # attr may be a zero-argument callable
qenvs validate --model my_model:MODEL --markdown
qenvs validate --model my_model:MODEL --reference my_reference:MODEL   # cross-check the matrices
```

`my_model.py` can be any importable module (the current directory is put at the front of
`sys.path` while loading and removed afterwards, so a file named like a standard-library
module, say `sched.py`, is found first; modules the interpreter has already imported, such as
`json`, stay cached and cannot be replaced); the attribute is a `qenvs.models.Model` instance
or a zero-argument callable returning one. `--reference` takes the same kind of path and needs
`--model`. The import runs your code, so only load modules you trust. Failures to import,
construct or type-check are reported as the JSON error code `invalid_model` with exit code
1:

```json
{"error": {"code": "invalid_model", "message": "could not load model 'my_model:build': ValueError: controls[0] is not Hermitian ..."}}
```

In Python, with full control of the options:

```python
from qenvs import toolkit
from qenvs.validation.generic import model_checks

results = model_checks(model, horizon=12, dt=None, action_bound=1.0, seed=0)
report = toolkit.validate_model(
    model, horizon=24, dt=0.1
)  # same dict structure as toolkit.validate()
```

The CLI enables float64 (as `qenvs validate` does) and uses the defaults, plus `--reference`;
use Python to set `dt`, `horizon` or `method`. The checks run in float32 as well (see the
tables below). The text table prints a footer whenever the check
`qutip_solver_cross_check` is present, stating that it does not verify the physics.

### The checks

All checks run a random piecewise-constant rollout of `horizon` steps with amplitudes
uniform in $[-b, b]$ (`action_bound`), **through the real `StateTransfer` environment**, so
the action clipping, float32 cast and solver call are the ones your agent will see.

| check | what it verifies | tolerance |
|---|---|---|
| `structure` | every operator is $n\times n$, finite, with dims whose product is $n$ and agreeing with the drift (a flat `(n,)` drift contradicts nothing); at least one control; an all-zero control is noted in the detail | 0 problems |
| `hermiticity` | $\max\lvert A - A^\dagger\rvert / \max\lvert A\rvert$ over drift, controls, observables, battery Hamiltonian | $10^{-6}$ |
| `initial_state_valid` | closed: ket with $\lvert\langle\psi\vert\psi\rangle - 1\rvert$ small; open: density matrix, Hermitian, trace 1, positive semidefinite; the state type matches `is_open` | $10^{-5}$ |
| `norm_conservation` (closed) | $\max_t \lvert\lVert\psi(t)\rVert - 1\rvert$ | $5\times10^{-5}$ |
| `density_matrix_valid` (open) | max of Hermiticity, $\lvert\operatorname{tr}\rho - 1\rvert$ and negativity (most negative eigenvalue) along the rollout | $5\times10^{-5}$ |
| `solver_convergence` | max entry-wise state difference between the method under test (default Tsit5) and a tight Tsit5 (`rtol=atol=1e-10`; `3e-7` in float32), and, for closed models or open models with $n \le 16$, the exact matrix exponential `Expm` | $5\times10^{-5}$ |
| `qutip_solver_cross_check` (no `reference`) | QuTiP 5 (`sesolve`/`mesolve`, tight tolerances) integrates **the model's own matrices** with the same piecewise-constant amplitudes; max entry-wise state difference at every step. It checks the solver stack only and **does not verify that the matrices are the physics you meant**. Skipped (status SKIP, never PASS) without QuTiP | $5\times10^{-5}$ |
| `qutip_cross_check` (with `reference=`) | the same comparison, but QuTiP integrates an independently built `reference` model of the intended physics; this is the only generic check that can detect a wrong operator | $5\times10^{-5}$ |
| `battery_consistency` (if $H_B$) | `battery_reduced` returns a valid density matrix (Hermitian, unit trace, positive semidefinite) whose shape conforms with $H_B$ at every step | $5\times10^{-5}$ |
| `ergotropy_bounds` (if $H_B$) | $0 \le \mathcal{E} \le E - E_{\min}$ along the rollout **and on 8 random density matrices of the battery space** (a closed model only visits pure states, so mixed states are exercised only by the random ones), with $E_{\min}$ the lowest eigenvalue of $H_B$; error is the violation divided by the width of the $H_B$ spectrum | $10^{-5}$ |

**Tolerance scaling.** The rollout-based checks (`norm_conservation`, `density_matrix_valid`,
`solver_convergence`, both QuTiP checks, `battery_consistency`, `ergotropy_bounds`)
accumulate integrator error along the episode, so their tolerance is the base value in the
table times

$$
s = \max\!\Big(1,\; \frac{\text{horizon}\cdot dt}{12\, dt_{\text{default}}}\Big),
$$

with $dt_{\text{default}} = 1/w$ the time step rule below. $s$ is the integrated time in units
of the model's fastest rate: it is 1 for the default rollout (12 steps at the default `dt`) and
grows in proportion to a longer episode or a coarser `dt`. A valid model therefore passes at
`horizon=400` or at `dt = 3 * default` (`TwoLevel()` used to fail `battery_consistency` at
`horizon=400`: error $1.6\times10^{-5}$ against a tolerance of $10^{-5}$, which was ten times
tighter than `norm_conservation`). The reported `tolerance` is the scaled one and the detail
names the factor. `structure`, `hermiticity` and `initial_state_valid` are not scaled.

The base tolerances are float32-safe. At `horizon=12` and the default `dt` they sit 44 to 702
times above the worst error measured in either precision for the models below (the default
adaptive solver has `rtol = atol = 1e-6`, which sets the floor, not the floating-point
precision); the margins at other horizons and steps are in the second table.

**Time step rule** (`dt=None`). With $b$ the action bound and $\lVert\cdot\rVert$ the
spectral norm,

$$
w = \lVert H_0\rVert + b\sum_k \lVert H_k\rVert + \sum_j \lVert L_j\rVert^2 ,
\qquad dt = 1/w ,
$$

(and $dt = 1$ if $w = 0$). $w$ bounds the fastest rate in the dynamics, so one control
interval rotates the state by about a radian at most and the rollout spans `horizon / w`.
Models whose drift is much larger than the controls and rates (a lab-frame Hamiltonian
without a rotating frame) get a tiny `dt` and a short, nearly trivial episode: pass `dt`
explicitly for those.

### Measured results

Errors measured with this repository's checks (dynamiqs 0.3.6, jax 0.10.2, QuTiP 5.3.1, CPU,
`action_bound=1`, seed 0). All tables are regenerated by a script, so they can be refreshed after
any change to the checks or to a dependency:

```bash
JAX_PLATFORMS=cpu python tests/gen_custom_model_tables.py > /tmp/tables.md   # about 4 minutes
```

(`tests/gen_custom_model_tables.py` is a test helper, not part of the package; it runs the
checks in a fresh float64 and a fresh float32 interpreter.)

**Valid models at `horizon=12` and the default `dt`**, float64 and float32 (every margin in this
table is qualified by that horizon and step; other rollouts are in the next table):

| model | check | error x64 | error f32 | tol | margin (worst precision) |
|---|---|---|---|---|---|
| two qubits, XY coupler, H_B (closed) | `norm_conservation` | 3.3e-08 | 7.1e-08 | 5e-05 | 702 |
|  | `solver_convergence` | 8.0e-08 | 3.1e-07 | 5e-05 | 163 |
|  | `qutip_solver_cross_check` | 7.0e-08 | 1.8e-07 | 5e-05 | 271 |
|  | `battery_consistency` | 6.6e-08 | 1.6e-07 | 5e-05 | 320 |
|  | `ergotropy_bounds` | 2.2e-16 | 1.8e-07 | 1e-05 | 55.9 |
| same, with T1 on both qubits (open) | `density_matrix_valid` | 2.2e-16 | 2.5e-07 | 5e-05 | 203 |
|  | `solver_convergence` | 8.4e-07 | 4.8e-07 | 5e-05 | 59.5 |
|  | `qutip_solver_cross_check` | 8.4e-07 | 5.2e-07 | 5e-05 | 59.6 |
|  | `battery_consistency` | 2.2e-16 | 2.5e-07 | 5e-05 | 203 |
|  | `ergotropy_bounds` | 2.2e-16 | 6.0e-08 | 1e-05 | 168 |
| driven anharmonic qutrit | `norm_conservation` | 1.4e-08 | 1.6e-07 | 5e-05 | 309 |
|  | `solver_convergence` | 1.6e-08 | 3.6e-07 | 5e-05 | 139 |
|  | `qutip_solver_cross_check` | 2.5e-08 | 2.7e-07 | 5e-05 | 182 |
| DickeBattery(n_cells=2) | `norm_conservation` | 1.0e-07 | 4.2e-07 | 5e-05 | 118 |
|  | `solver_convergence` | 4.6e-07 | 1.1e-06 | 5e-05 | 44.1 |
|  | `qutip_solver_cross_check` | 3.0e-07 | 9.9e-07 | 5e-05 | 50.7 |
|  | `battery_consistency` | 2.1e-07 | 8.3e-07 | 5e-05 | 60.5 |
|  | `ergotropy_bounds` | 2.2e-16 | 6.0e-08 | 1e-05 | 168 |
| TwoLevel() | `norm_conservation` | 2.3e-07 | 2.6e-07 | 5e-05 | 195 |
|  | `solver_convergence` | 3.0e-07 | 5.4e-07 | 5e-05 | 93.2 |
|  | `qutip_solver_cross_check` | 4.7e-07 | 6.5e-07 | 5e-05 | 77.1 |
|  | `battery_consistency` | 4.5e-07 | 5.1e-07 | 5e-05 | 98.7 |
|  | `ergotropy_bounds` | 5.6e-17 | 6.0e-08 | 1e-05 | 168 |

Margins (`tol / error`) at `horizon=12` and the default `dt`: 44.1 to 702 over the rows above.

`structure`, `hermiticity` and `initial_state_valid` measure 0 for every valid model. The
built-in `TwoLevel`, `QubitBattery`, `DickeBattery` (closed and with cavity loss) and
`LambdaSystem` (closed and open) pass every generic check in both precisions: no false
positives (`tests/test_custom_model.py`).

**Long and coarse rollouts of valid models** (tolerance scaled by the integrated time, the rule
above; the row shows the check with the smallest margin):

| model | rollout | precision | worst check | error | scaled tol | margin |
|---|---|---|---|---|---|---|
| TwoLevel() | horizon=400 | x64 | `qutip_solver_cross_check` | 1.6e-05 | 1.7e-03 | 102 |
| TwoLevel() | dt=3*default, horizon=60 | x64 | `qutip_solver_cross_check` | 1.2e-05 | 7.5e-04 | 61.8 |
| two qubits, XY coupler, H_B (closed) | horizon=400 | x64 | `battery_consistency` | 2.0e-06 | 1.7e-03 | 842 |
| two qubits, XY coupler, H_B (closed) | dt=3*default, horizon=60 | x64 | `battery_consistency` | 4.2e-06 | 7.5e-04 | 179 |
| TwoLevel() | horizon=400 | f32 | `qutip_solver_cross_check` | 1.4e-05 | 1.7e-03 | 123 |
| TwoLevel() | dt=3*default, horizon=60 | f32 | `battery_consistency` | 3.6e-06 | 7.5e-04 | 206 |
| two qubits, XY coupler, H_B (closed) | horizon=400 | f32 | `solver_convergence` | 2.2e-06 | 1.7e-03 | 743 |
| two qubits, XY coupler, H_B (closed) | dt=3*default, horizon=60 | f32 | `battery_consistency` | 3.4e-06 | 7.5e-04 | 221 |

**Deliberately broken models** that construct fine (or are built by bypassing validation), float64,
`horizon=12` and the default `dt` unless the row says otherwise:

| broken model | failing checks (error) | passes |
|---|---|---|
| control with anti-Hermitian part 0.1i XI | `hermiticity` (0.39), `norm_conservation` (0.02), `qutip_solver_cross_check` (0.027), `battery_consistency` (0.039) | `structure`, `initial_state_valid`, `solver_convergence`, `ergotropy_bounds` |
| norm loss: drift with -0.01i P1 (P1 = projector on level 1) | `hermiticity` (0.067), `norm_conservation` (0.12), `qutip_solver_cross_check` (0.19) | `structure`, `initial_state_valid`, `solver_convergence` |
| control asymmetry 1e-07 (relative) | none | **all checks pass** |
| control asymmetry 1e-04 (relative) | `hermiticity` (0.0001) | `structure`, `initial_state_valid`, `norm_conservation`, `solver_convergence`, `qutip_solver_cross_check`, `battery_consistency`, `ergotropy_bounds` |
| Euler(dt=0.5), closed | `norm_conservation` (0.69), `solver_convergence` (0.69), `qutip_solver_cross_check` (1.6), `battery_consistency` (1.9) | `structure`, `hermiticity`, `initial_state_valid`, `ergotropy_bounds` |
| Euler(dt=0.5), open | `density_matrix_valid` (1.1), `solver_convergence` (1), `qutip_solver_cross_check` (1), `battery_consistency` (1.1), `ergotropy_bounds` (1.1) | `structure`, `hermiticity`, `initial_state_valid` |
| Euler(dt=0.005), open qubit | `solver_convergence` (0.0017), `qutip_solver_cross_check` (0.0017) | `structure`, `hermiticity`, `initial_state_valid`, `density_matrix_valid` |
| Tsit5(rtol=atol=1e-2), qubit at dt=3*default, horizon=4 | `norm_conservation` (0.00068), `solver_convergence` (0.00073), `qutip_solver_cross_check` (0.00084), `battery_consistency` (0.0014) | `structure`, `hermiticity`, `initial_state_valid`, `ergotropy_bounds` |
| wrong jump operator, reference=intended | `qutip_cross_check` (0.13) | `structure`, `hermiticity`, `initial_state_valid`, `density_matrix_valid`, `solver_convergence`, `battery_consistency`, `ergotropy_bounds` |
| jump rate 5 percent too large, reference=intended | `qutip_cross_check` (0.011) | `structure`, `hermiticity`, `initial_state_valid`, `density_matrix_valid`, `solver_convergence`, `battery_consistency`, `ergotropy_bounds` |
| the same wrong jump operator WITHOUT reference | none | **all checks pass** |
| non-Hermitian H_B injected | `hermiticity` (0.15), `ergotropy_bounds` (0.011) | `structure`, `initial_state_valid`, `norm_conservation`, `solver_convergence`, `qutip_solver_cross_check`, `battery_consistency` |
| initial state scaled by 2 | `initial_state_valid` (3), `norm_conservation` (1), `battery_consistency` (3) | `structure`, `hermiticity`, `solver_convergence`, `qutip_solver_cross_check`, `ergotropy_bounds` |
| initial density matrix with trace 1.001 | `initial_state_valid` (0.001), `density_matrix_valid` (1), `solver_convergence` (1), `qutip_solver_cross_check` (1) | `structure`, `hermiticity` |
| initial density matrix with eigenvalue -1e-3 | `initial_state_valid` (0.001), `density_matrix_valid` (1), `solver_convergence` (1), `qutip_solver_cross_check` (1) | `structure`, `hermiticity` |

An error of exactly 1.0 with a detail starting `could not run` means the check raised; the
exception text is in `detail`. Rows with a bypassed model use `eqx.tree_at`, which skips the
construction checks.

**Sensitivity of the reference cross-check.** The reference check compares the dynamiqs rollout
of the model with a QuTiP rollout of the `reference` (tolerance $5\times10^{-5}$ at
`horizon=12`). A T1 rate that is off by a relative error:

| rate gamma | T | relative error | gamma * T * rel | measured error | caught |
|---|---|---|---|---|---|
| 0.02 | 14.6 | 1e-04 | 2.9e-05 | 1.6e-05 | no |
| 0.02 | 14.6 | 1e-03 | 2.9e-04 | 1.6e-04 | yes |
| 0.02 | 14.6 | 1e-02 | 2.9e-03 | 1.6e-03 | yes |
| 0.02 | 14.6 | 1e-01 | 2.9e-02 | 1.6e-02 | yes |
| 0.2 | 12 | 1e-04 | 2.4e-04 | 3.2e-05 | no |
| 0.2 | 12 | 1e-03 | 2.4e-03 | 3.2e-04 | yes |
| 0.2 | 12 | 1e-02 | 2.4e-02 | 3.2e-03 | yes |
| 0.2 | 12 | 1e-01 | 2.4e-01 | 3.0e-02 | yes |
| 1 | 6.67 | 1e-04 | 6.7e-04 | 3.5e-05 | no |
| 1 | 6.67 | 1e-03 | 6.7e-03 | 3.5e-04 | yes |
| 1 | 6.67 | 1e-02 | 6.7e-02 | 3.5e-03 | yes |
| 1 | 6.67 | 1e-01 | 6.7e-01 | 3.3e-02 | yes |

and a control operator $\sigma_x/2$ whose amplitude is off by a relative error (random
piecewise-constant amplitudes, so the contributions partly cancel):

| relative error | T | ||H_k|| * T * rel | measured error | caught |
|---|---|---|---|---|
| 1e-05 | 15 | 7.5e-05 | 5.4e-06 | no |
| 1e-04 | 15 | 7.5e-04 | 5.3e-05 | yes |
| 1e-03 | 15 | 7.5e-03 | 5.3e-04 | yes |
| 1e-02 | 15 | 7.5e-02 | 5.3e-03 | yes |


## What the checks can and cannot guarantee

They establish **numerics and internal consistency**: your operators are what they claim to
be structurally (Hermitian, same size, valid state), the solver stack integrates them
accurately and preserves the physical constraints, QuTiP agrees on the result, and the
battery quantities obey the thermodynamic bounds. They do **not** establish that the
Hamiltonian is the right physics.

* A sign error in a coupling, controls in the wrong order, a swapped basis ordering, the
  wrong detuning convention, or a lowering operator that raises are all perfectly valid
  Hamiltonians of a different system. They pass every check. The test-suite includes swapped
  controls, a flipped coupling sign and a raising "decay" operator as passing models, on
  purpose.
* Without `reference=` the QuTiP check is named `qutip_solver_cross_check`: it is built from
  the same matrices as the model, so it tests the solver stack, not your matrices (its detail
  and the CLI table footer say so). It becomes a check of the matrices only when you pass
  `reference=` (CLI: `--reference MOD:ATTR`): an independently built model of the intended
  physics (a second implementation, ideally by a different route, such as QuTiP operators
  against NumPy ones). The check is then named `qutip_cross_check` and fails if the two
  disagree (tables above).
* The reference cross-check has a sensitivity floor. An error is caught only if it moves the
  state by more than the tolerance ($5\times10^{-5}$ at `horizon=12`) at some step. For a
  parameter that is off by a relative error $\varepsilon$, the state error is bounded by about
  $\gamma T\varepsilon$ ($\gamma$ the rate, or $\lVert H_k\rVert$ for a control, $T$ the
  episode length), so an error with $\gamma T\varepsilon$ below the tolerance is certainly
  missed, and one above it is caught only if it does not average out. Measured ratios of the
  state error to $\gamma T\varepsilon$: 0.55 for a weak decay ($\gamma T = 0.29$) down to 0.05
  for a strong one ($\gamma T = 6.7$, where the error saturates near $\varepsilon/e$ because the
  decay has run its course), and 0.07 for a control with random amplitudes. In the sensitivity
  tables above, a T1 rate that is $10^{-3}$ too large is caught at every $\gamma$ measured and
  $10^{-4}$ too large never is; a control amplitude that is $10^{-4}$ off is only just caught
  ($5.3\times10^{-5}$ against $5\times10^{-5}$) and $10^{-5}$ off is missed. A longer
  `horizon` does not lower the floor, because the tolerance is scaled with the integrated
  time (rule above); a smaller discrepancy needs a state-resolved test of your own (a
  closed-form limit, below).
* Passing is not a statement about validity regimes: RWA, Markov and weak-coupling
  approximations, Fock truncation, frame choices are your responsibility.

Practical advice: always compare one or two limits you know (below), print
`model.hamiltonian(u)` for a known `u` and read it, and give the matrices the same scrutiny
as a unit test.

## When to write model-specific checks

Add checks of your own (the built-in models each ship a `validation/<model>.py`) as soon as
any of these apply:

* **A closed-form limit exists.** Resonant Rabi oscillations, the iSWAP time above, a
  decay rate, a stationary state. These are the only checks that tie the matrices to the
  physics. Put them in a test with a stated tolerance.
* **A truncation is involved.** The generic checks cannot see that a Fock cutoff is too
  small (the states stay valid, only wrong). Monitor the population of the highest kept level
  as `DickeBattery.top_fock` does and test convergence in the cutoff.
* **A symmetry or conserved quantity exists** (excitation number, parity). Test that it is
  conserved by the dynamics that should conserve it.
* **The task is a gate or has a non-trivial reward.** Check the reward on a trajectory whose
  answer you know.
* **The battery is a subsystem**, or the energy reference is not the ground state. Test
  `battery_reduced` against a hand computation.

## Assumptions and limitations

* Dense matrices only. The checks run `Expm` on the Liouvillian for open models up to
  dimension 16 (superoperator size $n^2$), and the QuTiP check builds dense `Qobj`s; large
  Hilbert spaces are slow.
* Operators are validated at construction and need concrete arrays: build the model outside
  `jit`/`vmap`. Replacing fields with `eqx.tree_at` is allowed and skips validation.
* Time-independent $H_0$, $H_k$, $L_j$; the agent's controls are piecewise constant over `dt`
  and clipped to $[-b, b]$ (`action_bound`) by the environments. A drive with a carrier
  frequency has to be in a rotating frame (RWA) or sampled finely.
* Arrays are stored in the active JAX float precision (complex64 unless x64 is enabled).
  Hermiticity and state validity are decided in float64 on the inputs.
* `battery_hamiltonian` is on the full space; subsystem batteries need a subclass (above).
* dynamiqs 0.3.6 drops the subsystem dims of a composite density matrix integrated with
  `Expm`; `HamiltonianModel.evolve` restores them, the built-in composite models (for
  example the open `DickeBattery`) do not, so do not use `method=Expm()` in an environment
  around them. The generic `solver_convergence` check works around this internally.
* `qenvs validate --model` imports and runs your module. It validates with the library's
  random rollout only; the only option besides `--model` is `--reference`, there is no CLI
  flag for `dt`, `horizon` or `method`.
* The tolerances are empirical: about $5\times10^{-5}$ on state errors is what the default
  adaptive solver delivers at `horizon=12`. The solver check catches fixed-step Euler (state
  error $10^{-3}$ to 1 depending on the step) and a loose adaptive solver on a coarse step
  (`Tsit5(rtol=1e-2, atol=1e-2)` at `dt = 3 * default`, `horizon=4`: $7\times10^{-4}$ closed,
  $1.4\times10^{-2}$ open), both between the tolerance and $10^3$ times the tolerance. A loosened
  adaptive tolerance at the *default* `dt` is not detected, because Tsit5 is much more accurate
  than its tolerance when the steps are already small: errors much below $5\times10^{-5}$ are
  not distinguishable from the default solver.

## References

* G. Lindblad, "On the generators of quantum dynamical semigroups", Commun. Math. Phys. 48,
  119-130 (1976).
* V. Gorini, A. Kossakowski and E. C. G. Sudarshan, "Completely positive dynamical
  semigroups of N-level systems", J. Math. Phys. 17, 821-825 (1976).
* A. E. Allahverdyan, R. Balian and Th. M. Nieuwenhuizen, "Maximal work extraction from
  finite quantum systems", EPL 67, 565 (2004) (ergotropy, used by the battery checks).
* N. Lambert et al., "QuTiP 5: The Quantum Toolbox in Python", Phys. Rep. 1153, 1-62 (2026),
  arXiv:2412.04705 (the reference implementation of the cross-check).

## Selecting a battery subsystem

For a composite system, pass `battery_subsystems` and `subsystem_hamiltonian`
instead of a full-space `battery_hamiltonian`. The same keywords work with
`HamiltonianModel.from_operators`.

```python
import numpy as np
from qenvs.models import HamiltonianModel
from qenvs.observables import energy, ergotropy

model = HamiltonianModel(
    drift=np.zeros((4, 4)),
    controls=[np.kron(np.eye(2), np.array([[0, 1], [1, 0]]))],
    initial_state=np.array([1, 0, 0, 1]) / np.sqrt(2),
    dims=(2, 2),
    battery_subsystems=(1,),
    subsystem_hamiltonian=np.diag([0.0, 1.0]),
)
rho_b, h_b = model.battery_reduced(model.initial_state())
print(energy(rho_b, h_b))     # 0.5
print(ergotropy(rho_b, h_b))  # 0.0, unlike the pure joint state's work
```

Indices must form a nonempty, increasing tuple without duplicates. The subsystem
Hamiltonian acts on those factors in tensor order and must have ground energy
zero, within `1e-8 * max(1, spectral norm)`. The constructor rejects a shifted
Hamiltonian instead of changing it. `battery_hamiltonian()` returns its full-space
embedding, while `battery_reduced()` returns the reduced density matrix and the
local Hamiltonian. `BatteryCharging` therefore uses the subsystem's work.

Existing calls using `battery_hamiltonian` retain their full-state behavior.
