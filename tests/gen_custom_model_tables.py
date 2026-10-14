"""Regenerate the measured-results tables of ``docs/guides/custom-models.md``.

A test helper, not part of the package (pytest does not collect it). Run it from the
repository root and paste the Markdown it prints into the guide::

    JAX_PLATFORMS=cpu python tests/gen_custom_model_tables.py > /tmp/tables.md

It runs the generic checks of :mod:`qenvs.validation.generic` in two fresh interpreters
(float64 and float32, the library default) and prints four tables:

1. valid models at ``horizon=12`` and the default ``dt``: error per check in both precisions,
   tolerance and margin (``tol / error``);
2. valid models at ``horizon=400`` and at ``dt = 3 * default`` (scaled tolerances);
3. deliberately broken models and which checks catch them (float64);
4. the sensitivity floor of the ``reference=`` cross-check.
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np

SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]])
SZ = np.diag([1.0, -1.0]).astype(complex)
I2 = np.eye(2, dtype=complex)
LOWER = np.array([[0, 1], [0, 0]], dtype=complex)  # |0><1|
XY = 0.5 * (np.kron(SX, SX) + np.kron(SY, SY))
HB2 = np.diag([0.0, 1.0, 1.0, 2.0])


def _models() -> dict[str, object]:
    from qenvs.models import DickeBattery, TwoLevel
    from qenvs.models.custom import HamiltonianModel

    def two_qubit(**kw: object) -> HamiltonianModel:
        base = {
            "drift": 0.1 * np.kron(SZ, I2) + 0.05 * np.kron(I2, SZ),
            "controls": [XY, np.kron(SX, I2) / 2, np.kron(I2, SX) / 2],
            "initial_state": np.array([0, 1, 0, 0]),
            "battery_hamiltonian": HB2,
            "dims": (2, 2),
        }
        base.update(kw)
        return HamiltonianModel.from_operators(**base)

    a = np.diag(np.sqrt([1.0, 2.0]), 1).astype(complex)
    ad = a.conj().T
    qutrit = HamiltonianModel.from_operators(
        0.1 * ad @ a + 0.5 * (-0.3) * ad @ ad @ a @ a, [(a + ad) / 2, 1j * (ad - a) / 2]
    )
    open_jumps = [0.3 * np.kron(LOWER, I2), 0.2 * np.kron(I2, LOWER)]
    return {
        "two qubits, XY coupler, H_B (closed)": two_qubit(),
        "same, with T1 on both qubits (open)": two_qubit(jump_ops=open_jumps),
        "driven anharmonic qutrit": qutrit,
        "DickeBattery(n_cells=2)": DickeBattery(n_cells=2),
        "TwoLevel()": TwoLevel(),
    }


def _row(results: list[object]) -> dict[str, list[float]]:
    return {r.name: [r.error, r.tolerance, bool(r.passed or r.skipped)] for r in results}


def _valid_worker() -> dict[str, object]:
    from qenvs.validation.generic import default_dt, model_checks

    out: dict[str, object] = {"default": {}, "long": {}}
    for name, m in _models().items():
        out["default"][name] = _row(model_checks(m))  # type: ignore[index]
    for name in ("TwoLevel()", "two qubits, XY coupler, H_B (closed)"):
        m = _models()[name]
        d = default_dt(m)
        out["long"][name] = {  # type: ignore[index]
            "horizon=400": _row(model_checks(m, horizon=400)),
            "dt=3*default, horizon=60": _row(model_checks(m, horizon=60, dt=3 * d)),
        }
    return out


def _broken_worker() -> dict[str, object]:
    import dynamiqs as dq
    import equinox as eqx
    import jax.numpy as jnp

    from qenvs.models.custom import HamiltonianModel
    from qenvs.validation.generic import default_dt, model_checks

    models = _models()
    closed = models["two qubits, XY coupler, H_B (closed)"]
    opened = models["same, with T1 on both qubits (open)"]
    rows: dict[str, object] = {}

    def run(label: str, m: object, **kw: object) -> None:
        rows[label] = _row(model_checks(m, **kw))

    bad = closed.control_matrices.at[1].add(0.1j * np.kron(SX, I2))
    run(
        "control with anti-Hermitian part 0.1i XI",
        eqx.tree_at(lambda q: q.control_matrices, closed, bad),
    )
    p1 = np.diag([0.0, 1.0])
    qubit = HamiltonianModel(0.3 * SZ, [SX / 2], initial_state=[0, 1])
    lossy = eqx.tree_at(lambda q: q.drift_matrix, qubit, jnp.asarray(0.3 * SZ - 0.01j * p1))
    run("norm loss: drift with -0.01i P1 (P1 = projector on level 1)", lossy)
    q0 = HamiltonianModel(0.3 * SZ, [SX / 2], battery_hamiltonian=np.diag([0.0, 1.0]))
    for eps in (1e-7, 1e-4):
        m = eqx.tree_at(
            lambda q: q.control_matrices, q0, jnp.asarray((SX / 2 + eps * LOWER / 2)[None])
        )
        run(f"control asymmetry {eps:.0e} (relative)", m)
    run("Euler(dt=0.5), closed", closed, method=dq.method.Euler(dt=0.5))
    run("Euler(dt=0.5), open", opened, method=dq.method.Euler(dt=0.5))
    q_open = HamiltonianModel(0.3 * SZ, [SX / 2], jump_ops=[0.2 * LOWER], initial_state=[0, 1])
    run("Euler(dt=0.005), open qubit", q_open, method=dq.method.Euler(dt=0.005), horizon=4)
    run(
        "Tsit5(rtol=atol=1e-2), qubit at dt=3*default, horizon=4",
        q0,
        method=dq.method.Tsit5(rtol=1e-2, atol=1e-2),
        horizon=4,
        dt=3 * default_dt(q0),
    )
    wrong = HamiltonianModel.from_operators(
        closed.drift_matrix,
        list(closed.control_matrices),
        jump_ops=[0.3 * np.kron(LOWER, I2), 0.2 * np.kron(I2, LOWER.T)],
        initial_state=np.array([0, 1, 0, 0]),
        battery_hamiltonian=HB2,
        dims=(2, 2),
    )
    off = HamiltonianModel.from_operators(
        closed.drift_matrix,
        list(closed.control_matrices),
        jump_ops=[0.3 * np.kron(LOWER, I2), 0.21 * np.kron(I2, LOWER)],
        initial_state=np.array([0, 1, 0, 0]),
        battery_hamiltonian=HB2,
        dims=(2, 2),
    )
    run("wrong jump operator, reference=intended", wrong, reference=opened)
    run("jump rate 5 percent too large, reference=intended", off, reference=opened)
    run("the same wrong jump operator WITHOUT reference", wrong)
    hb = closed.battery_matrix.at[0, 1].set(0.3)
    run("non-Hermitian H_B injected", eqx.tree_at(lambda q: q.battery_matrix, closed, hb))
    run(
        "initial state scaled by 2",
        eqx.tree_at(lambda q: q.initial_matrix, closed, 2.0 * closed.initial_matrix),
    )
    for label, rho in (
        ("initial density matrix with trace 1.001", np.diag([1.001, 0.0])),
        ("initial density matrix with eigenvalue -1e-3", np.diag([1.001, -0.001])),
    ):
        run(label, eqx.tree_at(lambda q: q.initial_matrix, q_open, jnp.asarray(rho)), horizon=4)
    return rows


def _sensitivity_worker() -> dict[str, list[dict[str, float]]]:
    from qenvs.models.custom import HamiltonianModel
    from qenvs.validation.generic import default_dt, model_checks

    def decay(rate: float) -> HamiltonianModel:
        return HamiltonianModel(
            0.3 * SZ, [SX / 2], jump_ops=[np.sqrt(rate) * LOWER], initial_state=[0, 1]
        )

    rows = []
    ctrl_rows = []
    for rel in (1e-5, 1e-4, 1e-3, 1e-2):
        ref = HamiltonianModel(0.3 * SZ, [SX / 2])
        off = HamiltonianModel(0.3 * SZ, [(1 + rel) * SX / 2])
        chk = {x.name: x for x in model_checks(off, reference=ref)}["qutip_cross_check"]
        span = 12 * default_dt(ref)
        ctrl_rows.append(
            {
                "rel": rel,
                "T": span,
                "product": 0.5 * span * rel,
                "error": chk.error,
                "tol": chk.tolerance,
                "caught": not chk.passed,
            }
        )
    for rate in (0.02, 0.2, 1.0):
        for rel in (1e-4, 1e-3, 1e-2, 1e-1):
            ref = decay(rate)
            r = {x.name: x for x in model_checks(decay(rate * (1 + rel)), reference=ref)}
            chk = r["qutip_cross_check"]
            span = 12 * default_dt(ref)
            rows.append(
                {
                    "rate": rate,
                    "T": span,
                    "rel": rel,
                    "product": rate * span * rel,
                    "error": chk.error,
                    "tol": chk.tolerance,
                    "caught": not chk.passed,
                }
            )
    return {"decay": rows, "control": ctrl_rows}


def worker(mode: str) -> None:
    import jax

    if mode == "x64":
        jax.config.update("jax_enable_x64", True)
    out: dict[str, object] = {"valid": _valid_worker()}
    if mode == "x64":
        out["broken"] = _broken_worker()
        out["sensitivity"] = _sensitivity_worker()
    from qenvs.validation.core import environment_info

    out["env"] = environment_info()
    sys.stdout.write("JSON:" + json.dumps(out) + "\n")


def _run_worker(mode: str) -> dict[str, object]:
    proc = subprocess.run(
        [sys.executable, __file__, "--worker", mode],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-3000:])
        raise SystemExit(f"worker {mode} failed")
    line = next(x for x in proc.stdout.splitlines() if x.startswith("JSON:"))
    return json.loads(line[5:])


def _fmt(x: float) -> str:
    return f"{x:.1e}"


def render(x64: dict[str, object], f32: dict[str, object]) -> str:
    lines = []
    env = x64["env"]
    lines.append(
        f"<!-- qenvs {env['qenvs']}, dynamiqs {env['dynamiqs']}, jax {env['jax']}, "
        f"backend {env['jax_backend']} -->"
    )
    lines += ["", "Valid models at `horizon=12` and the default `dt`, float64 / float32:", ""]
    lines += ["| model | check | error x64 | error f32 | tol | margin (worst precision) |"]
    lines += ["|---|---|---|---|---|---|"]
    margins = []
    d64, d32 = x64["valid"]["default"], f32["valid"]["default"]
    for model, checks in d64.items():
        first = True
        for name, (e64, tol, ok64) in checks.items():
            e32, _, ok32 = d32[model][name]
            if e64 == 0 and e32 == 0:
                continue
            worst = max(e64, e32)
            margins.append(tol / worst)
            tag = model if first else ""
            first = False
            flag = "" if ok64 and ok32 else " **FAIL**"
            lines.append(
                f"| {tag} | `{name}` | {_fmt(e64)} | {_fmt(e32)} | {tol:.0e} | "
                f"{tol / worst:.3g}{flag} |"
            )
    lines += [
        "",
        f"Margins (`tol / error`) at `horizon=12` and the default `dt`: {min(margins):.3g} to "
        f"{max(margins):.3g} over the rows above.",
        "",
        "Long and coarse rollouts of valid models (tolerances scaled by the integrated time, "
        "worst check per row):",
        "",
        "| model | rollout | precision | worst check | error | scaled tol | margin |",
        "|---|---|---|---|---|---|---|",
    ]
    for prec, data in (("x64", x64), ("f32", f32)):
        for model, cases in data["valid"]["long"].items():
            for case, checks in cases.items():
                name, (err, tol, _) = max(
                    ((k, v) for k, v in checks.items() if v[1] > 0 and k != "hermiticity"),
                    key=lambda kv: kv[1][0] / kv[1][1],
                )
                lines.append(
                    f"| {model} | {case} | {prec} | `{name}` | {_fmt(err)} | {tol:.1e} | "
                    f"{tol / err:.3g} |"
                )
    lines += [
        "",
        "Broken models (float64):",
        "",
        "| broken model | failing checks (error) | passes |",
    ]
    lines += ["|---|---|---|"]
    for label, checks in x64["broken"].items():
        fails = [f"`{k}` ({v[0]:.2g})" for k, v in checks.items() if not v[2]]
        ok = [f"`{k}`" for k, v in checks.items() if v[2]]
        lines.append(
            f"| {label} | {', '.join(fails) or 'none'} | "
            f"{'**all checks pass**' if not fails else ', '.join(ok)} |"
        )
    lines += [
        "",
        "Sensitivity of the `reference=` cross-check (qubit, T1 rate off by a relative error; "
        "`T = 12 * default_dt`, tolerance 5e-5):",
        "",
        "| rate gamma | T | relative error | gamma * T * rel | measured error | caught |",
        "|---|---|---|---|---|---|",
    ]
    for r in x64["sensitivity"]["decay"]:
        lines.append(
            f"| {r['rate']:g} | {r['T']:.3g} | {r['rel']:.0e} | {r['product']:.1e} | "
            f"{_fmt(r['error'])} | {'yes' if r['caught'] else 'no'} |"
        )
    lines += [
        "",
        "Same, for a control operator `sigma_x / 2` whose amplitude is off by a relative error "
        "(`||H_k|| = 1/2`):",
        "",
        "| relative error | T | ||H_k|| * T * rel | measured error | caught |",
        "|---|---|---|---|---|",
    ]
    for r in x64["sensitivity"]["control"]:
        lines.append(
            f"| {r['rel']:.0e} | {r['T']:.3g} | {r['product']:.1e} | {_fmt(r['error'])} | "
            f"{'yes' if r['caught'] else 'no'} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        worker(sys.argv[2])
        return
    sys.stdout.write(render(_run_worker("x64"), _run_worker("f32")))


if __name__ == "__main__":
    main()
