"""CLI and agent toolkit: machine-readable output must round-trip through JSON."""

import json

import pytest

from qenvs import cli, toolkit


def _run(capsys, *argv):
    code = cli.main(list(argv))
    out = capsys.readouterr().out
    return code, out


def test_list_json(capsys):
    code, out = _run(capsys, "list", "--json")
    assert code == 0
    tasks = json.loads(out)
    assert any(t["id"] == "qubit/StateTransfer-v0" for t in tasks)


def test_info_json_has_defaults_and_usage(capsys):
    code, out = _run(capsys, "info", "qubit/StateTransfer-v0", "--json")
    assert code == 0
    info = json.loads(out)
    assert info["defaults"]["action_dim"] == 1
    assert "python" in info["usage"]
    assert info["references"]


def test_run_pi_pulse_json(capsys):
    code, out = _run(
        capsys,
        "run",
        "qubit/StateTransfer-v0",
        "--policy",
        "pi-pulse",
        "--set",
        "horizon=10",
        "--set",
        "dt=0.2",
        "--json",
    )
    assert code == 0
    res = json.loads(out)
    assert res["return"] == pytest.approx(1.0, abs=1e-6)
    assert res["final"]["fidelity"] == pytest.approx(1.0, abs=1e-6)


def test_run_open_system_kwargs(capsys):
    code, out = _run(
        capsys,
        "run",
        "qubit/StateTransfer-v0",
        "--policy",
        "zero",
        "--set",
        "gamma1=0.5",
        "--set",
        "horizon=4",
        "--json",
    )
    assert code == 0
    res = json.loads(out)
    assert res["final"]["fidelity"] == pytest.approx(0.0, abs=1e-8)


def test_validate_text_and_json(capsys):
    code, out = _run(capsys, "validate")
    assert code == 0
    assert "PASS" in out and "FAIL" not in out
    code, out = _run(capsys, "validate", "--json")
    res = json.loads(out)
    assert res["all_passed"]
    assert {"qenvs", "dynamiqs", "jax"} <= set(res["environment"])


def test_unknown_policy_raises():
    with pytest.raises(ValueError, match="unknown policy"):
        toolkit.run_rollout("qubit/StateTransfer-v0", policy="nope")


def test_doctor_json(capsys):
    code, out = _run(capsys, "doctor", "--json")
    assert code == 0
    res = json.loads(out)
    assert res["ok"] and res["smoke"] == "ok"
    assert "gymnasium" in res["optional"]


def test_unknown_task_json_error(capsys):
    code, out = _run(capsys, "info", "nope/Task-v0", "--json")
    assert code == 1
    assert json.loads(out)["error"]["code"] == "unknown_task"


def test_bad_set_syntax_json_error(capsys):
    code, out = _run(capsys, "run", "qubit/StateTransfer-v0", "--set", "horizon", "--json")
    assert code == 1
    assert json.loads(out)["error"]["code"] == "invalid_argument"


def test_run_output_schema(capsys):
    code, out = _run(capsys, "run", "qubit/StateTransfer-v0", "--set", "horizon=3", "--json")
    res = json.loads(out)
    assert code == 0
    for key in ("model", "model_params", "units", "solver", "actions", "rewards", "final"):
        assert key in res
    assert res["solver"]["method"] == "Tsit5"
    assert len(res["rewards"]) == 3


def test_validate_markdown_report(capsys):
    code, out = _run(capsys, "validate", "--markdown")
    assert code == 0
    assert out.startswith("**") and "| check | category |" in out
    for model in ("TwoLevel", "QubitBattery", "DickeBattery", "LambdaSystem"):
        assert f"## {model}" in out
    assert "FAIL" not in out
