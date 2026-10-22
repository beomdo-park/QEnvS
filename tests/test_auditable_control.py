"""Audit examples must preserve candidate identities, numerical evidence and provenance."""

import importlib.util
import json
import subprocess
from pathlib import Path

import dynamiqs as dq
import jax
import jax.numpy as jnp
import numpy as np
import pytest


def example():
    path = Path(__file__).parents[1] / "examples/auditable_control.py"
    assert path.is_file(), "auditable control example is missing"
    spec = importlib.util.spec_from_file_location("auditable_control", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selection_preserves_sources_and_negative_regret():
    ex = example()
    result = ex.select_candidates(np.array([[2, 1.8, 2.1, 1.7], [1, 1.2, 2, 1.9]]), [0, 0, 1, 1])
    assert result["selected_candidates"] == [0, 2]
    np.testing.assert_allclose(result["selected_matrix"], [[2, 2.1], [1, 2]])
    np.testing.assert_allclose(result["regret"], [[0, -0.05], [0.5, 0]], atol=1e-14)
    tie = ex.select_candidates(np.ones((2, 4)), [0, 0, 1, 1])
    assert tie["selected_candidates"] == [0, 2]
    zero = ex.select_candidates(np.array([[0.0, 1], [1, 2]]), [0, 1])
    assert zero["regret"][0] == [None, None]
    assert zero["undefined_reason"][0]


@pytest.mark.parametrize(
    "scores,sources",
    [
        (np.ones((2, 3)), [0, 0, 0]),
        (np.ones((2, 3)), [0, 1]),
        (np.array([[1, np.nan], [2, 3]]), [0, 1]),
        (np.ones((2, 2)), [0.0, 1.0]),
        (np.ones((2, 2)), [0, 2]),
    ],
)
def test_invalid_candidate_records(scores, sources):
    with pytest.raises(ValueError):
        example().select_candidates(scores, sources)


def test_evaluation_preserves_all_candidates_and_rejects_nonfinite():
    ex = example()
    models = [ex.make_model(0.0), ex.make_model(0.2)]
    candidates = np.full((4, 2, 1), 0.8)
    method = dq.method.Tsit5(rtol=1e-10, atol=1e-12)
    result = ex.evaluate_candidates(models, candidates, [0, 0, 1, 1], dt=0.4, method=method)
    assert np.shape(result["scores"]) == (2, 4)
    assert np.shape(result["peak_times"]) == (2, 4)
    assert np.shape(result["terminal"]) == (2, 4)
    assert np.isfinite(result["scores"]).all()
    candidates[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        ex.evaluate_candidates(models, candidates, [0, 0, 1, 1], dt=0.4, method=method)


def test_evaluation_matrix_matches_direct_rollouts():
    # distinct models and candidates, so a transposed matrix or a wrong reduction shows up
    from qenvs.trajectories import rollout_metrics

    ex = example()
    models = [ex.make_model(0.0), ex.make_model(0.2)]
    candidates = np.array([[[0.8], [0.8], [0.8]], [[1.0], [-1.0], [0.3]], [[0.2], [0.9], [-0.5]]])
    method = dq.method.Tsit5(rtol=1e-10, atol=1e-12)
    result = ex.evaluate_candidates(models, candidates, [0, 1, 1], dt=0.4, method=method)
    early_peak = False
    for i, model in enumerate(models):
        for j, actions in enumerate(candidates):
            times, values = rollout_metrics(
                model, jnp.asarray(actions), dt=0.4, metrics=ex.METRICS, method=method
            )
            work = np.asarray(values["work"])
            peak = int(work.argmax())
            early_peak |= peak < len(work) - 1
            assert result["scores"][i][j] == pytest.approx(work[peak], abs=1e-12)
            assert result["terminal"][i][j] == pytest.approx(work[-1], abs=1e-12)
            assert result["peak_times"][i][j] == pytest.approx(float(times[peak]), abs=1e-12)
    assert early_peak, "fixture must include a candidate whose peak is not the last step"
    assert not np.allclose(result["scores"][0], result["scores"][1])


@pytest.mark.skipif(not jax.config.jax_enable_x64, reason="the example requires x64")
def test_main_writes_the_audit_record(tmp_path, monkeypatch, capsys):
    pytest.importorskip("optax")
    ex = example()
    out = tmp_path / "run"
    monkeypatch.setattr("sys.argv", ["auditable_control.py", "--out", str(out)])
    ex.main()
    names = {p.name for p in out.iterdir()}
    expected = {f"candidate-{i}.json" for i in range(4)}  # two models, two starts each
    expected |= {"initial.json", "history.json", "manifest.json", "evaluation.json"}
    assert names == expected
    manifest = json.loads((out / "manifest.json").read_text())
    for key in ("seed", "dt", "source", "qenvs_imported_version", "versions", "candidates_sha256"):
        assert key in manifest
    evaluation = json.loads((out / "evaluation.json").read_text())
    assert np.shape(evaluation["scores"]) == (2, 4)
    assert json.loads(capsys.readouterr().out)["candidates"] == 4


def test_fixed_warmup_prefix_and_saved_score():
    ex = example()
    m = ex.make_model(0.0)
    initial = jnp.full((3, 1), 0.8)
    kwargs = {"warmup_steps": 1, "dt": 0.4, "method": dq.method.Tsit5(rtol=1e-8, atol=1e-10)}
    short = ex.optimize(m, initial, iterations=2, **kwargs)
    long = ex.optimize(m, initial, iterations=3, **kwargs)
    assert short["history"] == long["history"][:2]
    for row in long["history"]:
        assert np.isfinite([row["gradient_norm"], row["update_norm"]]).all()
    result = ex.evaluate_candidates(
        [m], np.array([short["actions"]]), [0], dt=0.4, method=kwargs["method"]
    )
    assert result["scores"][0][0] == pytest.approx(short["score"], abs=1e-9)


def test_provenance_does_not_claim_an_enclosing_repo(tmp_path):
    ex = example()
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    installed = tmp_path / "site-packages/qenvs/__init__.py"
    installed.parent.mkdir(parents=True)
    installed.write_text("")
    result = ex.source_provenance(installed)
    assert result["git_sha"] is None
    assert result["git_dirty"] is None
    assert result["source_sha256"]
    checkout = tmp_path / "src/qenvs/__init__.py"
    checkout.parent.mkdir(parents=True)
    checkout.write_text("# explicit source checkout fixture\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "src"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    actual = ex.source_provenance(checkout)
    expected = subprocess.check_output(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"], text=True
    ).strip()
    assert actual["git_sha"] == expected
    assert actual["git_dirty"] is True  # the unrelated installed fixture is untracked


def test_json_output_refuses_overwrite_and_nonfinite(tmp_path):
    ex = example()
    path = tmp_path / "result.json"
    ex.save_json(path, {"score": 1.0})
    with pytest.raises(FileExistsError):
        ex.save_json(path, {"score": 2.0})
    assert json.loads(path.read_text()) == {"score": 1.0}
    invalid = tmp_path / "invalid.json"
    with pytest.raises(ValueError):
        ex.save_json(invalid, {"score": float("nan")})
    assert not invalid.exists()


def test_policy_boundary_is_only_declared_observation():
    ex = example()
    seen = []

    def policy(obs):
        seen.append(np.asarray(obs))
        return jnp.zeros(1)

    ex.observation_demo(policy)
    assert len(seen) == 2
    assert all(obs.shape == (2,) for obs in seen)
    assert all(np.isfinite(obs).all() for obs in seen)
