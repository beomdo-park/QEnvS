"""Released tasks never change behaviour (docs/stability.md): compare with tests/golden."""

import json
import pathlib
import sys

import pytest

GOLDEN = pathlib.Path(__file__).with_name("golden")
sys.path.insert(0, str(GOLDEN))

import make_task_snapshot  # noqa: E402

SNAPSHOT = json.loads((GOLDEN / "tasks.json").read_text())


@pytest.mark.parametrize("task_id", sorted(SNAPSHOT))
def test_released_task_is_unchanged(task_id):
    import qenvs

    assert task_id in qenvs.registry, f"released task {task_id} was removed"
    now = make_task_snapshot.snapshot(task_id)
    old = SNAPSHOT[task_id]
    for key in ("horizon", "action_dim", "observation_dim", "open_system", "info_keys"):
        assert now[key] == old[key], key
    for key in ("dt", "action_bound", "zero_return", "fixed_pulse_return", "reference_return"):
        if key in old:
            assert now[key] == pytest.approx(old[key], abs=1e-6, rel=1e-6), key
    for k, v in old["fixed_pulse_info"].items():
        assert now["fixed_pulse_info"][k] == pytest.approx(v, abs=1e-6, rel=1e-6), k
