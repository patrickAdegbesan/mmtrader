import pytest
import torch

from cognition.learning.registry import ModelRegistry


def make_state(tag: float) -> dict:
    return {"weights": torch.tensor([tag])}


def test_save_creates_incrementing_immutable_versions(tmp_path):
    reg = ModelRegistry(tmp_path, "momentum")
    v1 = reg.save(make_state(1.0), {"note": "first"})
    v2 = reg.save(make_state(2.0), {"note": "second"})
    assert (v1, v2) == ("v0001", "v0002")
    assert reg.list_versions() == ["v0001", "v0002"]
    state1, meta1 = reg.load("v0001")
    assert state1["weights"].item() == 1.0
    assert meta1["note"] == "first"
    assert "saved_at" in meta1


def test_latest_points_to_most_recent_save(tmp_path):
    reg = ModelRegistry(tmp_path, "momentum")
    reg.save(make_state(1.0), {})
    reg.save(make_state(2.0), {})
    assert reg.latest_version() == "v0002"
    state, _ = reg.load()
    assert state["weights"].item() == 2.0


def test_rollback_by_activating_older_version(tmp_path):
    reg = ModelRegistry(tmp_path, "momentum")
    reg.save(make_state(1.0), {})
    reg.save(make_state(2.0), {})
    reg.activate("v0001")  # rollback — new version stays on disk untouched
    assert reg.latest_version() == "v0001"
    state, _ = reg.load()
    assert state["weights"].item() == 1.0
    assert reg.list_versions() == ["v0001", "v0002"]


def test_activating_missing_version_fails(tmp_path):
    reg = ModelRegistry(tmp_path, "momentum")
    reg.save(make_state(1.0), {})
    with pytest.raises(FileNotFoundError):
        reg.activate("v9999")


def test_load_from_empty_registry_fails(tmp_path):
    reg = ModelRegistry(tmp_path, "empty")
    with pytest.raises(FileNotFoundError):
        reg.load()
