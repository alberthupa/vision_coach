from __future__ import annotations

import json

from workout_ml.artifacts import (
    ArtifactLineage,
    fingerprint_payload,
    write_json_atomic,
)


def test_fingerprint_payload_is_stable_for_key_order() -> None:
    first = fingerprint_payload({"b": 2, "a": 1})
    second = fingerprint_payload({"a": 1, "b": 2})

    assert first == second


def test_write_json_atomic_replaces_file(tmp_path) -> None:
    path = tmp_path / "artifact.json"
    write_json_atomic(path, {"old": True})
    write_json_atomic(path, {"new": True})

    assert json.loads(path.read_text(encoding="utf-8")) == {"new": True}
    assert list(tmp_path.glob("*.tmp")) == []


def test_artifact_lineage_contains_required_stage_metadata(tmp_path) -> None:
    lineage = ArtifactLineage.create(
        stage="pose",
        input_fingerprint="input123",
        config_fingerprint="config123",
        code_version="abc123",
        model_version=None,
        path=tmp_path / "pose.parquet",
    )

    payload = lineage.to_dict()

    assert payload["artifact_id"]
    assert payload["stage"] == "pose"
    assert payload["input_fingerprint"] == "input123"
    assert payload["config_fingerprint"] == "config123"
    assert payload["code_version"] == "abc123"
    assert payload["path"].endswith("pose.parquet")
