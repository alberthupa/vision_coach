from __future__ import annotations

import json

from workout_ml.progress import StageFailure, append_failure


def test_append_failure_writes_jsonl_record(tmp_path) -> None:
    failure = StageFailure.create(
        stage="download",
        item_id="video-1",
        error="network timeout",
        context={"attempt": 2},
    )

    path = append_failure(tmp_path, failure)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert path.name == "download_failures.jsonl"
    assert records[0]["stage"] == "download"
    assert records[0]["item_id"] == "video-1"
    assert records[0]["context"] == {"attempt": 2}
