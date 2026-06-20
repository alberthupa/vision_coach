from __future__ import annotations

import pytest
from pydantic import ValidationError

from workout_ml.ingest.channels import load_source_registry


def test_load_source_registry_accepts_valid_sources(tmp_path) -> None:
    path = tmp_path / "sources.yaml"
    path.write_text(
        """
sources:
  - source_id: general_playlist
    url: https://www.youtube.com/playlist?list=abc
    source_type: playlist
    priority: general_bootstrap
    notes: bootstrap
    limit: 5
  - source_id: personal_video
    url: https://www.youtube.com/watch?v=abc
    source_type: video
    priority: personal_routine
""",
        encoding="utf-8",
    )

    registry = load_source_registry(path)

    assert len(registry.sources) == 2
    assert registry.sources[0].source_type == "playlist"
    assert registry.sources[1].priority == "personal_routine"


def test_load_source_registry_rejects_invalid_type_and_priority(tmp_path) -> None:
    path = tmp_path / "sources.yaml"
    path.write_text(
        """
sources:
  - source_id: bad
    url: https://www.youtube.com/watch?v=abc
    source_type: shorts
    priority: maybe
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_source_registry(path)
