from __future__ import annotations

import os

import pytest

from workout_ml.catalog import Catalog
from workout_ml.ingest.channels import source_from_cli
from workout_ml.ingest.download import ingest_sources
from workout_ml.settings import ensure_data_dirs, load_pipeline_settings


@pytest.mark.live_ingest
def test_live_playlist_ingest_acceptance() -> None:
    playlist_url = os.environ.get("WORKOUT_ML_LIVE_PLAYLIST_URL")
    if not playlist_url:
        pytest.skip("Set WORKOUT_ML_LIVE_PLAYLIST_URL to run the Phase 1 live ingest acceptance test.")

    settings = load_pipeline_settings()
    ensure_data_dirs(settings)
    source = source_from_cli(
        source_id="live_acceptance_playlist",
        url=playlist_url,
        source_type="playlist",
        limit=5,
    )

    with Catalog(settings.paths.catalog_path) as catalog:
        first = ingest_sources(
            sources=[source],
            settings=settings,
            catalog=catalog,
            workers=1,
        )
        second = ingest_sources(
            sources=[source],
            settings=settings,
            catalog=catalog,
            workers=1,
        )

    assert len(first) == 5
    assert all(result.status in {"downloaded", "skipped"} for result in first)
    assert all(result.status == "skipped" for result in second)
