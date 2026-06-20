from __future__ import annotations

from workout_ml.artifacts import ArtifactLineage
from workout_ml.catalog import Catalog, SourceRecord, VideoMetadata, utc_now


def test_catalog_schema_creation_is_idempotent(tmp_path) -> None:
    catalog_path = tmp_path / "catalog.duckdb"

    with Catalog(catalog_path) as catalog:
        catalog.initialize()
        assert catalog.table_count("videos") == 0
        assert catalog.table_count("sources") == 0

    with Catalog(catalog_path) as catalog:
        assert catalog.table_count("artifacts") == 0


def test_catalog_records_sources_videos_artifacts_and_stage_status(tmp_path) -> None:
    artifact = ArtifactLineage.create(
        stage="download",
        input_fingerprint="input123",
        config_fingerprint="config123",
        code_version="0.1.0",
        path=tmp_path / "raw" / "video-1",
    )

    with Catalog(tmp_path / "catalog.duckdb") as catalog:
        catalog.upsert_source(
            SourceRecord(
                source_id="source-1",
                url="https://example.com/playlist",
                source_type="playlist",
                priority="general_bootstrap",
                notes="test",
            )
        )
        catalog.upsert_video(
            VideoMetadata(
                video_id="video-1",
                url="https://example.com/watch?v=video-1",
                channel="Example",
                title="Workout",
                duration_s=120.0,
                fps=30.0,
                width=1280,
                height=720,
                downloaded_at=utc_now(),
            )
        )
        catalog.record_artifact(artifact)
        catalog.set_stage_status(
            video_id="video-1",
            stage="download",
            artifact_id=artifact.artifact_id,
            status="complete",
        )

        assert catalog.table_count("sources") == 1
        assert catalog.table_count("videos") == 1
        assert catalog.table_count("artifacts") == 1
        assert catalog.is_stage_complete(
            video_id="video-1",
            stage="download",
            artifact_id=artifact.artifact_id,
        )
        assert not catalog.is_stage_complete(
            video_id="video-1",
            stage="download",
            artifact_id="stale-artifact",
        )
