from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import duckdb

from workout_ml.artifacts import ArtifactLineage


StageName = Literal["download", "label", "pose", "dataset", "train", "export"]
StageState = Literal["pending", "running", "complete", "failed", "skipped"]
SourceType = Literal["video", "playlist", "channel"]
SourcePriority = Literal["general_bootstrap", "personal_routine"]


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    url: str
    source_type: SourceType
    priority: SourcePriority
    notes: str = ""


@dataclass(frozen=True)
class VideoMetadata:
    video_id: str
    url: str
    channel: str | None
    title: str | None
    duration_s: float | None
    fps: float | None
    width: int | None
    height: int | None
    downloaded_at: str


@dataclass(frozen=True)
class StageStatus:
    video_id: str
    stage: StageName
    artifact_id: str
    status: StageState
    updated_at: str
    error: str | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Catalog:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = duckdb.connect(str(path))
        self.initialize()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> Catalog:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def initialize(self) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS videos (
                video_id VARCHAR PRIMARY KEY,
                url VARCHAR NOT NULL,
                channel VARCHAR,
                title VARCHAR,
                duration_s DOUBLE,
                fps DOUBLE,
                width INTEGER,
                height INTEGER,
                downloaded_at VARCHAR NOT NULL
            )
            """
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS sources (
                source_id VARCHAR PRIMARY KEY,
                url VARCHAR NOT NULL,
                source_type VARCHAR NOT NULL,
                priority VARCHAR NOT NULL,
                notes VARCHAR
            )
            """
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS stage_status (
                video_id VARCHAR NOT NULL,
                stage VARCHAR NOT NULL,
                artifact_id VARCHAR NOT NULL,
                status VARCHAR NOT NULL,
                updated_at VARCHAR NOT NULL,
                error VARCHAR,
                PRIMARY KEY (video_id, stage, artifact_id)
            )
            """
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS artifacts (
                artifact_id VARCHAR PRIMARY KEY,
                stage VARCHAR NOT NULL,
                input_fingerprint VARCHAR NOT NULL,
                config_fingerprint VARCHAR NOT NULL,
                code_version VARCHAR NOT NULL,
                model_version VARCHAR,
                path VARCHAR NOT NULL,
                created_at VARCHAR NOT NULL
            )
            """
        )

    def upsert_source(self, source: SourceRecord) -> None:
        self.connection.execute("DELETE FROM sources WHERE source_id = ?", [source.source_id])
        self.connection.execute(
            """
            INSERT INTO sources (source_id, url, source_type, priority, notes)
            VALUES (?, ?, ?, ?, ?)
            """,
            [source.source_id, source.url, source.source_type, source.priority, source.notes],
        )

    def upsert_video(self, video: VideoMetadata) -> None:
        self.connection.execute("DELETE FROM videos WHERE video_id = ?", [video.video_id])
        self.connection.execute(
            """
            INSERT INTO videos (
                video_id, url, channel, title, duration_s, fps, width, height, downloaded_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                video.video_id,
                video.url,
                video.channel,
                video.title,
                video.duration_s,
                video.fps,
                video.width,
                video.height,
                video.downloaded_at,
            ],
        )

    def record_artifact(self, artifact: ArtifactLineage) -> None:
        self.connection.execute(
            "DELETE FROM artifacts WHERE artifact_id = ?",
            [artifact.artifact_id],
        )
        self.connection.execute(
            """
            INSERT INTO artifacts (
                artifact_id, stage, input_fingerprint, config_fingerprint, code_version,
                model_version, path, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                artifact.artifact_id,
                artifact.stage,
                artifact.input_fingerprint,
                artifact.config_fingerprint,
                artifact.code_version,
                artifact.model_version,
                artifact.path,
                artifact.created_at,
            ],
        )

    def set_stage_status(
        self,
        *,
        video_id: str,
        stage: StageName,
        artifact_id: str,
        status: StageState,
        error: str | None = None,
    ) -> None:
        self.connection.execute(
            """
            DELETE FROM stage_status
            WHERE video_id = ? AND stage = ? AND artifact_id = ?
            """,
            [video_id, stage, artifact_id],
        )
        self.connection.execute(
            """
            INSERT INTO stage_status (video_id, stage, artifact_id, status, updated_at, error)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [video_id, stage, artifact_id, status, utc_now(), error],
        )

    def get_stage_status(
        self,
        *,
        video_id: str,
        stage: StageName,
        artifact_id: str,
    ) -> StageStatus | None:
        row = self.connection.execute(
            """
            SELECT video_id, stage, artifact_id, status, updated_at, error
            FROM stage_status
            WHERE video_id = ? AND stage = ? AND artifact_id = ?
            """,
            [video_id, stage, artifact_id],
        ).fetchone()
        if row is None:
            return None
        return StageStatus(*row)

    def is_stage_complete(self, *, video_id: str, stage: StageName, artifact_id: str) -> bool:
        status = self.get_stage_status(
            video_id=video_id,
            stage=stage,
            artifact_id=artifact_id,
        )
        return status is not None and status.status == "complete"

    def table_count(self, table_name: str) -> int:
        if table_name not in {"videos", "sources", "stage_status", "artifacts"}:
            raise ValueError(f"Unsupported catalog table: {table_name}")
        return int(self.connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0])
