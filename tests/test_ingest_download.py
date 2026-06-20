from __future__ import annotations

import json
from pathlib import Path
import subprocess
from collections.abc import Sequence

from workout_ml.catalog import Catalog
from workout_ml.ingest.channels import source_from_cli
from workout_ml.ingest.download import (
    build_download_args,
    build_ffprobe_args,
    expand_source,
    ingest_sources,
    parse_ffprobe_json,
)
from workout_ml.settings import ensure_data_dirs, load_pipeline_settings


class FakeRunner:
    def __init__(self, *, fail_video_ids: set[str] | None = None) -> None:
        self.commands: list[list[str]] = []
        self.fail_video_ids = fail_video_ids or set()

    def __call__(self, args: Sequence[str]) -> subprocess.CompletedProcess[str]:
        command = list(args)
        self.commands.append(command)
        if command[:2] == ["yt-dlp", "--flat-playlist"]:
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps(
                    {
                        "entries": [
                            {"id": "video-good", "url": "video-good"},
                            {"id": "video-bad", "url": "video-bad"},
                        ]
                    }
                ),
                "",
            )
        if command[:2] == ["yt-dlp", "--no-warnings"]:
            return subprocess.CompletedProcess(command, 0, "video-good\n", "")
        if command[0] == "yt-dlp" and "-o" in command:
            url = command[-1]
            video_id = url.rsplit("=", 1)[-1]
            if video_id in self.fail_video_ids:
                return subprocess.CompletedProcess(command, 1, "", "download failed")
            output_template = Path(command[command.index("-o") + 1])
            raw_dir = output_template.parent
            raw_dir.mkdir(parents=True, exist_ok=True)
            (raw_dir / "video.mp4").write_bytes(b"fake mp4")
            (raw_dir / "video.info.json").write_text(
                json.dumps(
                    {
                        "id": video_id,
                        "webpage_url": url,
                        "channel": "Example Channel",
                        "title": "Example Workout",
                        "duration": 120.0,
                    }
                ),
                encoding="utf-8",
            )
            (raw_dir / "video.description").write_text("description", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, "downloaded", "")
        if command[0] == "ffprobe":
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps(
                    {
                        "streams": [
                            {
                                "width": 1280,
                                "height": 720,
                                "avg_frame_rate": "30000/1001",
                            }
                        ],
                        "format": {"duration": "123.45"},
                    }
                ),
                "",
            )
        raise AssertionError(f"Unexpected command: {command}")

    @property
    def download_call_count(self) -> int:
        return sum(1 for command in self.commands if command and command[0] == "yt-dlp" and "-o" in command)


def _settings(tmp_path):
    config = tmp_path / "pipeline.yaml"
    data_dir = tmp_path / "data"
    config.write_text(
        f"""
paths:
  data_dir: {data_dir}
  raw_dir: {data_dir / "raw"}
  labels_dir: {data_dir / "labels"}
  poses_dir: {data_dir / "poses"}
  datasets_dir: {data_dir / "datasets"}
  models_dir: {data_dir / "models"}
  logs_dir: {data_dir / "logs"}
  catalog_path: {data_dir / "catalog.duckdb"}
runtime:
  default_workers: 1
  worker_omp_threads: 2
artifacts:
  fingerprint_size: 16
video:
  max_height: 720
  download_retries: 2
  sleep_interval_seconds: 3
  max_sleep_interval_seconds: 8
""",
        encoding="utf-8",
    )
    settings = load_pipeline_settings(config)
    ensure_data_dirs(settings)
    return settings


def test_download_args_include_required_ytdlp_policy(tmp_path) -> None:
    settings = _settings(tmp_path)

    args = build_download_args(
        url="https://www.youtube.com/watch?v=abc",
        output_dir=tmp_path / "raw" / "abc",
        settings=settings,
    )

    assert "bv*[height<=720]+ba/b[height<=720]" in args
    assert "--write-info-json" in args
    assert "--write-description" in args
    assert "--write-auto-subs" in args
    assert "--sub-langs" in args
    assert "--sleep-interval" in args
    assert "--max-sleep-interval" in args


def test_parse_ffprobe_json_extracts_video_metadata() -> None:
    probe = parse_ffprobe_json(
        json.dumps(
            {
                "streams": [
                    {
                        "width": 640,
                        "height": 360,
                        "avg_frame_rate": "30000/1001",
                    }
                ],
                "format": {"duration": "65.5"},
            }
        )
    )

    assert probe.duration_s == 65.5
    assert round(probe.fps or 0, 2) == 29.97
    assert probe.width == 640
    assert probe.height == 360


def test_expand_source_uses_flat_playlist_and_limit() -> None:
    runner = FakeRunner()
    source = source_from_cli(
        source_id="playlist",
        url="https://www.youtube.com/playlist?list=abc",
        source_type="playlist",
        limit=1,
    )

    entries = expand_source(source, runner=runner)

    assert len(entries) == 1
    assert entries[0].video_id == "video-good"
    assert "--playlist-end" in runner.commands[0]


def test_ingest_is_resumable_and_unchanged_rerun_skips_download(tmp_path) -> None:
    settings = _settings(tmp_path)
    source = source_from_cli(
        source_id="single",
        url="https://www.youtube.com/watch?v=video-good",
        source_type="video",
    )
    runner = FakeRunner()

    with Catalog(settings.paths.catalog_path) as catalog:
        first = ingest_sources(
            sources=[source],
            settings=settings,
            catalog=catalog,
            workers=1,
            runner=runner,
        )
        second = ingest_sources(
            sources=[source],
            settings=settings,
            catalog=catalog,
            workers=1,
            runner=runner,
        )

        assert [result.status for result in first] == ["downloaded"]
        assert [result.status for result in second] == ["skipped"]
        assert runner.download_call_count == 1
        assert catalog.table_count("videos") == 1
        assert catalog.table_count("artifacts") == 1
        assert (settings.paths.raw_dir / "video-good" / "info.json").exists()
        assert (settings.paths.raw_dir / "video-good" / "description.txt").exists()


def test_ingest_logs_failed_video_and_continues(tmp_path) -> None:
    settings = _settings(tmp_path)
    source = source_from_cli(
        source_id="playlist",
        url="https://www.youtube.com/playlist?list=abc",
        source_type="playlist",
        limit=2,
    )
    runner = FakeRunner(fail_video_ids={"video-bad"})

    with Catalog(settings.paths.catalog_path) as catalog:
        results = ingest_sources(
            sources=[source],
            settings=settings,
            catalog=catalog,
            workers=1,
            runner=runner,
        )

        assert sorted(result.status for result in results) == ["downloaded", "failed"]
        assert catalog.table_count("videos") == 1
        failure_log = settings.paths.logs_dir / "download_failures.jsonl"
        assert failure_log.exists()
        assert "download failed" in failure_log.read_text(encoding="utf-8")


def test_build_ffprobe_args_points_at_video_file(tmp_path) -> None:
    video_path = tmp_path / "video.mp4"

    args = build_ffprobe_args(video_path)

    assert args[-1] == str(video_path)
    assert "-show_entries" in args
