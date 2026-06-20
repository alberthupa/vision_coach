from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Literal

from workout_ml import __version__
from workout_ml.artifacts import ArtifactLineage, fingerprint_payload
from workout_ml.catalog import Catalog, VideoMetadata, utc_now
from workout_ml.ingest.channels import SourceConfig
from workout_ml.progress import StageFailure, append_failure, report_progress
from workout_ml.settings import PipelineSettings


CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]
DownloadStatus = Literal["downloaded", "skipped", "failed"]


@dataclass(frozen=True)
class VideoEntry:
    url: str
    video_id: str | None = None


@dataclass(frozen=True)
class VideoProbe:
    duration_s: float | None
    fps: float | None
    width: int | None
    height: int | None


@dataclass(frozen=True)
class DownloadResult:
    status: DownloadStatus
    source_id: str
    url: str
    video_id: str | None
    artifact: ArtifactLineage | None = None
    metadata: VideoMetadata | None = None
    error: str | None = None


def run_command(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, check=False, text=True)


def _require_success(result: subprocess.CompletedProcess[str], label: str) -> str:
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        detail = stderr or stdout or f"exit code {result.returncode}"
        raise RuntimeError(f"{label} failed: {detail}")
    return result.stdout


def build_video_id_args(url: str) -> list[str]:
    return ["yt-dlp", "--no-warnings", "--skip-download", "--print", "id", url]


def build_flat_playlist_args(url: str, limit: int | None = None) -> list[str]:
    args = ["yt-dlp", "--flat-playlist", "--dump-single-json"]
    if limit is not None:
        args.extend(["--playlist-end", str(limit)])
    args.append(url)
    return args


def build_download_args(
    *,
    url: str,
    output_dir: Path,
    settings: PipelineSettings,
) -> list[str]:
    video = settings.video
    return [
        "yt-dlp",
        "--no-progress",
        "--retries",
        str(video.download_retries),
        "--fragment-retries",
        str(video.download_retries),
        "--sleep-interval",
        str(video.sleep_interval_seconds),
        "--max-sleep-interval",
        str(video.max_sleep_interval_seconds),
        "-f",
        f"bv*[height<={video.max_height}]+ba/b[height<={video.max_height}]",
        "--merge-output-format",
        video.merge_output_format,
        "--write-info-json",
        "--write-description",
        "--write-auto-subs",
        "--sub-langs",
        video.subtitle_languages,
        "-o",
        str(output_dir / "video.%(ext)s"),
        url,
    ]


def build_ffprobe_args(video_path: Path) -> list[str]:
    return [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate:format=duration",
        "-of",
        "json",
        str(video_path),
    ]


def parse_ffprobe_json(payload: str) -> VideoProbe:
    parsed = json.loads(payload)
    streams = parsed.get("streams") or []
    stream = streams[0] if streams else {}
    fmt = parsed.get("format") or {}
    return VideoProbe(
        duration_s=_optional_float(fmt.get("duration")),
        fps=_parse_fps(stream.get("avg_frame_rate")) or _parse_fps(stream.get("r_frame_rate")),
        width=_optional_int(stream.get("width")),
        height=_optional_int(stream.get("height")),
    )


def expand_source(source: SourceConfig, runner: CommandRunner = run_command) -> list[VideoEntry]:
    if source.source_type == "video":
        return [VideoEntry(url=str(source.url))]
    stdout = _require_success(
        runner(build_flat_playlist_args(str(source.url), source.limit)),
        f"source expansion for {source.source_id}",
    )
    payload = json.loads(stdout)
    entries = payload.get("entries") or []
    expanded: list[VideoEntry] = []
    for raw_entry in entries:
        if not isinstance(raw_entry, dict):
            continue
        video_id = raw_entry.get("id")
        url = raw_entry.get("webpage_url") or raw_entry.get("url")
        if isinstance(url, str) and url.startswith("http"):
            expanded.append(VideoEntry(url=url, video_id=video_id))
        elif isinstance(video_id, str) and video_id:
            expanded.append(
                VideoEntry(
                    url=f"https://www.youtube.com/watch?v={video_id}",
                    video_id=video_id,
                )
            )
    if source.limit is not None:
        return expanded[: source.limit]
    return expanded


def download_video(
    *,
    entry: VideoEntry,
    source: SourceConfig,
    settings: PipelineSettings,
    config_fingerprint: str,
    catalog_complete: bool = False,
    runner: CommandRunner = run_command,
) -> DownloadResult:
    try:
        video_id, artifact = create_download_artifact_for_entry(
            entry=entry,
            source=source,
            settings=settings,
            config_fingerprint=config_fingerprint,
            runner=runner,
        )
        raw_dir = settings.paths.raw_dir / video_id
        if catalog_complete and is_download_materialized(raw_dir):
            metadata = build_video_metadata(
                video_id=video_id,
                url=entry.url,
                raw_dir=raw_dir,
                runner=runner,
            )
            return DownloadResult(
                status="skipped",
                source_id=source.source_id,
                url=entry.url,
                video_id=video_id,
                artifact=artifact,
                metadata=metadata,
            )

        raw_dir.mkdir(parents=True, exist_ok=True)
        _require_success(
            runner(build_download_args(url=entry.url, output_dir=raw_dir, settings=settings)),
            f"download for {entry.url}",
        )
        normalize_download_artifacts(raw_dir)
        metadata = build_video_metadata(
            video_id=video_id,
            url=entry.url,
            raw_dir=raw_dir,
            runner=runner,
        )
        return DownloadResult(
            status="downloaded",
            source_id=source.source_id,
            url=entry.url,
            video_id=video_id,
            artifact=artifact,
            metadata=metadata,
        )
    except Exception as exc:
        return DownloadResult(
            status="failed",
            source_id=source.source_id,
            url=entry.url,
            video_id=entry.video_id,
            error=str(exc),
        )


def ingest_sources(
    *,
    sources: Iterable[SourceConfig],
    settings: PipelineSettings,
    catalog: Catalog,
    workers: int,
    runner: CommandRunner = run_command,
) -> list[DownloadResult]:
    config_fingerprint = fingerprint_payload(
        {
            "video": settings.video.model_dump(mode="json"),
            "raw_dir": str(settings.paths.raw_dir),
        },
        size=settings.artifacts.fingerprint_size,
    )
    prepared_sources = list(sources)
    for source in prepared_sources:
        catalog.upsert_source(source.to_catalog_record())

    entries: list[tuple[VideoEntry, SourceConfig]] = []
    for source in prepared_sources:
        try:
            for entry in expand_source(source, runner=runner):
                entries.append((entry, source))
        except Exception as exc:
            append_failure(
                settings.paths.logs_dir,
                StageFailure.create(
                    stage="download",
                    item_id=source.source_id,
                    error=str(exc),
                    context={"url": str(source.url), "source_type": source.source_type},
                ),
            )

    results: list[DownloadResult] = []
    completed = 0
    total = len(entries)
    if total == 0:
        report_progress("download", 0, 0, "no videos to ingest")
        return []

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = []
        for entry, source in entries:
            catalog_complete = False
            prepared_entry = entry
            try:
                video_id, artifact = create_download_artifact_for_entry(
                    entry=entry,
                    source=source,
                    settings=settings,
                    config_fingerprint=config_fingerprint,
                    runner=runner,
                )
                prepared_entry = VideoEntry(url=entry.url, video_id=video_id)
                catalog_complete = catalog.is_stage_complete(
                    video_id=video_id,
                    stage="download",
                    artifact_id=artifact.artifact_id,
                )
            except Exception:
                pass
            futures.append(
                executor.submit(
                    download_video,
                    entry=prepared_entry,
                    source=source,
                    settings=settings,
                    config_fingerprint=config_fingerprint,
                    catalog_complete=catalog_complete,
                    runner=runner,
                )
            )
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            record_result(result, settings=settings, catalog=catalog)
            completed += 1
            report_progress(
                "download",
                completed,
                total,
                f"{result.status}: {result.video_id or result.url}",
            )
    return results


def record_result(result: DownloadResult, *, settings: PipelineSettings, catalog: Catalog) -> None:
    if result.status == "failed":
        append_failure(
            settings.paths.logs_dir,
            StageFailure.create(
                stage="download",
                item_id=result.video_id or result.url,
                error=result.error or "unknown download error",
                context={"source_id": result.source_id, "url": result.url},
            ),
        )
        if result.video_id and result.artifact:
            catalog.set_stage_status(
                video_id=result.video_id,
                stage="download",
                artifact_id=result.artifact.artifact_id,
                status="failed",
                error=result.error,
            )
        return
    if result.artifact is None or result.metadata is None or result.video_id is None:
        raise ValueError("Successful download result is missing metadata or artifact")
    catalog.record_artifact(result.artifact)
    catalog.upsert_video(result.metadata)
    catalog.set_stage_status(
        video_id=result.video_id,
        stage="download",
        artifact_id=result.artifact.artifact_id,
        status="complete",
    )


def fetch_video_id(url: str, runner: CommandRunner = run_command) -> str:
    stdout = _require_success(runner(build_video_id_args(url)), f"video id lookup for {url}")
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    raise RuntimeError(f"yt-dlp did not return a video id for {url}")


def create_download_artifact(
    *,
    video_id: str,
    raw_dir: Path,
    input_fingerprint: str,
    config_fingerprint: str,
    fingerprint_size: int,
) -> ArtifactLineage:
    artifact_id = fingerprint_payload(
        {
            "stage": "download",
            "video_id": video_id,
            "input_fingerprint": input_fingerprint,
            "config_fingerprint": config_fingerprint,
            "code_version": __version__,
            "path": str(raw_dir),
        },
        size=fingerprint_size,
    )
    return ArtifactLineage(
        artifact_id=artifact_id,
        stage="download",
        input_fingerprint=input_fingerprint,
        config_fingerprint=config_fingerprint,
        code_version=__version__,
        model_version=None,
        path=str(raw_dir),
        created_at=utc_now(),
    )


def create_download_artifact_for_entry(
    *,
    entry: VideoEntry,
    source: SourceConfig,
    settings: PipelineSettings,
    config_fingerprint: str,
    runner: CommandRunner = run_command,
) -> tuple[str, ArtifactLineage]:
    video_id = entry.video_id or fetch_video_id(entry.url, runner)
    raw_dir = settings.paths.raw_dir / video_id
    input_fingerprint = fingerprint_payload(
        {
            "url": entry.url,
            "video_id": video_id,
            "source_id": source.source_id,
            "source_type": source.source_type,
        },
        size=settings.artifacts.fingerprint_size,
    )
    artifact = create_download_artifact(
        video_id=video_id,
        raw_dir=raw_dir,
        input_fingerprint=input_fingerprint,
        config_fingerprint=config_fingerprint,
        fingerprint_size=settings.artifacts.fingerprint_size,
    )
    return video_id, artifact


def is_download_materialized(raw_dir: Path) -> bool:
    return find_video_file(raw_dir) is not None and read_info_json(raw_dir) is not None


def normalize_download_artifacts(raw_dir: Path) -> None:
    _replace_if_present(raw_dir / "video.info.json", raw_dir / "info.json")
    _replace_if_present(raw_dir / "video.description", raw_dir / "description.txt")
    _replace_if_present(raw_dir / "video.description.txt", raw_dir / "description.txt")


def _replace_if_present(source: Path, destination: Path) -> None:
    if source.exists() and not destination.exists():
        source.replace(destination)


def build_video_metadata(
    *,
    video_id: str,
    url: str,
    raw_dir: Path,
    runner: CommandRunner = run_command,
) -> VideoMetadata:
    info = read_info_json(raw_dir) or {}
    video_path = find_video_file(raw_dir)
    probe = probe_video(video_path, runner=runner) if video_path else VideoProbe(None, None, None, None)
    return VideoMetadata(
        video_id=video_id,
        url=str(info.get("webpage_url") or url),
        channel=_optional_str(info.get("channel") or info.get("uploader")),
        title=_optional_str(info.get("title")),
        duration_s=probe.duration_s or _optional_float(info.get("duration")),
        fps=probe.fps,
        width=probe.width,
        height=probe.height,
        downloaded_at=utc_now(),
    )


def probe_video(video_path: Path, runner: CommandRunner = run_command) -> VideoProbe:
    stdout = _require_success(runner(build_ffprobe_args(video_path)), f"ffprobe for {video_path}")
    return parse_ffprobe_json(stdout)


def read_info_json(raw_dir: Path) -> dict[str, object] | None:
    for path in (raw_dir / "info.json", raw_dir / "video.info.json"):
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def find_video_file(raw_dir: Path) -> Path | None:
    preferred = raw_dir / "video.mp4"
    if preferred.exists():
        return preferred
    for path in sorted(raw_dir.glob("video.*")):
        if path.suffix.lower() not in {".json", ".description", ".txt", ".vtt", ".srt"}:
            return path
    return None


def _parse_fps(value: object) -> float | None:
    if not isinstance(value, str) or value in {"0/0", ""}:
        return None
    if "/" not in value:
        return _optional_float(value)
    numerator, denominator = value.split("/", 1)
    denominator_float = _optional_float(denominator)
    if denominator_float in {None, 0.0}:
        return None
    numerator_float = _optional_float(numerator)
    if numerator_float is None:
        return None
    return numerator_float / denominator_float


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None
