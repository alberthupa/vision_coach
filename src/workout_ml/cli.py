from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from workout_ml import __version__
from workout_ml.artifacts import fingerprint_payload
from workout_ml.catalog import Catalog
from workout_ml.ingest.channels import load_source_registry, source_from_cli
from workout_ml.ingest.download import ingest_sources
from workout_ml.progress import StageFailure, append_failure, report_progress
from workout_ml.settings import ensure_data_dirs, load_pipeline_settings


app = typer.Typer(no_args_is_help=True, help="Local workout recognition pipeline.")


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", help="Show package version and exit."),
    ] = False,
) -> None:
    if version:
        typer.echo(__version__)
        raise typer.Exit()


@app.command()
def doctor(
    config: Annotated[
        Path,
        typer.Option("--config", help="Path to pipeline YAML config."),
    ] = Path("configs/pipeline.yaml"),
) -> None:
    settings = load_pipeline_settings(config)
    ensure_data_dirs(settings)
    report_progress("doctor", 1, 1, "configuration loaded")
    typer.echo(
        fingerprint_payload(
            {
                "paths": settings.paths.model_dump(mode="json"),
                "runtime": settings.runtime.model_dump(mode="json"),
                "artifacts": settings.artifacts.model_dump(mode="json"),
                "video": settings.video.model_dump(mode="json"),
            },
            size=settings.artifacts.fingerprint_size,
        )
    )


@app.command()
def log_failure(
    stage: Annotated[str, typer.Option(help="Stage name for the failure log.")],
    item_id: Annotated[str, typer.Option(help="Video, dataset, or model item id.")],
    error: Annotated[str, typer.Option(help="Failure message.")],
    config: Annotated[
        Path,
        typer.Option("--config", help="Path to pipeline YAML config."),
    ] = Path("configs/pipeline.yaml"),
) -> None:
    settings = load_pipeline_settings(config)
    path = append_failure(
        settings.paths.logs_dir,
        StageFailure.create(stage=stage, item_id=item_id, error=error),
    )
    typer.echo(path)


def _placeholder(stage: str) -> None:
    report_progress(stage, 0, 0, "not implemented until its phase")
    raise typer.Exit(code=2)


@app.command()
def ingest(
    video: Annotated[
        list[str] | None,
        typer.Option("--video", help="Single YouTube video URL to ingest."),
    ] = None,
    playlist: Annotated[
        list[str] | None,
        typer.Option("--playlist", help="YouTube playlist URL to ingest."),
    ] = None,
    channel: Annotated[
        list[str] | None,
        typer.Option("--channel", help="YouTube channel URL to ingest."),
    ] = None,
    sources: Annotated[
        Path,
        typer.Option("--sources", help="YAML source registry to ingest when provided."),
    ] = Path("configs/sources.yaml"),
    limit: Annotated[
        int | None,
        typer.Option("--limit", min=1, help="Optional per-source video limit."),
    ] = None,
    workers: Annotated[
        int | None,
        typer.Option("--workers", min=1, help="Parallel download workers."),
    ] = None,
    config: Annotated[
        Path,
        typer.Option("--config", help="Path to pipeline YAML config."),
    ] = Path("configs/pipeline.yaml"),
) -> None:
    settings = load_pipeline_settings(config)
    ensure_data_dirs(settings)
    source_configs = []
    for index, url in enumerate(video or [], start=1):
        source_configs.append(
            source_from_cli(
                source_id=f"cli_video_{index}",
                url=url,
                source_type="video",
                limit=limit,
            )
        )
    for index, url in enumerate(playlist or [], start=1):
        source_configs.append(
            source_from_cli(
                source_id=f"cli_playlist_{index}",
                url=url,
                source_type="playlist",
                limit=limit,
            )
        )
    for index, url in enumerate(channel or [], start=1):
        source_configs.append(
            source_from_cli(
                source_id=f"cli_channel_{index}",
                url=url,
                source_type="channel",
                limit=limit,
            )
        )
    if not source_configs:
        source_configs = load_source_registry(sources).sources

    with Catalog(settings.paths.catalog_path) as catalog:
        results = ingest_sources(
            sources=source_configs,
            settings=settings,
            catalog=catalog,
            workers=workers or settings.runtime.default_workers,
        )
    downloaded = sum(1 for result in results if result.status == "downloaded")
    skipped = sum(1 for result in results if result.status == "skipped")
    failed = sum(1 for result in results if result.status == "failed")
    typer.echo(f"downloaded={downloaded} skipped={skipped} failed={failed}")


@app.command()
def label() -> None:
    _placeholder("label")


@app.command()
def pose() -> None:
    _placeholder("pose")


@app.command()
def dataset() -> None:
    _placeholder("dataset")


@app.command()
def train() -> None:
    _placeholder("train")


@app.command("app")
def app_command() -> None:
    _placeholder("app")
