from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, PositiveInt
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PIPELINE_CONFIG = PROJECT_ROOT / "configs" / "pipeline.yaml"


class PathSettings(BaseModel):
    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    labels_dir: Path = Path("data/labels")
    poses_dir: Path = Path("data/poses")
    datasets_dir: Path = Path("data/datasets")
    models_dir: Path = Path("data/models")
    logs_dir: Path = Path("data/logs")
    catalog_path: Path = Path("data/catalog.duckdb")


class RuntimeSettings(BaseModel):
    default_workers: PositiveInt = 4
    worker_omp_threads: PositiveInt = 2
    progress_refresh_seconds: float = Field(default=1.0, gt=0)


class ArtifactSettings(BaseModel):
    fingerprint_size: PositiveInt = 16
    atomic_suffix: str = ".tmp"


class VideoSettings(BaseModel):
    target_fps: float = Field(default=1.0, gt=0)
    max_height: PositiveInt = 720
    download_retries: PositiveInt = 2


class PipelineSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="WORKOUT_ML_",
        env_nested_delimiter="__",
        extra="forbid",
    )

    paths: PathSettings = Field(default_factory=PathSettings)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    artifacts: ArtifactSettings = Field(default_factory=ArtifactSettings)
    video: VideoSettings = Field(default_factory=VideoSettings)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping at {path}")
    return loaded


def load_pipeline_settings(path: Path = DEFAULT_PIPELINE_CONFIG) -> PipelineSettings:
    return PipelineSettings(**_read_yaml(path))


def worker_environment(settings: PipelineSettings) -> dict[str, str]:
    threads = str(settings.runtime.worker_omp_threads)
    return {
        "OMP_NUM_THREADS": threads,
        "OPENBLAS_NUM_THREADS": threads,
        "MKL_NUM_THREADS": threads,
        "NUMEXPR_NUM_THREADS": threads,
    }


def ensure_data_dirs(settings: PipelineSettings) -> None:
    for directory in (
        settings.paths.data_dir,
        settings.paths.raw_dir,
        settings.paths.labels_dir,
        settings.paths.poses_dir,
        settings.paths.datasets_dir,
        settings.paths.models_dir,
        settings.paths.logs_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
