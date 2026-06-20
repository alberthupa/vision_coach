from __future__ import annotations

from workout_ml.settings import ensure_data_dirs, load_pipeline_settings, worker_environment


def test_load_pipeline_settings_from_yaml(tmp_path) -> None:
    config = tmp_path / "pipeline.yaml"
    config.write_text(
        """
paths:
  data_dir: custom_data
runtime:
  default_workers: 2
  worker_omp_threads: 1
artifacts:
  fingerprint_size: 12
video:
  target_fps: 2.0
""",
        encoding="utf-8",
    )

    settings = load_pipeline_settings(config)

    assert settings.paths.data_dir.as_posix() == "custom_data"
    assert settings.runtime.default_workers == 2
    assert settings.artifacts.fingerprint_size == 12
    assert settings.video.target_fps == 2.0


def test_ensure_data_dirs_creates_configured_directories(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    settings = load_pipeline_settings()

    ensure_data_dirs(settings)

    assert (tmp_path / "data/raw").is_dir()
    assert (tmp_path / "data/logs").is_dir()


def test_worker_environment_caps_nested_parallelism() -> None:
    settings = load_pipeline_settings()

    assert worker_environment(settings)["OMP_NUM_THREADS"] == "2"
    assert worker_environment(settings)["OPENBLAS_NUM_THREADS"] == "2"
