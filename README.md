# Workout ML

Local workout-recognition pipeline scaffold.

Phase 0 sets up the CPU Python environment, base configuration files, CLI entry points, artifact helpers, structured progress logging, and focused unit tests.

Phase 1 adds a DuckDB catalog, validated source registry, and resumable YouTube ingestion through `yt-dlp` plus `ffprobe`.

## Verify

```bash
make verify
```

This runs the required import check and the unit test suite through `uv`.

## Useful Commands

```bash
uv run workout-ml doctor
uv run workout-ml ingest --video "https://www.youtube.com/watch?v=..."
uv run workout-ml ingest --playlist "https://www.youtube.com/playlist?list=..." --limit 5
uv run workout-ml ingest --channel "https://www.youtube.com/@Example/videos" --limit 5
uv run workout-ml ingest --sources configs/sources.yaml
uv run workout-ml log-failure --stage download --item-id example --error "example failure"
make mlflow
```

Later-stage commands (`label`, `pose`, `dataset`, `train`, `app`) are present as CLI placeholders and intentionally fail until their implementation phases.

## Tests

The default test suite is offline and uses fake `yt-dlp`/`ffprobe` runners for ingestion behavior:

```bash
make verify
```

The Phase 1 live acceptance test is skipped unless a playlist URL is provided:

```bash
WORKOUT_ML_LIVE_PLAYLIST_URL="https://www.youtube.com/playlist?list=..." make live-ingest
```
