# Workout ML

Local workout-recognition pipeline scaffold.

Phase 0 sets up the CPU Python environment, base configuration files, CLI entry points, artifact helpers, structured progress logging, and focused unit tests.

## Verify

```bash
make verify
```

This runs the required import check and the unit test suite through `uv`.

## Useful Commands

```bash
uv run workout-ml doctor
uv run workout-ml log-failure --stage download --item-id example --error "example failure"
make mlflow
```

Later-stage commands (`ingest`, `label`, `pose`, `dataset`, `train`, `app`) are present as CLI placeholders and intentionally fail until their implementation phases.
