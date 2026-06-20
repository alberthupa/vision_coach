.PHONY: verify imports test mlflow ingest label pose dataset train app

UV_CACHE_DIR ?= .cache/uv
MPLCONFIGDIR ?= .cache/matplotlib
UV := env UV_CACHE_DIR=$(UV_CACHE_DIR) MPLCONFIGDIR=$(MPLCONFIGDIR) PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True uv

verify: imports test

imports:
	$(UV) run python -c "import mediapipe, paddleocr, faster_whisper, torch, duckdb; print('ok')"

test:
	$(UV) run pytest

mlflow:
	$(UV) run mlflow server --backend-store-uri sqlite:///data/mlflow.db --default-artifact-root data/mlruns --host 127.0.0.1

ingest:
	$(UV) run workout-ml ingest

label:
	$(UV) run workout-ml label

pose:
	$(UV) run workout-ml pose

dataset:
	$(UV) run workout-ml dataset

train:
	$(UV) run workout-ml train

app:
	$(UV) run workout-ml app
