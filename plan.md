# Workout Recognition Pipeline — Implementation Plan

**Target machine:** GMKtec EVO-X2 — AMD Ryzen AI Max+ 395 (Strix Halo), 16C/32T Zen 5, Radeon 8060S iGPU (RDNA 3.5, `gfx1151`), ~100 GB unified LPDDR5X RAM, Ubuntu 24.04, fast NVMe.

**Goal:** Build a fully local pipeline that (1) downloads YouTube follow-along workout videos, (2) auto-labels exercise segments from description chapters + on-screen OCR + ASR, (3) extracts MediaPipe pose features, (4) trains an exercise classifier, and (5) runs a real-time webcam app that reports: current exercise, start/stop, duration, and rep count.

**Design principles for this hardware:**
- Everything local: Parquet + DuckDB instead of Delta/Databricks, local MLflow with file backend.
- CPU-first. The models are tiny (≤2M params); 16 Zen 5 cores train them in minutes. The 100 GB unified RAM means whole datasets fit in memory — exploit that, don't over-engineer streaming.
- GPU (ROCm on gfx1151) is an *optional* accelerator behind a flag, never a dependency. Accelerator selection and fallback must always be logged.
- Parallelism across videos, not within a video. Benchmark 1, 2, 4, 8, and 12 workers on the target machine instead of assuming the largest pool is fastest.
- Every stage is an idempotent, resumable CLI command. Outputs carry an artifact fingerprint derived from inputs, configuration, model versions, and code version, so stale outputs are not mistaken for completed work.

### Bootstrap strategy

The first training run should be as hands-free as possible:

- Start with a broad vocabulary of roughly 25 exercise families rather than requiring a manually curated 6–10 exercise dataset.
- Seed the corpus with both suitable public follow-along channels and the specific workout videos the user follows personally. Mark those sources as `priority: personal_routine` so reports show whether the model covers the exercises that matter most.
- Use chapters, OCR, ASR, confidence thresholds, and automatic quality gates to produce the first dataset without manually labeling every segment.
- Manual work is limited to a statistically useful audit sample and correction of systematic errors. It is not a prerequisite for creating the first model.
- Treat the first model as a bootstrap model. Its purpose is to expose taxonomy, labeling, pose, and domain-gap failures before scaling the corpus.

### Success levels

1. **Pipeline success:** ingestion through training runs unattended, resumes after interruption, and reports failures.
2. **Weak-label success:** audited segment precision is at least 90% for accepted high-confidence labels, with results broken down by class and source.
3. **Model success:** held-out-channel macro-F1 is at least 0.85 after corpus scaling; priority personal-routine exercises are reported separately.
4. **Product success:** a personal follow-along session produces sensible exercise boundaries, durations, and rep counts without manual interaction.

---

## 0. Agent operating instructions

These rules apply to every phase below.

1. Work phase by phase, in order. Do not start phase N+1 until phase N's **acceptance criteria** pass.
2. Each phase ends with a runnable verification command. Run it and show output before moving on.
3. Prefer boring, debuggable code: `typer` CLIs, plain functions, dataclasses/pydantic models, type hints everywhere.
4. All configuration lives in `configs/*.yaml`, loaded via pydantic-settings. No magic constants in code.
5. All long-running stages must: write progress to stdout, be killable and resumable, log per-item failures to `data/logs/<stage>_failures.jsonl` and continue (never crash the whole batch on one bad video).
6. Write unit tests for pure logic (label normalization, segment merging, windowing, rep counting on synthetic signals). Skip tests for thin I/O wrappers.
7. Commit at the end of each phase with message `phase-N: <summary>`.
8. If a library fails to install or a model fails to load, do not silently substitute a different approach — report it and propose options.

---

## 1. Repository layout

```
workout-ml/
├── configs/
│   ├── pipeline.yaml          # paths, parallelism, fps, thresholds
│   ├── vocabulary.yaml        # canonical exercise vocabulary + aliases
│   ├── sources.yaml           # channels/playlists/videos + priority metadata
│   └── train.yaml             # model + training hyperparameters
├── data/                      # gitignored
│   ├── catalog.duckdb         # central metadata DB
│   ├── raw/<video_id>/        # video.mp4, info.json, description.txt, subs.vtt
│   ├── labels/<video_id>/<artifact_id>.json # weak label segments
│   ├── poses/<video_id>/<artifact_id>.parquet
│   ├── datasets/<dataset_version>/   # train/val/test parquet + manifest.json
│   ├── models/                # exported .pt / .onnx + metadata
│   └── logs/
├── src/workout_ml/
│   ├── catalog.py             # DuckDB helpers, stage-status tracking
│   ├── ingest/
│   │   ├── download.py        # yt-dlp wrapper
│   │   └── channels.py        # curated channel/playlist lists
│   ├── labeling/
│   │   ├── chapters.py        # description/chapter timestamp parser
│   │   ├── ocr.py             # frame sampling + PaddleOCR + stable-region detection
│   │   ├── asr.py             # faster-whisper transcript segments
│   │   ├── normalize.py       # vocabulary matching (rapidfuzz + embeddings)
│   │   └── fuse.py            # signal fusion -> final segments + confidence
│   ├── pose/
│   │   ├── extract.py         # MediaPipe Pose Landmarker (video mode)
│   │   ├── normalize.py       # hip-centering, scaling, smoothing
│   │   └── features.py        # joint angles + velocities
│   ├── dataset/
│   │   └── windows.py         # sliding windows, splits, class balancing
│   ├── models/
│   │   ├── classifier.py      # baseline temporal classifier (PyTorch Lightning)
│   │   ├── train.py           # training entrypoint, MLflow logging
│   │   └── export.py          # TorchScript + ONNX export
│   ├── reps/
│   │   └── counter.py         # exercise-specific rep state machines
│   ├── realtime/
│   │   ├── app.py             # webcam loop, overlay UI
│   │   ├── state_machine.py   # hysteresis exercise state machine
│   │   └── session_log.py     # sets/reps/durations -> jsonl + summary
│   └── cli.py                 # typer app: workout-ml <stage> ...
├── tests/
├── notebooks/                 # exploration only, never imported by src
├── Makefile                   # make ingest / label / pose / dataset / train / app
└── pyproject.toml             # uv-managed
```

---

## 2. Phase 0 — Environment setup

### Epic A — Foundation and reproducibility

- [x] Scaffold the repository, configuration models, CLI, tests, and Makefile.
- [x] Create CPU environment and verify all required imports and system tools.
- [x] Implement artifact fingerprints, atomic writes, and stage lineage metadata.
- [x] Add structured progress reporting and per-item failure logs.

Use `uv` for environment management. Keep the supported CPU environment on Python 3.11 for broad MediaPipe and OCR wheel coverage. The optional ROCm environment is separate and follows AMD's supported Python/ROCm matrix.

```bash
curl -LsSf https://astral.sh/uv | sh
uv init --python 3.11
uv add yt-dlp mediapipe opencv-python numpy scipy pandas pyarrow duckdb \
       paddleocr paddlepaddle faster-whisper rapidfuzz sentence-transformers \
       torch torchvision pytorch-lightning mlflow typer pydantic pydantic-settings \
       rich pytest
sudo apt-get install -y ffmpeg libgl1
```

Notes:
- `paddlepaddle` CPU build is sufficient. If PaddleOCR proves troublesome on this stack, the approved fallback is `easyocr` (slower but simpler); note it and switch.
- `torch` default CPU build first. ROCm is Phase 8 (optional acceleration), not a prerequisite.
- Set sensible thread caps for nested parallelism: workers use `OMP_NUM_THREADS=2` when running under the process pool.
- Use `opencv-python`, not `opencv-python-headless`, because the real-time phase requires a local OpenCV window.

**MLflow:** run `mlflow server --backend-store-uri sqlite:///data/mlflow.db --default-artifact-root data/mlruns --host 127.0.0.1` (add a `make mlflow` target).

**Acceptance criteria:**
- `uv run python -c "import mediapipe, paddleocr, faster_whisper, torch, duckdb; print('ok')"` prints `ok`.
- `ffmpeg -version` works.
- `pytest` runs (zero tests is fine at this point).

### Phase 0 implementation notes — 2026-06-20

Completed in this repository:

- Initialized a `uv` package named `workout-ml` on Python 3.11 (`.python-version`, `pyproject.toml`, `uv.lock`, `.venv/`).
- Added the Phase 0 CPU dependency set: `yt-dlp`, MediaPipe, OpenCV, NumPy/SciPy/Pandas/PyArrow, DuckDB, PaddleOCR/PaddlePaddle, Faster Whisper, RapidFuzz, Sentence Transformers, PyTorch Lightning, MLflow, Typer, Pydantic, pydantic-settings, Rich, and pytest.
- Adjusted PyTorch to a CPU-only lock: `torch==2.5.1+cpu` and `torchvision==0.20.1+cpu` from the PyTorch CPU wheel index. The first unconstrained resolve selected current CUDA-related dependencies, which contradicts the CPU-first Phase 0 requirement.
- Added `pyyaml` as a direct dependency because the project configuration contract is YAML.
- Added base config files under `configs/`: `pipeline.yaml`, `train.yaml`, `vocabulary.yaml`, and `sources.yaml`.
- Added planned package boundaries under `src/workout_ml/`: `ingest`, `labeling`, `pose`, `dataset`, `models`, `reps`, and `realtime`.
- Added `src/workout_ml/settings.py` with pydantic-settings-backed pipeline config models, data-directory creation, and worker thread-cap environment generation (`OMP_NUM_THREADS=2`, `OPENBLAS_NUM_THREADS=2`, `MKL_NUM_THREADS=2`, `NUMEXPR_NUM_THREADS=2` by default).
- Added `src/workout_ml/artifacts.py` with stable canonical JSON fingerprints, file fingerprints, `ArtifactLineage`, atomic writes, and atomic JSON writes.
- Added `src/workout_ml/progress.py` with JSON progress output and per-stage JSONL failure logs at `data/logs/<stage>_failures.jsonl`.
- Added `src/workout_ml/catalog.py` with initial typed stage-status metadata. DuckDB table creation and catalog persistence remain Phase 1 work.
- Added a Typer CLI at `workout-ml` with `doctor`, `log-failure`, and placeholder commands for later stages (`ingest`, `label`, `pose`, `dataset`, `train`, `app`). The placeholders intentionally exit non-zero until their phases are implemented.
- Added a `Makefile` with `verify`, `imports`, `test`, `mlflow`, and stage targets. `make mlflow` uses `sqlite:///data/mlflow.db` and `data/mlruns` as planned.
- Added a short `README.md` with Phase 0 verification and basic CLI commands.
- Added `.gitignore` for `.venv`, Python caches, test caches, build metadata, and `data/`.

Environment notes:

- `uv` was already installed (`uv 0.5.9`).
- `ffmpeg` was already installed (`ffmpeg 6.1.1-3ubuntu5`), so no `apt-get` step was required.
- The sandboxed `uv` cache under `~/.cache` was read-only, so commands used `UV_CACHE_DIR=/tmp/uv-cache`.
- The large PyTorch CPU wheel needed network access and a longer timeout; the successful install used `UV_HTTP_TIMEOUT=180`.
- `make verify` sets `MPLCONFIGDIR=/tmp/workout-ml-matplotlib` to avoid Matplotlib cache warnings in the restricted home directory.
- `make verify` sets `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True` to skip PaddleOCR's import-time model host connectivity check. A direct import without this variable still imports successfully and prints `ok`, but Paddle reports that no remote model hoster is reachable. Phase 2 should explicitly handle OCR model availability/local caching before running OCR extraction.

Verification output:

```text
$ env UV_CACHE_DIR=/tmp/uv-cache uv run python -c "import mediapipe, paddleocr, faster_whisper, torch, duckdb; print('ok')"
Checking connectivity to the model hosters, this may take a while. To bypass this check, set `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK` to `True`.
No model hoster is available! Please check your network connection to one of the following model hoster: HuggingFace (https://huggingface.co), ModelScope (https://modelscope.cn), AIStudio (https://aistudio.baidu.com), or BOS (https://paddle-model-ecology.bj.bcebos.com). Otherwise, only local models can be used.
ok

$ ffmpeg -version
ffmpeg version 6.1.1-3ubuntu5

$ env UV_CACHE_DIR=/tmp/uv-cache uv run pytest
collected 7 items
7 passed in 0.10s

$ make verify
Connectivity check to the model hoster has been skipped because `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK` is enabled.
ok
collected 7 items
7 passed in 0.11s
```

Testing decision:

- Tests are appropriate already because Phase 0 now contains pure logic, not only packaging. Added focused unit tests for deterministic fingerprints, atomic JSON replacement, artifact lineage metadata, failure JSONL logging, config loading, data-directory creation, and nested-parallelism thread caps.
- No tests were added for thin wrappers such as CLI placeholder commands or `Makefile` targets. Those are covered by direct verification commands at this stage.

---

## 3. Phase 1 — Catalog + ingestion

### Epic B — Automated corpus ingestion

- [ ] Implement single-video, playlist, and channel ingestion.
- [ ] Add a configurable source registry, including `priority: personal_routine` videos.
- [ ] Store probed video metadata and stage state in the catalog.
- [ ] Prove ingestion is resumable and an unchanged rerun is a no-op.

### catalog.py
DuckDB database `data/catalog.duckdb` with tables:
- `videos(video_id PK, url, channel, title, duration_s, fps, width, height, downloaded_at)`
- `sources(source_id PK, url, source_type, priority, notes)` — includes channels, playlists, and individual videos. Videos the user actually follows use `priority=personal_routine`.
- `stage_status(video_id, stage, artifact_id, status, updated_at, error)` — stages: `download`, `label`, `pose`. A stage is complete only for the current artifact fingerprint.
- `artifacts(artifact_id PK, stage, input_fingerprint, config_fingerprint, code_version, model_version, path, created_at)` — lineage for per-video outputs, datasets, and models.
- Worker processes never write to DuckDB directly. Workers return results to the parent process, which owns the single catalog writer.

### ingest/download.py
- `workout-ml ingest --playlist <url>` and `--channel <url> --limit N` and `--video <url>`.
- yt-dlp options: `-f "bv*[height<=720]+ba/b[height<=720]"`, write `info.json`, description, and auto-subs (`--write-auto-subs --sub-langs en`). Output to `data/raw/<video_id>/`.
- Rate-limit politely (`--sleep-interval 3 --max-sleep-interval 8`), retry each video up to 2 times, then record failure and continue.
- After download, probe with ffprobe and fill the `videos` row.

### configs/sources.yaml + ingest/channels.py
Seed list of follow-along channels/playlists and personally followed videos. Selection criteria documented in the module docstring: single trainer, full body visible, exercise names on screen or timestamped descriptions, minimal jump cuts. Reports must distinguish personal-routine sources from general bootstrap sources.

**Acceptance criteria:**
- Ingest 5 test videos from one playlist; `data/raw/` contains video + info.json + description for each; catalog rows exist; re-running the same command downloads nothing (idempotency proven by log output).

---

## 4. Phase 2 — Weak auto-labeling

### Epic C — Hands-free weak labeling

- [ ] Define the exercise-family vocabulary, aliases, metadata, and versioning.
- [ ] Implement chapter and description timestamp extraction.
- [ ] Implement OCR overlay discovery and text extraction.
- [ ] Implement lazy ASR fallback and source fusion.
- [ ] Build confidence, coverage, boundary, and class-distribution reports.
- [ ] Audit a stratified sample and reach the weak-label quality target.

Output per video: `data/labels/<video_id>/<artifact_id>.json`:

```json
{
  "video_id": "...",
  "segments": [
    {"t_start": 45.0, "t_end": 75.0, "label": "squat", "raw_text": "SQUATS",
     "source": "chapters|ocr|asr|fused", "confidence": 0.92}
  ],
  "vocabulary_version": "v1",
  "coverage": 0.87
}
```

### 4.1 vocabulary.yaml
Canonical vocabulary, start with ~25 exercises + 3 meta-classes. Structure:

```yaml
version: v1
meta_classes: [rest, transition, unknown]
exercises:
  squat:
    aliases: [squats, air squat, bodyweight squat]
    variants: [sumo_squat, goblet_squat]
    equipment: [none, dumbbell, kettlebell]
    rep_counter: knee_hip_cycle
  push_up:
    aliases: [pushup, push-ups, press up, incline push up]
    variants: [incline_push_up, knee_push_up]
    equipment: [none]
    rep_counter: elbow_cycle
  lunge:
    aliases: [lunges, reverse lunge, forward lunge, alternating lunges]
    variants: [reverse_lunge, forward_lunge]
    alternating: true
  # ... burpee, plank, mountain_climber, jumping_jack, glute_bridge, crunch,
  # bicycle_crunch, russian_twist, high_knees, deadlift, shoulder_press,
  # bicep_curl, bent_over_row, tricep_dip, side_plank, leg_raise, superman,
  # bird_dog, wall_sit, calf_raise, hip_thrust, butt_kicks
```

The classifier target remains the broad `exercise_family` for the bootstrap model. Preserve variant, side, equipment, `alternating`, and `timed_only` metadata when the source provides it so the taxonomy can become more specific later without relabeling from scratch.

### 4.2 labeling/chapters.py — cheapest signal first
- Parse YouTube chapters from `info.json` (`chapters` key) and, as fallback, regex timestamps from the description (`^\s*(\d{1,2}:)?\d{1,2}:\d{2}\s+(.+)$`).
- Each chapter title goes through `normalize.py`. Chapters that normalize to a known exercise become segments with `confidence=0.95`.

### 4.3 labeling/ocr.py
- Sample frames at 1 fps with OpenCV.
- Pass 1 (stable-region detection): run PaddleOCR detection on every 10th sampled frame across the video; cluster text boxes by position (IoU grid voting); regions present in >30% of frames are "overlay regions" (exercise name, timer). Hard-exclude nothing yet — timers are useful (`REST`, `0:30`).
- Pass 2: run OCR (det+rec) only on the union of overlay-region crops for *all* sampled frames. Crop OCR at 1 fps on 16 cores is fast.
- Emit raw per-second OCR tokens: `(t, text, region_id, ocr_conf)`.

### 4.4 labeling/asr.py
- `faster-whisper` `small` model, CPU, int8. Transcribe to word-level segments. Used only as a fusion tiebreaker — keep it lazy (run only when chapters are absent AND OCR coverage < 0.5, configurable).

### 4.5 labeling/normalize.py
- `normalize(text) -> (canonical_label | None, score)`.
- Pipeline: lowercase/strip punctuation → exact alias hit → `rapidfuzz.fuzz.WRatio` against all aliases (accept ≥ 88) → embedding similarity via `sentence-transformers/all-MiniLM-L6-v2` against alias centroids (accept ≥ 0.75) → else `None`.
- Special tokens: `rest|break|water` → `rest`; pure timer patterns → no label change (timers coexist with exercise names).
- Unit-test heavily: misspellings, ALL CAPS, "PUSH UPS — 40 SEC", Polish trainer overlays if those channels are included.

### 4.6 labeling/fuse.py
- Per second of video, collect candidate labels from each source.
- Run-length encode OCR labels into runs; merge adjacent runs of the same label with gaps < 3 s; drop runs < 5 s.
- Fusion priority: chapters > OCR runs > ASR. When chapters exist, use OCR only to refine boundaries (±5 s search for the OCR transition nearest the chapter timestamp).
- Gaps between exercise segments: label `rest` if a rest cue was seen, else `transition`.
- `coverage` = fraction of video duration assigned a non-`unknown` label. Videos with coverage < 0.5 get `stage_status=needs_review`.
- CLI: `workout-ml label --all` / `--video <id>`; `workout-ml label-report` prints a per-video coverage/segment table (rich).

### 4.7 Manual audit hook
- `workout-ml audit --sample 50` exports stratified 8–12 second clips named `<label>__<video_id>__<t>.mp4` into `data/audit/`. Sampling covers class, channel, source, confidence band, segment interior, and segment boundaries.
- A simple `audit_results.csv` records label correctness and approximate boundary error. Report precision by class and labeling source, not only one aggregate number.
- Target: ≥ 90% precision for accepted high-confidence segments before scaling. The first bootstrap model may still be trained earlier as a diagnostic run, with its weak-label status clearly recorded.

**Acceptance criteria:**
- On the 5 test videos: label JSONs exist, mean coverage ≥ 0.7, `label-report` renders, unit tests for `normalize.py` and run-length merging pass.

---

## 5. Phase 3 — Pose extraction & features

### Epic D — Pose and causal features

- [ ] Extract MediaPipe image and world landmarks with timestamps and quality flags.
- [ ] Implement one shared stateful causal feature pipeline for offline replay and live use.
- [ ] Add pose visualization, dropped-frame replay, and offline/live parity tests.
- [ ] Benchmark worker counts and select the default from measured throughput.

### pose/extract.py
- MediaPipe **Pose Landmarker** (Tasks API), `pose_landmarker_heavy.task`, `running_mode=VIDEO` (gives temporal smoothing for free).
- Decode with OpenCV at native fps; process every frame.
- Multi-person handling: Pose Landmarker tracks one pose; if detection confidence drops below 0.5 for > 1 s, mark frames invalid.
- Parallelize across videos with a process pool. Benchmark 1, 2, 4, 8, and 12 workers with thread caps; store the measured default in configuration.
- Output Parquet per video: columns `frame_idx, t, valid, conf`, `wl_{i}_{x,y,z}` for 33 world landmarks, `il_{i}_{x,y}` image landmarks, `vis_{i}`.

### pose/normalize.py
- Operate on **world landmarks** (already hip-centered, metric) — verify and re-center on mid-hip anyway.
- Implement normalization as a stateful causal transformer shared by batch and live inference.
- Scale by a robust running torso-length estimate rather than a whole-video median.
- Rotate around the vertical axis so the shoulder line's horizontal projection aligns frame-to-frame is *not* done globally (it destroys exercises with rotation); instead store the raw normalized pose and let augmentation handle rotation invariance.
- Gap handling: causal interpolation/hold for short invalid runs, with explicit validity and missing-landmark masks; longer runs stay invalid.
- Smoothing: causal EMA or One Euro filter. Do not use centered Savitzky–Golay smoothing because it depends on future frames unavailable in live mode.

### pose/features.py
- Joint angles (degrees): elbows L/R, shoulders L/R, hips L/R, knees L/R, trunk inclination vs vertical, hip–shoulder torsion. Calculate derivatives using real timestamp deltas.
- Include explicit validity/missing-landmark features; define the final dimensionality from the feature schema rather than hard-coding 119 throughout the system.
- Append feature columns to the same Parquet; bump a `features_version` field in the file metadata.
- Add an offline live-replay mode that feeds frames with realistic timing and dropped-frame patterns through the exact same stateful transformer used by the webcam app.

CLI: `workout-ml pose --all`, resumable via `stage_status`.

**Acceptance criteria:**
- Pose Parquet exists for the 5 test videos; a debug command `workout-ml pose-viz --video <id> --t 60` writes an annotated frame PNG proving landmarks land on the body; invalid-frame fraction reported and < 10% on test videos; throughput logged (frames/s).
- Offline batch mode and simulated live replay produce equivalent features within defined tolerance for the same accepted frames.

---

## 6. Phase 4 — Dataset construction

### Epic E — Versioned dataset

- [ ] Join labels and poses into fixed-duration windows.
- [ ] Add `other_activity`, `no_pose`, rest, and transition examples.
- [ ] Create leakage-safe, persisted train/validation/test splits.
- [ ] Write immutable dataset manifests with source and artifact lineage.

### dataset/windows.py
- Join pose Parquet with label segments (DuckDB does this nicely).
- Windows: 75 frames target at a canonical 25 fps — first resample each video's feature stream to 25 fps so all windows are uniform (3.0 s). Stride 12 frames (~0.5 s).
- Window label = segment covering ≥ 80% of the window; windows straddling boundaries → `transition`. Windows with > 20% invalid frames are dropped.
- Add hard-negative windows for `other_activity`: standing, walking, adjusting the camera, entering/leaving frame, unsupported movement, and setup between exercises. `no_pose` is handled primarily as a runtime quality gate, but examples are retained for testing.
- **Splits: by channel** when ≥ 4 channels exist, else by video. Persist split assignment in `manifest.json` (with vocabulary version, feature version, per-class counts, source video lists). Never resplit silently — a new split is a new dataset version.
- Class balancing: compute inverse-frequency sampling weights, store in the train Parquet (used by `WeightedRandomSampler`); cap `rest`/`transition` at 2× the median exercise class count.
- Output: `data/datasets/v001/{train,val,test}.parquet` + `manifest.json`. With ~100 GB RAM, loading the full dataset as numpy in memory is the *intended* design. At roughly 120–140 float32 features, 75 frames, and a 0.5-second stride, 100 hours of video is approximately 26–30 GB before dataframe/loader overhead; measure actual peak RAM.

CLI: `workout-ml dataset build --version v001`.

**Acceptance criteria:**
- Dataset builds from test videos; manifest shows per-class counts; assertion tests prove no video_id appears in two splits; loading train.parquet into memory works and reports RAM usage.

---

## 7. Phase 5 — Exercise classifier

### Epic F — Classifier

- [ ] Train and evaluate one baseline temporal classifier.
- [ ] Apply landmark augmentation before derived-feature calculation.
- [ ] Report per-class, held-out-channel, and personal-routine performance.
- [ ] Export and validate the selected model with ONNX Runtime.
- [ ] Benchmark an alternative architecture only if the baseline exposes a clear need.

### models/classifier.py
- Architecture v1 (default): input `(B, 75, n_features)` → LayerNorm → 2-layer bidirectional GRU (hidden 128) → mean+max pool over time → MLP head → `n_classes` logits. ~1.5M params.
- Implement and validate this baseline first. Add a TCN alternative only after the baseline produces evidence that architecture comparison is useful.
- Loss: cross-entropy with label smoothing 0.1; per-sample weights = segment confidence from labeling.

### Augmentation (in the Dataset class, on-the-fly, CPU)
- Apply augmentation to raw normalized landmarks first, then recompute angles and velocities so derived features remain physically consistent.
- Horizontal flip with correct left/right landmark index swap (build the swap index map once, test it).
- Random 3D rotation of world coords: yaw ±30°, pitch ±10°, roll ±5° — this is the main camera-domain-gap weapon.
- Keypoint dropout (p=0.05 per landmark per frame, zero + explicit mask), Gaussian jitter (σ=0.01), temporal speed scale 0.8–1.2× with resampling, random crop-pad of window start.

### models/train.py
- PyTorch Lightning; CPU trainer default (`accelerator=auto` picks up ROCm later if present); batch 256; AdamW 3e-4; cosine schedule; early stopping on val macro-F1; ~30 epochs.
- MLflow: log params, per-class F1, confusion matrix image, dataset version, git sha. Register best checkpoint.
- Expected wall-clock on 16 Zen 5 cores: minutes-to-tens-of-minutes per run. If an epoch exceeds ~10 min, profile the dataloader before touching the model.

### models/export.py
- Export best model to TorchScript and ONNX. Validate ONNX output ≡ PyTorch (atol 1e-4). The real-time app loads ONNX via `onnxruntime` (CPU EP) — add `onnxruntime` dependency here.

CLI: `workout-ml train --config configs/train.yaml`, `workout-ml export --run <mlflow_run_id>`.

**Acceptance criteria:**
- Training completes; val macro-F1 reported (with only 5 test videos this is a smoke test — the real bar comes after scaling ingestion: macro-F1 ≥ 0.85 on a held-out channel); MLflow UI shows the run; exported ONNX passes the equivalence check; single-window ONNX inference latency printed.
- Reports include per-class results, hard-negative rejection, held-out-channel results, and a separate slice for `priority: personal_routine` sources.

---

## 8. Phase 6 — Rep counter

### Epic G — Rep counting

- [ ] Implement exercise-specific signal selection and hysteresis counters for common exercises.
- [ ] Add explicit handling for alternating exercises and timed holds.
- [ ] Validate counters with synthetic signals and manually counted clips.
- [ ] Investigate a generic PCA or learned counter only after the deterministic baseline.

### reps/counter.py
- Input: feature stream of one exercise segment (or live ring buffer) at 25 fps.
- V1 uses deterministic, exercise-specific signal selection and hysteresis state machines. Examples: knee/hip angles for squats, elbow angle for curls and push-ups, wrist/ankle separation for jumping jacks.
- Per-exercise configuration defines signal, direction, thresholds, minimum range of motion, and min/max rep period.
- Alternating exercises use explicit left/right states and pairing rules rather than dividing a generic peak count by two.
- Static holds (plank, wall sit): listed in config as `timed_only`; rep counter disabled, duration reported instead.
- A generic rolling-PCA or learned periodicity counter is a later experiment. Rolling PCA is not the default because component direction and sign can change during a live segment.
- Tests: synthetic angle cycles + noise at known frequencies → exact counts; partial repetitions and flat signals → zero; threshold-crossing jitter does not double-count.

**Validation:** `workout-ml reps-eval --video <id>` plots the selected exercise signal, hysteresis states, and counted events per segment; user manually counts reps in 10 stratified clips and compares. Target: ±1 rep on ≥ 80% of segments.

**Acceptance criteria:** synthetic-signal tests pass; eval plots generated for test videos.

---

## 9. Phase 7 — Real-time webcam app

### Epic H — Real-time product

- [ ] Build webcam capture, pose-quality gating, timestamp resampling, and ring buffering.
- [ ] Add confidence/entropy rejection and exercise state hysteresis.
- [ ] Add overlay UI and session logging.
- [ ] Run an early personal-camera smoke test before corpus scaling.
- [ ] Complete an end-to-end personal workout acceptance test.

### realtime/app.py
- OpenCV capture (1280×720 @ 30 fps) → MediaPipe Pose Landmarker `running_mode=LIVE_STREAM` → the same stateful causal normalization and feature transformer used by offline live replay → ring buffer (last 3 s at 25 fps via timestamp-based resampling).
- MediaPipe live mode may drop newly submitted frames while inference is busy. Use result timestamps, process the latest available result, and test with equivalent dropped-frame patterns offline.
- Clear or invalidate the ring buffer after prolonged pose loss. Do not infer an exercise from stale frames.
- Every 0.5 s: run ONNX classifier on the buffer → push prediction into the state machine.

### realtime/state_machine.py
- States: `IDLE`, `ACTIVE(exercise)`, `RESTING`.
- Enter `ACTIVE(e)` after 3 consecutive windows predicting `e` with mean prob ≥ 0.6; exit after 4 consecutive windows disagreeing. Switching exercises passes through a 1-window grace transition.
- Before state transitions, reject predictions when pose quality is insufficient, `other_activity` wins, maximum probability is too low, or prediction entropy is too high. Thresholds are calibrated from validation data rather than fixed permanently at initial guesses.
- On `ACTIVE` entry: start segment timer, reset live rep counter. On exit: finalize set record `(exercise, t_start, t_end, reps, duration)`.

### UI + session log
- OpenCV window overlay: skeleton, current exercise, live rep count, set timer, last-3-sets summary strip. Keys: `q` quit, `r` reset session, `s` save snapshot.
- `realtime/session_log.py`: appends finalized sets to `data/sessions/<date>.jsonl`; `workout-ml session-summary` prints a per-exercise table.

### Calibration loop (important)
- `workout-ml record --minutes 10` records webcam video locally. The user performs known exercises announcing them; the same labeling path (ASR-only) or a quick manual JSON labels it. These videos enter the dataset as a `self` channel pinned to the train split, and the classifier is fine-tuned for 5 epochs at LR 1e-4. This 10-minute step closes most of the YT→webcam domain gap.
- Before scaling to 150–250 videos, run a short personal-camera smoke test covering standing, floor exercises, equipment occlusion, entering/leaving frame, and ordinary setup movement. This validates the camera and feature pipeline; it does not block the hands-free bootstrap training run.

**Acceptance criteria:**
- App sustains ≥ 20 fps end-to-end on CPU (log fps); doing 10 squats in front of the camera yields one `squat` set with 10 ± 1 reps in the session log.

---

## 10. Phase 8 (optional) — ROCm acceleration on gfx1151

### Epic I — Optional acceleration

- [ ] Create and verify the separate ROCm environment for `gfx1151`.
- [ ] Benchmark CPU versus GPU training before selecting an accelerator.

Only after everything works on CPU:
- Use a separate environment matching AMD's current supported matrix. As of June 2026, AMD lists Ryzen AI Max+ 395 (`gfx1151`) with ROCm 7.2.1, PyTorch 2.9.1, Python 3.12, Ubuntu 24.04.4, and the required OEM kernel.
- Prefer AMD's supported wheels and installation instructions. Do not use `HSA_OVERRIDE_GFX_VERSION` as a normal setup path.
- Gate via `train.yaml: accelerator: auto|cpu|gpu`. Benchmark: training epoch time CPU vs GPU; keep whichever wins (for a 1.5M-param GRU, CPU may genuinely win).
- Do **not** move MediaPipe or OCR to GPU — not worth the integration cost here.

---

## 11. Phase 9 — Scaling the corpus

### Epic J — Corpus scaling

- [ ] Scale the corpus only after the end-to-end vertical slice works.
- [ ] Re-audit labels and retrain on the expanded corpus.

Only start this phase after the pipeline works end to end:

1. Expand `channels.py` to ~10 channels, target 150–250 videos (≈ 80–120 h).
2. Run `make ingest label pose dataset` overnight (disk estimate: ~150 GB raw video, ~10 GB poses — confirm NVMe headroom; raw video can be deleted after pose extraction, keep a config flag `delete_video_after_pose: true`).
3. Run the audit (Phase 2.7) on 200 segments; fix the top normalization misses.
4. Retrain; evaluate on a fully held-out channel; iterate on augmentation if the held-out-channel gap exceeds ~5 F1 points.
5. Record 2–3 personal sessions; fine-tune; re-test live.

**Acceptance criteria:**
- Expanded dataset manifest records 150–250 source videos and complete artifact lineage.
- The 200-segment stratified audit reaches the weak-label precision target.
- Retraining reports held-out-channel, per-class, and personal-routine results; the live personal workout test is repeated.

---

## 12. Risk register

| Risk | Mitigation |
|---|---|
| OCR misses stylized overlay fonts | PaddleOCR first; easyocr fallback; chapters cover many videos anyway |
| Label noise poisons training | confidence-weighted loss, label smoothing, audit loop, coverage gating |
| YT camera angles ≠ webcam | world landmarks, 3D rotation augmentation, self-recorded fine-tune set |
| Offline/live feature skew | one causal stateful transformer, timestamp-aware derivatives, live-replay parity tests |
| Ordinary movement classified as exercise | `other_activity` negatives, pose-quality gate, confidence/entropy rejection |
| Alternating exercises double-count reps | per-exercise period priors + pair-counting config list |
| MediaPipe loses tracking on floor exercises | mark invalid frames, interpolate short gaps, include floor exercises in audit |
| ROCm instability on gfx1151 | GPU strictly optional; CPU is the supported path |
| yt-dlp throttling/blocks | sleep intervals, small batches, resumable catalog |
| Stale artifacts reused after config/code changes | artifact fingerprints, lineage metadata, atomic output writes |

## 13. Milestone summary

| # | Deliverable | Verify with |
|---|---|---|
| 0 | Env + repo skeleton | import check |
| 1 | 5 videos ingested, catalog | `workout-ml ingest`, rerun no-op |
| 2 | Weak labels, coverage ≥ 0.7 | `workout-ml label-report` |
| 3 | Pose Parquet + viz | `workout-ml pose-viz` |
| 4 | Dataset v001 + manifest | split-leak tests |
| 5 | Trained + exported classifier | MLflow run, ONNX check |
| 6 | Rep counter validated | synthetic tests + eval plots |
| 7 | Live app, real squat set logged | session jsonl |
| 8 | (opt) ROCm benchmark | epoch-time comparison |
| 9 | Expanded corpus and retrained model | audit report + held-out evaluation |

## 14. Technical references

- [MediaPipe Pose Landmarker for Python](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python)
- [DuckDB concurrency](https://duckdb.org/docs/current/connect/concurrency)
- [OpenCV Python package variants](https://pypi.org/project/opencv-python/)
- [AMD ROCm support matrix for Ryzen APUs on Linux](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityryz/native_linux/native_linux_compatibility.html)
