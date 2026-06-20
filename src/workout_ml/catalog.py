from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


StageName = Literal["download", "label", "pose", "dataset", "train", "export"]
StageState = Literal["pending", "running", "complete", "failed", "skipped"]


@dataclass(frozen=True)
class StageStatus:
    video_id: str
    stage: StageName
    artifact_id: str
    status: StageState
    error: str | None = None
