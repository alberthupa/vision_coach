from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from rich.console import Console


console = Console()


@dataclass(frozen=True)
class StageFailure:
    stage: str
    item_id: str
    error: str
    context: dict[str, Any]
    created_at: str

    @classmethod
    def create(
        cls,
        *,
        stage: str,
        item_id: str,
        error: str,
        context: dict[str, Any] | None = None,
    ) -> StageFailure:
        return cls(
            stage=stage,
            item_id=item_id,
            error=error,
            context=context or {},
            created_at=datetime.now(timezone.utc).isoformat(),
        )


def report_progress(stage: str, completed: int, total: int, message: str) -> None:
    console.print_json(
        data={
            "stage": stage,
            "completed": completed,
            "total": total,
            "message": message,
        },
    )


def append_failure(logs_dir: Path, failure: StageFailure) -> Path:
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = logs_dir / f"{failure.stage}_failures.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(failure), sort_keys=True))
        handle.write("\n")
    return path
