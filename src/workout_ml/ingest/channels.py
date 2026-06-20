from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl, PositiveInt
import yaml

from workout_ml.catalog import SourcePriority, SourceRecord, SourceType


SourceTypeValue = Literal["video", "playlist", "channel"]
SourcePriorityValue = Literal["general_bootstrap", "personal_routine"]


class SourceConfig(BaseModel):
    source_id: str = Field(min_length=1)
    url: HttpUrl
    source_type: SourceTypeValue
    priority: SourcePriorityValue = "general_bootstrap"
    notes: str = ""
    limit: PositiveInt | None = None

    def to_catalog_record(self) -> SourceRecord:
        return SourceRecord(
            source_id=self.source_id,
            url=str(self.url),
            source_type=self.source_type,
            priority=self.priority,
            notes=self.notes,
        )


class SourceRegistry(BaseModel):
    sources: list[SourceConfig] = Field(default_factory=list)


def load_source_registry(path: Path) -> SourceRegistry:
    if not path.exists():
        return SourceRegistry()
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in source registry: {path}")
    return SourceRegistry.model_validate(payload)


def source_from_cli(
    *,
    source_id: str,
    url: str,
    source_type: SourceType,
    priority: SourcePriority = "general_bootstrap",
    notes: str = "CLI-provided source",
    limit: int | None = None,
) -> SourceConfig:
    payload: dict[str, Any] = {
        "source_id": source_id,
        "url": url,
        "source_type": source_type,
        "priority": priority,
        "notes": notes,
    }
    if limit is not None:
        payload["limit"] = limit
    return SourceConfig.model_validate(payload)
