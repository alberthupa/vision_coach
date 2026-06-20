from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import tempfile
from typing import Any, Iterator


JsonValue = Any


def canonical_json(value: JsonValue) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def fingerprint_payload(payload: Mapping[str, JsonValue], size: int = 16) -> str:
    digest = sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return digest[:size]


def fingerprint_file(path: Path, size: int = 16) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:size]


@dataclass(frozen=True)
class ArtifactLineage:
    artifact_id: str
    stage: str
    input_fingerprint: str
    config_fingerprint: str
    code_version: str
    model_version: str | None
    path: str
    created_at: str

    @classmethod
    def create(
        cls,
        *,
        stage: str,
        input_fingerprint: str,
        config_fingerprint: str,
        code_version: str,
        path: Path,
        model_version: str | None = None,
        fingerprint_size: int = 16,
    ) -> ArtifactLineage:
        created_at = datetime.now(timezone.utc).isoformat()
        artifact_id = fingerprint_payload(
            {
                "stage": stage,
                "input_fingerprint": input_fingerprint,
                "config_fingerprint": config_fingerprint,
                "code_version": code_version,
                "model_version": model_version,
                "path": str(path),
                "created_at": created_at,
            },
            size=fingerprint_size,
        )
        return cls(
            artifact_id=artifact_id,
            stage=stage,
            input_fingerprint=input_fingerprint,
            config_fingerprint=config_fingerprint,
            code_version=code_version,
            model_version=model_version,
            path=str(path),
            created_at=created_at,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@contextmanager
def atomic_writer(path: Path, mode: str = "w", encoding: str = "utf-8") -> Iterator[Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode=mode,
        encoding=None if "b" in mode else encoding,
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        tmp_path = Path(handle.name)
        try:
            yield handle
            handle.flush()
            if hasattr(handle, "fileno"):
                import os

                os.fsync(handle.fileno())
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
    tmp_path.replace(path)


def write_json_atomic(path: Path, payload: Mapping[str, JsonValue]) -> None:
    with atomic_writer(path) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
