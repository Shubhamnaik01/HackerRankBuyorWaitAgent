from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class EvidenceCache:
    def __init__(self, root: Path | None):
        self.root = root

    @staticmethod
    def key(kind: str, evidence_id: str, model: str, content: bytes, schema_version: str = "v1") -> str:
        digest = hashlib.sha256()
        for part in (kind.encode(), evidence_id.encode(), model.encode(), schema_version.encode(), content):
            digest.update(len(part).to_bytes(8, "big"))
            digest.update(part)
        return digest.hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        if self.root is None:
            return None
        path = self.root / f"{key}.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def put(self, key: str, value: dict[str, Any]) -> None:
        if self.root is None:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{key}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
        temporary.replace(path)
