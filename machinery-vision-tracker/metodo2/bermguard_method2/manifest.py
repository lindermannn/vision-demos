"""Input-manifest validation and reproducible hashing."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(items: list[dict[str, Any]]) -> dict[str, Any]:
    records = []
    for item in items:
        path = Path(item["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append({**item, "path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {"schema_version": 1, "inputs": records}


def save_manifest(manifest: dict[str, Any], output: Path) -> None:
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def validate_manifest(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema_version") != 1:
        errors.append("unsupported_schema_version")
    for index, item in enumerate(manifest.get("inputs", [])):
        path = Path(str(item.get("path", "")))
        if not path.is_file():
            errors.append(f"input_{index}_missing")
        elif item.get("sha256") != sha256_file(path):
            errors.append(f"input_{index}_hash_mismatch")
        if item.get("kind") == "berm_mask" and not item.get("approved", False):
            errors.append(f"input_{index}_berm_mask_not_approved")
    return errors
