"""Single-use ledger for the reserved post-freeze blind evaluation."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path


class BlindRunLedger:
    """Persist a freeze contract and atomically claim one blind run."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def freeze(self, code_sha256: str, config_sha256: str) -> None:
        if self.path.exists():
            raise RuntimeError("blind ledger already exists")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "code_sha256": code_sha256,
            "config_sha256": config_sha256,
            "blind_run_claimed": False,
        }
        self._write_exclusive(payload)

    def claim(self, source_sha256: str, code_sha256: str, config_sha256: str) -> None:
        if not self.path.is_file():
            raise RuntimeError("blind run requires a freeze ledger")
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("blind_run_claimed") is True:
            raise RuntimeError("blind run was already claimed")
        if payload.get("code_sha256") != code_sha256:
            raise RuntimeError("code changed after the blind freeze")
        if payload.get("config_sha256") != config_sha256:
            raise RuntimeError("configuration changed after the blind freeze")
        payload.update(
            {
                "blind_run_claimed": True,
                "blind_source_sha256": source_sha256,
                "claimed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        temporary = self.path.with_suffix(self.path.suffix + ".claiming")
        if temporary.exists():
            raise RuntimeError("blind run claim is already in progress")
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        temporary.replace(self.path)

    def _write_exclusive(self, payload: dict[str, object]) -> None:
        with self.path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
