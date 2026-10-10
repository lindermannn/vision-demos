"""Block common accidental publication of private or oversized artifacts."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".pt", ".pth", ".onnx", ".safetensors", ".sqlite3"}
PRIVATE_NAMES = {".env", "credentials.json", "secrets.json"}
RISKY = re.compile(r"(?i)(?:api[_-]?key|access[_-]?token|secret|password)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{12,}|bearer\s+[A-Za-z0-9._\-]{20,}")
ABSOLUTE = re.compile(r"(?i)((?<![A-Za-z0-9_])[A-Z]:\\|/Users/|/home/)")


def main() -> int:
    findings: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        relative = path.relative_to(ROOT)
        if path.suffix.lower() in PRIVATE_SUFFIXES or path.name.lower() in PRIVATE_NAMES:
            findings.append(f"PRIVATE_ARTIFACT {relative}")
        if path.stat().st_size > 50 * 1024 * 1024:
            findings.append(f"OVERSIZED {relative} ({path.stat().st_size} bytes)")
        if path.suffix.lower() in {".md", ".py", ".yaml", ".yml", ".json", ".toml", ".txt"} and path.name != "validate_publication.py":
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if RISKY.search(text): findings.append(f"SECRET_LIKE_TEXT {relative}")
            if ABSOLUTE.search(text): findings.append(f"ABSOLUTE_PATH {relative}")
    for line in findings:
        print(line)
    if findings:
        print(f"BLOCKED: {len(findings)} finding(s) require review")
        return 1
    print("PUBLICATION_AUDIT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
