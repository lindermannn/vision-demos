"""Strict JSON conversion for operational artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Optional


def encode_json(data: Any) -> str:
    """Serialize data while rejecting NaN and infinity."""

    return json.dumps(
        data,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


def write_json_exclusive(path: Path, data: Any) -> None:
    """Publish a complete JSON artifact atomically without overwriting."""

    encoded = encode_json(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            stream.write(encoded)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

        # A same-volume hard link publishes the completed temporary file and fails
        # atomically when the destination already exists. This preserves both the
        # no-overwrite contract and crash safety on the supported local filesystems.
        os.link(temporary_name, path)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
