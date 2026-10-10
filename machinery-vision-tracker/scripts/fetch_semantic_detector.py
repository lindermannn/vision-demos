"""Fetch and verify the pinned semantic detector during development/image build."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Optional, Sequence


MODEL_ID = "IDEA-Research/grounding-dino-tiny"
REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"
WEIGHT_SHA256 = "1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3"
FILES = (
    "README.md",
    "added_tokens.json",
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.txt",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument(
        "--bundle",
        type=Path,
        default=None,
        help="local snapshot copy shipped with the release; used instead of downloading when its weight matches the pinned SHA-256",
    )
    args = parser.parse_args(argv)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("install the optional learned dependency set first") from exc

    destination = args.destination.expanduser().resolve(strict=False)
    bundle = args.bundle.expanduser() if args.bundle is not None else None
    bundled_weight = bundle / "model.safetensors" if bundle is not None else None
    if bundled_weight is not None and bundled_weight.is_file() and _sha256(bundled_weight) == WEIGHT_SHA256:
        # Offline path: the release ZIP carries the pinned snapshot, so the
        # evaluator's image build needs no network at all.
        import shutil
        destination.mkdir(parents=True, exist_ok=True)
        for entry in bundle.iterdir():
            if entry.is_file():
                shutil.copy2(entry, destination / entry.name)
        source = "bundle"
    else:
        snapshot_download(
            repo_id=MODEL_ID,
            revision=REVISION,
            local_dir=destination,
            allow_patterns=list(FILES),
        )
        source = "huggingface_hub"
    weight = destination / "model.safetensors"
    actual = _sha256(weight)
    if actual != WEIGHT_SHA256:
        raise SystemExit("downloaded semantic-detector weight failed SHA-256 verification")
    print(
        json.dumps(
            {
                "model_id": MODEL_ID,
                "revision": REVISION,
                "weight_sha256": actual,
                "weight_size_bytes": weight.stat().st_size,
                "status": "verified",
                "source": source,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
