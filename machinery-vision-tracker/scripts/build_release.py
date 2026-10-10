"""Build a reviewable ZIP from one clean Git commit and validated output artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Iterable, Optional, Sequence
import zipfile


def _run_git(repository: Path, *arguments: str) -> bytes:
    return subprocess.check_output(
        ["git", "-C", str(repository), *arguments],
        stderr=subprocess.STDOUT,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _tracked_files(repository: Path) -> list[Path]:
    raw = _run_git(repository, "ls-files", "-z")
    return [Path(item.decode("utf-8")) for item in raw.split(b"\0") if item]


def _regular_files(root: Path) -> Iterable[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_zip(source: Path, archive: Path) -> None:
    with zipfile.ZipFile(
        archive,
        mode="x",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as destination:
        for path in _regular_files(source):
            destination.write(path, path.relative_to(source.parent).as_posix())


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument(
        "--bundle-models",
        type=Path,
        default=None,
        help="directory whose snapshot subdirectories are copied into models/ so the image builds offline",
    )
    args = parser.parse_args(argv)

    repository = args.repository.expanduser().resolve(strict=True)
    artifacts = args.artifacts.expanduser().resolve(strict=True)
    destination = args.destination.expanduser().resolve(strict=False)
    if not repository.is_dir() or not artifacts.is_dir():
        raise SystemExit("repository and artifacts must be directories")
    if _run_git(repository, "status", "--porcelain").strip():
        raise SystemExit("release requires a clean Git worktree")

    commit = _run_git(repository, "rev-parse", "HEAD").decode("ascii").strip()
    package_name = f"BermGuard_Assessment_G5_3_{commit[:7]}"
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / f"{package_name}.zip"
    checksum_sidecar = destination / f"{package_name}.zip.sha256"
    release_sidecar = destination / f"{package_name}.release.json"
    for target in (archive, checksum_sidecar, release_sidecar):
        if target.exists():
            raise SystemExit(f"release target already exists: {target.name}")

    tracked = _tracked_files(repository)
    if not tracked:
        raise SystemExit("repository has no tracked files")
    artifact_files = list(_regular_files(artifacts))
    run_manifests = sorted(artifacts.glob("run_manifest*.json"))
    if not artifact_files or not run_manifests:
        raise SystemExit("artifact directory is incomplete")

    with tempfile.TemporaryDirectory(prefix="bermguard-release-", dir=destination) as raw:
        package_root = Path(raw) / package_name
        package_root.mkdir()
        for relative in tracked:
            source = repository / relative
            if not source.is_file() or source.is_symlink():
                raise SystemExit(f"tracked release input is not a regular file: {relative}")
            target = package_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        shutil.copytree(artifacts, package_root / "output")
        bundled: list[str] = []
        if args.bundle_models is not None:
            for snapshot in sorted(args.bundle_models.iterdir()):
                if snapshot.is_dir() and (snapshot / "model.safetensors").is_file():
                    shutil.copytree(snapshot, package_root / "models" / snapshot.name)
                    bundled.append(snapshot.name)

        release_manifest = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": commit,
            "source_file_count": len(tracked),
            "output_file_count": len(artifact_files),
            "output_manifests_sha256": {
                path.name: _sha256(path) for path in run_manifests
            },
            "model_weight_included": bool(bundled),
            "bundled_snapshots": bundled,
            "docker_image_tar_included": False,
            "scope": "assessment source, documentation and validated processed outputs",
        }
        _write_json(package_root / "RELEASE_MANIFEST.json", release_manifest)

        manifest_path = package_root / "MANIFEST.sha256"
        manifest_lines = [
            f"{_sha256(path)}  {path.relative_to(package_root).as_posix()}"
            for path in _regular_files(package_root)
            if path != manifest_path
        ]
        manifest_path.write_text(
            "\n".join(manifest_lines) + "\n", encoding="utf-8", newline="\n"
        )
        _write_zip(package_root, archive)

    archive_hash = _sha256(archive)
    checksum_sidecar.write_text(
        f"{archive_hash}  {archive.name}\n", encoding="utf-8", newline="\n"
    )
    _write_json(
        release_sidecar,
        {
            "schema_version": 1,
            "archive": archive.name,
            "archive_sha256": archive_hash,
            "archive_size_bytes": archive.stat().st_size,
            "git_commit": commit,
        },
    )
    print(release_sidecar.read_text(encoding="utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
