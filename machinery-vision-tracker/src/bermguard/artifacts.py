"""Transactional output layout for one video and one method."""

from __future__ import annotations

from contextlib import contextmanager
import csv
from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Any, Iterable, Iterator, Mapping, Sequence
from uuid import uuid4

from .contracts import MethodSelection
from .exceptions import OutputExistsError


@dataclass(frozen=True)
class RunFiles:
    """Root-level files written once per method selection."""

    manifest: Path
    benchmark_table: Path
    benchmark_summary: Path

    def as_names(self) -> dict[str, str]:
        return {
            "run_manifest": self.manifest.name,
            "benchmark_comparison": self.benchmark_table.name,
            "benchmark_summary": self.benchmark_summary.name,
        }


def prepare_output_root(input_root: Path, output_root: Path) -> Path:
    """Resolve and create the output root without requiring it to be empty.

    The official evaluation command mounts one host directory at ``/app/output``
    and may be repeated for ``--method 1`` and ``--method 2``.  Overwrite
    protection therefore lives at two finer levels: :func:`reserve_run_files`
    for the root files of one method selection and
    :func:`method_output_transaction` for each ``<video>/method_<n>`` directory.
    """

    source = input_root.expanduser().resolve(strict=True)
    destination = output_root.expanduser().resolve(strict=False)
    if destination == source or source in destination.parents:
        raise OutputExistsError("output directory must not be inside the input corpus")
    if destination.exists():
        if not destination.is_dir():
            raise OutputExistsError("output path exists and is not a directory")
    else:
        destination.mkdir(parents=True)
    return destination


def reserve_run_files(output_root: Path, method: MethodSelection) -> RunFiles:
    """Resolve the root-level run files for one selection without writing them.

    Selections ``1``, ``2`` and ``all`` coexist in one output root.  Repeating a
    selection into the same root is refused here, before any video is decoded,
    so a mistaken re-run fails fast instead of after minutes of processing.
    """

    suffix = f"method_{method.value}"
    files = RunFiles(
        manifest=output_root / f"run_manifest_{suffix}.json",
        benchmark_table=output_root / f"benchmark_comparison_{suffix}.csv",
        benchmark_summary=output_root / f"benchmark_summary_{suffix}.json",
    )
    existing = [
        path.name
        for path in (files.manifest, files.benchmark_table, files.benchmark_summary)
        if path.exists()
    ]
    if existing:
        raise OutputExistsError(
            "run files already exist for this method selection; refusing overwrite: "
            + ", ".join(existing)
        )
    return files


@contextmanager
def method_output_transaction(
    output_root: Path, video_id: str, method: MethodSelection
) -> Iterator[Path]:
    if method is MethodSelection.ALL:
        raise ValueError("transaction requires one concrete method")
    final_directory = output_root / video_id / f"method_{method.value}"
    if final_directory.exists():
        raise OutputExistsError(f"method output already exists: {video_id}/method_{method.value}")

    temporary_parent = output_root / ".work"
    temporary_parent.mkdir(exist_ok=True)
    temporary_directory = temporary_parent / uuid4().hex
    temporary_directory.mkdir()
    try:
        yield temporary_directory
        if final_directory.exists():
            raise OutputExistsError(
                f"method output appeared during processing: {video_id}/method_{method.value}"
            )
        final_directory.parent.mkdir(parents=True, exist_ok=True)
        temporary_directory.rename(final_directory)
    except BaseException:
        if temporary_directory.exists():
            shutil.rmtree(temporary_directory)
        raise
    finally:
        if temporary_parent.exists() and not any(temporary_parent.iterdir()):
            temporary_parent.rmdir()


def write_csv_header(path: Path, columns: Sequence[str]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerow(columns)


def write_csv_rows(
    path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, Any]]
) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
