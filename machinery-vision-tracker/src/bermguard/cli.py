"""Command-line entry point for the assessment pipeline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Optional, Sequence

from .contracts import MethodSelection
from .exceptions import BermGuardError
from .pipeline import PipelineRequest, run_pipeline
from .serialization import encode_json


@dataclass(frozen=True)
class CliOptions:
    input_directory: Path
    output_directory: Path
    method: MethodSelection
    config: Optional[Path]
    model_directory: Optional[Path]
    detector_model_directory: Optional[Path]


def build_parser() -> argparse.ArgumentParser:
    model_from_environment = os.environ.get("BERMGUARD_MODEL_DIR")
    detector_model_from_environment = os.environ.get(
        "BERMGUARD_DETECTOR_MODEL_DIR"
    )
    parser = argparse.ArgumentParser(prog="bermguard")
    parser.add_argument("--input", required=True, type=Path, dest="input_directory")
    parser.add_argument("--output", required=True, type=Path, dest="output_directory")
    parser.add_argument(
        "--method",
        required=True,
        choices=[item.value for item in MethodSelection],
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(model_from_environment) if model_from_environment else None,
        help="local, complete Depth Anything snapshot; no runtime downloads",
    )
    parser.add_argument(
        "--detector-model-dir",
        type=Path,
        default=(
            Path(detector_model_from_environment)
            if detector_model_from_environment
            else None
        ),
        help=(
            "optional local, complete Grounding DINO snapshot; enables semantic "
            "heavy-equipment detection without runtime downloads"
        ),
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> CliOptions:
    namespace = build_parser().parse_args(argv)
    return CliOptions(
        input_directory=namespace.input_directory,
        output_directory=namespace.output_directory,
        method=MethodSelection(namespace.method),
        config=namespace.config,
        model_directory=namespace.model_dir,
        detector_model_directory=namespace.detector_model_dir,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    options = parse_args(argv)
    try:
        summary = run_pipeline(
            PipelineRequest(
                input_directory=options.input_directory,
                output_directory=options.output_directory,
                method=options.method,
                config=options.config,
                model_directory=options.model_directory,
                detector_model_directory=options.detector_model_directory,
            )
        )
    except BermGuardError as exc:
        print(f"bermguard: {exc}", file=sys.stderr)
        return 2

    print(encode_json(summary.__dict__))
    return 0 if summary.status == "complete" else 1
