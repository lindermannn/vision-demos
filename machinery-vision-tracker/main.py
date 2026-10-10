"""Official entry point integrating approved Method 1 and executable Method 2."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Optional, Sequence

HERE = Path(__file__).resolve().parent
V8_DIRECTORY = HERE / "metodo1_v8"
V8_MODULES = ("fixed_camera_v8.py", "berm_metrics.py", "normativa_ds132.py",
              "anchored_search.py", "fusion_tracking.py", "groundplane_proximity.py")


def _ensure_paths() -> None:
    for directory in (HERE / "src", HERE / "metodo2", V8_DIRECTORY):
        if directory.is_dir() and str(directory) not in sys.path:
            sys.path.insert(0, str(directory))


def _requested(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--method", required=True, choices=("1", "2", "all"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    known, _ = parser.parse_known_args(list(argv) if argv is not None else None)
    return known


def _replace_method(argv: Sequence[str], method: str) -> list[str]:
    result = list(argv)
    index = result.index("--method")
    result[index + 1] = method
    return result


def _method1_args(argv: Sequence[str]) -> list[str]:
    allowed_with_value = {"--input", "--output", "--method", "--config", "--model-dir", "--detector-model-dir"}
    result = []; index = 0
    while index < len(argv):
        item = argv[index]
        if item in allowed_with_value and index + 1 < len(argv):
            result.extend((item, argv[index + 1])); index += 2
        else:
            index += 1
    return result


def _method2_args(argv: Sequence[str]) -> list[str]:
    excluded = {"--method", "--config", "--model-dir", "--detector-model-dir"}
    result = []; index = 0
    while index < len(argv):
        if argv[index] in excluded:
            index += 2
        else:
            result.append(argv[index]); index += 1
    return result


def _install_method_1_v8() -> None:
    from bermguard import processing, rendering
    from bermguard.contracts import ProximityLevel
    import anchored_search
    from fixed_camera_v8 import MeasuringFixedCameraValidator, render_experiment
    from fusion_tracking import FusionResistantTracker
    from groundplane_proximity import GroundPlaneProximityEngine
    processing.TemporalBermValidator = MeasuringFixedCameraValidator
    processing.ProximityEngine = GroundPlaneProximityEngine
    processing.CausalTracker = FusionResistantTracker
    rendering._LEVEL_COLOURS[ProximityLevel.UNKNOWN] = rendering._LEVEL_COLOURS[ProximityLevel.GREEN]
    processing.render_frame = render_experiment
    processing.ClassicalBermSegmenter = anchored_search.DaylightOnlyClassicalSegmenter
    anchored_search.reset_counts()


def _write_method1_provenance(output_root: Path, exit_code: int) -> None:
    import anchored_search, berm_metrics, normativa_ds132 as normativa
    written = berm_metrics.write_records(output_root)
    provenance = {
        "package": "metodo_1_listo", "policy": "measure_v8", "method": "1", "exit_code": exit_code,
        "sidecar_profile_files": written, "search_sweeps": dict(anchored_search.COUNTS),
        "assumption": "fixed camera for the entire input clip, including lighting cuts",
        "berm_size_norm": {"regulation": "DS 132 Reglamento de Seguridad Minera (Chile)",
                           "article_in_force": normativa.ARTICLE_DUMP,
                           "article_reported_alongside": normativa.ARTICLE_ROAD,
                           "dump_edge_minimum_m": round(normativa.DUMP_EDGE_MINIMUM_M, 3),
                           "steep_road_minimum_m": round(normativa.STEEP_ROAD_MINIMUM_M, 3),
                           "assumed_wheel_diameter_m": normativa.WHEEL_DIAMETER_M,
                           "assumed_truck_height_m": normativa.TRUCK_HEIGHT_M,
                           "primary_unit": "metres", "secondary_unit": "wheel diameters"},
        "code_sha256": {name: hashlib.sha256((V8_DIRECTORY / name).read_bytes()).hexdigest()
                        for name in V8_MODULES},
    }
    (output_root / "procedencia_metodo_1_v8.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")


def _run_method1(argv: Sequence[str], output: Path) -> int:
    from bermguard.cli import main
    _install_method_1_v8()
    code = main(_method1_args(_replace_method(argv, "1")))
    if output.is_dir(): _write_method1_provenance(output, code)
    return code


def _run_method2(argv: Sequence[str]) -> int:
    from runner import run
    return run(_method2_args(argv))


def run(argv: Optional[Sequence[str]] = None) -> int:
    _ensure_paths()
    arguments = list(argv) if argv is not None else sys.argv[1:]
    if "--help" in arguments or "-h" in arguments:
        print("usage: main.py --input INPUT --output OUTPUT --method {1,2,all} "
              "[--berm-mask-root ROOT] [--device cpu] [--detect-every 1]")
        return 0
    requested = _requested(arguments)
    if requested.method == "1": return _run_method1(arguments, requested.output)
    if requested.method == "2": return _run_method2(arguments)
    code1 = _run_method1(arguments, requested.output)
    code2 = _run_method2(arguments)
    return 0 if code1 == 0 and code2 == 0 else 1


if __name__ == "__main__":
    raise SystemExit(run())
