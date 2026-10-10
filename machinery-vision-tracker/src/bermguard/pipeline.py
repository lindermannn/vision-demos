"""Two-method G3 assessment pipeline with a shared analytical downstream."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import platform
from time import perf_counter
from typing import Any, Optional

from .artifacts import (
    method_output_transaction,
    prepare_output_root,
    reserve_run_files,
    write_csv_rows,
)
from .charts import (
    write_height_curve,
    write_minimum_distance_matrix,
    write_vehicle_dispersion,
)
from .config import LoadedConfiguration, load_configuration
from .contracts import MethodSelection
from .exceptions import ConfigurationError
from .g2_config import G2Settings
from .processing import (
    AnalyzedVideoResult,
    process_analyzed_video,
    process_classical_video,
)
from .serialization import write_json_exclusive
from .video import (
    VideoInfo,
    discover_videos,
    inspect_video,
)


@dataclass(frozen=True)
class PipelineRequest:
    input_directory: Path
    output_directory: Path
    method: MethodSelection
    config: Optional[Path] = None
    model_directory: Optional[Path] = None
    detector_model_directory: Optional[Path] = None


@dataclass(frozen=True)
class RunSummary:
    status: str
    discovered_videos: int
    completed_outputs: int
    failures: int
    output_directory: str
    wall_s: float


def _methods(selection: MethodSelection) -> tuple[MethodSelection, ...]:
    if selection is MethodSelection.ALL:
        return (MethodSelection.METHOD_1, MethodSelection.METHOD_2)
    return (selection,)


def _with_identity(
    rows: list[dict[str, Any]], video_id: str, method: MethodSelection
) -> list[dict[str, Any]]:
    return [
        {"video_id": video_id, "method": method.value, **row}
        for row in rows
    ]


def _write_g2_artifacts(
    directory: Path,
    video: VideoInfo,
    method: MethodSelection,
    result: AnalyzedVideoResult,
) -> list[str]:
    tables = {
        "height_series.csv": (
            (
                "video_id", "method", "scene_id", "frame_index", "timestamp_s",
                "status", "height_px", "height_m_estimated", "height_m_vehicle_scale",
                "vehicle_px_per_m", "crest_y_median", "toe_y_median", "scale_source",
                "confidence",
                "profile_source", "quality_reason", "measurement_basis",
                "profile_age_frames", "lighting", "operating_surface_far_y",
                "operating_surface_near_y",
                "segmentation_preprocess_ms",
                "segmentation_inference_ms", "segmentation_postprocess_ms",
            ),
            result.height_rows,
        ),
        "vehicle_positions.csv": (
            (
                "video_id", "method", "scene_id", "frame_index", "timestamp_s",
                "track_id", "label", "confidence", "x_px", "y_px", "bbox_x1",
                "bbox_y1", "bbox_x2", "bbox_y2", "proximity_level",
            ),
            result.vehicle_rows,
        ),
        "minimum_distance.csv": (
            (
                "video_id", "method", "scene_id", "track_id_a", "track_id_b",
                "minimum_distance_px", "minimum_distance_m_estimated",
                "proximity_level", "frame_index", "timestamp_s",
            ),
            result.minimum_rows,
        ),
        "proximity_events.csv": (
            (
                "video_id", "method", "event_id", "scene_id", "track_id_a",
                "track_id_b", "start_frame", "end_frame", "start_s", "end_s",
                "minimum_distance_px", "minimum_distance_m_estimated", "level",
            ),
            result.event_rows,
        ),
    }
    for filename, (columns, rows) in tables.items():
        write_csv_rows(
            directory / filename,
            columns,
            _with_identity(rows, video.video_id, method),
        )
    write_height_curve(directory / "height_curve.png", result.height_rows)
    write_vehicle_dispersion(
        directory / "vehicle_dispersion.png",
        result.vehicle_rows,
        video.width,
        video.height,
    )
    write_minimum_distance_matrix(
        directory / "minimum_distance_matrix.png", result.minimum_rows
    )
    return sorted([*tables, "height_curve.png", "vehicle_dispersion.png", "minimum_distance_matrix.png"])


def _analysis_metadata(
    video: VideoInfo,
    method: MethodSelection,
    configuration: LoadedConfiguration,
    result: AnalyzedVideoResult,
    artifact_names: list[str],
    stage: str,
    implementation: str,
    learned_weights: bool,
    segmenter_runtime: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    import cv2
    import numpy

    detector_backend = str(result.summary.get("vehicle_detector_backend", "unknown"))
    semantic_detector = detector_backend.startswith("grounding_dino")
    alert_count = int(result.summary.get("persistent_red_events", 0))
    proximity_alert = alert_count > 0
    pair_levels = dict(result.summary.get("pair_level_counts", {}))
    frame_count = max(int(result.stats.frame_count), 1)
    return {
        "schema_version": 1,
        "stage": stage,
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "video": video.public_metadata(),
        "method": {
            "id": method.value,
            "implementation": implementation,
            "cv_inference": True,
            "learned_weights": learned_weights,
            "segmenter_runtime": segmenter_runtime or {},
        },
        "configuration": {
            "source": configuration.source,
            "sha256": configuration.sha256,
        },
        "processing": asdict(result.stats),
        "analysis": result.summary,
        # Keys named by the assessment brief/PDF: proximity_alert follows the brief
        # simulator (true only for a persistent red pair); the PDF metadata list is
        # FPS, per-frame time, alert count and analysed resolution.
        "proximity_alert": proximity_alert,
        "proximity_alert_count": alert_count,
        "evaluation_summary": {
            "average_processing_fps": result.stats.observed_fps,
            "processing_ms_per_frame": 1000.0 * result.stats.wall_s / frame_count,
            "analyzed_resolution": {"width": video.width, "height": video.height},
            "source_fps": video.fps,
            "proximity_alert": proximity_alert,
            "proximity_alert_count": alert_count,
            "proximity_red_pair_frames": int(pair_levels.get("red", 0)),
            "alert_semantics": (
                "proximity_alert is true only when at least one red (< 10 m "
                "estimated) pair persisted for the configured frames; yellow and "
                "green never raise it. The count is persistent events, not frames."
            ),
        },
        "runtime": {
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "numpy": numpy.__version__,
            "platform": platform.system().lower(),
        },
        "artifacts": ["processed.mp4", "metadata.json", *artifact_names],
        "warnings": [
            "Berm crest/base are image-space inferred profiles; they are not ground truth.",
            "Meters use the assessment-authorized assumed ratio of 5 px = 1 m and are not survey-grade.",
            (
                "Heavy-equipment boxes use zero-shot semantic detection without labelled local ground truth."
                if semantic_detector
                else "Heavy-equipment outputs are motion/appearance candidates, not validated semantic classifications."
            ),
            "Tracking identities are causal and reset at detected scene cuts.",
            "No labelled reference masks or boxes were supplied; valid rate is an internal quality-gate rate, not accuracy.",
        ],
    }


def _benchmark_records(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    columns = (
        "video_id", "method", "stage", "processed_frames", "wall_s",
        "observed_fps", "segmentation_s", "segmentation_fps",
        "height_valid_frames", "height_valid_rate", "height_m_estimated_median",
        "unique_scene_tracks", "persistent_red_events",
    )
    return [{column: record.get(column) for column in columns} for record in results]


def _benchmark_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    methods: dict[str, dict[str, Any]] = {}
    for method in ("1", "2"):
        rows = [record for record in results if record.get("method") == method]
        if not rows:
            continue
        frames = sum(int(row["processed_frames"]) for row in rows)
        valid_frames = sum(int(row["height_valid_frames"]) for row in rows)
        wall = sum(float(row["wall_s"]) for row in rows)
        segmentation = sum(float(row["segmentation_s"]) for row in rows)
        methods[method] = {
            "videos": len(rows),
            "frames": frames,
            "sum_video_wall_s": wall,
            "aggregate_observed_fps": frames / wall if wall else 0.0,
            "aggregate_segmentation_fps": frames / segmentation if segmentation else 0.0,
            "aggregate_height_valid_rate": valid_frames / frames if frames else 0.0,
            "mean_height_valid_rate": sum(
                float(row["height_valid_rate"]) for row in rows
            ) / len(rows),
            "median_height_m_estimated_by_video": {
                row["video_id"]: row["height_m_estimated_median"] for row in rows
            },
        }
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "descriptive_without_ground_truth",
        "shared_downstream": [
            "scene_cut_detector", "vehicle_detector", "causal_tracker",
            "pixel_scale_geometry", "proximity_events", "renderer", "artifact_writer",
        ],
        "controlled_variable": "berm_segmentation_signal_and_profile_extraction",
        "methods": methods,
        "limitations": [
            "One local corpus run is reported; it is not a repeated performance study.",
            "No labelled masks, crest/base surveys or vehicle boxes were supplied.",
            "Height valid rate measures internal acceptance, not correctness.",
        ],
    }


def run_pipeline(request: PipelineRequest) -> RunSummary:
    started = perf_counter()
    configuration = load_configuration(request.config)
    settings = G2Settings.from_mapping(configuration.data)
    sources = discover_videos(request.input_directory)
    input_root = request.input_directory.expanduser().resolve(strict=True)
    methods = _methods(request.method)
    learned_segmenter = None
    if MethodSelection.METHOD_2 in methods:
        if request.model_directory is None:
            raise ConfigurationError(
                "method 2 requires --model-dir with the pinned local snapshot"
            )
        from .segmentation.pretrained import DepthAnythingV2Segmenter

        learned_segmenter = DepthAnythingV2Segmenter(
            settings, request.model_directory
        )
    vehicle_detector = None
    if request.detector_model_directory is not None:
        from .semantic_detection import GroundingDinoVehicleDetector

        vehicle_detector = GroundingDinoVehicleDetector(
            settings, request.detector_model_directory
        )
    output_root = prepare_output_root(input_root, request.output_directory)
    run_files = reserve_run_files(output_root, request.method)

    input_records: list[dict[str, Any]] = []
    result_records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for source in sources:
        relative_name = source.relative_to(input_root).as_posix()
        try:
            video = inspect_video(input_root, source)
            input_records.append(video.public_metadata())
        except Exception as exc:
            failures.append(
                {"input": relative_name, "stage": "inspect", "error": str(exc)}
            )
            if configuration.fail_fast:
                raise
            continue

        for method in methods:
            try:
                with method_output_transaction(output_root, video.video_id, method) as work:
                    if method is MethodSelection.METHOD_1:
                        result = process_classical_video(
                            video,
                            work / "processed.mp4",
                            settings,
                            vehicle_detector,
                        )
                        stats = result.stats
                        artifacts = _write_g2_artifacts(work, video, method, result)
                        metadata = _analysis_metadata(
                            video,
                            method,
                            configuration,
                            result,
                            artifacts,
                            "G2_classical_analytics",
                            "classical_horizontal_profile_shared_vehicle_detector",
                            False,
                        )
                        stage = "G2_classical_analytics"
                    else:
                        if learned_segmenter is None:
                            raise RuntimeError("learned segmenter was not initialized")
                        result = process_analyzed_video(
                            video,
                            work / "processed.mp4",
                            settings,
                            learned_segmenter,
                            "metodo 2 profundidad",
                            vehicle_detector,
                        )
                        stats = result.stats
                        artifacts = _write_g2_artifacts(work, video, method, result)
                        metadata = _analysis_metadata(
                            video,
                            method,
                            configuration,
                            result,
                            artifacts,
                            "G3_learned_depth_analytics",
                            "depth_anything_v2_relative_depth_profile_shared_vehicle_detector",
                            True,
                            learned_segmenter.runtime_metadata(),
                        )
                        stage = "G3_learned_depth_analytics"
                    write_json_exclusive(work / "metadata.json", metadata)
                result_records.append(
                    {
                        "video_id": video.video_id,
                        "method": method.value,
                        "directory": f"{video.video_id}/method_{method.value}",
                        "processed_frames": stats.frame_count,
                        "wall_s": stats.wall_s,
                        "observed_fps": stats.observed_fps,
                        "stage": stage,
                        "segmentation_s": stats.segmentation_s,
                        "segmentation_fps": (
                            stats.frame_count / stats.segmentation_s
                            if stats.segmentation_s
                            else 0.0
                        ),
                        "height_valid_rate": result.summary["height_valid_rate"],
                        "height_valid_frames": result.summary["height_valid_frames"],
                        "height_m_estimated_median": result.summary[
                            "height_m_estimated_median"
                        ],
                        "unique_scene_tracks": result.summary["unique_scene_tracks"],
                        "persistent_red_events": result.summary[
                            "persistent_red_events"
                        ],
                    }
                )
            except Exception as exc:
                failures.append(
                    {
                        "input": relative_name,
                        "method": method.value,
                        "stage": (
                            "G2_classical_analytics"
                            if method is MethodSelection.METHOD_1
                            else "G3_learned_depth_analytics"
                        ),
                        "error": str(exc),
                    }
                )
                if configuration.fail_fast:
                    raise

    status = "complete" if not failures else ("partial" if result_records else "failed")
    manifest = {
        "schema_version": 1,
        "stage": "G3_two_method_comparison",
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "source": configuration.source,
            "sha256": configuration.sha256,
        },
        "method_selection": request.method.value,
        "methods": [method.value for method in methods],
        "run_files": run_files.as_names(),
        "vehicle_detector": (
            vehicle_detector.name
            if vehicle_detector is not None
            else "conservative_motion_appearance_visual_candidates"
        ),
        "inputs": input_records,
        "results": result_records,
        "failures": failures,
    }
    benchmark_rows = _benchmark_records(result_records)
    write_csv_rows(
        run_files.benchmark_table,
        tuple(benchmark_rows[0]) if benchmark_rows else (
            "video_id", "method", "stage", "processed_frames", "wall_s",
            "observed_fps", "segmentation_s", "segmentation_fps",
            "height_valid_frames", "height_valid_rate", "height_m_estimated_median",
            "unique_scene_tracks", "persistent_red_events",
        ),
        benchmark_rows,
    )
    write_json_exclusive(
        run_files.benchmark_summary, _benchmark_summary(result_records)
    )
    write_json_exclusive(run_files.manifest, manifest)

    return RunSummary(
        status=status,
        discovered_videos=len(sources),
        completed_outputs=len(result_records),
        failures=len(failures),
        output_directory=str(output_root),
        wall_s=perf_counter() - started,
    )
