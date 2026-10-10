"""Frame-causal G2 classical processing and structured evidence collection."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any, Protocol

import cv2
import numpy as np

from .contracts import FrameContext, ObservationStatus
from .detection import ClassicalVehicleDetector
from .exceptions import VideoProcessingError
from .g2_config import G2Settings
from .geometry import bbox_bottom_center
from .proximity import ProximityEngine
from .rendering import render_frame
from .scene import SceneCutDetector
from .segmentation.classical import ClassicalBermSegmenter
from .segmentation.result import SegmentationResult
from .segmentation.validation import LOW_LIGHT_ESTIMATE_REASON, TemporalBermValidator
from .tracking import CausalTracker
from .video import ProcessingStats, VideoInfo, inspect_video


@dataclass(frozen=True)
class AnalyzedVideoResult:
    stats: ProcessingStats
    height_rows: list[dict[str, Any]]
    vehicle_rows: list[dict[str, Any]]
    minimum_rows: list[dict[str, Any]]
    event_rows: list[dict[str, Any]]
    summary: dict[str, Any]


class DetailedSegmenter(Protocol):
    name: str

    def segment(self, frame: Any, context: FrameContext) -> SegmentationResult:
        """Return one mask/profile observation for a source frame."""


ClassicalVideoResult = AnalyzedVideoResult


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _validate_output(video: VideoInfo, path: Path, decoded_frames: int) -> VideoInfo:
    if decoded_frames <= 0 or not path.is_file() or path.stat().st_size <= 0:
        raise VideoProcessingError("video processing produced no frames")
    if abs(video.frame_count - decoded_frames) > 1:
        raise VideoProcessingError("decoded frame count differs from source declaration")
    output = inspect_video(path.parent, path)
    if (output.width, output.height) != (video.width, video.height):
        raise VideoProcessingError("encoded video changed frame dimensions")
    if abs(output.fps - video.fps) > max(0.01, video.fps * 0.001):
        raise VideoProcessingError("encoded video changed frame rate")
    if abs(output.frame_count - decoded_frames) > 1:
        raise VideoProcessingError("encoded video frame count failed validation")
    return output


def process_analyzed_video(
    video: VideoInfo,
    output_path: Path,
    settings: G2Settings,
    segmenter: DetailedSegmenter,
    method_label: str,
    vehicle_detector: Any | None = None,
) -> AnalyzedVideoResult:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise VideoProcessingError(f"output already exists: {output_path.name}")
    capture = cv2.VideoCapture(str(video.source_path))
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        video.fps,
        (video.width, video.height),
    )
    if not capture.isOpened() or not writer.isOpened():
        capture.release()
        writer.release()
        raise VideoProcessingError("cannot open input or create MP4 output")

    scene_detector = SceneCutDetector(settings)
    detector = vehicle_detector or ClassicalVehicleDetector(settings)
    if hasattr(detector, "reset_run"):
        detector.reset_run()
    else:
        detector.reset()
    tracker = CausalTracker(settings)
    proximity = ProximityEngine(settings)
    berm_validator = TemporalBermValidator(settings)
    # Operating surface: rows where confirmed machinery touches the ground in
    # the current scene.  The berm toe cannot lie below its nearest contact.
    surface_contacts: deque[tuple[int, float]] = deque(maxlen=400)
    # Vehicle-referenced scale: apparent height of confirmed machinery whose
    # ground contact sits at the berm toe depth, divided by the documented
    # reference vehicle height.  An estimate, never survey grade.
    scale_samples: deque[tuple[int, float]] = deque(maxlen=600)
    height_rows: list[dict[str, Any]] = []
    vehicle_rows: list[dict[str, Any]] = []
    scene_cuts: list[dict[str, Any]] = []
    pair_level_counts = {"green": 0, "yellow": 0, "red": 0, "unknown": 0}
    unique_tracks: set[tuple[int, int]] = set()
    tracker_scene_diagnostics: list[dict[str, int | float]] = []
    detector_rejection_totals: dict[str, int] = {}
    timings = {
        "decode": 0.0,
        "scene": 0.0,
        "segmentation": 0.0,
        "detection": 0.0,
        "tracking": 0.0,
        "analytics": 0.0,
        "rendering": 0.0,
        "encode": 0.0,
    }
    frame_index = 0
    scene_id = 0
    started = perf_counter()
    try:
        while True:
            stage = perf_counter()
            readable, frame = capture.read()
            timings["decode"] += perf_counter() - stage
            if not readable:
                break
            if frame is None or frame.shape[:2] != (video.height, video.width):
                raise VideoProcessingError(f"unexpected frame geometry at frame {frame_index}")

            stage = perf_counter()
            scene_decision = scene_detector.update(frame)
            timings["scene"] += perf_counter() - stage
            if scene_decision.is_cut:
                tracker_scene_diagnostics.append(tracker.diagnostics())
                scene_id += 1
                if hasattr(detector, "reset_scene"):
                    detector.reset_scene()
                else:
                    detector.reset()
                tracker.reset()
                proximity.reset_scene_state()
                berm_validator.reset()
                surface_contacts.clear()
                scale_samples.clear()
                scene_cuts.append(
                    {
                        "frame_index": frame_index,
                        "timestamp_s": frame_index / video.fps,
                        "histogram_correlation": scene_decision.histogram_correlation,
                        "mean_difference": scene_decision.mean_difference,
                        "mean_intensity": scene_decision.mean_intensity,
                        "illumination_transition": scene_decision.illumination_transition,
                        "structural_drift": scene_decision.structural_drift,
                        "lagged_structural_correlation": scene_decision.lagged_structural_correlation,
                    }
                )

            context = FrameContext(
                video_id=video.video_id,
                scene_id=scene_id,
                frame_index=frame_index,
                timestamp_s=frame_index / video.fps,
                width=video.width,
                height=video.height,
            )
            stage = perf_counter()
            raw_detections = detector.detect(frame, context)
            timings["detection"] += perf_counter() - stage
            for key, value in detector.last_diagnostics.items():
                if key.startswith("rejected_") or key in {
                    "raw_contours",
                    "raw_semantic_candidates",
                    "pre_nms_candidates",
                    "accepted_candidates",
                    "propagated_candidates",
                }:
                    detector_rejection_totals[key] = (
                        detector_rejection_totals.get(key, 0) + int(value)
                    )
            stage = perf_counter()
            tracked = tracker.update(raw_detections, context)
            timings["tracking"] += perf_counter() - stage
            # Only daylight contacts define the operating surface: at night the
            # boxes follow headlights and glare, and a spurious high contact would
            # make the remembered berm look like it lies inside the surface.
            if scene_decision.mean_intensity >= settings.scene_day_above_mean:
                for detection in tracked:
                    if detection.track_id is not None:
                        surface_contacts.append((frame_index, bbox_bottom_center(detection.bbox)[1]))
            recent_contacts = [y for f, y in surface_contacts if frame_index - f <= 150]
            operating_surface = None
            if len(recent_contacts) >= 3:
                operating_surface = (
                    float(np.percentile(recent_contacts, 5)),
                    float(np.percentile(recent_contacts, 95)),
                )
            if hasattr(segmenter, "set_operating_surface"):
                segmenter.set_operating_surface(operating_surface)
            stage = perf_counter()
            raw_segmentation = segmenter.segment(frame, context)
            segmentation = berm_validator.refine(
                raw_segmentation,
                raw_detections,
                frame,
                detector.last_exclusion_mask,
                operating_surface=operating_surface,
            )
            timings["segmentation"] += perf_counter() - stage
            observation = segmentation.observation
            diagnostics = observation.diagnostics
            if observation.base and observation.status.value in {"observed", "temporal_estimate"}:
                toe_median = float(np.median([y for _, y in observation.base]))
                for detection in tracked:
                    box = detection.bbox
                    if detection.track_id is None or box.y2 <= box.y1:
                        continue
                    if abs(box.y2 - toe_median) <= 0.10 * video.height:
                        scale_samples.append(
                            (frame_index, (box.y2 - box.y1) / settings.proximity_reference_vehicle_height_m)
                        )
            recent_scale = [s for f, s in scale_samples if frame_index - f <= 150]
            vehicle_px_per_m = float(np.median(recent_scale)) if len(recent_scale) >= 3 else None
            height_m_vehicle = (
                observation.height_px / vehicle_px_per_m
                if observation.height_px is not None and vehicle_px_per_m
                else None
            )
            height_rows.append(
                {
                    "scene_id": scene_id,
                    "frame_index": frame_index,
                    "timestamp_s": context.timestamp_s,
                    "status": observation.status.value,
                    "height_px": observation.height_px,
                    "height_m_estimated": observation.height_m_estimated,
                    "height_m_vehicle_scale": height_m_vehicle,
                    "crest_y_median": float(np.median([y for _, y in observation.crest])) if observation.crest else None,
                    "toe_y_median": float(np.median([y for _, y in observation.base])) if observation.base else None,
                    "vehicle_px_per_m": vehicle_px_per_m,
                    "scale_source": settings.scale_source,
                    "confidence": segmentation.confidence,
                    "profile_source": diagnostics.get("profile_source", "unknown"),
                    "quality_reason": diagnostics.get("reason", ""),
                    "measurement_basis": diagnostics.get("measurement_basis"),
                    "profile_age_frames": diagnostics.get("profile_age_frames"),
                    "lighting": diagnostics.get("lighting"),
                    "operating_surface_far_y": diagnostics.get("operating_surface_far_y"),
                    "operating_surface_near_y": diagnostics.get("operating_surface_near_y"),
                    "free_profile_fraction": diagnostics.get("free_profile_fraction"),
                    "occluding_vehicle_count": diagnostics.get(
                        "occluding_vehicle_count"
                    ),
                    "temporal_height_change_ratio": diagnostics.get(
                        "temporal_height_change_ratio"
                    ),
                    "segmentation_preprocess_ms": diagnostics.get("preprocess_ms"),
                    "segmentation_inference_ms": diagnostics.get("inference_ms"),
                    "segmentation_postprocess_ms": diagnostics.get("postprocess_ms"),
                }
            )
            stage = perf_counter()
            samples, levels = proximity.update(tracked, context)
            timings["analytics"] += perf_counter() - stage
            for sample in samples:
                pair_level_counts[sample.level.value] += 1
            for detection in tracked:
                if detection.track_id is None:
                    continue
                unique_tracks.add((scene_id, detection.track_id))
                point = bbox_bottom_center(detection.bbox)
                vehicle_rows.append(
                    {
                        "scene_id": scene_id,
                        "frame_index": frame_index,
                        "timestamp_s": context.timestamp_s,
                        "track_id": detection.track_id,
                        "label": detection.label,
                        "confidence": detection.confidence,
                        "x_px": point[0],
                        "y_px": point[1],
                        "bbox_x1": detection.bbox.x1,
                        "bbox_y1": detection.bbox.y1,
                        "bbox_x2": detection.bbox.x2,
                        "bbox_y2": detection.bbox.y2,
                        "proximity_level": levels.get(detection.track_id).value,
                    }
                )

            stage = perf_counter()
            rendered = render_frame(
                frame,
                segmentation,
                tracked,
                levels,
                scene_id,
                method_label,
                raw_detections,
                fps=video.fps,
                vehicle_px_per_m=vehicle_px_per_m,
            )
            timings["rendering"] += perf_counter() - stage
            stage = perf_counter()
            writer.write(rendered)
            timings["encode"] += perf_counter() - stage
            frame_index += 1
    finally:
        capture.release()
        finalize_started = perf_counter()
        writer.release()
        finalize_s = perf_counter() - finalize_started
    wall_s = perf_counter() - started

    output = _validate_output(video, output_path, frame_index)
    tracker_scene_diagnostics.append(tracker.diagnostics())
    minimum_rows = proximity.minimum_rows()
    event_rows = proximity.event_rows()
    # Heights carried from a daytime reference through low light are published
    # as estimates but are deliberately excluded from the "valid" benchmark
    # count: they are memory, not current evidence.
    valid_heights = [
        float(row["height_m_estimated"])
        for row in height_rows
        if row["status"]
        in {ObservationStatus.VALID.value, ObservationStatus.OCCLUDED.value}
        and row["height_m_estimated"] is not None
        and row["quality_reason"] != LOW_LIGHT_ESTIMATE_REASON
    ]
    height_status_counts: dict[str, int] = {}
    for row in height_rows:
        height_status_counts[row["status"]] = height_status_counts.get(row["status"], 0) + 1
    low_light_estimate_frames = sum(
        1 for row in height_rows if row["quality_reason"] == LOW_LIGHT_ESTIMATE_REASON
    )
    stage_latency: dict[str, dict[str, float]] = {}
    for field, label in (
        ("segmentation_preprocess_ms", "preprocess"),
        ("segmentation_inference_ms", "inference"),
        ("segmentation_postprocess_ms", "postprocess"),
    ):
        values = [float(row[field]) for row in height_rows if row[field] is not None]
        if values:
            stage_latency[label] = {
                "median_ms": median(values),
                "p95_ms": _percentile(values, 0.95),
            }
    summary = {
        "segmentation_backend": segmenter.name,
        "segmentation_stage_latency": stage_latency,
        "scene_count": scene_id + 1,
        "scene_cuts": scene_cuts,
        "height_valid_frames": len(valid_heights),
        "height_status_counts": height_status_counts,
        "vehicle_px_per_m_median": (
            median([float(r["vehicle_px_per_m"]) for r in height_rows if r["vehicle_px_per_m"]])
            if any(r["vehicle_px_per_m"] for r in height_rows) else None
        ),
        "height_m_vehicle_scale_median": (
            median([float(r["height_m_vehicle_scale"]) for r in height_rows if r["height_m_vehicle_scale"]])
            if any(r["height_m_vehicle_scale"] for r in height_rows) else None
        ),
        "height_low_light_estimate_frames": low_light_estimate_frames,
        "height_valid_rate": len(valid_heights) / frame_index if frame_index else 0.0,
        "height_m_estimated_median": median(valid_heights) if valid_heights else None,
        "height_m_estimated_min": min(valid_heights) if valid_heights else None,
        "height_m_estimated_max": max(valid_heights) if valid_heights else None,
        "vehicle_observations": len(vehicle_rows),
        "unique_scene_tracks": len(unique_tracks),
        "pair_samples": sum(pair_level_counts.values()),
        "pair_level_counts": pair_level_counts,
        "persistent_red_events": len(event_rows),
        "detector_diagnostics": detector_rejection_totals,
        "vehicle_detector_backend": detector.name,
        "vehicle_detector_runtime": (
            detector.runtime_metadata()
            if hasattr(detector, "runtime_metadata")
            else {"backend": detector.name}
        ),
        "tracker_scene_diagnostics": tracker_scene_diagnostics,
        "scale": {
            "pixels_per_meter": settings.pixels_per_meter,
            "source": settings.scale_source,
            "physical_status": "estimated_not_survey_grade",
        },
        "proximity_scale": {
            "mode": settings.proximity_scale_mode,
            "reference_vehicle_height_m": (
                settings.proximity_reference_vehicle_height_m
            ),
            "physical_status": "perspective_adjusted_estimate_not_survey_grade",
        },
    }
    stats = ProcessingStats(
        frame_count=frame_index,
        decode_s=timings["decode"],
        encode_s=timings["encode"],
        finalize_s=finalize_s,
        wall_s=wall_s,
        observed_fps=frame_index / wall_s if wall_s else 0.0,
        output_sha256=output.sha256,
        output_size_bytes=output.size_bytes,
        output_codec_fourcc=output.codec_fourcc,
        scene_s=timings["scene"],
        segmentation_s=timings["segmentation"],
        detection_s=timings["detection"],
        tracking_s=timings["tracking"],
        analytics_s=timings["analytics"],
        rendering_s=timings["rendering"],
    )
    return AnalyzedVideoResult(
        stats=stats,
        height_rows=height_rows,
        vehicle_rows=vehicle_rows,
        minimum_rows=minimum_rows,
        event_rows=event_rows,
        summary=summary,
    )


def process_classical_video(
    video: VideoInfo,
    output_path: Path,
    settings: G2Settings,
    vehicle_detector: Any | None = None,
) -> AnalyzedVideoResult:
    return process_analyzed_video(
        video,
        output_path,
        settings,
        ClassicalBermSegmenter(settings),
        "metodo 1 clasico",
        vehicle_detector,
    )
