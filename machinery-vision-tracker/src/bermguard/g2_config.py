"""Typed and validated G2 settings derived from the YAML snapshot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple

from .exceptions import ConfigurationError


def _section(data: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = data.get(name)
    if not isinstance(value, dict):
        raise ConfigurationError(f"configuration {name} section must be a mapping")
    return value


def _positive_float(section: Mapping[str, Any], key: str) -> float:
    value = section.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ConfigurationError(f"{key} must be a positive number")
    return float(value)


def _positive_int(section: Mapping[str, Any], key: str) -> int:
    value = section.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigurationError(f"{key} must be a positive integer")
    return value


def _non_empty_string(section: Mapping[str, Any], key: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{key} must be a non-empty string")
    return value.strip()


def _string_sequence(section: Mapping[str, Any], key: str) -> Tuple[str, ...]:
    value = section.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ConfigurationError(f"{key} must be a sequence of strings")
    cleaned = tuple(
        item.strip() for item in value if isinstance(item, str) and item.strip()
    )
    if not cleaned or len(cleaned) != len(value):
        raise ConfigurationError(f"{key} must contain only non-empty strings")
    return cleaned


def _ratio_pair(section: Mapping[str, Any], key: str) -> Tuple[float, float]:
    value = section.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ConfigurationError(f"{key} must contain two ratios")
    lower, upper = value
    if not all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value):
        raise ConfigurationError(f"{key} ratios must be numeric")
    lower_float, upper_float = float(lower), float(upper)
    if not 0 <= lower_float < upper_float <= 1:
        raise ConfigurationError(f"{key} must satisfy 0 <= lower < upper <= 1")
    return lower_float, upper_float


@dataclass(frozen=True)
class G2Settings:
    pixels_per_meter: float
    scale_source: str
    red_below_m: float
    yellow_at_or_below_m: float
    proximity_scale_mode: str
    proximity_reference_vehicle_height_m: float
    scene_histogram_threshold: float
    scene_mean_difference_threshold: float
    scene_night_below_mean: float
    scene_day_above_mean: float
    scene_cooldown_frames: int
    scene_drift_lag_frames: int
    scene_drift_correlation: float
    scene_drift_frames: int
    segmentation_process_width: int
    segmentation_roi_x: Tuple[float, float]
    segmentation_crest_y: Tuple[float, float]
    segmentation_base_offset_y: Tuple[float, float]
    segmentation_expected_height_y: float
    segmentation_max_expected_height_deviation_ratio: float
    segmentation_min_confidence: float
    segmentation_min_free_profile_fraction: float
    segmentation_max_interpolation_gap_ratio: float
    segmentation_max_temporal_height_change_ratio: float
    segmentation_max_height_iqr_ratio: float
    segmentation_occlusion_margin_ratio: float
    segmentation_height_ema_alpha: float
    segmentation_minimum_valid_frames: int
    segmentation_reference_expiry_frames: int
    segmentation_recovery_frames: int
    segmentation_low_light_registration_tolerance_frames: int
    segmentation_reconfirmation_failure_frames: int
    segmentation_bootstrap_settle_frames: int
    segmentation_min_pair_contrast: float
    segmentation_min_paired_support_fraction: float
    segmentation_min_contiguous_support_fraction: float
    segmentation_observed_coverage_fraction: float
    segmentation_min_current_support_for_estimate: float
    segmentation_profile_ema_alpha: float
    segmentation_max_profile_shift_ratio: float
    segmentation_max_registration_shift_ratio: float
    segmentation_max_profile_residual_ratio: float
    segmentation_max_saturated_fraction: float
    segmentation_ridge_polarity: str
    segmentation_toe_nearest_evidence_ratio: float
    learned_model_id: str
    learned_revision: str
    learned_weight_sha256: str
    learned_input_size: int
    learned_min_confidence: float
    learned_device: str
    detection_min_area_ratio: float
    detection_max_area_ratio: float
    detection_min_width_ratio: float
    detection_min_height_ratio: float
    detection_max_width_ratio: float
    detection_max_height_ratio: float
    detection_min_aspect_ratio: float
    detection_max_aspect_ratio: float
    detection_min_vertical_center_ratio: float
    detection_max_vertical_center_ratio: float
    detection_min_motion_fill: float
    detection_min_texture_std: float
    detection_max_night_bright_fraction: float
    detection_min_confidence: float
    detection_max_detections: int
    detection_warmup_frames: int
    semantic_model_id: str
    semantic_revision: str
    semantic_weight_sha256: str
    semantic_device: str
    semantic_labels: Tuple[str, ...]
    semantic_box_threshold: float
    semantic_text_threshold: float
    semantic_nms_iou_threshold: float
    semantic_inference_interval: int
    semantic_input_short_edge: int
    semantic_input_long_edge: int
    semantic_min_area_ratio: float
    semantic_max_area_ratio: float
    semantic_min_width_ratio: float
    semantic_min_height_ratio: float
    semantic_max_detections: int
    tracking_max_center_distance_ratio: float
    tracking_min_iou: float
    tracking_velocity_alpha: float
    tracking_max_size_change_ratio: float
    tracking_max_missed_frames: int
    tracking_minimum_hits: int
    event_persistence_frames: int
    event_recovery_frames: int

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "G2Settings":
        scale = _section(data, "scale")
        proximity = _section(data, "proximity")
        scene = _section(data, "scene")
        segmentation = _section(data, "segmentation")
        learned = _section(data, "learned")
        detection = _section(data, "detection")
        semantic = _section(data, "semantic_detection")
        tracking = _section(data, "tracking")
        events = _section(data, "events")

        red = _positive_float(proximity, "red_below_m")
        yellow = _positive_float(proximity, "yellow_at_or_below_m")
        if yellow <= red:
            raise ConfigurationError("yellow threshold must be greater than red threshold")
        min_area = _positive_float(detection, "min_area_ratio")
        max_area = _positive_float(detection, "max_area_ratio")
        if max_area <= min_area or max_area > 1:
            raise ConfigurationError("detection area ratios are invalid")

        settings = cls(
            pixels_per_meter=_positive_float(scale, "pixels_per_meter"),
            scale_source=str(scale.get("source", "unknown")),
            red_below_m=red,
            yellow_at_or_below_m=yellow,
            proximity_scale_mode=_non_empty_string(proximity, "scale_mode"),
            proximity_reference_vehicle_height_m=_positive_float(
                proximity, "reference_vehicle_height_m"
            ),
            scene_histogram_threshold=_positive_float(scene, "histogram_threshold"),
            scene_mean_difference_threshold=_positive_float(
                scene, "mean_difference_threshold"
            ),
            scene_night_below_mean=_positive_float(scene, "night_below_mean"),
            scene_day_above_mean=_positive_float(scene, "day_above_mean"),
            scene_cooldown_frames=_positive_int(scene, "cooldown_frames"),
            scene_drift_lag_frames=_positive_int(scene, "drift_lag_frames"),
            scene_drift_correlation=_positive_float(scene, "drift_correlation"),
            scene_drift_frames=_positive_int(scene, "drift_frames"),
            segmentation_process_width=_positive_int(segmentation, "process_width"),
            segmentation_roi_x=_ratio_pair(segmentation, "roi_x"),
            segmentation_crest_y=_ratio_pair(segmentation, "crest_y"),
            segmentation_base_offset_y=_ratio_pair(segmentation, "base_offset_y"),
            segmentation_expected_height_y=_positive_float(
                segmentation, "expected_height_y"
            ),
            segmentation_max_expected_height_deviation_ratio=_positive_float(
                segmentation, "max_expected_height_deviation_ratio"
            ),
            segmentation_min_confidence=_positive_float(
                segmentation, "min_confidence"
            ),
            segmentation_min_free_profile_fraction=_positive_float(
                segmentation, "min_free_profile_fraction"
            ),
            segmentation_max_interpolation_gap_ratio=_positive_float(
                segmentation, "max_interpolation_gap_ratio"
            ),
            segmentation_max_temporal_height_change_ratio=_positive_float(
                segmentation, "max_temporal_height_change_ratio"
            ),
            segmentation_max_height_iqr_ratio=_positive_float(
                segmentation, "max_height_iqr_ratio"
            ),
            segmentation_occlusion_margin_ratio=_positive_float(
                segmentation, "occlusion_margin_ratio"
            ),
            segmentation_height_ema_alpha=_positive_float(
                segmentation, "height_ema_alpha"
            ),
            segmentation_minimum_valid_frames=_positive_int(
                segmentation, "minimum_valid_frames"
            ),
            segmentation_reference_expiry_frames=_positive_int(
                segmentation, "reference_expiry_frames"
            ),
            segmentation_low_light_registration_tolerance_frames=_positive_int(
                segmentation, "low_light_registration_tolerance_frames"
            ),
            segmentation_reconfirmation_failure_frames=_positive_int(
                segmentation, "reconfirmation_failure_frames"
            ),
            segmentation_bootstrap_settle_frames=_positive_int(
                segmentation, "bootstrap_settle_frames"
            ),
            segmentation_recovery_frames=_positive_int(
                segmentation, "recovery_frames"
            ),
            segmentation_min_pair_contrast=_positive_float(
                segmentation, "min_pair_contrast"
            ),
            segmentation_min_paired_support_fraction=_positive_float(
                segmentation, "min_paired_support_fraction"
            ),
            segmentation_min_contiguous_support_fraction=_positive_float(
                segmentation, "min_contiguous_support_fraction"
            ),
            segmentation_observed_coverage_fraction=_positive_float(
                segmentation, "observed_coverage_fraction"
            ),
            segmentation_min_current_support_for_estimate=_positive_float(
                segmentation, "min_current_support_for_estimate"
            ),
            segmentation_profile_ema_alpha=_positive_float(
                segmentation, "profile_ema_alpha"
            ),
            segmentation_max_registration_shift_ratio=_positive_float(
                segmentation, "max_registration_shift_ratio"
            ),
            segmentation_max_profile_shift_ratio=_positive_float(
                segmentation, "max_profile_shift_ratio"
            ),
            segmentation_max_profile_residual_ratio=_positive_float(
                segmentation, "max_profile_residual_ratio"
            ),
            segmentation_ridge_polarity=str(segmentation.get("ridge_polarity", "both")),
            segmentation_toe_nearest_evidence_ratio=float(
                segmentation.get("toe_nearest_evidence_ratio", 0.0)
            ),
            segmentation_max_saturated_fraction=_positive_float(
                segmentation, "max_saturated_fraction"
            ),
            learned_model_id=_non_empty_string(learned, "model_id"),
            learned_revision=_non_empty_string(learned, "revision"),
            learned_weight_sha256=_non_empty_string(learned, "weight_sha256"),
            learned_input_size=_positive_int(learned, "input_size"),
            learned_min_confidence=_positive_float(learned, "min_confidence"),
            learned_device=_non_empty_string(learned, "device"),
            detection_min_area_ratio=min_area,
            detection_max_area_ratio=max_area,
            detection_min_width_ratio=_positive_float(
                detection, "min_width_ratio"
            ),
            detection_min_height_ratio=_positive_float(
                detection, "min_height_ratio"
            ),
            detection_max_width_ratio=_positive_float(
                detection, "max_width_ratio"
            ),
            detection_max_height_ratio=_positive_float(
                detection, "max_height_ratio"
            ),
            detection_min_aspect_ratio=_positive_float(
                detection, "min_aspect_ratio"
            ),
            detection_max_aspect_ratio=_positive_float(
                detection, "max_aspect_ratio"
            ),
            detection_min_vertical_center_ratio=_positive_float(
                detection, "min_vertical_center_ratio"
            ),
            detection_max_vertical_center_ratio=_positive_float(
                detection, "max_vertical_center_ratio"
            ),
            detection_min_motion_fill=_positive_float(
                detection, "min_motion_fill"
            ),
            detection_min_texture_std=_positive_float(
                detection, "min_texture_std"
            ),
            detection_max_night_bright_fraction=_positive_float(
                detection, "max_night_bright_fraction"
            ),
            detection_min_confidence=_positive_float(
                detection, "min_confidence"
            ),
            detection_max_detections=_positive_int(detection, "max_detections"),
            detection_warmup_frames=_positive_int(detection, "warmup_frames"),
            semantic_model_id=_non_empty_string(semantic, "model_id"),
            semantic_revision=_non_empty_string(semantic, "revision"),
            semantic_weight_sha256=_non_empty_string(semantic, "weight_sha256"),
            semantic_device=_non_empty_string(semantic, "device"),
            semantic_labels=_string_sequence(semantic, "labels"),
            semantic_box_threshold=_positive_float(semantic, "box_threshold"),
            semantic_text_threshold=_positive_float(semantic, "text_threshold"),
            semantic_nms_iou_threshold=_positive_float(
                semantic, "nms_iou_threshold"
            ),
            semantic_inference_interval=_positive_int(
                semantic, "inference_interval"
            ),
            semantic_input_short_edge=_positive_int(semantic, "input_short_edge"),
            semantic_input_long_edge=_positive_int(semantic, "input_long_edge"),
            semantic_min_area_ratio=_positive_float(semantic, "min_area_ratio"),
            semantic_max_area_ratio=_positive_float(semantic, "max_area_ratio"),
            semantic_min_width_ratio=_positive_float(semantic, "min_width_ratio"),
            semantic_min_height_ratio=_positive_float(semantic, "min_height_ratio"),
            semantic_max_detections=_positive_int(semantic, "max_detections"),
            tracking_max_center_distance_ratio=_positive_float(
                tracking, "max_center_distance_ratio"
            ),
            tracking_min_iou=_positive_float(tracking, "min_iou"),
            tracking_velocity_alpha=_positive_float(
                tracking, "velocity_alpha"
            ),
            tracking_max_size_change_ratio=_positive_float(
                tracking, "max_size_change_ratio"
            ),
            tracking_max_missed_frames=_positive_int(
                tracking, "max_missed_frames"
            ),
            tracking_minimum_hits=_positive_int(tracking, "minimum_hits"),
            event_persistence_frames=_positive_int(events, "persistence_frames"),
            event_recovery_frames=_positive_int(events, "recovery_frames"),
        )
        if settings.scene_histogram_threshold > 1:
            raise ConfigurationError("scene histogram threshold must be <= 1")
        if settings.scene_mean_difference_threshold > 1:
            raise ConfigurationError("scene mean difference threshold must be <= 1")
        if settings.scene_day_above_mean <= settings.scene_night_below_mean:
            raise ConfigurationError("day illumination threshold must exceed night threshold")
        if settings.scene_day_above_mean > 255:
            raise ConfigurationError("illumination thresholds must be in image range")
        if settings.segmentation_expected_height_y > 1:
            raise ConfigurationError("expected height ratio must be <= 1")
        if settings.segmentation_max_expected_height_deviation_ratio > 1:
            raise ConfigurationError(
                "maximum expected-height deviation ratio must be <= 1"
            )
        if settings.segmentation_min_confidence > 1:
            raise ConfigurationError("segmentation minimum confidence must be <= 1")
        for name, value in (
            ("min_free_profile_fraction", settings.segmentation_min_free_profile_fraction),
            ("max_interpolation_gap_ratio", settings.segmentation_max_interpolation_gap_ratio),
            (
                "max_temporal_height_change_ratio",
                settings.segmentation_max_temporal_height_change_ratio,
            ),
            ("max_height_iqr_ratio", settings.segmentation_max_height_iqr_ratio),
            ("occlusion_margin_ratio", settings.segmentation_occlusion_margin_ratio),
            ("height_ema_alpha", settings.segmentation_height_ema_alpha),
            ("min_pair_contrast", settings.segmentation_min_pair_contrast),
            (
                "min_paired_support_fraction",
                settings.segmentation_min_paired_support_fraction,
            ),
            (
                "min_contiguous_support_fraction",
                settings.segmentation_min_contiguous_support_fraction,
            ),
            (
                "observed_coverage_fraction",
                settings.segmentation_observed_coverage_fraction,
            ),
            (
                "min_current_support_for_estimate",
                settings.segmentation_min_current_support_for_estimate,
            ),
            ("profile_ema_alpha", settings.segmentation_profile_ema_alpha),
            ("max_profile_shift_ratio", settings.segmentation_max_profile_shift_ratio),
            (
                "max_profile_residual_ratio",
                settings.segmentation_max_profile_residual_ratio,
            ),
            ("max_saturated_fraction", settings.segmentation_max_saturated_fraction),
        ):
            if value > 1:
                raise ConfigurationError(f"segmentation {name} must be <= 1")
        if (
            settings.segmentation_observed_coverage_fraction
            < settings.segmentation_min_current_support_for_estimate
        ):
            raise ConfigurationError(
                "observed coverage must be >= minimum temporal-estimate support"
            )
        if len(settings.learned_revision) != 40 or any(
            character not in "0123456789abcdef"
            for character in settings.learned_revision.casefold()
        ):
            raise ConfigurationError("learned revision must be a 40-character Git SHA")
        if len(settings.learned_weight_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in settings.learned_weight_sha256.casefold()
        ):
            raise ConfigurationError("learned weight checksum must be SHA-256")
        if settings.learned_min_confidence > 1:
            raise ConfigurationError("learned minimum confidence must be <= 1")
        if settings.learned_device not in {"auto", "cpu", "cuda"}:
            raise ConfigurationError("learned device must be auto, cpu or cuda")
        if settings.proximity_scale_mode not in {"fixed_pixels", "vehicle_reference"}:
            raise ConfigurationError(
                "proximity scale_mode must be fixed_pixels or vehicle_reference"
            )
        if len(settings.semantic_revision) != 40 or any(
            character not in "0123456789abcdef"
            for character in settings.semantic_revision.casefold()
        ):
            raise ConfigurationError(
                "semantic detection revision must be a 40-character Git SHA"
            )
        if len(settings.semantic_weight_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in settings.semantic_weight_sha256.casefold()
        ):
            raise ConfigurationError(
                "semantic detection weight checksum must be SHA-256"
            )
        if settings.semantic_device not in {"auto", "cpu", "cuda"}:
            raise ConfigurationError(
                "semantic detection device must be auto, cpu or cuda"
            )
        for name, value in (
            ("box_threshold", settings.semantic_box_threshold),
            ("text_threshold", settings.semantic_text_threshold),
            ("nms_iou_threshold", settings.semantic_nms_iou_threshold),
            ("min_area_ratio", settings.semantic_min_area_ratio),
            ("max_area_ratio", settings.semantic_max_area_ratio),
            ("min_width_ratio", settings.semantic_min_width_ratio),
            ("min_height_ratio", settings.semantic_min_height_ratio),
        ):
            if value > 1:
                raise ConfigurationError(
                    f"semantic detection {name} must be <= 1"
                )
        if settings.semantic_max_area_ratio <= settings.semantic_min_area_ratio:
            raise ConfigurationError(
                "semantic detection area ratios are invalid"
            )
        if settings.semantic_input_long_edge < settings.semantic_input_short_edge:
            raise ConfigurationError(
                "semantic input_long_edge must be >= input_short_edge"
            )
        if not 0 < settings.detection_max_width_ratio <= 1:
            raise ConfigurationError("detection max_width_ratio must be <= 1")
        if not 0 < settings.detection_max_height_ratio <= 1:
            raise ConfigurationError("detection max_height_ratio must be <= 1")
        if settings.detection_max_aspect_ratio <= settings.detection_min_aspect_ratio:
            raise ConfigurationError("detection aspect ratio bounds are invalid")
        if settings.detection_min_vertical_center_ratio >= 1:
            raise ConfigurationError("detection minimum vertical center must be < 1")
        if not (
            settings.detection_min_vertical_center_ratio
            < settings.detection_max_vertical_center_ratio
            < 1
        ):
            raise ConfigurationError("detection vertical center bounds are invalid")
        if settings.detection_min_motion_fill > 1:
            raise ConfigurationError("detection minimum motion fill must be <= 1")
        if settings.detection_max_night_bright_fraction > 1:
            raise ConfigurationError("detection night bright fraction must be <= 1")
        if settings.detection_min_confidence > 1:
            raise ConfigurationError("detection minimum confidence must be <= 1")
        if settings.tracking_max_center_distance_ratio > 1:
            raise ConfigurationError("tracking maximum center distance must be <= 1")
        if settings.tracking_min_iou > 1:
            raise ConfigurationError("tracking minimum IoU must be <= 1")
        if settings.tracking_velocity_alpha > 1:
            raise ConfigurationError("tracking velocity alpha must be <= 1")
        if settings.tracking_max_size_change_ratio <= 1:
            raise ConfigurationError("tracking maximum size change ratio must exceed 1")
        return settings
