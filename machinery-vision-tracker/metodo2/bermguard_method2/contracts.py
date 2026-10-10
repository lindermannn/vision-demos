"""Typed integration contracts. No model inference is performed here."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class ObservationStatus(str, Enum):
    OBSERVED = "observed"
    TEMPORAL_ESTIMATE = "temporal_estimate"
    HISTORICAL_REFERENCE = "historical_reference"
    UNKNOWN = "unknown"


class CalibrationStatus(str, Enum):
    CALIBRATED = "calibrated"
    ESTIMATED = "estimated"
    PIXEL_ONLY = "pixel_only"
    UNAVAILABLE = "unavailable"
    INVALID = "invalid"


@dataclass(frozen=True)
class VehicleObservation:
    frame_index: int
    timestamp_s: float
    bbox: tuple[float, float, float, float]
    confidence: float
    class_name: str = "vehicle"
    mask: Optional[np.ndarray] = None
    source: str = "yolo_seg"

    def __post_init__(self) -> None:
        if self.frame_index < 0 or self.timestamp_s < 0:
            raise ValueError("frame_index and timestamp_s must be non-negative")
        if len(self.bbox) != 4 or not np.isfinite(self.bbox).all():
            raise ValueError("bbox must contain four finite values")
        x1, y1, x2, y2 = self.bbox
        if x2 <= x1 or y2 <= y1:
            raise ValueError("bbox must have positive area")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.mask is not None and (np.asarray(self.mask).ndim != 2 or not np.asarray(self.mask).any()):
            raise ValueError("vehicle mask must be a non-empty 2-D array")


@dataclass(frozen=True)
class BermObservation:
    frame_index: int
    timestamp_s: float
    mask: Optional[np.ndarray]
    observation_status: ObservationStatus
    confidence: float
    source: str
    approved: bool = False
    diagnostics: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.frame_index < 0 or self.timestamp_s < 0:
            raise ValueError("frame_index and timestamp_s must be non-negative")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.mask is not None and np.asarray(self.mask).ndim != 2:
            raise ValueError("berm mask must be 2-D")
        if self.observation_status == ObservationStatus.OBSERVED and not self.approved:
            raise ValueError("observed berm masks must be explicitly approved")
        if self.approved and self.mask is None:
            raise ValueError("approved berm observation requires a mask")


@dataclass(frozen=True)
class FrameIntegrationResult:
    frame_index: int
    timestamp_s: float
    track_id: int
    track_state: str
    observation_status: ObservationStatus
    calibration_status: CalibrationStatus
    distance_px: Optional[float]
    distance_m: Optional[float]
    height_px: Optional[float]
    height_m: Optional[float]
    confidence: float
    diagnostics: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.calibration_status not in (CalibrationStatus.CALIBRATED, CalibrationStatus.ESTIMATED):
            if self.distance_m is not None or self.height_m is not None:
                raise ValueError("metric values require calibrated or explicitly estimated status")
        if self.observation_status in (ObservationStatus.HISTORICAL_REFERENCE, ObservationStatus.UNKNOWN):
            if self.distance_px is not None or self.distance_m is not None:
                raise ValueError("historical/unknown berm state cannot publish current distance")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
