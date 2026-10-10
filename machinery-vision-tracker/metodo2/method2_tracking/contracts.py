"""Typed contracts for Sprint 5.1 temporal tracking."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import numpy as np

BBox = tuple[float, float, float, float]

class ObservationStatus(str, Enum):
    OBSERVED = "observed"
    TEMPORAL_ESTIMATE = "temporal_estimate"
    HISTORICAL_REFERENCE = "historical_reference"
    UNKNOWN = "unknown"

class TrackLifecycle(str, Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"
    REMOVED = "removed"

class PositionSource(str, Enum):
    OBSERVED = "observed"
    PREDICTED = "predicted"

@dataclass(frozen=True)
class VehicleDetection:
    bbox: BBox
    mask: Optional[np.ndarray] = None
    class_name: str = "vehicle"
    confidence: float = 1.0
    source: str = "detector"
    def __post_init__(self) -> None:
        x1,y1,x2,y2=(float(v) for v in self.bbox)
        if not all(np.isfinite([x1,y1,x2,y2])) or x2<=x1 or y2<=y1:
            raise ValueError("bbox must contain finite x1,y1,x2,y2 with positive area")
        if not np.isfinite(self.confidence) or not 0.0<=float(self.confidence)<=1.0:
            raise ValueError("confidence must be in [0,1]")
        if self.mask is not None:
            m=np.asarray(self.mask)
            if m.ndim!=2 or m.size==0:
                raise ValueError("mask must be a non-empty HxW array")

Detection=VehicleDetection

@dataclass
class TrackResult:
    track_id: int
    state: TrackLifecycle
    bbox: BBox
    position_source: PositionSource
    confidence: float
    class_name: str
    age_s: float
    age_since_observation_s: float
    observed_bbox: Optional[BBox]=None
    predicted_bbox: Optional[BBox]=None
    observed_mask: Optional[np.ndarray]=None
    propagated_mask: Optional[np.ndarray]=None
    observation_status: ObservationStatus=ObservationStatus.UNKNOWN
    diagnostics: dict[str,Any]=field(default_factory=dict)

TrackOutput=TrackResult

@dataclass(frozen=True)
class DistanceRecord:
    track_id: int
    frame_index: int
    timestamp: float
    observation_status: ObservationStatus
    calibration_status: str
    distance_px: Optional[float]
    distance_m: Optional[float]
    source: str="current_geometry"
    confidence: float=0.0
    diagnostics: tuple[str,...]=()

@dataclass
class BermState:
    observation_status: ObservationStatus
    confidence: float
    age_since_observation_s: float
    source: str
    geometry: Any=None
    diagnostics: list[str]=field(default_factory=list)

@dataclass
class TemporalBermState:
    observation_status: ObservationStatus|str
    mask: Optional[np.ndarray]
    confidence: float
    age_since_observation_s: Optional[float]
    source: str
    diagnostics: dict[str,Any]=field(default_factory=dict)

@dataclass
class FusionResult:
    frame_index: int
    timestamp: float
    track_id: int
    track_state: TrackLifecycle
    position_source: PositionSource
    vehicle_bbox: BBox
    vehicle_mask: Optional[np.ndarray]
    berm_state: ObservationStatus
    berm_age_s: float
    distance_px: Optional[float]
    distance_m: Optional[float]
    calibration_status: str
    confidence: float
    reason: str
    diagnostics: list[str]=field(default_factory=list)

@dataclass
class FusionOutput:
    frame_index: int
    track: TrackResult
    berm: Any
    distance_px: Optional[float]
    distance_m: Optional[float]
    calibration_status: str
    confidence: float
    diagnostics: dict[str,Any]=field(default_factory=dict)
