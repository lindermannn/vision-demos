"""Typed boundaries shared by pipeline components."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Sequence, Tuple


Point = Tuple[float, float]


class MethodSelection(str, Enum):
    METHOD_1 = "1"
    METHOD_2 = "2"
    ALL = "all"


class ObservationStatus(str, Enum):
    OBSERVED = "observed"
    TEMPORAL_ESTIMATE = "temporal_estimate"
    HISTORICAL_REFERENCE = "historical_reference"
    UNKNOWN = "unknown"

    # Compatibility aliases for internal callers written before the G5.1 contract.
    VALID = "observed"
    OCCLUDED = "temporal_estimate"
    INVALID = "unknown"


class ProximityLevel(str, Enum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class FrameContext:
    video_id: str
    scene_id: int
    frame_index: int
    timestamp_s: float
    width: int
    height: int

    def __post_init__(self) -> None:
        if not self.video_id:
            raise ValueError("video_id must not be empty")
        if self.scene_id < 0 or self.frame_index < 0:
            raise ValueError("scene_id and frame_index must be non-negative")
        if self.timestamp_s < 0:
            raise ValueError("timestamp_s must be non-negative")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("frame dimensions must be positive")


@dataclass(frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError("bounding-box maximums must not be below minimums")


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    bbox: BoundingBox
    track_id: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("detection label must not be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if self.track_id is not None and self.track_id <= 0:
            raise ValueError("track_id must be positive when present")


@dataclass(frozen=True)
class BermObservation:
    context: FrameContext
    status: ObservationStatus
    crest: Sequence[Point] = field(default_factory=tuple)
    base: Sequence[Point] = field(default_factory=tuple)
    height_px: Optional[float] = None
    height_m_estimated: Optional[float] = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


class BermSegmenter(Protocol):
    name: str

    def segment(self, frame: Any, context: FrameContext) -> BermObservation:
        """Return one observation without mutating the input frame."""


class VehicleDetector(Protocol):
    name: str

    def detect(self, frame: Any, context: FrameContext) -> Sequence[Detection]:
        """Return detections for one frame."""


class ArtifactWriter(Protocol):
    def video_method_directory(
        self, output_root: Path, video_id: str, method: MethodSelection
    ) -> Path:
        """Resolve an isolated destination without writing it."""
