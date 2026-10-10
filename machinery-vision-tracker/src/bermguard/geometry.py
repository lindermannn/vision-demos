"""Small, unit-explicit geometry primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from .contracts import BoundingBox, ProximityLevel


@dataclass(frozen=True)
class PixelScale:
    pixels_per_meter: float
    source: str = "assumed_pixel_ratio"

    def __post_init__(self) -> None:
        if not math.isfinite(self.pixels_per_meter) or self.pixels_per_meter <= 0:
            raise ValueError("pixels_per_meter must be finite and positive")
        if not self.source:
            raise ValueError("scale source must not be empty")

    def to_estimated_meters(self, pixels: float) -> float:
        if not math.isfinite(pixels) or pixels < 0:
            raise ValueError("pixel distance must be finite and non-negative")
        return pixels / self.pixels_per_meter


def bbox_bottom_center(bbox: BoundingBox) -> Tuple[float, float]:
    return ((bbox.x1 + bbox.x2) / 2.0, bbox.y2)


def classify_proximity(
    distance_m_estimated: Optional[float],
    red_below_m: float = 10.0,
    yellow_at_or_below_m: float = 20.0,
) -> ProximityLevel:
    if red_below_m <= 0 or yellow_at_or_below_m <= red_below_m:
        raise ValueError("proximity thresholds must be positive and ordered")
    if distance_m_estimated is None:
        return ProximityLevel.UNKNOWN
    if not math.isfinite(distance_m_estimated) or distance_m_estimated < 0:
        raise ValueError("distance must be finite and non-negative")
    if distance_m_estimated < red_below_m:
        return ProximityLevel.RED
    if distance_m_estimated <= yellow_at_or_below_m:
        return ProximityLevel.YELLOW
    return ProximityLevel.GREEN
