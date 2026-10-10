"""Defensible pixel geometry for berm masks and vehicle contact points."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class BoundaryGeometry:
    crest_segments: tuple[np.ndarray, ...]
    toe_segments: tuple[np.ndarray, ...]
    visible_columns: int
    excluded_columns: int
    confidence: float
    diagnostics: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class PixelMeasurement:
    distance_px: Optional[float]
    height_px: Optional[float]
    vehicle_ground_point: Optional[tuple[float, float]]
    confidence: float
    diagnostics: tuple[str, ...] = field(default_factory=tuple)


def _segments(points: np.ndarray, valid: np.ndarray, min_points: int = 2) -> tuple[np.ndarray, ...]:
    indices = np.flatnonzero(valid)
    if indices.size == 0:
        return ()
    cuts = np.flatnonzero(np.diff(indices) > 1) + 1
    groups = np.split(indices, cuts)
    return tuple(points[g].astype(np.float64) for g in groups if len(g) >= min_points)


def extract_boundaries(
    berm_mask: np.ndarray,
    exclusion_mask: np.ndarray | None = None,
    min_column_height_px: int = 2,
) -> BoundaryGeometry:
    mask = np.asarray(berm_mask, dtype=bool)
    if mask.ndim != 2 or not mask.any():
        return BoundaryGeometry((), (), 0, 0, 0.0, ("empty_berm_mask",))
    h, w = mask.shape
    excluded = np.zeros_like(mask, dtype=bool)
    if exclusion_mask is not None:
        excluded = np.asarray(exclusion_mask, dtype=bool)
        if excluded.shape != mask.shape:
            raise ValueError("exclusion_mask shape must match berm_mask")

    xs = np.arange(w)
    has = mask.any(axis=0)
    crest_y = np.argmax(mask, axis=0)
    toe_y = h - 1 - np.argmax(mask[::-1], axis=0)
    sufficient = has & ((toe_y - crest_y + 1) >= int(min_column_height_px))
    occluded_columns = excluded.any(axis=0)
    valid = sufficient & ~occluded_columns
    crest = np.column_stack((xs, crest_y))
    toe = np.column_stack((xs, toe_y))
    visible = int(valid.sum())
    possible = max(1, int(sufficient.sum()))
    confidence = float(np.clip(visible / possible, 0.0, 1.0))
    diagnostics = () if visible else ("no_defensible_visible_columns",)
    return BoundaryGeometry(
        _segments(crest, valid),
        _segments(toe, valid),
        visible,
        int((sufficient & occluded_columns).sum()),
        confidence,
        diagnostics,
    )


def vehicle_ground_point(mask: np.ndarray | None, bbox: tuple[float, float, float, float], image_shape: tuple[int, int]) -> tuple[tuple[float, float] | None, str]:
    h, w = image_shape
    if mask is not None:
        candidate = np.asarray(mask, dtype=bool)
        if candidate.shape == (h, w) and candidate.any():
            ys, xs = np.nonzero(candidate)
            bottom = int(ys.max())
            bottom_x = xs[ys >= max(0, bottom - 1)]
            point = (float(np.median(bottom_x)), float(bottom))
            if 0 <= point[0] < w and 0 <= point[1] < h:
                return point, "mask_bottom_center"
    x1, _, x2, y2 = map(float, bbox)
    point = ((x1 + x2) / 2.0, min(float(h - 1), y2))
    if 0 <= point[0] < w and 0 <= point[1] < h:
        return point, "bbox_bottom_center_inferred"
    return None, "ground_point_outside_image"


def point_to_segments_distance(point: tuple[float, float], segments: tuple[np.ndarray, ...]) -> float | None:
    p = np.asarray(point, dtype=np.float64)
    best = np.inf
    for segment in segments:
        if len(segment) < 2:
            continue
        a, b = segment[:-1], segment[1:]
        direction = b - a
        denominator = np.einsum("ij,ij->i", direction, direction)
        valid = denominator > 1e-12
        if not valid.any():
            continue
        t = np.zeros_like(denominator)
        t[valid] = np.clip(np.einsum("ij,ij->i", p - a[valid], direction[valid]) / denominator[valid], 0.0, 1.0)
        projected = a + t[:, None] * direction
        best = min(best, float(np.linalg.norm(projected - p, axis=1).min()))
    return None if not np.isfinite(best) else best


def visible_height_px(boundary: BoundaryGeometry) -> float | None:
    if not boundary.crest_segments or not boundary.toe_segments:
        return None
    crest = {int(x): float(y) for segment in boundary.crest_segments for x, y in segment}
    toe = {int(x): float(y) for segment in boundary.toe_segments for x, y in segment}
    values = [abs(toe[x] - y) for x, y in crest.items() if x in toe]
    return float(np.median(values)) if values else None


def measure_pixels(
    berm_mask: np.ndarray,
    vehicle_mask: np.ndarray | None,
    bbox: tuple[float, float, float, float],
    exclusion_mask: np.ndarray | None = None,
) -> PixelMeasurement:
    h, w = np.asarray(berm_mask).shape
    combined = np.zeros((h, w), dtype=bool)
    if exclusion_mask is not None:
        combined |= np.asarray(exclusion_mask, dtype=bool)
    if vehicle_mask is not None and np.asarray(vehicle_mask).shape == (h, w):
        combined |= np.asarray(vehicle_mask, dtype=bool)
    boundary = extract_boundaries(berm_mask, combined)
    point, source = vehicle_ground_point(vehicle_mask, bbox, (h, w))
    if point is None:
        return PixelMeasurement(None, visible_height_px(boundary), None, 0.0, (source,))
    distance = point_to_segments_distance(point, boundary.toe_segments)
    diagnostics = list(boundary.diagnostics)
    diagnostics.append(source)
    if distance is None:
        diagnostics.append("no_defensible_toe_segment")
    return PixelMeasurement(distance, visible_height_px(boundary), point, boundary.confidence, tuple(diagnostics))
