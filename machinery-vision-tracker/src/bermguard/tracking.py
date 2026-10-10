"""Causal constant-velocity tracker with conservative visual identities."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from .contracts import BoundingBox, Detection, FrameContext
from .g2_config import G2Settings


@dataclass
class _Track:
    track_id: int
    bbox: BoundingBox
    hits: int
    consecutive_hits: int
    age: int
    missed: int
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    confirmed: bool = False


def _center(box: BoundingBox) -> tuple[float, float]:
    return ((box.x1 + box.x2) / 2.0, (box.y1 + box.y2) / 2.0)


def _area(box: BoundingBox) -> float:
    return max(0.0, box.x2 - box.x1) * max(0.0, box.y2 - box.y1)


def _iou(first: BoundingBox, second: BoundingBox) -> float:
    x1, y1 = max(first.x1, second.x1), max(first.y1, second.y1)
    x2, y2 = min(first.x2, second.x2), min(first.y2, second.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = _area(first) + _area(second) - intersection
    return intersection / union if union > 0 else 0.0


def _predicted_center(track: _Track) -> tuple[float, float]:
    center_x, center_y = _center(track.bbox)
    horizon = min(track.missed + 1, 3)
    return (
        center_x + track.velocity_x * horizon,
        center_y + track.velocity_y * horizon,
    )


class CausalTracker:
    def __init__(self, settings: G2Settings) -> None:
        self.settings = settings
        self.reset()

    def reset(self) -> None:
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1
        self._created_tracks = 0
        self._confirmed_track_ids: set[int] = set()
        self._retired_short_tracks = 0
        self._associations = 0

    def diagnostics(self) -> dict[str, int | float]:
        lifetimes = [track.age for track in self._tracks.values() if track.confirmed]
        return {
            "created_tracks": self._created_tracks,
            "confirmed_tracks": len(self._confirmed_track_ids),
            "retired_short_tracks": self._retired_short_tracks,
            "associations": self._associations,
            "active_tracks": len(self._tracks),
            "active_confirmed_tracks": sum(
                1 for track in self._tracks.values() if track.confirmed
            ),
            "active_confirmed_mean_age": (
                sum(lifetimes) / len(lifetimes) if lifetimes else 0.0
            ),
        }

    def update(
        self, detections: Sequence[Detection], context: FrameContext
    ) -> tuple[Detection, ...]:
        diagonal = math.hypot(context.width, context.height)
        maximum_distance = diagonal * self.settings.tracking_max_center_distance_ratio
        candidates: list[tuple[float, int, int]] = []
        for track_id, track in self._tracks.items():
            predicted_x, predicted_y = _predicted_center(track)
            track_area = max(_area(track.bbox), 1.0)
            for detection_index, detection in enumerate(detections):
                detection_x, detection_y = _center(detection.bbox)
                distance = math.hypot(
                    predicted_x - detection_x,
                    predicted_y - detection_y,
                )
                overlap = _iou(track.bbox, detection.bbox)
                detection_area = max(_area(detection.bbox), 1.0)
                size_change = max(track_area, detection_area) / min(
                    track_area, detection_area
                )
                if size_change > self.settings.tracking_max_size_change_ratio:
                    continue
                if (
                    distance > maximum_distance
                    or (
                        overlap < self.settings.tracking_min_iou
                        and distance > diagonal * 0.020
                    )
                ):
                    continue
                score = (
                    distance / max(maximum_distance, 1.0)
                    + 0.45 * (1.0 - overlap)
                    + 0.15 * abs(math.log(size_change))
                )
                candidates.append((score, track_id, detection_index))
        candidates.sort()

        assigned_tracks: set[int] = set()
        assigned_detections: set[int] = set()
        detection_to_track: dict[int, int] = {}
        for _, track_id, detection_index in candidates:
            if track_id in assigned_tracks or detection_index in assigned_detections:
                continue
            assigned_tracks.add(track_id)
            assigned_detections.add(detection_index)
            detection_to_track[detection_index] = track_id

        for track_id, track in list(self._tracks.items()):
            track.age += 1
            if track_id not in assigned_tracks:
                track.missed += 1
                track.consecutive_hits = 0
                if track.missed > self.settings.tracking_max_missed_frames:
                    if not track.confirmed:
                        self._retired_short_tracks += 1
                    del self._tracks[track_id]

        for detection_index, detection in enumerate(detections):
            track_id = detection_to_track.get(detection_index)
            if track_id is None:
                track_id = self._next_id
                self._next_id += 1
                self._created_tracks += 1
                self._tracks[track_id] = _Track(
                    track_id=track_id,
                    bbox=detection.bbox,
                    hits=1,
                    consecutive_hits=1,
                    age=1,
                    missed=0,
                )
                detection_to_track[detection_index] = track_id
                continue

            track = self._tracks[track_id]
            old_x, old_y = _center(track.bbox)
            new_x, new_y = _center(detection.bbox)
            alpha = self.settings.tracking_velocity_alpha
            track.velocity_x = alpha * (new_x - old_x) + (1.0 - alpha) * track.velocity_x
            track.velocity_y = alpha * (new_y - old_y) + (1.0 - alpha) * track.velocity_y
            track.bbox = detection.bbox
            track.hits += 1
            track.consecutive_hits += 1
            track.missed = 0
            self._associations += 1

        tracked: list[Detection] = []
        for detection_index, detection in enumerate(detections):
            track_id = detection_to_track[detection_index]
            track = self._tracks[track_id]
            if (
                not track.confirmed
                and track.consecutive_hits >= self.settings.tracking_minimum_hits
            ):
                track.confirmed = True
                self._confirmed_track_ids.add(track_id)
            if track.confirmed:
                tracked.append(
                    Detection(
                        label=detection.label,
                        confidence=detection.confidence,
                        bbox=detection.bbox,
                        track_id=track_id,
                    )
                )
        return tuple(tracked)
