"""Pairwise proximity with stable states and explicit unavailable distance."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Sequence

from .contracts import Detection, FrameContext, ProximityLevel
from .g2_config import G2Settings
from .geometry import PixelScale, bbox_bottom_center, classify_proximity


@dataclass(frozen=True)
class ProximitySample:
    scene_id: int
    frame_index: int
    timestamp_s: float
    track_id_a: int
    track_id_b: int
    distance_px: float
    distance_m_estimated: float
    level: ProximityLevel


@dataclass
class _PairState:
    candidate: ProximityLevel = ProximityLevel.UNKNOWN
    consecutive: int = 0
    stable: ProximityLevel = ProximityLevel.UNKNOWN
    active_event_id: Optional[int] = None


class ProximityEngine:
    def __init__(self, settings: G2Settings) -> None:
        self.settings = settings
        self.scale = PixelScale(settings.pixels_per_meter, settings.scale_source)
        self.reset()

    def _estimated_distance_m(
        self, distance_px: float, first: Detection, second: Detection
    ) -> float:
        if self.settings.proximity_scale_mode == "fixed_pixels":
            return self.scale.to_estimated_meters(distance_px)
        first_height = max(1.0, first.bbox.y2 - first.bbox.y1)
        second_height = max(1.0, second.bbox.y2 - second.bbox.y1)
        reference_height = self.settings.proximity_reference_vehicle_height_m
        first_pixels_per_meter = first_height / reference_height
        second_pixels_per_meter = second_height / reference_height
        pair_pixels_per_meter = math.sqrt(
            first_pixels_per_meter * second_pixels_per_meter
        )
        return distance_px / max(pair_pixels_per_meter, 1e-6)

    def reset(self) -> None:
        self._states: dict[tuple[int, int, int], _PairState] = {}
        self._minimums: dict[tuple[int, int, int], ProximitySample] = {}
        self._events: list[dict[str, object]] = []
        self._active_events: dict[int, dict[str, object]] = {}
        self._next_event_id = 1

    def reset_scene_state(self) -> None:
        """Close active scene-local events while retaining run-wide evidence."""

        self._events.extend(self._active_events.values())
        self._active_events.clear()
        self._states.clear()

    def update(
        self, detections: Sequence[Detection], context: FrameContext
    ) -> tuple[tuple[ProximitySample, ...], dict[int, ProximityLevel]]:
        samples: list[ProximitySample] = []
        colours = {
            detection.track_id: ProximityLevel.UNKNOWN
            for detection in detections
            if detection.track_id is not None
        }
        for first_index, first in enumerate(detections):
            if first.track_id is None:
                continue
            if self._is_border_truncated(first, context):
                continue
            for second in detections[first_index + 1 :]:
                if second.track_id is None:
                    continue
                if self._is_border_truncated(second, context):
                    continue
                first_point = bbox_bottom_center(first.bbox)
                second_point = bbox_bottom_center(second.bbox)
                distance_px = math.hypot(
                    first_point[0] - second_point[0],
                    first_point[1] - second_point[1],
                )
                distance_m = self._estimated_distance_m(
                    distance_px, first, second
                )
                level = classify_proximity(
                    distance_m,
                    self.settings.red_below_m,
                    self.settings.yellow_at_or_below_m,
                )
                id_a, id_b = sorted((first.track_id, second.track_id))
                sample = ProximitySample(
                    scene_id=context.scene_id,
                    frame_index=context.frame_index,
                    timestamp_s=context.timestamp_s,
                    track_id_a=id_a,
                    track_id_b=id_b,
                    distance_px=distance_px,
                    distance_m_estimated=distance_m,
                    level=level,
                )
                samples.append(sample)
                key = (context.scene_id, id_a, id_b)
                previous_minimum = self._minimums.get(key)
                if previous_minimum is None or sample.distance_px < previous_minimum.distance_px:
                    self._minimums[key] = sample
                stable_level = self._update_state(key, sample)
                colours[first.track_id] = self._more_severe(
                    colours[first.track_id], stable_level
                )
                colours[second.track_id] = self._more_severe(
                    colours[second.track_id], stable_level
                )
        return tuple(samples), colours

    @staticmethod
    def _is_border_truncated(
        detection: Detection, context: FrameContext
    ) -> bool:
        margin = 1.0
        box = detection.bbox
        return (
            box.x1 <= margin
            or box.y1 <= margin
            or box.x2 >= context.width - margin
            or box.y2 >= context.height - margin
        )

    def _update_state(
        self, key: tuple[int, int, int], sample: ProximitySample
    ) -> ProximityLevel:
        state = self._states.setdefault(key, _PairState())
        if sample.level is state.candidate:
            state.consecutive += 1
        else:
            state.candidate = sample.level
            state.consecutive = 1
        required = (
            self.settings.event_recovery_frames
            if state.stable is ProximityLevel.RED and sample.level is not ProximityLevel.RED
            else self.settings.event_persistence_frames
        )
        if state.consecutive < required or state.stable is sample.level:
            if state.active_event_id is not None:
                active = self._active_events[state.active_event_id]
                active["end_frame"] = sample.frame_index
                active["end_s"] = sample.timestamp_s
                active["minimum_distance_px"] = min(
                    float(active["minimum_distance_px"]), sample.distance_px
                )
                active["minimum_distance_m_estimated"] = min(
                    float(active["minimum_distance_m_estimated"]),
                    sample.distance_m_estimated,
                )
            return state.stable

        previous = state.stable
        state.stable = sample.level
        if sample.level is ProximityLevel.RED and previous is not ProximityLevel.RED:
            event_id = self._next_event_id
            self._next_event_id += 1
            event = {
                "event_id": event_id,
                "scene_id": sample.scene_id,
                "track_id_a": sample.track_id_a,
                "track_id_b": sample.track_id_b,
                "start_frame": sample.frame_index,
                "end_frame": sample.frame_index,
                "start_s": max(0.0, sample.timestamp_s),
                "end_s": sample.timestamp_s,
                "minimum_distance_px": sample.distance_px,
                "minimum_distance_m_estimated": sample.distance_m_estimated,
                "level": "red",
            }
            self._active_events[event_id] = event
            state.active_event_id = event_id
        elif previous is ProximityLevel.RED and state.active_event_id is not None:
            event_id = state.active_event_id
            self._events.append(self._active_events.pop(event_id))
            state.active_event_id = None
        return state.stable

    @staticmethod
    def _more_severe(
        first: ProximityLevel, second: ProximityLevel
    ) -> ProximityLevel:
        rank = {
            ProximityLevel.UNKNOWN: 0,
            ProximityLevel.GREEN: 1,
            ProximityLevel.YELLOW: 2,
            ProximityLevel.RED: 3,
        }
        return second if rank[second] > rank[first] else first

    def minimum_rows(self) -> list[dict[str, object]]:
        return [
            {
                "scene_id": sample.scene_id,
                "track_id_a": sample.track_id_a,
                "track_id_b": sample.track_id_b,
                "minimum_distance_px": sample.distance_px,
                "minimum_distance_m_estimated": sample.distance_m_estimated,
                "proximity_level": sample.level.value,
                "frame_index": sample.frame_index,
                "timestamp_s": sample.timestamp_s,
            }
            for _, sample in sorted(self._minimums.items())
        ]

    def event_rows(self) -> list[dict[str, object]]:
        completed = list(self._events)
        completed.extend(self._active_events.values())
        return sorted(completed, key=lambda event: int(event["event_id"]))
