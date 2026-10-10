"""Isolated fixed-camera ground-plane proximity experiment.

This module changes only proximity analytics.  Berm extraction and its fixed
camera validator are untouched.  Metric thresholds remain the assessment's
literal <10 m / <=20 m / >20 m contract.
"""
from __future__ import annotations

from collections import defaultdict, deque
import math
from statistics import median

from bermguard.contracts import Detection, ProximityLevel
from bermguard.proximity import ProximityEngine


class GroundPlaneProximityEngine(ProximityEngine):
    """Estimate separation between ground contacts with a pinhole ground model."""

    ASSUMED_HORIZONTAL_FOV_DEG = 70.0
    HEIGHT_HISTORY = 7
    SAFETY_ENVELOPE_PER_VEHICLE_M = 0.75
    MAX_DEPTH_GAP_FOR_ENVELOPE_M = 12.0

    def reset(self) -> None:
        super().reset()
        self._height_history = defaultdict(lambda: deque(maxlen=self.HEIGHT_HISTORY))
        self._frame_width = 1

    def reset_scene_state(self) -> None:
        super().reset_scene_state()
        self._height_history.clear()

    def update(self, detections, context):
        self._frame_width = context.width
        for detection in detections:
            if detection.track_id is not None:
                height = max(1.0, detection.bbox.y2 - detection.bbox.y1)
                self._height_history[detection.track_id].append(height)
        samples, levels = super().update(detections, context)
        # CamVision's requested presentation has only the semaphore states.
        levels = {
            track_id: ProximityLevel.GREEN if level is ProximityLevel.UNKNOWN else level
            for track_id, level in levels.items()
        }
        return samples, levels

    def _stable_height(self, detection: Detection) -> float:
        current = max(1.0, detection.bbox.y2 - detection.bbox.y1)
        if detection.track_id is None:
            return current
        history = self._height_history.get(detection.track_id)
        return float(median(history)) if history else current

    def _ground_point_m(self, detection: Detection) -> tuple[float, float]:
        """Return lateral/depth coordinates in estimated metres.

        With known reference object height H and image height h, pinhole
        geometry gives X=H*(x-cx)/h and Z=f*H/h.  Absolute accuracy remains an
        estimate because FOV and vehicle height are assumptions, but unlike the
        old isotropic pixel distance it handles vehicles at different depths.
        """
        box = detection.bbox
        height_px = self._stable_height(detection)
        reference_height_m = self.settings.proximity_reference_vehicle_height_m
        center_x = 0.5 * (box.x1 + box.x2)
        principal_x = 0.5 * self._frame_width
        focal_px = self._frame_width / (
            2.0 * math.tan(math.radians(self.ASSUMED_HORIZONTAL_FOV_DEG) / 2.0)
        )
        lateral_m = reference_height_m * (center_x - principal_x) / height_px
        depth_m = focal_px * reference_height_m / height_px
        return lateral_m, depth_m

    def _estimated_distance_m(self, distance_px, first, second):
        first_x, first_z = self._ground_point_m(first)
        second_x, second_z = self._ground_point_m(second)
        depth_gap = abs(first_z - second_z)
        center_distance = math.hypot(first_x - second_x, first_z - second_z)
        # Report estimated clearance, not centre-to-centre distance.  Apply the
        # small physical envelope only when both objects occupy a comparable
        # depth plane; perspective-separated vehicles retain the v3 result.
        if depth_gap <= self.MAX_DEPTH_GAP_FOR_ENVELOPE_M:
            center_distance -= 2.0 * self.SAFETY_ENVELOPE_PER_VEHICLE_M
        return max(0.0, center_distance)
