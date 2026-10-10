"""Read-only adapter around the hardened Sprint 5.1 tracker."""
from __future__ import annotations

from typing import Any

import numpy as np

def _load_tracker():
    from method2_tracking.contracts import VehicleDetection
    from method2_tracking.vehicle_tracker import VehicleTracker

    return VehicleDetection, VehicleTracker


class TrackerAdapter:
    def __init__(self, fps: float, **kwargs: Any):
        detection_type, tracker_type = _load_tracker()
        self._detection_type = detection_type
        self._tracker = tracker_type(fps=fps, **kwargs)

    def update(self, detections: list[dict[str, Any]], dt: float, camera_discontinuity: bool = False):
        typed = [
            self._detection_type(
                bbox=tuple(float(value) for value in item["bbox"]),
                mask=None if item.get("mask") is None else np.asarray(item["mask"], dtype=bool),
                class_name=str(item.get("class_name", "vehicle")),
                confidence=float(item.get("confidence", 0.0)),
                source=str(item.get("source", "yolo_seg")),
            )
            for item in detections
        ]
        return self._tracker.update(typed, dt=dt, camera_discontinuity=camera_discontinuity)

    def reset(self) -> None:
        self._tracker.reset()
