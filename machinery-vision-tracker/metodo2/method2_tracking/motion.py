"""Deterministic bounded alpha-beta motion model."""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .contracts import BBox


def _center_size(bbox: BBox) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = map(float, bbox)
    return ((x1+x2)/2.0, (y1+y2)/2.0, x2-x1, y2-y1)


def _bbox(cx: float, cy: float, w: float, h: float) -> BBox:
    w, h = max(1.0, w), max(1.0, h)
    return (cx-w/2.0, cy-h/2.0, cx+w/2.0, cy+h/2.0)


@dataclass
class MotionState:
    cx: float
    cy: float
    width: float
    height: float
    vx: float = 0.0
    vy: float = 0.0
    uncertainty: float = 1.0

    @classmethod
    def from_bbox(cls, bbox: BBox) -> "MotionState":
        return cls(*_center_size(bbox))

    @property
    def bbox(self) -> BBox:
        return _bbox(self.cx, self.cy, self.width, self.height)

    def predict(self, dt: float, uncertainty_growth: float = 12.0) -> BBox:
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and > 0")
        self.cx += self.vx * dt
        self.cy += self.vy * dt
        self.uncertainty = min(1e6, self.uncertainty + uncertainty_growth * dt)
        return self.bbox

    def update(
        self,
        observed_bbox: BBox,
        dt: float,
        max_speed_px_s: float,
        alpha: float = 0.82,
        beta: float = 0.32,
    ) -> BBox:
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and > 0")
        ox, oy, ow, oh = _center_size(observed_bbox)
        rx, ry = ox-self.cx, oy-self.cy
        self.cx += alpha*rx
        self.cy += alpha*ry
        proposed_vx = self.vx + beta*rx/dt
        proposed_vy = self.vy + beta*ry/dt
        speed = math.hypot(proposed_vx, proposed_vy)
        if speed > max_speed_px_s > 0:
            scale = max_speed_px_s/speed
            proposed_vx *= scale
            proposed_vy *= scale
        self.vx, self.vy = proposed_vx, proposed_vy
        self.width = 0.75*self.width + 0.25*ow
        self.height = 0.75*self.height + 0.25*oh
        self.uncertainty = max(1.0, self.uncertainty*0.45)
        return self.bbox

    def reset(self, bbox: BBox) -> None:
        self.cx, self.cy, self.width, self.height = _center_size(bbox)
        self.vx = self.vy = 0.0
        self.uncertainty = 1.0
