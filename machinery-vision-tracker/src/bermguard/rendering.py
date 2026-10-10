"""Deterministic OpenCV overlay for G2 evidence videos.

Berm states are drawn with unambiguous styles (G5.3):

* ``observed``: solid crest/toe lines and a mask tint, a current measurement;
* ``temporal_estimate``: dashed lines, either a partially occluded current
  profile completed from the reference or a daytime reference carried through
  low light while the camera is verified stationary, always with its age;
* ``historical_reference``: thin dim lines only, no mask and no height;
* ``unknown``: nothing is drawn and the panel states the reason.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import cv2
import numpy as np

from .contracts import Detection, ProximityLevel
from .segmentation.result import SegmentationResult
from .segmentation.validation import LOW_LIGHT_ESTIMATE_REASON


_LEVEL_COLOURS = {
    ProximityLevel.GREEN: (40, 210, 70),
    ProximityLevel.YELLOW: (20, 210, 240),
    ProximityLevel.RED: (30, 30, 235),
    ProximityLevel.UNKNOWN: (255, 175, 40),
}
_CREST_COLOUR = (30, 30, 240)
_TOE_COLOUR = (30, 170, 255)
_CREST_DIM = (70, 60, 130)
_TOE_DIM = (60, 100, 140)

_UNKNOWN_REASONS_ES = {
    "night_without_reference": "noche sin referencia diurna",
    "low_light_without_reference": "baja luz sin referencia diurna",
    "historical_reference_expired": "referencia vencida",
    "reference_not_reconfirmed_after_low_light": "referencia no reconfirmada al volver la luz",
    "reference_bootstrap": "acumulando evidencia",
}


def _polyline(points: Sequence[tuple[float, float]]) -> np.ndarray:
    return np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))


def _draw_dashed_polyline(
    image: np.ndarray,
    points: Sequence[tuple[float, float]],
    colour: tuple[int, int, int],
    thickness: int,
) -> None:
    coordinates = np.rint(np.asarray(points, dtype=np.float64)).astype(np.int32)
    if len(coordinates) < 2:
        return
    dash = max(3, len(coordinates) // 32)
    for index in range(len(coordinates) - 1):
        if (index // dash) % 2 == 0:
            cv2.line(
                image,
                tuple(int(v) for v in coordinates[index]),
                tuple(int(v) for v in coordinates[index + 1]),
                colour,
                thickness,
                cv2.LINE_AA,
            )


def _overlaps_tracked(candidate: Detection, tracked: Sequence[Detection]) -> bool:
    candidate_box = candidate.bbox
    candidate_area = max(
        1.0,
        (candidate_box.x2 - candidate_box.x1) * (candidate_box.y2 - candidate_box.y1),
    )
    for detection in tracked:
        box = detection.bbox
        intersection = max(0.0, min(candidate_box.x2, box.x2) - max(candidate_box.x1, box.x1)) * max(
            0.0, min(candidate_box.y2, box.y2) - max(candidate_box.y1, box.y1)
        )
        tracked_area = max(1.0, (box.x2 - box.x1) * (box.y2 - box.y1))
        union = candidate_area + tracked_area - intersection
        if intersection / union >= 0.45 or intersection / min(candidate_area, tracked_area) >= 0.75:
            return True
    return False


def _draw_dashed_rectangle(
    image: np.ndarray,
    first: tuple[int, int],
    second: tuple[int, int],
    colour: tuple[int, int, int],
) -> None:
    x1, y1 = first
    x2, y2 = second
    dash = 14
    for start in range(x1, x2, dash * 2):
        cv2.line(image, (start, y1), (min(start + dash, x2), y1), colour, 2)
        cv2.line(image, (start, y2), (min(start + dash, x2), y2), colour, 2)
    for start in range(y1, y2, dash * 2):
        cv2.line(image, (x1, start), (x1, min(start + dash, y2)), colour, 2)
        cv2.line(image, (x2, start), (x2, min(start + dash, y2)), colour, 2)


def _age_seconds(segmentation: SegmentationResult, fps: Optional[float]) -> float:
    diagnostics = segmentation.observation.diagnostics
    age_frames = float(diagnostics.get("profile_age_frames", 0) or 0)
    context = segmentation.observation.context
    if fps and fps > 0:
        return age_frames / fps
    if context.frame_index > 0 and context.timestamp_s > 0:
        return age_frames * context.timestamp_s / context.frame_index
    return 0.0


def berm_status_texts(
    segmentation: SegmentationResult,
    fps: Optional[float] = None,
    vehicle_px_per_m: Optional[float] = None,
) -> tuple[str, str, str]:
    """Return the panel status label, the height line and the detail line."""

    observation = segmentation.observation
    diagnostics = observation.diagnostics
    reason = str(diagnostics.get("reason", "") or "")
    status = observation.status.value
    age_s = _age_seconds(segmentation, fps)
    has_height = (
        observation.height_px is not None and observation.height_m_estimated is not None
    )
    if has_height:
        metres = (
            f"h~{observation.height_px:.1f} px | {observation.height_m_estimated:.1f} m"
            " (5 px/m supuesto)"
        )
        if vehicle_px_per_m:
            metres += f" | {observation.height_px / vehicle_px_per_m:.1f} m (ref. CAEX)"
    else:
        metres = "h~N/D"
    if status == "observed":
        return "OBSERVADO", metres, "medicion actual; metros equivalentes, no topografia"
    if status == "temporal_estimate":
        if reason == LOW_LIGHT_ESTIMATE_REASON:
            return "ESTIMADO", metres, f"ref. diurna hace {age_s:.1f} s, camara estatica verificada"
        return "PARCIAL (OCLUIDO)", metres, "columnas ocluidas completadas desde la referencia"
    if status == "historical_reference":
        return "REF. HISTORICA", metres, f"referencia de hace {age_s:.1f} s, sin medicion actual"
    detail = _UNKNOWN_REASONS_ES.get(reason, "observacion no confiable")
    return "N/D", metres, detail


def render_frame(
    frame: np.ndarray,
    segmentation: SegmentationResult,
    detections: Sequence[Detection],
    levels: Mapping[int, ProximityLevel],
    scene_id: int,
    method_label: str,
    raw_detections: Sequence[Detection] = (),
    fps: Optional[float] = None,
    vehicle_px_per_m: Optional[float] = None,
) -> np.ndarray:
    rendered = frame.copy()
    observation = segmentation.observation
    status = observation.status.value
    if status in {"observed", "temporal_estimate"} and np.any(segmentation.mask):
        tint = np.zeros_like(rendered)
        tint[:, :, 2] = segmentation.mask
        rendered = cv2.addWeighted(
            rendered, 1.0, tint, 0.12 if status == "observed" else 0.07, 0.0
        )
    if observation.crest and observation.base:
        if status == "observed":
            cv2.polylines(rendered, [_polyline(observation.crest)], False, _CREST_COLOUR, 3)
            cv2.polylines(rendered, [_polyline(observation.base)], False, _TOE_COLOUR, 3)
            crest_label, toe_label = "CRESTA PRETIL", "PIE / RASANTE"
            crest_colour, toe_colour = _CREST_COLOUR, _TOE_COLOUR
        elif status == "temporal_estimate":
            _draw_dashed_polyline(rendered, observation.crest, _CREST_COLOUR, 3)
            _draw_dashed_polyline(rendered, observation.base, _TOE_COLOUR, 3)
            crest_label, toe_label = "CRESTA (estimada)", "PIE (estimado)"
            crest_colour, toe_colour = _CREST_COLOUR, _TOE_COLOUR
        else:
            cv2.polylines(rendered, [_polyline(observation.crest)], False, _CREST_DIM, 1)
            cv2.polylines(rendered, [_polyline(observation.base)], False, _TOE_DIM, 1)
            crest_label, toe_label = "cresta (ref. historica)", "pie (ref. historica)"
            crest_colour, toe_colour = _CREST_DIM, _TOE_DIM
        crest_start = tuple(int(round(value)) for value in observation.crest[0])
        base_start = tuple(int(round(value)) for value in observation.base[0])
        cv2.putText(
            rendered,
            crest_label,
            (crest_start[0], max(22, crest_start[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            crest_colour,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            rendered,
            toe_label,
            (base_start[0], min(rendered.shape[0] - 10, base_start[1] + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            toe_colour,
            2,
            cv2.LINE_AA,
        )

    for candidate in raw_detections:
        if _overlaps_tracked(candidate, detections):
            continue
        box = candidate.bbox
        first = (int(round(box.x1)), int(round(box.y1)))
        second = (int(round(box.x2)), int(round(box.y2)))
        colour = _LEVEL_COLOURS[ProximityLevel.UNKNOWN]
        _draw_dashed_rectangle(rendered, first, second, colour)
        cv2.putText(
            rendered,
            "maquinaria detectada | trayectoria sin confirmar",
            (first[0], max(22, first[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            colour,
            2,
            cv2.LINE_AA,
        )

    for detection in detections:
        if detection.track_id is None:
            continue
        colour = _LEVEL_COLOURS[levels.get(detection.track_id, ProximityLevel.UNKNOWN)]
        box = detection.bbox
        first = (int(round(box.x1)), int(round(box.y1)))
        second = (int(round(box.x2)), int(round(box.y2)))
        cv2.rectangle(rendered, first, second, colour, 3)
        level = levels.get(detection.track_id, ProximityLevel.UNKNOWN)
        proximity_text = (
            "proximidad N/D (sin par)"
            if level is ProximityLevel.UNKNOWN
            else f"proximidad {level.value.upper()} est."
        )
        label = f"maquinaria ID {detection.track_id} | {proximity_text}"
        cv2.putText(
            rendered,
            label,
            (first[0], max(22, first[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            colour,
            2,
            cv2.LINE_AA,
        )

    status_label, height_text, detail_text = berm_status_texts(
        segmentation, fps, vehicle_px_per_m
    )
    panel_width = min(rendered.shape[1] - 16, 760)
    cv2.rectangle(rendered, (8, 8), (panel_width, 108), (12, 12, 12), -1)
    cv2.putText(
        rendered,
        detail_text,
        (20, 96),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        rendered,
        f"BermGuard G5 | {method_label} | escena {scene_id} | pretil {status_label}",
        (20, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.64,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        rendered,
        height_text,
        (20, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (70, 235, 255),
        2,
        cv2.LINE_AA,
    )
    return rendered
