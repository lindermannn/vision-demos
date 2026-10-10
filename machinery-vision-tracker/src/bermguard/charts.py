"""Dependency-light PNG charts generated from raw G2 tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


_WIDTH = 1100
_HEIGHT = 620
_PLOT_LEFT = 100
_PLOT_TOP = 70
_PLOT_RIGHT = 1040
_PLOT_BOTTOM = 535


def _canvas(title: str, x_label: str, y_label: str) -> np.ndarray:
    image = np.full((_HEIGHT, _WIDTH, 3), 250, dtype=np.uint8)
    cv2.putText(image, title, (35, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.88, (30, 30, 30), 2)
    cv2.rectangle(image, (_PLOT_LEFT, _PLOT_TOP), (_PLOT_RIGHT, _PLOT_BOTTOM), (80, 80, 80), 2)
    cv2.putText(image, x_label, (470, 590), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (45, 45, 45), 1)
    cv2.putText(image, y_label, (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (45, 45, 45), 1)
    return image


def _write(path: Path, image: np.ndarray) -> None:
    if path.exists():
        raise OSError(f"cannot create chart: {path.name}")
    encoded, payload = cv2.imencode(".png", image)
    if not encoded:
        raise OSError(f"cannot encode chart: {path.name}")
    with path.open("xb") as stream:
        stream.write(payload.tobytes())


def _placeholder(path: Path, title: str, message: str) -> None:
    image = _canvas(title, "", "")
    cv2.putText(image, message, (280, 305), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (80, 80, 80), 2)
    _write(path, image)


def write_height_curve(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    valid = [
        row
        for row in rows
        if row.get("height_m_estimated") not in (None, "")
        and row.get("status") in {"observed", "temporal_estimate"}
    ]
    if not valid:
        _placeholder(path, "Altura proyectada del pretil", "Sin observaciones validas")
        return
    times = np.asarray([float(row["timestamp_s"]) for row in valid], dtype=np.float64)
    heights = np.asarray([float(row["height_m_estimated"]) for row in valid], dtype=np.float64)
    minimum_t, maximum_t = float(times.min()), float(times.max())
    maximum_h = max(float(heights.max()) * 1.08, 1.0)
    image = _canvas(
        "Altura proyectada del pretil (estimacion monocular)",
        "tiempo [s]",
        "altura estimada [m]",
    )
    last_point = None
    last_scene = None
    for row, time_value, height_value in zip(valid, times, heights):
        x = _PLOT_LEFT + int(round((time_value - minimum_t) / max(maximum_t - minimum_t, 1e-9) * (_PLOT_RIGHT - _PLOT_LEFT)))
        y = _PLOT_BOTTOM - int(round(height_value / maximum_h * (_PLOT_BOTTOM - _PLOT_TOP)))
        point = (x, y)
        scene = int(row["scene_id"])
        if last_point is not None and scene == last_scene:
            cv2.line(image, last_point, point, (190, 85, 30), 2, cv2.LINE_AA)
        last_point, last_scene = point, scene
    cv2.putText(image, f"0", (70, _PLOT_BOTTOM + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (60, 60, 60), 1)
    cv2.putText(image, f"{maximum_h:.1f}", (50, _PLOT_TOP + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (60, 60, 60), 1)
    _write(path, image)


def write_vehicle_dispersion(
    path: Path, rows: Sequence[Mapping[str, Any]], frame_width: int, frame_height: int
) -> None:
    if not rows:
        _placeholder(path, "Distribucion espacial de vehiculos", "Sin tracks confirmados")
        return
    image = _canvas(
        "Distribucion espacial de vehiculos en el tiempo (punto inferior de BBOX)",
        "x [px imagen]",
        "y [px imagen]",
    )
    # Time is encoded as colour (blue = inicio, rojo = fin) and each track is
    # joined into a thin trajectory so the evolution, not only the spread, is
    # visible, as requested by the assessment e-mail.
    times = [float(row.get("timestamp_s", 0.0) or 0.0) for row in rows]
    t_min, t_max = min(times), max(times)
    span = max(t_max - t_min, 1e-6)

    def point(row: Mapping[str, Any]) -> tuple[int, int]:
        x = _PLOT_LEFT + int(round(float(row["x_px"]) / frame_width * (_PLOT_RIGHT - _PLOT_LEFT)))
        y = _PLOT_TOP + int(round(float(row["y_px"]) / frame_height * (_PLOT_BOTTOM - _PLOT_TOP)))
        return x, y

    def colour_for(t: float) -> tuple[int, int, int]:
        ratio = np.uint8(round(255 * (t - t_min) / span))
        b, g, r = (int(v) for v in cv2.applyColorMap(np.array([[ratio]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0])
        return b, g, r

    tracks: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        tracks.setdefault((str(row.get("scene_id", 0)), str(row.get("track_id", 0))), []).append(row)
    for members in tracks.values():
        members.sort(key=lambda item: float(item.get("timestamp_s", 0.0) or 0.0))
        for first, second in zip(members, members[1:]):
            cv2.line(image, point(first), point(second), (170, 170, 170), 1, cv2.LINE_AA)
    for row in rows:
        cv2.circle(image, point(row), 3, colour_for(float(row.get("timestamp_s", 0.0) or 0.0)), -1, cv2.LINE_AA)
    bar_left, bar_top, bar_width, bar_height = _PLOT_RIGHT - 260, _PLOT_TOP - 30, 200, 12
    for offset in range(bar_width):
        cv2.line(
            image,
            (bar_left + offset, bar_top),
            (bar_left + offset, bar_top + bar_height),
            colour_for(t_min + span * offset / max(bar_width - 1, 1)),
            1,
        )
    cv2.putText(image, f"t={t_min:.1f}s", (bar_left - 60, bar_top + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(image, f"t={t_max:.1f}s", (bar_left + bar_width + 6, bar_top + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1, cv2.LINE_AA)
    _write(path, image)


def write_minimum_distance_matrix(
    path: Path, rows: Sequence[Mapping[str, Any]]
) -> None:
    if not rows:
        _placeholder(path, "Matriz de distancia minima", "Sin pares simultaneos")
        return
    identities = sorted(
        {(int(row["scene_id"]), int(row["track_id_a"])) for row in rows}
        | {(int(row["scene_id"]), int(row["track_id_b"])) for row in rows}
    )
    size = len(identities)
    index = {identity: position for position, identity in enumerate(identities)}
    matrix = np.full((size, size), np.nan, dtype=np.float64)
    np.fill_diagonal(matrix, 0.0)
    for row in rows:
        first = index[(int(row["scene_id"]), int(row["track_id_a"]))]
        second = index[(int(row["scene_id"]), int(row["track_id_b"]))]
        value = float(row["minimum_distance_m_estimated"])
        matrix[first, second] = value
        matrix[second, first] = value

    cell = min(70, max(24, int(480 / max(size, 1))))
    image = np.full(
        (max(620, 145 + size * cell), max(800, 190 + size * cell), 3),
        250,
        dtype=np.uint8,
    )
    cv2.putText(image, "Matriz de distancia minima estimada [m]", (25, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (30, 30, 30), 2)
    origin_x, origin_y = 150, 105
    for row_index, identity in enumerate(identities):
        label = f"S{identity[0]}-T{identity[1]}"
        cv2.putText(image, label, (18, origin_y + row_index * cell + int(cell * 0.65)), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (40, 40, 40), 1)
        cv2.putText(image, label, (origin_x + row_index * cell, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 40), 1)
        for column_index in range(size):
            value = matrix[row_index, column_index]
            if np.isnan(value):
                colour, text = (220, 220, 220), "-"
            elif row_index == column_index:
                colour, text = (235, 235, 235), "0"
            elif value < 10:
                colour, text = (70, 70, 235), f"{value:.1f}"
            elif value <= 20:
                colour, text = (60, 215, 245), f"{value:.1f}"
            else:
                colour, text = (70, 205, 90), f"{value:.1f}"
            first = (origin_x + column_index * cell, origin_y + row_index * cell)
            second = (first[0] + cell, first[1] + cell)
            cv2.rectangle(image, first, second, colour, -1)
            cv2.rectangle(image, first, second, (120, 120, 120), 1)
            cv2.putText(image, text, (first[0] + 5, first[1] + int(cell * 0.62)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (25, 25, 25), 1)
    _write(path, image)
