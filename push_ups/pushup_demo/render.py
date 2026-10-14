"""Push-up analysis video: reel (1080x1920) or horizontal (1920x1080).

Grades come from the athlete's own depth calibration; without one, reps are
shown as detected movement only. Force, tension and effort are never inferred.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from functools import lru_cache

from PIL import ImageFont

from .pose_contract import LANDMARK_INDEX
from .pushup_calibration import GRADES, VALID_GRADES

LINKS = (
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
)


@lru_cache(maxsize=16)
def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for name in ("arialbd.ttf" if bold else "arial.ttf", "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def _pose_index(timestamps: np.ndarray, seconds: float) -> int:
    index = int(np.searchsorted(timestamps, seconds))
    if index >= len(timestamps):
        return len(timestamps) - 1
    if index and abs(float(timestamps[index - 1]) - seconds) < abs(float(timestamps[index]) - seconds):
        return index - 1
    return index


def _angle(track, frame_index: int, side: str) -> float | None:
    names = (f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist")
    points = [track.image_landmarks[frame_index, LANDMARK_INDEX[name], :2] for name in names]
    scores = [track.visibility[frame_index, LANDMARK_INDEX[name]] for name in names]
    if not all(np.isfinite(point).all() for point in points) or min(scores) < 0.5:
        return None
    first, center, last = points
    a, b = first - center, last - center
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return math.degrees(math.acos(float(np.clip(np.dot(a, b) / norm, -1, 1)))) if norm > 0 else None


FPS = 30
SUMMARY_S = 4.0
LAYOUTS = {"reel": (1080, 1920), "horizontal": (1920, 1080)}
# Instagram/TikTok UI covers these bands of a 1080x1920 reel.
REEL_SAFE_TOP, REEL_SAFE_BOTTOM, REEL_SAFE_RIGHT = 230, 1536, 930
PANEL_BG = (9, 14, 22)
GRADE_COLORS = {
    "rom_full_alto": "#2fe0b0", "rom_semi_alto": "#5ab8ff", "rom_full": "#b4e35a",
    "bajo_minimo": "#ffb547", "sin_medida": "#8a96a3", None: "#36edca",
}
GRADE_TEXT = {
    "rom_full_alto": "ROM FULL ALTO", "rom_semi_alto": "ROM SEMI ALTO", "rom_full": "ROM FULL",
    "bajo_minimo": "BAJO EL MÍNIMO", "sin_medida": "SIN MEDIDA", None: "REP DETECTADA",
}
THRESHOLD_LINES = (("rom_full_alto", "alto"), ("rom_semi_alto", "semi"), ("rom_full", "mín."))
GRADE_SHORT = {"rom_full_alto": "ALTO", "rom_semi_alto": "SEMI", "rom_full": "FULL",
               "bajo_minimo": "BAJO", "sin_medida": "--", None: "REP"}


@dataclass(frozen=True)
class Rep:
    number: int
    start: float
    peak: float
    end: float
    height: float | None
    grade: str | None
    pause: float | None = None  # Rest time since the previous rep ended.

    @property
    def up(self) -> float:
        return self.end - self.peak

    @property
    def down(self) -> float:
        return self.peak - self.start


@dataclass(frozen=True)
class Series:
    reps: tuple[Rep, ...]
    times: np.ndarray
    values: np.ndarray
    shoulder_y: float | None
    thresholds: dict | None
    low: float
    high: float


def build_series(events: list[dict], curve: dict, calibration: dict | None) -> Series:
    ordered = sorted(events, key=lambda e: float(e["end"]))
    reps = tuple(Rep(index, float(e["start"]), float(e["peak"]), float(e["end"]),
                     e.get("height_above_floor"), e.get("grade") if calibration else None,
                     max(0.0, float(e["start"]) - float(ordered[index - 2]["end"])) if index > 1 else None)
                 for index, e in enumerate(ordered, 1))
    times = np.asarray(curve["times"], dtype=float)
    values = np.asarray(curve["values"], dtype=float)
    thresholds = dict(calibration["thresholds"]) if calibration else None
    # Scale only over the set itself: mounting and dismounting move the body
    # and would flatten every rep in the graph.
    active = np.isfinite(values)
    if reps:
        active &= (times >= reps[0].start - 1) & (times <= reps[-1].end + 1)
    finite = values[active]
    marks = [0.0] + [r.height for r in reps if r.height is not None] + list((thresholds or {}).values())
    curve_range = [float(np.percentile(finite, 1)), float(np.percentile(finite, 99))] if len(finite) else []
    return Series(reps, times, values, curve.get("shoulder_y"),
                  thresholds, min(curve_range + marks) - .08, max(curve_range + marks) + .08)


def _num(value: float, digits: int = 2, sign: bool = False) -> str:
    return f"{value:{'+' if sign else ''}.{digits}f}".replace(".", ",")


def _clock(series: Series, seconds: float) -> str:
    """Set timer: starts at the first rep, stops at the last one."""
    if not series.reps:
        return "00:00"
    total = series.reps[-1].end - series.reps[0].start
    elapsed = round(total) if seconds >= series.reps[-1].end else max(0.0, seconds - series.reps[0].start)
    return f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"


def _stats(series: Series) -> dict:
    reps = series.reps
    if not reps:
        return {}
    duration = reps[-1].end - reps[0].start
    measured = [r for r in reps if r.height is not None]
    pauses = [r for r in reps if r.pause is not None]
    value = {"duration": duration, "per_rep": duration / len(reps),
             "best": max(measured, key=lambda r: r.height) if measured else None,
             "mean_height": sum(r.height for r in measured) / len(measured) if measured else None,
             "longest_pause": max(pauses, key=lambda r: r.pause) if pauses else None}
    if len(reps) >= 4:
        k = 5 if len(reps) >= 10 else len(reps) // 2
        first, last = reps[:k], reps[-k:]
        value["k"] = k
        value["up"] = (sum(r.up for r in first) / k, sum(r.up for r in last) / k)
        if all(r.height is not None for r in first + last):
            value["height"] = (sum(r.height for r in first) / k, sum(r.height for r in last) / k)
    return value


def _change(before: float, after: float) -> str:
    return f" ({after / before - 1:+.0%})".replace(".", ",") if before > 0 else ""


def _mix(color: str, amount: float, base: tuple[int, int, int] = PANEL_BG) -> tuple[int, int, int]:
    rgb = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
    return tuple(round(b + (c - b) * amount) for c, b in zip(rgb, base))


def _darken(canvas: Image.Image, box: tuple[int, int, int, int], alpha: float = .74) -> None:
    region = canvas.crop(box)
    canvas.paste(Image.blend(region, Image.new("RGB", region.size, PANEL_BG), alpha), box[:2])


def _text(draw: ImageDraw.ImageDraw, xy, value: str, size: int, color="white", bold=False, anchor="la"):
    draw.text(xy, value, font=_font(size, bold), fill=color, anchor=anchor)


def _state(series: Series, seconds: float) -> tuple[int, Rep | None]:
    # A push-up finishes only after the ascent back to plank. The flexed
    # position (peak) is used for depth grading, never for the live count.
    count = sum(rep.end <= seconds for rep in series.reps)
    return count, series.reps[count - 1] if count else None


def _dashed(draw, x0, x1, y, color, width=2, dash=12):
    for x in range(int(x0), int(x1), dash * 2):
        draw.line((x, y, min(x + dash, x1), y), fill=color, width=width)


def _graph(draw, box, series: Series, seconds: float, window: float, size: int) -> None:
    x0, y0, x1, y1 = box
    start = seconds - window * .85
    end = start + window
    def sx(t): return x0 + (t - start) / window * (x1 - x0)
    def sy(v): return y1 - (v - series.low) / (series.high - series.low) * (y1 - y0)
    for rep in series.reps:
        if rep.end < start or rep.start > end:
            continue
        done = rep.end <= seconds
        left, right = max(x0, sx(rep.start)), min(x1, sx(rep.end))
        draw.rectangle((left, y0, right, y1), fill=_mix(GRADE_COLORS[rep.grade] if done else "#5c6672", .16))
        if x0 + size <= sx(rep.end) <= x1 - size:
            _text(draw, (sx(rep.end), y0 + 6), str(rep.number), size,
                  GRADE_COLORS[rep.grade] if done else "#7d8894", bold=True, anchor="ma")
    lines = []
    if series.thresholds:
        lines += [(series.thresholds[key], label, GRADE_COLORS[key]) for key, label in THRESHOLD_LINES]
    labelled: list[float] = []
    for value, label, color in lines:
        y = sy(value)
        if not y0 <= y <= y1:
            continue
        _dashed(draw, x0, x1 - size * 3, y, color, width=2)
        if all(abs(y - other) >= size * .9 for other in labelled):
            _text(draw, (x1, y), label, round(size * .8), color, anchor="rm")
            labelled.append(y)
    mask = (series.times >= start) & (series.times <= seconds) & np.isfinite(series.values)
    points = [(sx(t), min(y1, max(y0, sy(v)))) for t, v in zip(series.times[mask], series.values[mask])]
    if len(points) > 1:
        draw.line(points, fill="#4fa3ff", width=max(3, size // 6), joint="curve")
        cx, cy = points[-1]
        draw.ellipse((cx - 7, cy - 7, cx + 7, cy + 7), fill="#4fa3ff")
    draw.line((sx(seconds), y0, sx(seconds), y1), fill="#e8eef4", width=2)


def _bars(draw, box, series: Series, seconds: float, size: int, numbers: bool,
          labels: bool = False) -> None:
    x0, y0, x1, y1 = box
    count = max(1, len(series.reps))
    if labels and series.thresholds:
        x1 -= size * 3
    slot = (x1 - x0) / count
    marks = [r.height for r in series.reps if r.height is not None] + list((series.thresholds or {}).values())
    low = min(marks + [0.0]) - .06
    high = max(marks + [low + .3]) + .03
    def sy(v): return y1 - (v - low) / (high - low) * (y1 - y0)
    if series.thresholds:
        placed: list[float] = []
        for key, label in THRESHOLD_LINES:
            y = sy(series.thresholds[key])
            _dashed(draw, x0, x1, y, _mix(GRADE_COLORS[key], .55), width=1, dash=6)
            if labels and all(abs(y - other) >= size for other in placed):
                _text(draw, (x1 + 8, y), label, size, GRADE_COLORS[key], anchor="lm")
                placed.append(y)
    for index, rep in enumerate(series.reps):
        left = x0 + index * slot + slot * .14
        right = x0 + (index + 1) * slot - slot * .14
        top = sy(rep.height) if rep.height is not None else y1 - 6
        done = rep.end <= seconds
        fill = GRADE_COLORS[rep.grade] if done else "#2a333e"
        draw.rounded_rectangle((left, min(top, y1 - 4), right, y1), radius=3, fill=fill)
        if numbers and (rep.number in (1, count) or rep.number % 5 == 0):
            _text(draw, ((left + right) / 2, y1 + 6), str(rep.number), size, "#8a96a3", anchor="ma")


def _minimap(draw, box, series: Series, seconds: float, window: float) -> None:
    x0, y0, x1, y1 = box
    total = max(float(series.times[-1]) if len(series.times) else 1.0, 1.0)
    def sx(t): return x0 + t / total * (x1 - x0)
    def sy(v): return y1 - (v - series.low) / (series.high - series.low) * (y1 - y0)
    finite = np.isfinite(series.values)
    points = [(sx(t), min(y1, max(y0, sy(v))))
              for t, v in zip(series.times[finite][::2], series.values[finite][::2])]
    if len(points) > 1:
        draw.line(points, fill="#36506b", width=2)
    start = seconds - window * .85
    draw.rectangle((sx(max(0, start)), y0 - 3, sx(min(total, start + window)), y1 + 3), outline="#e8eef4", width=2)


def _video_mapping(width: int, height: int, box_w: int, box_h: int, cover: bool):
    scale = (max if cover else min)(box_w / width, box_h / height)
    out_w, out_h = round(width * scale) // 2 * 2, round(height * scale) // 2 * 2
    return scale, out_w, out_h, (out_w - box_w) / 2 if cover else 0, (out_h - box_h) / 2 if cover else 0


def _skeleton(image: Image.Image, track, index: int, width: int, height: int,
              ox: float, oy: float, line: int) -> None:
    draw = ImageDraw.Draw(image)
    def point(name):
        joint = LANDMARK_INDEX[name]
        xy = track.image_landmarks[index, joint, :2]
        if not np.isfinite(xy).all() or not track.visibility[index, joint] >= .5:
            return None
        return float(xy[0]) * width - ox, float(xy[1]) * height - oy
    for first, second in LINKS:
        a, b = point(first), point(second)
        if a and b:
            draw.line((a, b), fill="#07222c", width=line + 4)
            draw.line((a, b), fill="#36edca", width=line)
    for name in {item for link in LINKS for item in link}:
        p = point(name)
        if p:
            r = line + 2
            draw.ellipse((p[0] - r, p[1] - r, p[0] + r, p[1] + r), fill="#ffcf65", outline="#132736", width=2)


def _badge(draw, xy, rep: Rep | None, size: int) -> None:
    label = GRADE_TEXT[rep.grade] if rep else "ESPERANDO REP"
    color = GRADE_COLORS[rep.grade] if rep else "#8a96a3"
    x, y = xy
    width = draw.textlength(label, font=_font(size, True)) + size * 1.2
    draw.rounded_rectangle((x, y, x + width, y + size * 1.8), radius=size // 2, fill=PANEL_BG,
                           outline=color, width=3)
    _text(draw, (x + size * .6, y + size * .9), label, size, color, bold=True, anchor="lm")


def _rep_line(rep: Rep | None) -> str:
    if rep is None:
        return "Esperando la primera rep"
    height = f" · profundidad {_num(rep.height, sign=True)}" if rep.height is not None else ""
    return f"Rep {rep.number} · bajada {_num(rep.down)} s{height}"


def _card(draw, box, label: str, value: str, value_size: int) -> None:
    draw.rounded_rectangle(box, radius=14, fill="#16212d", outline="#2a3a4a", width=2)
    _text(draw, (box[0] + 18, box[1] + 14), label, 22, "#8a96a3", bold=True)
    _text(draw, (box[0] + 18, box[3] - 14), value, value_size, "white", bold=True, anchor="ld")


def _grade_line(draw, xy, series: Series, size: int) -> None:
    x, y = xy
    parts = [(f"{sum(r.grade == key for r in series.reps)} {GRADES[key]}", GRADE_COLORS[key])
             for key in GRADES if any(r.grade == key for r in series.reps)]
    for index, (value, color) in enumerate(parts):
        if index:
            _text(draw, (x, y), "·", size, "#5c6672")
            x += draw.textlength("·", font=_font(size)) + size * .6
        _text(draw, (x, y), value, size, color, bold=True)
        x += draw.textlength(value, font=_font(size, True)) + size * .6


def _fatigue_lines(stats: dict) -> list[str]:
    lines = []
    if "height" in stats:
        before, after = stats["height"]
        lines.append(f"Profundidad: {_num(before, sign=True)} primeras {stats['k']} → "
                     f"{_num(after, sign=True)} últimas {stats['k']}{_change(before, after)}")
    if "up" in stats:
        before, after = stats["up"]
        lines.append(f"Subida: {_num(before)} s → {_num(after)} s{_change(before, after)}")
    return lines


def _summary_cards(stats: dict, wide: bool) -> list[tuple[str, str]]:
    best = stats.get("best")
    cards = [("MEJOR REP", f"#{best.number} · {_num(best.height, sign=True)}" if best else "--"),
             ("PROF. PROM.", _num(stats["mean_height"], sign=True) if stats.get("mean_height") is not None else "--"),
             ("RITMO", f"1 rep / {_num(stats['per_rep'], 1)} s")]
    return [("DURACIÓN", f"{_num(stats['duration'], 0)} s")] + cards if wide else cards


def _pause_line(stats: dict) -> str | None:
    rep = stats.get("longest_pause")
    if rep is None or rep.pause < .05:
        return None
    return f"Pausa más larga: {_num(rep.pause, 1)} s antes de la rep {rep.number}"


def _blocks_summary(series: Series, block_size: int = 10) -> list[dict]:
    reps = series.reps
    blocks = []
    for i in range(0, len(reps), block_size):
        chunk = reps[i:i + block_size]
        start_num = chunk[0].number
        end_num = chunk[-1].number
        heights = [r.height for r in chunk if r.height is not None]
        mean_h = sum(heights) / len(heights) if heights else None
        mean_up = sum(r.up for r in chunk) / len(chunk)
        valid_count = sum(r.grade in VALID_GRADES for r in chunk) if series.thresholds else None
        blocks.append({
            "label": f"Reps {start_num:02d}–{end_num:02d}",
            "mean_height": mean_h,
            "mean_up": mean_up,
            "valid": valid_count,
            "total": len(chunk),
        })
    return blocks



def _rep_table(draw, xy, series: Series, offsets: tuple, column_w: float, row_h: int, size: int) -> None:
    x, y = xy
    titles = ("#", "PROF.", "GRADO", "SUBIDA", "PAUSA")[:len(offsets)]
    # Cap columns to prevent canvas overflow (max 3 columns = 30 reps visible)
    max_columns = 3
    columns = min(max(1, math.ceil(len(series.reps) / 10)), max_columns)
    for column in range(columns):
        for offset, title in zip(offsets, titles):
            _text(draw, (x + column * column_w + offset, y), title, 20, "#8a96a3", bold=True)
    # Only render reps that fit in the capped columns
    max_reps = columns * 10
    for index, rep in enumerate(series.reps[:max_reps]):
        cx = x + (index // 10) * column_w
        cy = y + 34 + (index % 10) * row_h
        cells = [(str(rep.number), "white", True),
                 (_num(rep.height, sign=True) if rep.height is not None else "--", "white", False),
                 (GRADE_SHORT[rep.grade], GRADE_COLORS[rep.grade], True),
                 (f"{_num(rep.up)} s", "#c9d4df", False),
                 (f"{_num(rep.pause, 1)} s" if rep.pause is not None else "--", "#c9d4df", False)]
        for offset, (value, color, bold) in zip(offsets, cells):
            _text(draw, (cx + offset, cy), value, size, color, bold=bold)


def _reel_frame(canvas: Image.Image, series: Series, seconds: float) -> None:
    count, last = _state(series, seconds)
    draw = ImageDraw.Draw(canvas)

    # 1. Top HUD Card (Pegado arriba y centrado: x: 60..1020, y: 50..190)
    top_box = (60, 50, 1020, 190)
    _darken(canvas, top_box, .85)
    draw.rounded_rectangle(top_box, radius=18, outline="#2a3a4a", width=2)

    # Left: Conteo de repeticiones
    _text(draw, (96, 68), "FLEXIONES", 20, "#8a96a3", bold=True)
    _text(draw, (96, 98), f"{count:02d} / {len(series.reps):02d}", 60,
          GRADE_COLORS[last.grade] if last else "white", bold=True)

    # Center: Tiempo (Pegado arriba en el centro)
    _text(draw, (540, 68), "TIEMPO", 20, "#8a96a3", bold=True, anchor="ma")
    _text(draw, (540, 98), _clock(series, seconds), 60, "white", bold=True, anchor="ma")

    # Right: Badge de grado
    label = GRADE_TEXT[last.grade] if last else "ESPERANDO REP"
    badge_w = draw.textlength(label, font=_font(22, True)) + 22 * 1.2
    badge_x = 984 - badge_w
    _badge(draw, (badge_x, 98), last, 22)

    # 2. Bottom HUD Card (Pegado abajo y centrado: x: 60..1020, y: 1450..1860)
    bot_box = (60, 1450, 1020, 1860)
    _darken(canvas, bot_box, .85)
    draw.rounded_rectangle(bot_box, radius=18, outline="#2a3a4a", width=2)

    _text(draw, (96, 1468), _rep_line(last), 30, "white", bold=True)
    _graph(draw, (96, 1512, 984, 1692), series, seconds, 6.0, 24)

    draw.line((96, 1708, 984, 1708), fill="#1f2d3d", width=1)

    cells = (("SUBIDA", last.up if last else None),
             ("BAJADA", last.down if last and seconds >= last.end else None),
             ("PAUSA", last.pause if last else None))
    col_w = (984 - 96) / 3
    for index, (label, value) in enumerate(cells):
        cx = 96 + index * col_w
        _text(draw, (cx + 14, 1724), label, 20, "#8a96a3", bold=True)
        _text(draw, (cx + 14, 1756), f"{_num(value, 1)} s" if value is not None else "--", 56, "white", bold=True)



def _horizontal_frame(canvas: Image.Image, series: Series, seconds: float, px: int,
                      track, pose_index: int) -> None:
    count, last = _state(series, seconds)
    draw = ImageDraw.Draw(canvas)
    right = canvas.width - 40
    _text(draw, (px, 34), f"Reps {count} / {len(series.reps)}", 68, "white", bold=True)
    _badge(draw, (px + 470, 46), last, 28)
    _text(draw, (right, 28), f"Tiempo {_clock(series, seconds)}", 32, "white", bold=True, anchor="ra")
    done = [rep for rep in series.reps if rep.end <= seconds]
    if done:
        _text(draw, (right, 70), f"Subida prom. {_num(sum(r.up for r in done) / len(done))} s",
              28, "#c9d4df", anchor="ra")
    angle = _angle(track, pose_index, "left") or _angle(track, pose_index, "right")
    _text(draw, (right, 106), f"Codo {angle:.0f}°" if angle is not None else "Codo --", 28,
          "#c9d4df", anchor="ra")
    _text(draw, (px, 150), "Profundidad del pecho bajo hombros · torsos", 24, "#8a96a3")
    _graph(draw, (px, 184, right, 580), series, seconds, 12.0, 26)
    _minimap(draw, (px, 606, right, 650), series, seconds, 12.0)

    # Tabla de últimas repeticiones (limpia y espaciosa, sin las barras en vivo)
    _text(draw, (px, 686), "Últimas repeticiones", 24, "#8a96a3", bold=True)
    headers = ("REP", "PROF.", "GRADO", "SUBIDA", "BAJADA", "PAUSA")
    columns = [px, px + 100, px + 250, px + 560, px + 730, px + 900]
    for x, title in zip(columns, headers):
        _text(draw, (x, 726), title, 22, "#8a96a3", bold=True)
    for row, rep in enumerate(done[-6:]):
        y = 766 + row * 44
        cells = (str(rep.number), _num(rep.height, sign=True) if rep.height is not None else "--",
                 GRADE_TEXT[rep.grade], f"{_num(rep.up)} s",
                 f"{_num(rep.down)} s" if seconds >= rep.end else "--",
                 f"{_num(rep.pause, 1)} s" if rep.pause is not None else "--")
        for column, (x, value) in enumerate(zip(columns, cells)):
            _text(draw, (x, y), value, 26, GRADE_COLORS[rep.grade] if column == 2 else "white",
                  bold=column == 0)


def _summary(canvas: Image.Image, series: Series, layout: str) -> None:
    """Closing card; in a reel it stays inside the safe zones."""
    stats = _stats(series)
    if layout == "reel":
        _summary_reel(canvas, series, stats)
    else:
        _summary_horizontal(canvas, series, stats)


def _summary_reel(canvas: Image.Image, series: Series, stats: dict) -> None:
    # Safe zones: REEL_SAFE_TOP = 230, REEL_SAFE_BOTTOM = 1536, REEL_SAFE_RIGHT = 930
    _darken(canvas, (0, 0, canvas.width, canvas.height), .80)
    box = (60, 180, 1020, 1740)
    _darken(canvas, box, .95)
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle(box, radius=18, outline="#2a3a4a", width=2)

    px = box[0] + 36
    inner_w = (box[2] - box[0]) - 72
    y = box[1] + 28

    _text(draw, (px, y), "FLEXIONES · RESUMEN FINAL", 24, "#8a96a3", bold=True)
    y += 34
    if not series.reps:
        return
    _text(draw, (px, y), f"{len(series.reps)} REPS · {_clock(series, math.inf)}", 54, "white", bold=True)
    y += 64

    if any(r.grade for r in series.reps):
        _grade_line(draw, (px, y), series, 26)
        y += 44
    else:
        _text(draw, (px, y), f"Ritmo promedio: 1 rep / {_num(stats['per_rep'], 1)} s", 26, "#c9d4df")
        y += 42

    # Cards 2x2
    best = stats.get("best")
    cards = [
        ("PROF. PROM.", _num(stats["mean_height"], sign=True) if stats.get("mean_height") is not None else "--"),
        ("MEJOR REP", f"#{best.number} · {_num(best.height, sign=True)}" if best else "--"),
        ("RITMO", f"1 rep / {_num(stats['per_rep'], 1)} s"),
        ("SUBIDA PROM.", f"{_num(sum(r.up for r in series.reps) / len(series.reps))} s"),
    ]
    card_w = (inner_w - 16) // 2
    card_h = 88
    for i, (label, val) in enumerate(cards):
        col = i % 2
        row = i // 2
        cx = px + col * (card_w + 16)
        cy = y + row * (card_h + 12)
        _card(draw, (cx, cy, cx + card_w, cy + card_h), label, val, 28)
    y += 2 * (card_h + 12) + 18

    # Gráfico de barras de profundidad
    _text(draw, (px, y), "PROFUNDIDAD POR REP", 22, "#8a96a3", bold=True)
    y += 30
    chart_h = 240
    chart_box = (px, y, px + inner_w, y + chart_h)
    _bars(draw, chart_box, series, math.inf, 20, numbers=True, labels=True)
    y += chart_h + 38

    # Diagnóstico de fatiga
    fatigue = _fatigue_lines(stats)
    pause = _pause_line(stats)
    if fatigue or pause:
        _text(draw, (px, y), "DIAGNÓSTICO DE FATIGA", 22, "#8a96a3", bold=True)
        y += 30
        for line in fatigue:
            _text(draw, (px, y), f"• {line}", 24, "#c9d4df")
            y += 30
        if pause:
            _text(draw, (px, y), f"• {pause}", 24, "#c9d4df")
            y += 32
        y += 8

    # Bloques o detalle adicional
    if len(series.reps) > 20:
        _text(draw, (px, y), "EVOLUCIÓN POR BLOQUES (DECENAS)", 22, "#8a96a3", bold=True)
        y += 30
        blocks = _blocks_summary(series, 10)
        for b in blocks[:4]:
            h_str = f"Prof. {_num(b['mean_height'], sign=True)}" if b['mean_height'] is not None else "--"
            val_str = f" · {b['valid']}/{b['total']} ROM" if b['valid'] is not None else ""
            line_text = f"{b['label']}: {h_str} · Subida {_num(b['mean_up'])} s{val_str}"
            _text(draw, (px, y), line_text, 23, "#e8eef4")
            y += 30
    else:
        _text(draw, (px, y), "DETALLE DE REPETICIONES", 22, "#8a96a3", bold=True)
        y += 28
        _rep_table(draw, (px, y), series, (0, 48, 145, 235), 380, 32, 22)
        y += 32 + min(10, len(series.reps)) * 32

    pauses = [r.pause for r in series.reps if r.pause is not None]
    down_prom = sum(r.down for r in series.reps) / len(series.reps)
    pause_prom = sum(pauses) / len(pauses) if pauses else None
    _text(draw, (px, y + 10),
          f"Bajada prom: {_num(down_prom)} s  ·  Pausa prom: {_num(pause_prom, 1) if pause_prom else '--'} s",
          22, "#8a96a3")



def _summary_horizontal(canvas: Image.Image, series: Series, stats: dict) -> None:
    _darken(canvas, (0, 0, canvas.width, canvas.height), .94)
    draw = ImageDraw.Draw(canvas)

    # --- PANEL IZQUIERDO: Métricas, KPIs y Diagnóstico de Fatiga ---
    x = 80
    y = 60
    _text(draw, (x, y), "Resumen de la serie", 52, "white", bold=True)
    y += 64
    if not series.reps:
        return
    _text(draw, (x, y), f"{len(series.reps)} flexiones · {_num(stats['duration'], 0)} s", 30, "#c9d4df")
    y += 44
    if any(r.grade for r in series.reps):
        _grade_line(draw, (x, y), series, 26)
        y += 44

    # 4 Cards en cuadrícula 2x2 (ancho total 760px, dos columnas de 372px)
    best = stats.get("best")
    cards = [
        ("DURACIÓN", f"{_num(stats['duration'], 0)} s"),
        ("RITMO", f"1 rep / {_num(stats['per_rep'], 1)} s"),
        ("PROF. PROM.", _num(stats["mean_height"], sign=True) if stats.get("mean_height") is not None else "--"),
        ("MEJOR REP", f"#{best.number} · {_num(best.height, sign=True)}" if best else "--"),
    ]
    card_w = 372
    card_h = 88
    for i, (label, val) in enumerate(cards):
        col = i % 2
        row = i // 2
        cx = x + col * (card_w + 16)
        cy = y + row * (card_h + 12)
        _card(draw, (cx, cy, cx + card_w, cy + card_h), label, val, 30)
    y += 2 * (card_h + 12) + 20

    # Diagnóstico de fatiga
    fatigue = _fatigue_lines(stats)
    pause = _pause_line(stats)
    if fatigue or pause:
        _text(draw, (x, y), "DIAGNÓSTICO DE FATIGA", 22, "#8a96a3", bold=True)
        y += 32
        for line in fatigue:
            _text(draw, (x, y), f"• {line}", 24, "#c9d4df")
            y += 32
        if pause:
            _text(draw, (x, y), f"• {pause}", 24, "#c9d4df")
            y += 34
        y += 12

    # Desglose de repeticiones: si son <= 20 reps, tabla compacta (max 2 columnas); si son > 20 reps, bloques de 10
    if len(series.reps) <= 20:
        _rep_table(draw, (x, y), series, (0, 48, 145, 235), 370, 36, 24)
    else:
        _text(draw, (x, y), "EVOLUCIÓN POR BLOQUES (DECENAS)", 22, "#8a96a3", bold=True)
        y += 32
        blocks = _blocks_summary(series, 10)
        for b in blocks[:5]:
            h_str = f"Prof. {_num(b['mean_height'], sign=True)}" if b['mean_height'] is not None else "Prof. --"
            up_str = f"Subida {_num(b['mean_up'])} s"
            val_str = f" · {b['valid']}/{b['total']} ROM válido" if b['valid'] is not None else ""
            line_text = f"{b['label']}:  {h_str}  ·  {up_str}{val_str}"
            _text(draw, (x, y), line_text, 23, "#e8eef4")
            y += 32
        if len(series.reps) > 50:
            _text(draw, (x, y), f"... y {len(series.reps) - 50} reps más", 20, "#8a96a3")

    # --- PANEL DERECHO: Gráfica de Profundidad por Rep y Métricas de Serie ---
    chart_x = 900
    chart_y = 60
    _text(draw, (chart_x, chart_y), "Profundidad por rep", 34, "white", bold=True)
    _text(draw, (chart_x, chart_y + 42), "Consistencia del movimiento a lo largo de la serie", 22, "#8a96a3")

    chart_box = (chart_x, 140, 1840, 720)
    _bars(draw, chart_box, series, math.inf, 22, numbers=True, labels=True)

    # Métricas adicionales bajo el gráfico (Subida prom, Bajada prom, Pausa prom)
    pauses = [r.pause for r in series.reps if r.pause is not None]
    sub_prom = sum(r.up for r in series.reps) / len(series.reps) if series.reps else 0
    down_prom = sum(r.down for r in series.reps) / len(series.reps) if series.reps else 0
    pause_prom = sum(pauses) / len(pauses) if pauses else None

    sub_cards = [
        ("SUBIDA PROM.", f"{_num(sub_prom)} s"),
        ("BAJADA PROM.", f"{_num(down_prom)} s"),
        ("PAUSA PROM.", f"{_num(pause_prom, 1)} s" if pause_prom is not None else "--"),
    ]
    sub_w = 301
    card_y = 770
    for idx, (label, val) in enumerate(sub_cards):
        cx = chart_x + idx * (sub_w + 18)
        _card(draw, (cx, card_y, cx + sub_w, card_y + 84), label, val, 28)




def _probe(video: Path) -> tuple[int, int]:
    output = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                             "stream=width,height:stream_side_data=rotation", "-of", "json", str(video)],
                            capture_output=True, text=True, check=True).stdout
    stream = json.loads(output)["streams"][0]
    width, height = int(stream["width"]), int(stream["height"])
    rotation = next((abs(int(item.get("rotation", 0))) for item in stream.get("side_data_list", [])), 0)
    return (height, width) if rotation in (90, 270) else (width, height)


def render(video_path: Path, track, events: list[dict], curve: dict, calibration: dict | None,
           layout: str, output_path: Path) -> None:
    if layout not in LAYOUTS:
        raise ValueError(f"Formato desconocido: {layout}")
    if not len(track.timestamps):
        raise ValueError("PoseTrack vacío")
    series = build_series(events, curve, calibration)
    canvas_w, canvas_h = LAYOUTS[layout]
    width, height = _probe(video_path)
    reel = layout == "reel"
    box_w, box_h = (canvas_w, canvas_h) if reel else (min(round(width * canvas_h / height), 1100), canvas_h)
    scale, out_w, out_h, ox, oy = _video_mapping(width, height, box_w, box_h, cover=reel)
    vf = f"fps={FPS},scale={out_w}:{out_h}" + (f",crop={canvas_w}:{canvas_h}" if reel else "")
    frame_w, frame_h = (canvas_w, canvas_h) if reel else (out_w, out_h)
    video_x = 0 if reel else max(0, (box_w - out_w) // 2)
    video_y = 0 if reel else (canvas_h - out_h) // 2
    px = video_x + frame_w + 56
    output_path.parent.mkdir(parents=True, exist_ok=True)
    decode = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(video_path), "-vf", vf,
                               "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    encode = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                               "-s", f"{canvas_w}x{canvas_h}", "-r", str(FPS), "-i", "-",
                               "-i", str(video_path), "-map", "0:v", "-map", "1:a?",
                               "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                               "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
                               "-movflags", "+faststart", str(output_path)],
                              stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_bytes = frame_w * frame_h * 3
    frames = 0
    canvas = closing_base = None
    line = 6 if reel else 5
    try:
        assert decode.stdout is not None and encode.stdin is not None
        while True:
            raw = decode.stdout.read(frame_bytes)
            if not raw:
                break
            if len(raw) != frame_bytes:
                raise RuntimeError("fotograma RGB incompleto")
            seconds = frames / FPS
            pose_index = _pose_index(track.timestamps, seconds)
            video = Image.frombytes("RGB", (frame_w, frame_h), raw)
            _skeleton(video, track, pose_index, round(width * scale), round(height * scale), ox, oy, line)
            if reel:
                if series.reps and closing_base is None and seconds >= series.reps[-1].end:
                    closing_base = video.copy()
                canvas = video
                _reel_frame(canvas, series, seconds)
            else:
                canvas = Image.new("RGB", (canvas_w, canvas_h), PANEL_BG)
                canvas.paste(video, (video_x, video_y))
                if series.reps and closing_base is None and seconds >= series.reps[-1].end:
                    closing_base = canvas.copy()
                _horizontal_frame(canvas, series, seconds, px, track, pose_index)
            encode.stdin.write(canvas.tobytes())
            frames += 1
        if canvas is not None:
            closing = closing_base or canvas.copy()
            _summary(closing, series, layout)
            for _ in range(round(SUMMARY_S * FPS)):
                encode.stdin.write(closing.tobytes())
        encode.stdin.close()
        if decode.wait() or encode.wait():
            detail = encode.stderr.read().decode(errors="replace")[-400:] if encode.stderr else ""
            raise RuntimeError(f"FFmpeg no pudo crear el video. {detail}".strip())
    finally:
        for process in (decode, encode):
            if process.poll() is None:
                process.kill()
        if decode.stdout:
            decode.stdout.close()
        if encode.stderr:
            encode.stderr.close()
    if not frames:
        raise RuntimeError("el video no produjo fotogramas")
