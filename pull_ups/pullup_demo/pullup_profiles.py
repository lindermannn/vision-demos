"""Pull-up profile experiments over PoseFrame v1 tracks.

This module never promotes a repetition to technically valid. A head/shoulder
cycle is only a movement estimate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import math
from typing import Any, Mapping

import numpy as np

from .calisthenics import CycleStateMachine, POLICIES, analyze_track, extract_signals
from .pose_contract import PoseTrack


@dataclass(frozen=True)
class PullupTuning:
    # Existing arm-based state machine. Defaults are deliberately unchanged.
    minimum_landmark_score: float = 0.55
    minimum_visible_fraction: float = 0.70
    minimum_rom_degrees: float = 30.0
    flexed_angle_degrees: float = 105.0
    minimum_duration_s: float = 0.45
    maximum_duration_s: float = 14.0
    stable_rest_s: float = 0.10
    rest_score: float = 0.25
    movement_score: float = 0.42
    peak_score: float = 0.70
    head_motion_weight: float = 0.50
    # Independent, conditional head/shoulder route.
    head_fallback_enabled: bool = True
    head_visibility_score: float = 0.55
    head_cycle_coverage: float = 0.80
    arm_cycle_coverage_trigger: float = 0.70
    head_rise_fraction: float = 0.04
    shoulder_rise_fraction: float = 0.025
    peak_spacing_s: float = 0.45
    local_peak_window_s: float = 0.65
    long_cycle_split_s: float = 2.30
    camera_stable_confirmed: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PullupTuning":
        allowed = {item.name for item in fields(cls)}
        extra = set(value) - allowed
        if extra:
            raise ValueError(f"Parámetros desconocidos: {', '.join(sorted(extra))}")
        defaults = cls()
        parsed: dict[str, Any] = {}
        for item in fields(cls):
            raw = value.get(item.name, getattr(defaults, item.name))
            if item.name in {"head_fallback_enabled", "camera_stable_confirmed"}:
                if type(raw) is not bool:
                    raise ValueError(f"{item.name} debe ser verdadero o falso")
                parsed[item.name] = raw
            else:
                if type(raw) not in (int, float) or not math.isfinite(float(raw)):
                    raise ValueError(f"{item.name} debe ser un número finito")
                parsed[item.name] = float(raw)
        result = cls(**parsed)
        fractions = (
            "minimum_landmark_score", "minimum_visible_fraction", "head_motion_weight",
            "head_visibility_score", "head_cycle_coverage", "arm_cycle_coverage_trigger",
            "head_rise_fraction", "shoulder_rise_fraction", "rest_score",
            "movement_score", "peak_score",
        )
        if any(not 0 <= getattr(result, name) <= 1 for name in fractions):
            raise ValueError("Visibilidad, proporciones y puntuaciones deben estar entre 0 y 1")
        if not 0 <= result.rest_score < result.movement_score < result.peak_score <= 1:
            raise ValueError("Debe cumplirse reposo < inicio < pico")
        if not 5 <= result.minimum_rom_degrees <= 120 or not 40 <= result.flexed_angle_degrees <= 170:
            raise ValueError("ROM o ángulo de codo fuera de límites seguros de prueba")
        if not 0.2 <= result.minimum_duration_s < result.maximum_duration_s <= 20:
            raise ValueError("Duraciones de ciclo inválidas")
        if not 0.05 <= result.stable_rest_s <= 1:
            raise ValueError("Pausa estable fuera de rango")
        if not 0.3 <= result.peak_spacing_s <= 3:
            raise ValueError("Separación entre picos fuera de rango")
        if not result.peak_spacing_s <= 2 * result.local_peak_window_s <= 4:
            raise ValueError("La ventana local debe cubrir la separación entre picos")
        if not 1 <= result.long_cycle_split_s <= 10:
            raise ValueError("Límite de ciclo largo fuera de rango")
        return result

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _smoothed(values: np.ndarray, radius: int = 2) -> np.ndarray:
    finite = np.isfinite(values)
    if finite.sum() < 3:
        return np.full(len(values), np.nan)
    locations = np.arange(len(values))
    filled = np.interp(locations, locations[finite], values[finite])
    return np.convolve(np.pad(filled, (radius, radius), mode="edge"),
                       np.ones(2 * radius + 1) / (2 * radius + 1), mode="valid")


def _head_candidates(signals, tuning: PullupTuning) -> list[dict[str, Any]]:
    if len(signals) < 12:
        return []
    times = np.asarray([item.timestamp_s for item in signals], dtype=float)
    cadence = float(np.median(np.diff(times)))
    if not .005 <= cadence <= .25:
        return []
    head_raw = np.asarray([item.head_y for item in signals], dtype=float)
    shoulder_raw = np.asarray([item.shoulder_y for item in signals], dtype=float)
    wrist_raw = np.asarray([item.wrist_y for item in signals], dtype=float)
    arm_raw = np.asarray([item.elbow_angle for item in signals], dtype=float)
    head = _smoothed(head_raw)
    shoulder = _smoothed(shoulder_raw)
    if not np.isfinite(head).any() or not np.isfinite(shoulder).any():
        return []
    window = max(3, round(tuning.local_peak_window_s / cadence))
    spacing = max(3, round(tuning.peak_spacing_s / cadence))
    possible: list[tuple[int, float]] = []
    for index in range(window, len(times) - window):
        if not (head[index] <= head[index - 1] and head[index] < head[index + 1]):
            continue
        head_rise = min(float(np.max(head[index - window:index]) - head[index]),
                        float(np.max(head[index + 1:index + window + 1]) - head[index]))
        shoulder_rise = min(float(np.max(shoulder[index - window:index]) - shoulder[index]),
                            float(np.max(shoulder[index + 1:index + window + 1]) - shoulder[index]))
        if head_rise >= tuning.head_rise_fraction and shoulder_rise >= tuning.shoulder_rise_fraction:
            possible.append((index, head_rise + shoulder_rise))
    peaks: list[tuple[int, float]] = []
    for index, strength in possible:
        if peaks and index - peaks[-1][0] < spacing:
            if strength > peaks[-1][1]:
                peaks[-1] = (index, strength)
        else:
            peaks.append((index, strength))
    result: list[dict[str, Any]] = []
    for peak, strength in peaks:
        left = max(0, peak - window)
        right = min(len(times) - 1, peak + window)
        start = left + int(np.argmax(head[left:peak + 1]))
        end = peak + int(np.argmax(head[peak:right + 1]))
        if not start < peak < end:
            continue
        duration = float(times[end] - times[start])
        if not tuning.minimum_duration_s <= duration <= tuning.maximum_duration_s:
            continue
        segment = slice(start, end + 1)
        head_coverage = float(np.isfinite(head_raw[segment]).mean())
        shoulder_coverage = float(np.isfinite(shoulder_raw[segment]).mean())
        arm_coverage = float(np.isfinite(arm_raw[segment]).mean())
        wrist_coverage = float(np.isfinite(wrist_raw[segment]).mean())
        if min(head_coverage, shoulder_coverage) < tuning.head_cycle_coverage:
            continue
        # Absolute image movement is only admissible when camera stability was
        # explicitly confirmed. Otherwise the hands must anchor the measurement.
        if wrist_coverage < tuning.head_cycle_coverage and not tuning.camera_stable_confirmed:
            continue
        result.append({
            "start": round(float(times[start]), 3),
            "peak": round(float(times[peak]), 3),
            "end": round(float(times[end]), 3),
            "start_frame": signals[start].source_index,
            "peak_frame": signals[peak].source_index,
            "end_frame": signals[end].source_index,
            "head_coverage": round(head_coverage, 3),
            "shoulder_coverage": round(shoulder_coverage, 3),
            "arm_coverage": round(arm_coverage, 3),
            "wrist_coverage": round(wrist_coverage, 3),
            "strength": round(strength, 3),
        })
    return result


def _bar_and_torso(signals, tuning: PullupTuning) -> tuple[float, float]:
    """Clip-wide bar line (hands on the bar) and torso length, image units."""
    wrist = np.asarray([item.wrist_y for item in signals], dtype=float)
    torso = np.asarray([item.hip_y - item.shoulder_y for item in signals], dtype=float)
    torso = torso[np.isfinite(torso) & (torso > .02)]
    length = float(np.median(torso)) if len(torso) >= 3 else math.nan
    covered = np.isfinite(wrist).mean() if len(wrist) else 0.0
    # A clip-wide bar line assumes the camera never moved.
    bar = float(np.nanmedian(wrist)) if covered >= .3 and tuning.camera_stable_confirmed else math.nan
    return bar, length


def _rep_heights(signals, events: list[dict[str, Any]], tuning: PullupTuning) -> None:
    """Head height above the bar at each rep's peak, in torso lengths.

    0 means the head reference (nose, else ears) reached the hands' line.
    This is a measurement for calibration, not a chin-over-bar verdict.
    """
    if not signals:
        return
    times = np.asarray([item.timestamp_s for item in signals], dtype=float)
    head = _smoothed(np.asarray([item.head_y for item in signals], dtype=float))
    wrist = np.asarray([item.wrist_y for item in signals], dtype=float)
    clip_bar, torso = _bar_and_torso(signals, tuning)
    for event in events:
        event["height_above_bar"] = None
        event["height_reference"] = None
        segment = (times >= float(event["start"])) & (times <= float(event["end"]))
        if not segment.any() or not math.isfinite(torso) or not np.isfinite(head[segment]).any():
            continue
        # Hands visible during the rep anchor the bar locally; otherwise only a
        # confirmed static camera allows the clip-wide line.
        if np.isfinite(wrist[segment]).mean() >= .3:
            bar, reference = float(np.nanmedian(wrist[segment])), "rep"
        elif math.isfinite(clip_bar):
            bar, reference = clip_bar, "clip"
        else:
            continue
        event["height_above_bar"] = round((bar - float(np.nanmin(head[segment]))) / torso, 3)
        event["height_reference"] = reference


def _profile_policy(tuning: PullupTuning):
    return replace(
        POLICIES["pull_up"],
        minimum_landmark_score=tuning.minimum_landmark_score,
        minimum_visible_fraction=tuning.minimum_visible_fraction,
        minimum_rom=tuning.minimum_rom_degrees,
        flexed_angle=tuning.flexed_angle_degrees,
        minimum_duration_s=tuning.minimum_duration_s,
        maximum_duration_s=tuning.maximum_duration_s,
        stable_rest_s=tuning.stable_rest_s,
        rest_score=tuning.rest_score,
        movement_score=tuning.movement_score,
        peak_score=tuning.peak_score,
    )


def _profile_signals(track: PoseTrack, tuning: PullupTuning, policy=None):
    return extract_signals(
        track, "pull_up", head_motion_weight=tuning.head_motion_weight,
        policy=policy or _profile_policy(tuning), head_visibility_score=tuning.head_visibility_score,
    )


def height_curve(track: PoseTrack, tuning: PullupTuning) -> dict[str, Any]:
    """Per-frame head height above the hands, in torso lengths, for graphs.

    Rep grades use the per-rep values from ``analyze_pullup_profile``; this
    curve only draws the movement. ``bar_y`` is set when a static camera was
    confirmed, so a reference line can be drawn over the video.
    """
    signals = _profile_signals(track, tuning)
    times = np.asarray([item.timestamp_s for item in signals], dtype=float)
    head = _smoothed(np.asarray([item.head_y for item in signals], dtype=float))
    wrist_raw = np.asarray([item.wrist_y for item in signals], dtype=float)
    clip_bar, torso = _bar_and_torso(signals, tuning)
    values = np.full(len(times), np.nan)
    if math.isfinite(torso) and len(wrist_raw) and np.isfinite(wrist_raw).mean() >= .3:
        values = (_smoothed(wrist_raw, radius=4) - head) / torso
    elif math.isfinite(torso) and math.isfinite(clip_bar):
        values = (clip_bar - head) / torso
    return {"times": times, "values": values,
            "bar_y": clip_bar if math.isfinite(clip_bar) else None}


def analyze_pullup_profile(track: PoseTrack, tuning: PullupTuning) -> dict[str, Any]:
    """Compare a candidate profile with the arm-based baseline path."""
    baseline = analyze_track(track, "pull_up")
    policy = _profile_policy(tuning)
    signals = _profile_signals(track, tuning, policy)
    primary = CycleStateMachine(policy).detect(signals)
    counted: list[dict[str, Any]] = []
    for item in primary:
        if item.cycle_status == "complete" and item.peak is not None:
            counted.append({
                "start": item.start, "peak": item.peak, "end": item.end,
                "source": "brazos", "technical_status": "unverifiable",
                "reason": "bar_clearance_not_assessed",
                "confidence": item.confidence,
                "rom_degrees": item.rom_degrees,
            })
    head_candidates = _head_candidates(signals, tuning) if tuning.head_fallback_enabled else []
    for item in head_candidates:
        if any(abs(float(existing["peak"]) - item["peak"]) < tuning.peak_spacing_s for existing in counted):
            continue
        in_long_cycle = any(
            old.cycle_status == "complete" and old.end is not None
            and old.end - old.start >= tuning.long_cycle_split_s
            and old.start <= item["peak"] <= old.end
            for old in primary
        )
        rejected_for_visibility = any(
            old.cycle_status != "complete"
            and old.rejection_reason == "insufficient_pose_visibility"
            and old.end is not None and old.start <= item["peak"] <= old.end
            for old in primary
        )
        if (item["arm_coverage"] >= tuning.arm_cycle_coverage_trigger
                and not in_long_cycle and not rejected_for_visibility):
            continue
        counted.append({
            "start": item["start"], "peak": item["peak"], "end": item["end"],
            "source": "cabeza_hombros", "technical_status": "unverifiable",
            "reason": "head_motion_not_technique_validation",
            "confidence": min(0.85, round(item["strength"] / .30, 3)),
            "rom_degrees": None,
            "evidence": item,
        })
    counted.sort(key=lambda item: item["peak"])
    _rep_heights(signals, counted, tuning)
    return {
        "schema_version": "pullup-profile/1.0",
        "mode": "analysis_only",
        "exercise": "pull_up",
        "baseline_count": baseline["complete_cycles"],
        "profile_count": len(counted),
        "head_candidates": len(head_candidates),
        "head_fallback_count": sum(item["source"] == "cabeza_hombros" for item in counted),
        "events": counted,
        "primary_incomplete": [{
            "start": item.start, "peak": item.peak, "end": item.end,
            "reason": item.rejection_reason,
        } for item in primary if item.cycle_status != "complete"],
        "limitations": [
            "head_motion_is_not_chin_over_bar_proof",
            "camera_stability_must_be_confirmed_when_hands_are_hidden",
            "profile_never_promotes_automatically",
        ],
    }
