"""Push-up profile experiments over PoseFrame v1 tracks.

This module never promotes a repetition to technically valid. An elbow cycle
is only a movement estimate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import math
from typing import Any, Mapping

import numpy as np

from .calisthenics import CycleStateMachine, POLICIES, analyze_track, extract_signals
from .pose_contract import PoseTrack


@dataclass(frozen=True)
class PushupTuning:
    # Existing arm-based state machine. Defaults are deliberately unchanged.
    minimum_landmark_score: float = 0.55
    minimum_visible_fraction: float = 0.70
    minimum_rom_degrees: float = 30.0
    flexed_angle_degrees: float = 105.0
    minimum_duration_s: float = 0.35
    maximum_duration_s: float = 8.0
    stable_rest_s: float = 0.10
    rest_score: float = 0.25
    movement_score: float = 0.42
    peak_score: float = 0.70
    head_motion_weight: float = 0.0
    # Independent, conditional head/shoulder route (disabled for push-ups).
    head_fallback_enabled: bool = False
    head_visibility_score: float = 0.55
    head_cycle_coverage: float = 0.80
    arm_cycle_coverage_trigger: float = 0.70
    head_rise_fraction: float = 0.04
    shoulder_rise_fraction: float = 0.025
    peak_spacing_s: float = 0.35
    local_peak_window_s: float = 0.50
    long_cycle_split_s: float = 2.00
    camera_stable_confirmed: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PushupTuning":
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


def _shoulder_and_torso(signals, tuning: PushupTuning) -> tuple[float, float]:
    """Clip-wide shoulder line and torso length, image units."""
    shoulder = np.asarray([item.shoulder_y for item in signals], dtype=float)
    hip = np.asarray([item.hip_y for item in signals], dtype=float)
    torso = hip - shoulder
    torso = torso[np.isfinite(torso) & (torso > .02)]
    length = float(np.median(torso)) if len(torso) >= 3 else math.nan
    covered = np.isfinite(shoulder).mean() if len(shoulder) else 0.0
    shoulder_line = float(np.nanmedian(shoulder)) if covered >= .3 and tuning.camera_stable_confirmed else math.nan
    return shoulder_line, length


def _rep_depths(signals, events: list[dict[str, Any]], tuning: PushupTuning) -> None:
    """Chest depth below shoulders at each rep's peak, in torso lengths.

    0 means the chest reference (mid-shoulder to elbow line) reached the floor line.
    This is a measurement for calibration, not a chest-to-floor verdict.
    """
    if not signals:
        return
    times = np.asarray([item.timestamp_s for item in signals], dtype=float)
    shoulder = _smoothed(np.asarray([item.shoulder_y for item in signals], dtype=float))
    elbow = _smoothed(np.asarray([item.elbow_y for item in signals], dtype=float))
    clip_shoulder, torso = _shoulder_and_torso(signals, tuning)
    for event in events:
        event["height_above_floor"] = None
        event["height_reference"] = None
        segment = (times >= float(event["start"])) & (times <= float(event["end"]))
        if not segment.any() or not math.isfinite(torso) or not np.isfinite(shoulder[segment]).any():
            continue
        # Shoulder visible during the rep anchors the line locally; otherwise only a
        # confirmed static camera allows the clip-wide line.
        if np.isfinite(shoulder[segment]).mean() >= .3:
            line, reference = float(np.nanmedian(shoulder[segment])), "rep"
        elif math.isfinite(clip_shoulder):
            line, reference = clip_shoulder, "clip"
        else:
            continue
        # Chest depth estimated from shoulder-elbow midpoint at peak flexion
        chest_depth = (shoulder[segment] + elbow[segment]) / 2
        event["height_above_floor"] = round((float(np.nanmin(chest_depth)) - line) / torso, 3)
        event["height_reference"] = reference


def _profile_policy(tuning: PushupTuning):
    return replace(
        POLICIES["push_up"],
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


def _profile_signals(track: PoseTrack, tuning: PushupTuning, policy=None):
    return extract_signals(
        track, "push_up", head_motion_weight=tuning.head_motion_weight,
        policy=policy or _profile_policy(tuning), head_visibility_score=tuning.head_visibility_score,
    )


def height_curve(track: PoseTrack, tuning: PushupTuning) -> dict[str, Any]:
    """Per-frame chest depth below shoulders, in torso lengths, for graphs.

    Rep grades use the per-rep values from ``analyze_pushup_profile``; this
    curve only draws the movement. ``shoulder_y`` is set when a static camera was
    confirmed, so a reference line can be drawn over the video.
    """
    signals = _profile_signals(track, tuning)
    times = np.asarray([item.timestamp_s for item in signals], dtype=float)
    shoulder = _smoothed(np.asarray([item.shoulder_y for item in signals], dtype=float))
    elbow = _smoothed(np.asarray([item.elbow_y for item in signals], dtype=float))
    clip_shoulder, torso = _shoulder_and_torso(signals, tuning)
    values = np.full(len(times), np.nan)
    if math.isfinite(torso) and len(shoulder) and np.isfinite(shoulder).mean() >= .3:
        chest = (shoulder + elbow) / 2
        values = (chest - _smoothed(shoulder, radius=4)) / torso
    elif math.isfinite(torso) and math.isfinite(clip_shoulder):
        chest = (shoulder + elbow) / 2
        values = (chest - clip_shoulder) / torso
    return {"times": times, "values": values,
            "shoulder_y": clip_shoulder if math.isfinite(clip_shoulder) else None}


def analyze_pushup_profile(track: PoseTrack, tuning: PushupTuning) -> dict[str, Any]:
    """Compare a candidate profile with the arm-based baseline path."""
    baseline = analyze_track(track, "push_up")
    policy = _profile_policy(tuning)
    signals = _profile_signals(track, tuning, policy)
    primary = CycleStateMachine(policy).detect(signals)
    counted: list[dict[str, Any]] = []
    for item in primary:
        if item.cycle_status == "complete" and item.peak is not None:
            counted.append({
                "start": item.start, "peak": item.peak, "end": item.end,
                "source": "brazos", "technical_status": "unverifiable",
                "reason": "floor_depth_not_assessed",
                "confidence": item.confidence,
                "rom_degrees": item.rom_degrees,
            })
    counted.sort(key=lambda item: item["peak"])
    _rep_depths(signals, counted, tuning)
    return {
        "schema_version": "pushup-profile/1.0",
        "mode": "analysis_only",
        "exercise": "push_up",
        "baseline_count": baseline["complete_cycles"],
        "profile_count": len(counted),
        "events": counted,
        "primary_incomplete": [{
            "start": item.start, "peak": item.peak, "end": item.end,
            "reason": item.rejection_reason,
        } for item in primary if item.cycle_status != "complete"],
        "limitations": [
            "elbow_flexion_is_not_chest_to_floor_proof",
            "camera_stability_must_be_confirmed_when_shoulders_hidden",
            "profile_never_promotes_automatically",
        ],
    }


__all__ = [
    "PushupTuning", "analyze_pushup_profile", "height_curve",
]