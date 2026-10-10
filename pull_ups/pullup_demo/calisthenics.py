"""Calisthenics signals and cycle detection over the PoseFrame v1 contract.

The module only observes and proposes cycles. It does not decide technique.
It separates four responsibilities:

1. PoseTrack -> senales biomecanicas observadas.
2. Senales -> candidatos temporales deterministas.
3. Evidencia de escena humana -> decision tecnica.
4. Reporte estructurado, sin promocion automatica.

Todos los timestamps emitidos pertenecen al PoseTrack de entrada. Cuando falta
una referencia critica la decision es ``unverifiable``; nunca se infiere una
repeticion valida por regularidad o por el total declarado del archivo.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .pose_contract import PoseTrack, landmark_index


SCHEMA_VERSION = "calisthenics/1.0"
SUPPORTED_EXERCISES = frozenset({"pull_up", "chin_up", "dip", "push_up"})
TECHNICAL_STATUSES = frozenset({"valid", "invalid", "unverifiable"})
CYCLE_STATUSES = frozenset({"movement_detected", "complete", "incomplete"})


@dataclass(frozen=True)
class ExercisePolicy:
    exercise: str
    extended_angle: float
    movement_angle: float
    flexed_angle: float
    minimum_rom: float
    minimum_duration_s: float
    maximum_duration_s: float
    minimum_visible_fraction: float = 0.70
    minimum_landmark_score: float = 0.55
    stable_rest_s: float = 0.10
    rest_score: float = 0.25
    movement_score: float = 0.42
    peak_score: float = 0.70


POLICIES: dict[str, ExercisePolicy] = {
    "pull_up": ExercisePolicy("pull_up", 145.0, 135.0, 105.0, 30.0, 0.45, 14.0),
    "chin_up": ExercisePolicy("chin_up", 145.0, 135.0, 105.0, 30.0, 0.45, 14.0),
    "dip": ExercisePolicy("dip", 145.0, 135.0, 115.0, 25.0, 0.45, 14.0),
    "push_up": ExercisePolicy("push_up", 145.0, 135.0, 105.0, 30.0, 0.35, 8.0),
}


@dataclass(frozen=True)
class SignalFrame:
    source_index: int
    timestamp_s: float
    elbow_angle: float
    effort_score: float
    landmark_score: float
    shoulder_y: float
    elbow_y: float
    hip_y: float
    head_y: float = math.nan
    wrist_y: float = math.nan

    @property
    def observable(self) -> bool:
        return math.isfinite(self.elbow_angle) and math.isfinite(self.landmark_score)


@dataclass(frozen=True)
class CycleCandidate:
    start: float
    peak: float | None
    end: float | None
    start_index: int
    peak_index: int | None
    end_index: int | None
    cycle_status: str
    phase: str
    confidence: float
    visible_fraction: float
    rom_degrees: float
    evidence: tuple[str, ...] = ()
    rejection_reason: str | None = None

    def __post_init__(self) -> None:
        if self.cycle_status not in CYCLE_STATUSES:
            raise ValueError(f"cycle_status no permitido: {self.cycle_status}")
        stamps = [value for value in (self.start, self.peak, self.end) if value is not None]
        if any(not math.isfinite(float(value)) for value in stamps):
            raise ValueError("timestamps deben ser finitos")
        if self.peak is not None and self.peak < self.start:
            raise ValueError("peak no puede preceder start")
        if self.end is not None and self.peak is not None and self.end < self.peak:
            raise ValueError("end no puede preceder peak")


@dataclass(frozen=True)
class CycleDecision:
    candidate: CycleCandidate
    technical_status: str
    reason: str
    applied_rules: tuple[str, ...]
    evidence: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.technical_status not in TECHNICAL_STATUSES:
            raise ValueError(f"technical_status no permitido: {self.technical_status}")

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["candidate"]["evidence"] = list(self.candidate.evidence)
        result["applied_rules"] = list(self.applied_rules)
        result["evidence"] = list(self.evidence)
        result["counted"] = self.candidate.cycle_status == "complete"
        result["rep_label"] = (
            "excluded" if not result["counted"] else
            "doubtful" if self.technical_status == "unverifiable" else
            "confirmed" if self.technical_status == "valid" else "invalid_technique"
        )
        return result


@dataclass(frozen=True)
class SceneSample:
    timestamp_s: float
    camera_view: str
    rules: Mapping[str, bool]
    references: tuple[Mapping[str, Any], ...]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SceneSample":
        refs = payload.get("references", ())
        return cls(
            timestamp_s=float(payload["timestamp_s"]),
            camera_view=str(payload.get("camera_view", "unverifiable")),
            rules=dict(payload.get("rules", {})),
            references=tuple(dict(item) for item in refs if isinstance(item, Mapping)),
        )

    def visible_reference(self, kind: str) -> Mapping[str, Any] | None:
        for reference in self.references:
            if reference.get("kind") == kind and bool(reference.get("visible")):
                return reference
        return None


def scene_samples_from_review(review: Mapping[str, Any]) -> tuple[SceneSample, ...]:
    """Carga solo observabilidad; un borrador nunca se convierte en gold."""
    samples = review.get("samples", ())
    return tuple(
        sorted(
            (SceneSample.from_mapping(item) for item in samples if isinstance(item, Mapping)),
            key=lambda item: item.timestamp_s,
        )
    )


def _finite_mean(values: Sequence[float]) -> float:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(clean)) if clean else math.nan


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    if not (np.isfinite(a).all() and np.isfinite(b).all() and np.isfinite(c).all()):
        return math.nan
    left, right = a - b, c - b
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-8:
        return math.nan
    cosine = float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def extract_signals(
    track: PoseTrack, exercise: str, *, head_motion_weight: float = 0.0,
    policy: ExercisePolicy | None = None,
    head_visibility_score: float = 0.55,
) -> tuple[SignalFrame, ...]:
    """Extrae medidas observables sin tomar decisiones de repeticion."""
    if exercise not in SUPPORTED_EXERCISES:
        raise ValueError(f"ejercicio no soportado: {exercise}")
    if not 0.0 <= head_motion_weight <= 1.0:
        raise ValueError("head_motion_weight debe estar entre 0 y 1")
    if not 0.0 <= head_visibility_score <= 1.0:
        raise ValueError("head_visibility_score debe estar entre 0 y 1")
    active_policy = policy or POLICIES[exercise]
    idx = {name: landmark_index(name) for name in (
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "nose", "left_ear", "right_ear", "mouth_left", "mouth_right",
    )}
    raw: list[dict[str, float | int]] = []
    for position, timestamp in enumerate(track.timestamps):
        image = track.image_landmarks[position, :, :2]
        visibility = track.visibility[position]
        angles: list[float] = []
        side_scores: list[float] = []
        for side in ("left", "right"):
            names = (f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist")
            score = min(float(visibility[idx[name]]) for name in names)
            if score >= active_policy.minimum_landmark_score:
                angles.append(_angle(*(image[idx[name]] for name in names)))
                side_scores.append(score)
        shoulders = [float(image[idx[name], 1]) for name in ("left_shoulder", "right_shoulder")]
        elbows = [float(image[idx[name], 1]) for name in ("left_elbow", "right_elbow")]
        hips = [float(image[idx[name], 1]) for name in ("left_hip", "right_hip")]
        wrists = [
            float(image[idx[name], 1])
            for name in ("left_wrist", "right_wrist")
            if float(visibility[idx[name]]) >= active_policy.minimum_landmark_score
        ]
        head_y = math.nan
        if float(visibility[idx["nose"]]) >= head_visibility_score and math.isfinite(float(image[idx["nose"], 1])):
            head_y = float(image[idx["nose"], 1])
        else:
            for pair in (("left_ear", "right_ear"), ("mouth_left", "mouth_right")):
                if all(float(visibility[idx[name]]) >= head_visibility_score for name in pair):
                    head_y = _finite_mean([float(image[idx[name], 1]) for name in pair])
                    if math.isfinite(head_y):
                        break
        raw.append({
            "source_index": position,
            "timestamp_s": float(timestamp),
            "elbow_angle": _finite_mean(angles),
            "landmark_score": _finite_mean(side_scores),
            "shoulder_y": _finite_mean(shoulders),
            "elbow_y": _finite_mean(elbows),
            "hip_y": _finite_mean(hips),
            "head_y": head_y,
            "wrist_y": _finite_mean(wrists),
        })

    def robust_unit(values: Sequence[float]) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        finite = np.isfinite(array)
        result = np.full(len(array), np.nan, dtype=np.float64)
        if finite.sum() < 3:
            return result
        low, high = np.percentile(array[finite], [5, 95])
        if high - low < 1e-6:
            result[finite] = 0.0
            return result
        result[finite] = np.clip((array[finite] - low) / (high - low), 0.0, 1.0)
        return result

    angle_values = [float(item["elbow_angle"]) for item in raw]
    shoulder_values = [float(item["shoulder_y"]) for item in raw]
    head_values = [float(item["head_y"]) for item in raw]
    angle_unit = robust_unit(angle_values)
    shoulder_unit = robust_unit(shoulder_values)
    head_unit = robust_unit(head_values)
    finite_shoulders = np.asarray(shoulder_values, dtype=np.float64)
    finite_shoulders = finite_shoulders[np.isfinite(finite_shoulders)]
    shoulder_dynamic = bool(
        len(finite_shoulders) >= 3
        and np.percentile(finite_shoulders, 95) - np.percentile(finite_shoulders, 5) >= .01
    )
    paired = np.isfinite(shoulder_unit) & np.isfinite(head_unit)
    head_agrees = bool(
        paired.sum() >= max(6, int(.7 * len(raw)))
        and np.std(shoulder_unit[paired]) >= .08
        and np.std(head_unit[paired]) >= .08
        and np.corrcoef(shoulder_unit[paired], head_unit[paired])[0, 1] >= .6
    )
    if shoulder_dynamic and head_agrees and head_motion_weight > 0:
        # La cabeza refuerza el recorrido corporal solo si acompana al hombro.
        # Una inclinacion de cuello aislada nunca crea una repeticion.
        body_unit = shoulder_unit.copy()
        body_unit[paired] = (
            (1 - head_motion_weight) * shoulder_unit[paired]
            + head_motion_weight * head_unit[paired]
        )
    else:
        body_unit = shoulder_unit
    flexion = 1.0 - angle_unit
    normalized = "pull_up" if exercise == "chin_up" else exercise
    if normalized == "pull_up":
        # En tomas de espalda el codo puede no aparentar extension completa;
        # el ascenso vertical del hombro conserva cada ciclo.
        body_effort = (1.0 - body_unit) if shoulder_dynamic else np.zeros(len(raw))
        effort = (.55 * flexion + .45 * body_effort) if shoulder_dynamic else flexion
    elif normalized == "dip":
        # El fondo combina flexion y descenso del hombro. El codo mantiene el
        # peso principal para no convertir movimiento de camara en repeticion.
        body_effort = body_unit if shoulder_dynamic else np.zeros(len(raw))
        effort = (.75 * flexion + .25 * body_effort) if shoulder_dynamic else flexion
    else:
        # En flexiones el eje de camara puede ser frontal o cenital; el codo es
        # la senal estable y el desplazamiento vertical no es comparable.
        effort = flexion
    result: list[SignalFrame] = []
    for item, score in zip(raw, effort):
        result.append(SignalFrame(
            source_index=int(item["source_index"]),
            timestamp_s=float(item["timestamp_s"]),
            elbow_angle=float(item["elbow_angle"]),
            effort_score=float(score),
            landmark_score=float(item["landmark_score"]),
            shoulder_y=float(item["shoulder_y"]),
            elbow_y=float(item["elbow_y"]),
            hip_y=float(item["hip_y"]),
            head_y=float(item["head_y"]),
            wrist_y=float(item["wrist_y"]),
        ))
    return tuple(result)


class CycleStateMachine:
    """Maquina determinista: reposo -> esfuerzo -> pico -> retorno."""

    def __init__(self, policy: ExercisePolicy) -> None:
        self.policy = policy

    @staticmethod
    def _confidence(rom: float, visible: float, duration: float, policy: ExercisePolicy) -> float:
        rom_score = min(1.0, rom / max(policy.minimum_rom, 1.0))
        visibility_score = min(1.0, visible / max(policy.minimum_visible_fraction, .01))
        duration_score = 1.0 if policy.minimum_duration_s <= duration <= policy.maximum_duration_s else .25
        return round(.45 * rom_score + .35 * visibility_score + .20 * duration_score, 3)

    def detect(
        self,
        signals: Sequence[SignalFrame],
        *,
        cut_timestamps: Iterable[float] = (),
    ) -> tuple[CycleCandidate, ...]:
        if not signals:
            return ()
        timestamps = [frame.timestamp_s for frame in signals]
        if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
            raise ValueError("las senales requieren timestamps estrictamente crecientes")
        cadence = median(np.diff(timestamps)) if len(timestamps) > 1 else .05
        stable_frames = max(2, int(math.ceil(self.policy.stable_rest_s / max(float(cadence), .001))))
        cuts = sorted(float(value) for value in cut_timestamps)
        cut_position = 0
        state = "seek_rest"
        rest_count = 0
        start_i: int | None = None
        peak_i: int | None = None
        minimum_angle = math.inf
        visible_count = 0
        observed_count = 0
        events: list[CycleCandidate] = []

        def finish(index: int, status: str, reason: str | None) -> None:
            nonlocal state, rest_count, start_i, peak_i, minimum_angle, visible_count, observed_count
            if start_i is None:
                return
            resolved_peak = peak_i
            if resolved_peak is None:
                measured = [
                    (signals[pos].elbow_angle, pos)
                    for pos in range(start_i, index + 1)
                    if math.isfinite(signals[pos].elbow_angle)
                ]
                resolved_peak = min(measured)[1] if measured else None
            end_i = index if index >= start_i else None
            start_t = signals[start_i].timestamp_s
            peak_t = signals[resolved_peak].timestamp_s if resolved_peak is not None else None
            end_t = signals[end_i].timestamp_s if end_i is not None else None
            duration = (end_t - start_t) if end_t is not None else 0.0
            start_angle = signals[start_i].elbow_angle
            rom = max(0.0, start_angle - minimum_angle) if math.isfinite(minimum_angle) else 0.0
            visible_fraction = visible_count / observed_count if observed_count else 0.0
            evidence = (
                f"start_frame:{signals[start_i].source_index}",
                f"peak_frame:{signals[resolved_peak].source_index}" if resolved_peak is not None else "peak_missing",
                f"end_frame:{signals[end_i].source_index}" if end_i is not None else "end_missing",
            )
            events.append(CycleCandidate(
                start=round(start_t, 3),
                peak=round(peak_t, 3) if peak_t is not None else None,
                end=round(end_t, 3) if end_t is not None else None,
                start_index=signals[start_i].source_index,
                peak_index=signals[resolved_peak].source_index if resolved_peak is not None else None,
                end_index=signals[end_i].source_index if end_i is not None else None,
                cycle_status=status,
                phase="return_complete" if status == "complete" else state,
                confidence=self._confidence(rom, visible_fraction, duration, self.policy),
                visible_fraction=round(visible_fraction, 3),
                rom_degrees=round(rom, 2),
                evidence=evidence,
                rejection_reason=reason,
            ))
            state = "ready" if signals[index].effort_score <= self.policy.rest_score else "seek_rest"
            rest_count = 1 if state == "ready" else 0
            start_i = peak_i = None
            minimum_angle = math.inf
            visible_count = observed_count = 0

        for index, frame in enumerate(signals):
            while cut_position < len(cuts) and cuts[cut_position] <= frame.timestamp_s:
                if state in {"effort", "peak", "return"}:
                    finish(max(index - 1, start_i or 0), "incomplete", "camera_cut")
                state, rest_count = "seek_rest", 0
                cut_position += 1
            if not frame.observable or frame.landmark_score < self.policy.minimum_landmark_score:
                if state in {"effort", "peak", "return"}:
                    observed_count += 1
                continue
            if state == "seek_rest":
                rest_count = rest_count + 1 if frame.effort_score <= self.policy.rest_score else 0
                if rest_count >= stable_frames:
                    state = "ready"
                continue
            if state == "ready":
                if frame.effort_score <= self.policy.rest_score:
                    rest_count = min(stable_frames, rest_count + 1)
                    continue
                if frame.effort_score >= self.policy.movement_score:
                    start_i = max(0, index - 1)
                    state = "effort"
                    minimum_angle = frame.elbow_angle
                    observed_count = 1
                    visible_count = int(frame.landmark_score >= self.policy.minimum_landmark_score)
                continue

            assert start_i is not None
            observed_count += 1
            visible_count += int(frame.landmark_score >= self.policy.minimum_landmark_score)
            if frame.elbow_angle < minimum_angle:
                minimum_angle, peak_i = frame.elbow_angle, index
            duration = frame.timestamp_s - signals[start_i].timestamp_s
            if duration > self.policy.maximum_duration_s:
                finish(index, "incomplete", "duration_exceeded")
                continue
            if state == "effort" and frame.effort_score >= self.policy.peak_score:
                state = "peak"
                peak_i = index if peak_i is None else peak_i
                continue
            if state == "effort" and frame.effort_score <= self.policy.rest_score:
                finish(index, "incomplete", "partial_rom")
                continue
            if state == "peak" and frame.effort_score < self.policy.peak_score - .08:
                state = "return"
            if state in {"peak", "return"} and frame.effort_score <= self.policy.rest_score:
                rom = signals[start_i].elbow_angle - minimum_angle
                duration = frame.timestamp_s - signals[start_i].timestamp_s
                visible_fraction = visible_count / max(observed_count, 1)
                reason = None
                status = "complete"
                if rom < self.policy.minimum_rom or minimum_angle > self.policy.flexed_angle:
                    status, reason = "incomplete", "partial_rom"
                elif duration + 1e-6 < self.policy.minimum_duration_s:
                    status, reason = "incomplete", "too_fast_to_verify"
                elif visible_fraction < self.policy.minimum_visible_fraction:
                    status, reason = "incomplete", "insufficient_pose_visibility"
                finish(index, status, reason)

        if state in {"effort", "peak", "return"} and start_i is not None:
            finish(len(signals) - 1, "incomplete", "no_return")
        return tuple(events)


class HeadCycleDetector:
    """Detecta ciclos de dominada en cabeza/hombros relativos a las manos.

    La deteccion es solo temporal. Su salida no certifica menton ni rango
    completo; esos juicios pertenecen a una capa de tecnica independiente.
    """

    def __init__(self, policy: ExercisePolicy) -> None:
        self.policy = policy

    def detect(self, signals: Sequence[SignalFrame]) -> tuple[CycleCandidate, ...]:
        if len(signals) < 12:
            return ()
        times = np.asarray([item.timestamp_s for item in signals], dtype=np.float64)
        cadence = float(np.median(np.diff(times)))
        if not .005 <= cadence <= .25:
            return ()
        head = np.asarray([item.head_y for item in signals], dtype=np.float64)
        shoulder = np.asarray([item.shoulder_y for item in signals], dtype=np.float64)
        wrist = np.asarray([item.wrist_y for item in signals], dtype=np.float64)
        elbow = np.asarray([item.elbow_angle for item in signals], dtype=np.float64)
        if not (np.isfinite(head).all() and np.isfinite(shoulder).all() and np.isfinite(elbow).all()):
            return ()
        reference = wrist if np.isfinite(wrist).all() else np.zeros(len(signals))
        head_relative = head - reference
        shoulder_relative = shoulder - reference
        pad = np.pad(head_relative, (2, 2), mode="edge")
        smooth = np.convolve(pad, np.ones(5) / 5.0, mode="valid")
        window = max(3, round(.8 / cadence))
        spacing = max(3, round(.45 / cadence))
        possible: list[tuple[int, float]] = []
        for index in range(window, len(signals) - window):
            if not (smooth[index] <= smooth[index - 1] and smooth[index] < smooth[index + 1]):
                continue
            rise = float(np.max(smooth[index - window:index]) - smooth[index])
            descent = float(np.max(smooth[index + 1:index + window + 1]) - smooth[index])
            prominence = min(rise, descent)
            if prominence >= .04 and elbow[index] <= 110.0:
                possible.append((index, prominence))
        peaks: list[tuple[int, float]] = []
        for index, prominence in possible:
            if peaks and index - peaks[-1][0] < spacing:
                if prominence > peaks[-1][1]:
                    peaks[-1] = (index, prominence)
            else:
                peaks.append((index, prominence))
        if not peaks:
            return ()

        def lowest_position(left: int, right: int) -> int:
            # Empates se resuelven con un frame real, sin interpolar tiempo.
            return left + int(np.argmax(smooth[left:right + 1]))

        valleys = [lowest_position(0, peaks[0][0])]
        valleys.extend(
            lowest_position(peaks[pos][0], peaks[pos + 1][0])
            for pos in range(len(peaks) - 1)
        )
        valleys.append(lowest_position(peaks[-1][0], len(signals) - 1))
        candidates: list[CycleCandidate] = []
        for pos, (peak, _) in enumerate(peaks):
            start, end = valleys[pos], valleys[pos + 1]
            if not start < peak < end:
                continue
            duration = times[end] - times[start]
            head_rise = head_relative[start] - head_relative[peak]
            head_descent = head_relative[end] - head_relative[peak]
            shoulder_rise = shoulder_relative[start] - shoulder_relative[peak]
            shoulder_descent = shoulder_relative[end] - shoulder_relative[peak]
            elbow_rom = min(elbow[start], elbow[end]) - elbow[peak]
            if not (
                self.policy.minimum_duration_s <= duration <= self.policy.maximum_duration_s
                and min(head_rise, head_descent) >= .04
                and min(shoulder_rise, shoulder_descent) >= .025
                and elbow_rom >= 25.0
            ):
                continue
            evidence = (
                f"head_rise:{head_rise:.3f}",
                f"head_descent:{head_descent:.3f}",
                f"shoulder_rise:{shoulder_rise:.3f}",
                f"elbow_rom:{elbow_rom:.1f}",
                f"start_frame:{signals[start].source_index}",
                f"peak_frame:{signals[peak].source_index}",
                f"end_frame:{signals[end].source_index}",
            )
            candidates.append(CycleCandidate(
                start=round(float(times[start]), 3),
                peak=round(float(times[peak]), 3),
                end=round(float(times[end]), 3),
                start_index=signals[start].source_index,
                peak_index=signals[peak].source_index,
                end_index=signals[end].source_index,
                cycle_status="complete",
                phase="return_complete",
                confidence=CycleStateMachine._confidence(
                    float(elbow_rom), 1.0, float(duration), self.policy,
                ),
                visible_fraction=1.0,
                rom_degrees=round(float(elbow_rom), 2),
                evidence=evidence,
            ))
        return tuple(candidates)


def _segment_index(timestamp: float, boundaries: Sequence[float]) -> int:
    return sum(float(boundary) <= float(timestamp) for boundary in boundaries)


def _nearest_scene(
    samples: Sequence[SceneSample],
    timestamp: float,
    limit: float,
    calibration_timestamps: Sequence[float] = (),
) -> SceneSample | None:
    if not samples:
        return None
    segment = _segment_index(timestamp, calibration_timestamps)
    eligible = [
        item for item in samples
        if _segment_index(item.timestamp_s, calibration_timestamps) == segment
    ]
    if not eligible:
        return None
    nearest = min(eligible, key=lambda item: abs(item.timestamp_s - timestamp))
    return nearest if abs(nearest.timestamp_s - timestamp) <= limit else None


class BiomechanicsValidator:
    def __init__(self, *, max_scene_gap_s: float = .75) -> None:
        self.max_scene_gap_s = float(max_scene_gap_s)

    def validate(
        self,
        track: PoseTrack,
        exercise: str,
        candidates: Sequence[CycleCandidate],
        scene_samples: Sequence[SceneSample],
        calibration_timestamps: Sequence[float] = (),
    ) -> tuple[CycleDecision, ...]:
        decisions: list[CycleDecision] = []
        normalized_exercise = "pull_up" if exercise == "chin_up" else exercise
        rule_name = {"pull_up": "chin_over_bar", "dip": "dip_depth", "push_up": "floor_depth"}[normalized_exercise]
        reference_kind = {"pull_up": "pull_up_bar", "dip": "dip_bars", "push_up": "floor"}[normalized_exercise]
        for candidate in candidates:
            base_rules = ("stable_start", "opposite_phase", "measured_return", "pose_visibility")
            if candidate.cycle_status != "complete":
                uncertain_reasons = {"camera_cut", "insufficient_pose_visibility"}
                status = "unverifiable" if candidate.rejection_reason in uncertain_reasons else "invalid"
                decisions.append(CycleDecision(
                    candidate, status, candidate.rejection_reason or "incomplete_cycle",
                    base_rules, candidate.evidence,
                ))
                continue
            if candidate.peak is None or candidate.peak_index is None:
                decisions.append(CycleDecision(candidate, "unverifiable", "peak_missing", base_rules, candidate.evidence))
                continue
            if normalized_exercise == "pull_up":
                # El conteo del ciclo no depende de observar el menton. Un
                # angulo de espalda no permite certificar su cruce de barra.
                decisions.append(CycleDecision(
                    candidate, "unverifiable", "bar_clearance_not_assessed",
                    base_rules + ("head_and_shoulder_ascent", "elbow_flexion"),
                    candidate.evidence,
                ))
                continue
            sample = _nearest_scene(
                scene_samples,
                candidate.peak,
                self.max_scene_gap_s,
                calibration_timestamps,
            )
            if sample is None:
                decisions.append(CycleDecision(candidate, "unverifiable", "scene_evidence_missing_near_peak", base_rules, candidate.evidence))
                continue
            if not bool(sample.rules.get(rule_name, False)):
                decisions.append(CycleDecision(candidate, "unverifiable", f"{rule_name}_not_observable", base_rules + (rule_name,), candidate.evidence))
                continue
            reference = sample.visible_reference(reference_kind)
            if reference is None and normalized_exercise != "push_up":
                decisions.append(CycleDecision(candidate, "unverifiable", f"{reference_kind}_not_visible", base_rules + (rule_name,), candidate.evidence))
                continue

            if normalized_exercise == "dip":
                signal = extract_signals(track, exercise)[candidate.peak_index]
                if not (math.isfinite(signal.shoulder_y) and math.isfinite(signal.elbow_y)):
                    status, reason = "unverifiable", "shoulder_elbow_depth_occluded"
                elif signal.shoulder_y >= signal.elbow_y - .015:
                    status, reason = "valid", "shoulder_reached_elbow_depth"
                else:
                    status, reason = "invalid", "dip_depth_not_reached"
            else:
                signal = extract_signals(track, exercise)[candidate.peak_index]
                if not math.isfinite(signal.elbow_angle):
                    status, reason = "unverifiable", "elbow_depth_occluded"
                elif signal.elbow_angle <= POLICIES[exercise].flexed_angle:
                    status, reason = "valid", "push_up_depth_observed"
                else:
                    status, reason = "invalid", "push_up_depth_not_reached"
            decisions.append(CycleDecision(
                candidate, status, reason, base_rules + (rule_name,),
                candidate.evidence + (f"scene_sample:{sample.timestamp_s:.3f}",),
            ))
        return tuple(decisions)


def analyze_track(
    track: PoseTrack,
    exercise: str,
    *,
    scene_review: Mapping[str, Any] | None = None,
    cut_timestamps: Iterable[float] = (),
    calibration_timestamps: Iterable[float] = (),
    head_motion_weight: float = 0.5,
) -> dict[str, Any]:
    """Run the calisthenics analysis as an observation, without deciding technique."""
    normalized = "pull_up" if exercise == "chin_up" else exercise
    calibration_boundaries = tuple(sorted(float(value) for value in calibration_timestamps))
    hard_cuts = tuple(sorted(float(value) for value in cut_timestamps))
    effective_head_weight = head_motion_weight if not calibration_boundaries else 0.0
    signals = extract_signals(track, exercise, head_motion_weight=effective_head_weight)
    state_candidates = CycleStateMachine(POLICIES[exercise]).detect(signals, cut_timestamps=hard_cuts)
    candidates = state_candidates
    detector = "elbow_shoulder_state_machine"
    if normalized == "pull_up" and not calibration_boundaries and not hard_cuts:
        head_candidates = HeadCycleDetector(POLICIES[exercise]).detect(signals)
        complete = [item for item in state_candidates if item.cycle_status == "complete"]
        if (
            head_candidates
            and len(head_candidates) >= len(complete)
            and all(
                any(abs(float(head.peak) - float(old.peak)) <= .45 for head in head_candidates)
                for old in complete if old.peak is not None
            )
        ):
            candidates = head_candidates + tuple(
                item for item in state_candidates
                if item.cycle_status != "complete"
                and not any(
                    head.start <= item.start <= head.end
                    for head in head_candidates
                )
            )
            candidates = tuple(sorted(candidates, key=lambda item: item.start))
            detector = "head_shoulder_elbow_cycles"
    samples = scene_samples_from_review(scene_review or {})
    decisions = BiomechanicsValidator().validate(
        track,
        exercise,
        candidates,
        samples,
        calibration_boundaries,
    )
    statuses = {status: 0 for status in sorted(TECHNICAL_STATUSES)}
    for decision in decisions:
        statuses[decision.technical_status] += 1
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "analysis_only",
        "exercise": normalized,
        "source_frames": len(track.timestamps),
        "head_motion_weight": effective_head_weight,
        "detector": detector,
        "candidates": len(candidates),
        "complete_cycles": sum(item.cycle_status == "complete" for item in candidates),
        "technical_statuses": statuses,
        "events": [item.as_dict() for item in decisions],
        "calibration_resets": [round(value, 3) for value in calibration_boundaries],
        "automatic_promotion": False,
    }


__all__ = [
    "BiomechanicsValidator", "CycleCandidate", "CycleDecision", "CycleStateMachine", "HeadCycleDetector",
    "ExercisePolicy", "POLICIES", "SCHEMA_VERSION", "SceneSample", "SignalFrame",
    "analyze_track", "extract_signals", "scene_samples_from_review",
]
