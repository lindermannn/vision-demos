"""Per-athlete height calibration for pull-up reps.

The athlete marks reps they know are valid: the highest ones define
"ROM full alto", the hardest valid ones define the minimum "ROM full", and
every other verified rep can only lower that minimum. Grades are relative to
this human calibration and the camera setups it came from; they are not an
independent chin-over-bar verdict.
"""

from __future__ import annotations

import math
import re
import statistics
from typing import Any, Iterable, Mapping

GRADES = {
    "rom_full_alto": "ROM full alto",
    "rom_semi_alto": "ROM semi alto",
    "rom_full": "ROM full",
    "bajo_minimo": "Bajo el mínimo",
    "sin_medida": "Sin medida",
}
VALID_GRADES = frozenset({"rom_full_alto", "rom_semi_alto", "rom_full"})


def parse_reps(text: str, total: int) -> list[int]:
    """Parse "1-5, 8" into rep numbers (1-based) inside 1..total."""
    reps: set[int] = set()
    for part in filter(None, (item.strip() for item in str(text).split(","))):
        match = re.fullmatch(r"(\d+)\s*(?:-\s*(\d+))?", part)
        if not match:
            raise ValueError(f"Rango de reps inválido: «{part}». Usa el formato 1-5, 8")
        low = int(match.group(1))
        high = int(match.group(2) or low)
        if low > high or low < 1 or high > total:
            raise ValueError(f"«{part}» está fuera de las {total} reps detectadas")
        reps.update(range(low, high + 1))
    return sorted(reps)


def _measured(events: list[Mapping[str, Any]], reps: Iterable[int]) -> list[float]:
    values = (events[rep - 1].get("height_above_bar") for rep in reps)
    return [float(value) for value in values if value is not None and math.isfinite(float(value))]


def build_calibration(references: list[Mapping[str, Any]], *, tolerance: float = .02,
                      semi_fraction: float = .20) -> dict[str, Any]:
    """Thresholds from human-verified reps.

    Each reference holds the detected ``events`` of one clip plus the rep
    numbers the athlete marked as ``alto``, ``minimo`` and ``validas``.
    """
    if not 0 <= tolerance <= .2:
        raise ValueError("La tolerancia debe estar entre 0 y 0,20")
    if not .05 <= semi_fraction <= .95:
        raise ValueError("El inicio de semi alto debe estar entre 5 % y 95 %")
    high: list[float] = []
    minimum: list[float] = []
    valid: list[float] = []
    for reference in references:
        events = list(reference["events"])
        high += _measured(events, reference.get("alto", ()))
        minimum += _measured(events, reference.get("minimo", ()))
        valid += _measured(events, set(reference.get("alto", ())) | set(reference.get("minimo", ()))
                           | set(reference.get("validas", ())))
    if len(high) < 3 or len(minimum) < 3:
        raise ValueError("Marca al menos 3 reps altas y 3 reps mínimas con altura medida")
    alto_min = min(high) - tolerance
    full_reference = statistics.median(minimum)
    full_min = min(valid) - tolerance
    semi_min = full_reference + semi_fraction * (alto_min - full_reference)
    if not full_min < semi_min < alto_min:
        raise ValueError("Las reps altas deben subir claramente más que las mínimas")
    return {
        "schema_version": "pullup-calibration/1.0",
        "unit": "head_height_above_bar_in_torso_lengths",
        "thresholds": {key: round(value, 3) for key, value in (
            ("rom_full_alto", alto_min), ("rom_semi_alto", semi_min), ("rom_full", full_min))},
        "rules": {"tolerance": tolerance, "semi_fraction": semi_fraction},
        "evidence": {"alto": [round(v, 3) for v in high], "minimo": [round(v, 3) for v in minimum],
                     "validas": len(valid)},
    }


def grade(height: float | None, calibration: Mapping[str, Any]) -> str:
    if height is None or not math.isfinite(float(height)):
        return "sin_medida"
    thresholds = calibration["thresholds"]
    for key in ("rom_full_alto", "rom_semi_alto", "rom_full"):
        if float(height) >= float(thresholds[key]):
            return key
    return "bajo_minimo"


def apply_calibration(events: list[dict[str, Any]], calibration: Mapping[str, Any]) -> None:
    for event in events:
        event["grade"] = grade(event.get("height_above_bar"), calibration)
        event["grade_label"] = GRADES[event["grade"]]
