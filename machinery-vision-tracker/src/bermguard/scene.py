"""Causal scene-cut detection used to reset temporal state.

Two kinds of scene change are detected:

* a hard cut: appearance changes between consecutive frames without a
  structural match, exactly as before;
* structural drift (G5.3): the supplied synthetic clips can morph from one
  place into another over a second or more without any single frame looking
  like a cut.  Comparing the structure of the current frame with the frame one
  lag ago, only while both are daylight frames, exposes that drift.  When the
  lagged correlation stays below the drift threshold for several consecutive
  frames the detector emits a deferred cut once, then re-arms only after the
  lagged correlation recovers.  Day/night ramps never qualify because one of
  the two compared frames is not daylight.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import cv2
import numpy as np

from .g2_config import G2Settings


_DRIFT_REARM_CORRELATION = 0.5


@dataclass(frozen=True)
class SceneDecision:
    is_cut: bool
    histogram_correlation: float
    mean_difference: float
    mean_intensity: float
    illumination_transition: bool
    structural_correlation: float
    structural_match: bool
    lagged_structural_correlation: float = 1.0
    structural_drift: bool = False


class SceneCutDetector:
    def __init__(self, settings: G2Settings) -> None:
        self.settings = settings
        self._previous_gray: Optional[np.ndarray] = None
        self._previous_histogram: Optional[np.ndarray] = None
        self._previous_structure: Optional[np.ndarray] = None
        self._illumination_mode: Optional[str] = None
        self._frames_since_cut = settings.scene_cooldown_frames
        self._history: Deque[tuple[np.ndarray, float]] = deque(
            maxlen=settings.scene_drift_lag_frames + 1
        )
        self._drift_frames = 0
        self._drift_armed = True

    def reset(self) -> None:
        self._previous_gray = None
        self._previous_histogram = None
        self._previous_structure = None
        self._illumination_mode = None
        self._frames_since_cut = self.settings.scene_cooldown_frames
        self._history.clear()
        self._drift_frames = 0
        self._drift_armed = True

    def _lagged_drift(
        self, structure: np.ndarray, mean_intensity: float
    ) -> tuple[float, bool]:
        """Return the lagged structural correlation and whether drift fired."""

        lagged_correlation = 1.0
        drift = False
        day = self.settings.scene_day_above_mean
        if len(self._history) == self._history.maxlen:
            old_structure, old_intensity = self._history[0]
            both_day = mean_intensity >= day and old_intensity >= day
            if (
                both_day
                and float(np.std(old_structure)) >= 1e-5
                and float(np.std(structure)) >= 1e-5
            ):
                value = np.corrcoef(old_structure.reshape(-1), structure.reshape(-1))[0, 1]
                lagged_correlation = float(value) if np.isfinite(value) else 0.0
                if lagged_correlation < self.settings.scene_drift_correlation:
                    self._drift_frames += 1
                else:
                    self._drift_frames = 0
                    if lagged_correlation >= _DRIFT_REARM_CORRELATION:
                        self._drift_armed = True
                if (
                    self._drift_armed
                    and self._drift_frames >= self.settings.scene_drift_frames
                ):
                    drift = True
                    self._drift_armed = False
                    self._drift_frames = 0
            else:
                self._drift_frames = 0
        self._history.append((structure, mean_intensity))
        return lagged_correlation, drift

    def update(self, frame: np.ndarray) -> SceneDecision:
        gray = cv2.cvtColor(
            cv2.resize(frame, (96, 54), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2GRAY,
        )
        histogram = cv2.calcHist([gray], [0], None, [32], [0, 256])
        cv2.normalize(histogram, histogram)
        equalized = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 6)).apply(gray)
        gx = cv2.Sobel(equalized, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(equalized, cv2.CV_32F, 0, 1, ksize=3)
        structure = cv2.magnitude(gx, gy)
        mean_intensity = float(np.mean(gray))
        current_mode: Optional[str]
        if mean_intensity <= self.settings.scene_night_below_mean:
            current_mode = "night"
        elif mean_intensity >= self.settings.scene_day_above_mean:
            current_mode = "day"
        else:
            current_mode = None
        if self._previous_gray is None or self._previous_histogram is None:
            self._previous_gray = gray
            self._previous_histogram = histogram
            self._previous_structure = structure
            self._illumination_mode = current_mode
            self._history.append((structure, mean_intensity))
            return SceneDecision(False, 1.0, 0.0, mean_intensity, False, 1.0, True)

        correlation = float(
            cv2.compareHist(
                self._previous_histogram, histogram, cv2.HISTCMP_CORREL
            )
        )
        mean_difference = float(
            np.mean(cv2.absdiff(self._previous_gray, gray)) / 255.0
        )
        eligible = self._frames_since_cut >= self.settings.scene_cooldown_frames
        illumination_transition = (
            current_mode is not None
            and self._illumination_mode is not None
            and current_mode != self._illumination_mode
        )
        previous_structure = self._previous_structure
        if (
            previous_structure is None
            or float(np.std(previous_structure)) < 1e-5
            or float(np.std(structure)) < 1e-5
        ):
            structural_correlation = 0.0
        else:
            structural_correlation = float(
                np.corrcoef(previous_structure.reshape(-1), structure.reshape(-1))[0, 1]
            )
            if not np.isfinite(structural_correlation):
                structural_correlation = 0.0
        structural_match = structural_correlation >= 0.34
        appearance_change = (
            correlation < self.settings.scene_histogram_threshold
            or mean_difference > self.settings.scene_mean_difference_threshold
        )
        # Day/night ramps in the supplied fixed-camera clips are not scene cuts.
        # Reset temporal geometry only when the appearance change is also a
        # structural change.  Uniform black/white test frames intentionally have
        # no reliable structure and remain a cut.
        hard_cut = eligible and appearance_change and not structural_match
        lagged_correlation, drift = self._lagged_drift(structure, mean_intensity)
        is_cut = hard_cut or (eligible and drift)
        self._previous_gray = gray
        self._previous_histogram = histogram
        self._previous_structure = structure
        self._frames_since_cut = 0 if is_cut else self._frames_since_cut + 1
        if is_cut:
            self._history.clear()
            self._history.append((structure, mean_intensity))
            self._drift_frames = 0
        if current_mode is not None:
            self._illumination_mode = current_mode
        return SceneDecision(
            is_cut,
            correlation,
            mean_difference,
            mean_intensity,
            illumination_transition,
            structural_correlation,
            structural_match,
            lagged_structural_correlation=lagged_correlation,
            structural_drift=bool(drift and eligible),
        )
