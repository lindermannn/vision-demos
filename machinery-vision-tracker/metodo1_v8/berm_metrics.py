"""Berm size metrics for the V8 experiment.

Two ideas, both physical rather than tuned to a particular clip:

1. ``GroundScaleModel``.  For a pinhole camera looking at a flat operating
   grade, an object of fixed real height ``H`` standing on that grade projects
   to an apparent height that is *linear in the image row of its ground
   contact*: with camera height ``Hc`` above the grade and horizon row ``cy``,
   ``h(y) = (H / Hc) * (y - cy)``.  Fitting ``h = a*y + b`` over confirmed
   machinery therefore yields a scale valid at any row, including rows where no
   vehicle happens to stand.  V7 instead required a vehicle whose contact row
   was already within 10 % of the berm toe, which is why video 02 produced no
   metric scale at all.  The fitted horizon ``-b/a`` is a free sanity check.

2. ``profile_statistics``.  The assessment asks for a height *profile*, not a
   single number, so the supported columns are summarised as p10/median/p90
   instead of collapsing them to a median alone.

Metres remain an estimate under declared assumptions (reference vehicle height,
flat grade, berm crest roughly vertical above its toe), never topography.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


# Declared relative spread of the reference vehicle height.  The corpus mixes
# haul trucks with dozers, so the assumed 5 m is a class average, not a
# measurement; the published interval must say so.
REFERENCE_HEIGHT_RELATIVE_SPREAD = 0.20
MIN_SAMPLES = 12
MIN_ROW_SPREAD_RATIO = 0.06      # contacts must span enough rows to fit a slope
MAX_EXTRAPOLATION_RATIO = 0.35   # how far beyond the observed rows we trust it
SAMPLE_MEMORY = 240


@dataclass(frozen=True)
class ScaleEstimate:
    pixels_per_meter: float
    low: float
    high: float
    horizon_row: float
    samples: int
    residual_ratio: float
    basis: str
    disagreement_ratio: float = 1.0

    def to_meters(self, height_px: float) -> tuple[float, float, float]:
        """Return (estimate, low, high) metres for a pixel height."""

        return (
            height_px / self.pixels_per_meter,
            height_px / self.high,
            height_px / self.low,
        )


class GroundScaleModel:
    """Apparent vehicle height as a linear function of its ground-contact row.

    The scale is calibrated on the largest machine class, not on a fleet
    average.  DS 132 sizes a berm against "el camión de mayor envergadura", and
    the corpus mixes haul trucks with dozers, so taking the median apparent
    height and calling it a 5 m vehicle understates every metre it produces.
    ``calibration_quantile`` selects the tall end of the population and
    ``reference_height_m`` is the declared height of that class.
    """

    def __init__(self, reference_height_m: float, calibration_quantile: float = 0.80) -> None:
        self.reference_height_m = float(reference_height_m)
        self.calibration_quantile = float(calibration_quantile)
        self._samples: deque[tuple[float, float]] = deque(maxlen=SAMPLE_MEMORY)

    def reset(self) -> None:
        self._samples.clear()

    def observe(self, detections: Sequence[object]) -> int:
        """Record (contact row, apparent height) for each confirmed machine.

        Confirmed tracks only.  The detector's raw proposals include small and
        short-lived boxes that bias the apparent height downward and therefore
        inflate every metre produced from it; the tracker's confirmed output is
        also the population the proximity model already trusts.
        """

        added = 0
        for detection in detections:
            box = detection.bbox
            height = float(box.y2 - box.y1)
            width = float(box.x2 - box.x1)
            if height <= 8.0 or width <= 8.0:
                continue
            aspect = width / height
            if not 0.4 <= aspect <= 4.5:
                continue
            self._samples.append((float(box.y2), height))
            added += 1
        return added

    def estimate(self, row: float, frame_height: int) -> ScaleEstimate | None:
        """Scale at ``row``; ``None`` when the geometry does not support one.

        The direct rule wins whenever machinery actually stands near the target
        row, because it involves no model and no extrapolation.  The regression
        exists for the rows where nothing stands, which is the case the earlier
        version could not serve at all.
        """

        if len(self._samples) < MIN_SAMPLES:
            return None
        rows = np.array([sample[0] for sample in self._samples], dtype=np.float64)
        heights = np.array([sample[1] for sample in self._samples], dtype=np.float64)

        direct = self._nearest_row_fallback(row, rows, heights, frame_height)
        modelled = self._regression(row, rows, heights, frame_height)
        if direct is None:
            return modelled
        if modelled is None:
            return direct
        ratio = max(direct.pixels_per_meter, modelled.pixels_per_meter) / max(
            min(direct.pixels_per_meter, modelled.pixels_per_meter), 1e-6
        )
        return ScaleEstimate(
            pixels_per_meter=direct.pixels_per_meter,
            low=direct.low,
            high=direct.high,
            horizon_row=modelled.horizon_row,
            samples=direct.samples,
            residual_ratio=direct.residual_ratio,
            basis="nearest_contact_rows",
            disagreement_ratio=float(ratio),
        )

    def _regression(
        self,
        row: float,
        rows: np.ndarray,
        heights: np.ndarray,
        frame_height: int,
    ) -> ScaleEstimate | None:
        """Fit apparent height against contact row, with plausibility gates."""

        spread = float(rows.max() - rows.min())
        if spread < MIN_ROW_SPREAD_RATIO * frame_height:
            return None
        # One median per row bucket: a crowd of small distant boxes at the same
        # depth must not outvote the few machines that define the slope.
        bucket_size = max(4.0, frame_height / 100.0)
        buckets: dict[int, list[float]] = {}
        for sample_row, sample_height in zip(rows, heights):
            buckets.setdefault(int(sample_row // bucket_size), []).append(sample_height)
        if len(buckets) < 6:
            return None
        bucket_rows = np.array(
            [(key + 0.5) * bucket_size for key in sorted(buckets)], dtype=np.float64
        )
        bucket_heights = np.array(
            [float(np.quantile(buckets[key], self.calibration_quantile))
             for key in sorted(buckets)],
            dtype=np.float64,
        )

        slope, intercept, residual_ratio = _robust_line(bucket_rows, bucket_heights)
        if slope is None or not 0.05 <= slope <= 3.0:
            return None
        margin = MAX_EXTRAPOLATION_RATIO * spread
        if not (rows.min() - margin <= row <= rows.max() + margin):
            return None
        horizon = -intercept / slope
        # The target must sit well below the fitted horizon; close to it the
        # intercept error dominates and the scale diverges.
        if row - horizon < 0.05 * frame_height:
            return None
        predicted = slope * row + intercept
        if predicted <= 1.0:
            return None
        pixels_per_meter = predicted / self.reference_height_m
        spread_factor = 1.0 + REFERENCE_HEIGHT_RELATIVE_SPREAD + residual_ratio
        return ScaleEstimate(
            pixels_per_meter=pixels_per_meter,
            low=pixels_per_meter / spread_factor,
            high=pixels_per_meter * spread_factor,
            horizon_row=float(horizon),
            samples=len(self._samples),
            residual_ratio=float(residual_ratio),
            basis="ground_plane_row_regression",
        )

    def _nearest_row_fallback(
        self,
        row: float,
        rows: np.ndarray,
        heights: np.ndarray,
        frame_height: int,
    ) -> ScaleEstimate | None:
        """V7's rule: vehicles whose contact row is close to the target row."""

        near = np.abs(rows - row) <= 0.10 * frame_height
        if np.count_nonzero(near) < 3:
            return None
        predicted = float(np.quantile(heights[near], self.calibration_quantile))
        if predicted <= 1.0:
            return None
        pixels_per_meter = predicted / self.reference_height_m
        scatter = float(np.median(np.abs(heights[near] - predicted)) / predicted)
        spread_factor = 1.0 + REFERENCE_HEIGHT_RELATIVE_SPREAD + scatter
        return ScaleEstimate(
            pixels_per_meter=pixels_per_meter,
            low=pixels_per_meter / spread_factor,
            high=pixels_per_meter * spread_factor,
            horizon_row=float("nan"),
            samples=int(np.count_nonzero(near)),
            residual_ratio=scatter,
            basis="nearest_contact_rows",
        )


def _robust_line(rows: np.ndarray, heights: np.ndarray) -> tuple[float | None, float, float]:
    """Least squares with one trimming pass over the worst fifth of residuals."""

    if len(rows) < 3:
        return None, 0.0, 1.0
    slope, intercept = np.polyfit(rows, heights, 1)
    residuals = np.abs(heights - (slope * rows + intercept))
    if len(rows) >= MIN_SAMPLES:
        keep = residuals <= np.quantile(residuals, 0.80)
        if np.count_nonzero(keep) >= max(3, len(rows) // 2):
            slope, intercept = np.polyfit(rows[keep], heights[keep], 1)
            residuals = np.abs(heights[keep] - (slope * rows[keep] + intercept))
            heights = heights[keep]
    scale = max(float(np.median(heights)), 1.0)
    return float(slope), float(intercept), float(np.median(residuals) / scale)


@dataclass(frozen=True)
class ProfileStatistics:
    columns: int
    coverage: float
    p10_px: float
    median_px: float
    p90_px: float

    @property
    def dispersion_px(self) -> float:
        return self.p90_px - self.p10_px


def profile_statistics(heights: Iterable[float], total_columns: int) -> ProfileStatistics | None:
    values = np.asarray([value for value in heights if np.isfinite(value)], dtype=np.float64)
    if values.size == 0 or total_columns <= 0:
        return None
    p10, median, p90 = (float(value) for value in np.percentile(values, (10.0, 50.0, 90.0)))
    return ProfileStatistics(
        columns=int(values.size),
        coverage=float(values.size / total_columns),
        p10_px=p10,
        median_px=median,
        p90_px=p90,
    )


# The validator runs before the tracker inside a frame, so the scale model
# consumes the previous frame's confirmed tracks.  One frame of lag is
# irrelevant for a static berm and buys a far cleaner sample population than
# the detector's raw proposals.
CONFIRMED: list[object] = []


def publish_confirmed(detections) -> None:
    CONFIRMED[:] = [
        detection for detection in detections
        if getattr(detection, "track_id", None) is not None
    ]


def peek_confirmed() -> tuple[object, ...]:
    return tuple(CONFIRMED)


def clear_confirmed() -> None:
    CONFIRMED.clear()


# The height reported when current support exists is measured on the candidate's
# own columns, so it can differ from the anchored geometry by up to twice the
# matching tolerance.  The renderer draws it, otherwise the panel would show a
# number that corresponds to nothing on screen.
MEASURED: dict[str, object] = {}


def publish_measured(crest, toe) -> None:
    MEASURED["crest"] = crest
    MEASURED["toe"] = toe


def clear_measured() -> None:
    MEASURED.clear()


def peek_measured():
    return MEASURED.get("crest"), MEASURED.get("toe")


# Frame-level records written beside the standard artifacts.  The pipeline's own
# CSV schema is fixed and belongs to the delivery repository, so the experiment
# publishes its extra measurements in a sidecar instead of editing it.
RECORDS: list[dict[str, object]] = []

FIELDS = (
    "video_id",
    "method",
    "frame_index",
    "timestamp_s",
    "lighting",
    "state",
    "measurement_basis",
    "anchor_frame",
    "profile_age_frames",
    "raw_status",
    "search_mode",
    "matched_columns",
    "profile_columns",
    "match_fraction",
    "residual_median_px",
    "height_px",
    "height_p10_px",
    "height_p90_px",
    "height_m",
    "height_m_low",
    "height_m_high",
    "height_m_required",
    "norm_article",
    "norm_verdict",
    "wheel_ratio",
    "scale_reference_height_m",
    "pixels_per_meter",
    "scale_basis",
    "scale_samples",
    "scale_horizon_row",
    "crest_y_median",
    "toe_y_median",
)


def record(entry: dict[str, object]) -> None:
    RECORDS.append({field: entry.get(field) for field in FIELDS})


def write_records(destination) -> int:
    import csv
    from collections import defaultdict
    from pathlib import Path

    destination = Path(destination)
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for entry in RECORDS:
        grouped[str(entry.get("video_id"))].append(entry)
    written = 0
    for video_id, entries in grouped.items():
        path = destination / f"berm_profile_v8_{video_id}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(FIELDS))
            writer.writeheader()
            writer.writerows(entries)
        written += 1
    return written
