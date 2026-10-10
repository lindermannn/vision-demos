"""Paired crest/body/toe extraction utilities for G5.1.

The helpers in this module never invent a full-width band.  They select one joint
crest-toe hypothesis, retain only locally supported columns, and may reject every
candidate.  The two public extractors intentionally use different evidence maps:
signed photometric contrast for method 1 and relative-depth slope changes for
method 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable
import warnings

import cv2
import numpy as np

from ..contracts import BermObservation, FrameContext, ObservationStatus
from ..g2_config import G2Settings
from .result import SegmentationResult


@dataclass(frozen=True)
class _CandidateRow:
    crest_y: int
    height: int
    score: np.ndarray
    evidence: np.ndarray
    supported: np.ndarray
    bright_fraction: float = 0.0


def _largest_true_run(values: np.ndarray) -> int:
    maximum = current = 0
    for value in values:
        if bool(value):
            current += 1
            maximum = max(maximum, current)
        else:
            current = 0
    return maximum


def _runs(values: np.ndarray) -> Iterable[tuple[int, int]]:
    padded = np.pad(values.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return zip(starts.tolist(), ends.tolist())


def _clean_support(values: np.ndarray, width: int) -> np.ndarray:
    support = values.astype(np.uint8).reshape(1, -1)
    close_width = max(3, int(round(width * 0.018)) | 1)
    support = cv2.morphologyEx(
        support,
        cv2.MORPH_CLOSE,
        np.ones((1, close_width), dtype=np.uint8),
    ).reshape(-1).astype(bool)
    minimum_run = max(5, int(round(width * 0.035)))
    cleaned = np.zeros_like(support)
    for start, end in _runs(support):
        if end - start >= minimum_run:
            cleaned[start:end] = True
    return cleaned


def _median_smooth(values: np.ndarray, width: int) -> np.ndarray:
    kernel = max(3, int(round(width * 0.025)) | 1)
    radius = kernel // 2
    padded = np.pad(values.astype(np.float32), (radius, radius), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, kernel)
    median = np.median(windows, axis=1).astype(np.float32)
    gaussian = max(3, int(round(width * 0.018)) | 1)
    return cv2.GaussianBlur(median.reshape(1, -1), (gaussian, 1), 0).reshape(-1)


def _robust_linear_profile(
    xs: np.ndarray,
    values: np.ndarray,
    support: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Fit the low-frequency, near-linear geometry of a compacted-earth berm.

    Short-lived notches from vehicles, light blooms and dust are outliers, not
    topographic vertices.  Iteratively reweighted least squares keeps the broad
    grade trend while preventing those local objects from becoming corners in
    the rendered crest or toe.
    """

    valid = np.logical_and(support, np.isfinite(values))
    if np.count_nonzero(valid) < 3:
        return values.astype(np.float32, copy=True), float("inf"), 0.0
    centre = float(np.mean(xs[valid]))
    scale_x = max(float(np.ptp(xs[valid])), 1.0)
    normalized_x = (xs.astype(np.float64) - centre) / scale_x
    design = np.column_stack([np.ones(len(xs)), normalized_x])
    active_design = design[valid]
    active_values = values[valid].astype(np.float64)
    coefficients, *_ = np.linalg.lstsq(active_design, active_values, rcond=None)
    weights = np.ones(len(active_values), dtype=np.float64)
    for _ in range(6):
        residual = active_values - active_design @ coefficients
        median_residual = float(np.median(residual))
        robust_scale = 1.4826 * float(np.median(np.abs(residual - median_residual)))
        robust_scale = max(robust_scale, 0.75)
        normalized_residual = np.abs(residual - median_residual) / (2.5 * robust_scale)
        weights = 1.0 / np.maximum(1.0, normalized_residual)
        weighted_design = active_design * np.sqrt(weights)[:, None]
        weighted_values = active_values * np.sqrt(weights)
        coefficients, *_ = np.linalg.lstsq(
            weighted_design,
            weighted_values,
            rcond=None,
        )
    fitted = (design @ coefficients).astype(np.float32)
    residual = active_values - active_design @ coefficients
    median_absolute_residual = float(np.median(np.abs(residual)))
    slope_px_per_px = float(coefficients[1] / scale_x)
    return fitted, median_absolute_residual, slope_px_per_px


def _mask_from_supported_profiles(
    height: int,
    width: int,
    xs: np.ndarray,
    crest: np.ndarray,
    toe: np.ndarray,
    support: np.ndarray,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for start, end in _runs(support):
        if end - start < 2:
            continue
        run_x = xs[start:end]
        polygon = np.vstack(
            [
                np.column_stack([run_x, crest[start:end]]),
                np.column_stack([run_x[::-1], toe[start:end][::-1]]),
            ]
        )
        cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32)], 255)
    return mask


def _unknown(
    context: FrameContext,
    reason: str,
    *,
    confidence: float = 0.0,
    diagnostics: dict[str, Any] | None = None,
) -> SegmentationResult:
    details: dict[str, Any] = {
        "reason": reason,
        "confidence": confidence,
        "spatially_valid": False,
    }
    if diagnostics:
        details.update(diagnostics)
    return SegmentationResult(
        BermObservation(
            context=context,
            status=ObservationStatus.UNKNOWN,
            diagnostics=details,
        ),
        np.zeros((context.height, context.width), dtype=np.uint8),
        confidence,
    )


def _select_and_pack(
    rows: list[_CandidateRow],
    context: FrameContext,
    settings: G2Settings,
    *,
    target_width: int,
    target_height: int,
    x0: int,
    x1: int,
    evidence_threshold: float,
    confidence_threshold: float,
    source: str,
    source_diagnostics: dict[str, Any],
    operating_surface: tuple[float, float] | None = None,
    nearest_toe_ratio: float = 0.0,
) -> SegmentationResult:
    if not rows:
        return _unknown(
            context,
            "no_paired_candidates",
            diagnostics={"profile_source": source, **source_diagnostics},
        )
    roi_width = x1 - x0
    best: _CandidateRow | None = None
    best_quality = -1.0
    best_coverage = 0.0
    best_run = 0.0
    best_strength = 0.0
    rejected = {
        "weak_pair": 0, "short_support": 0, "low_coverage": 0,
        "below_operating_surface": 0, "inside_operating_surface": 0,
        "contains_far_contact": 0,
    }
    # The PDF defines crest and base relative to the operating surface the
    # machinery drives on.  When confirmed tracks have revealed that surface
    # (far and near contact rows in source pixels), no candidate whose toe lies
    # below the nearest contact can be the berm: it is road or foreground.
    scale_y_rows = target_height / context.height
    surface = None
    if operating_surface is not None:
        surface = (operating_surface[0] * scale_y_rows, operating_surface[1] * scale_y_rows)
    surface_margin = 0.03 * target_height
    surface_far_slack = 0.06 * target_height
    diagnostic_max_coverage = 0.0
    diagnostic_max_run = 0.0
    diagnostic_max_strength = 0.0
    qualified_candidates: list[dict[str, float]] = []
    qualified_rows: list[tuple[_CandidateRow, float, float, float]] = []
    crest_low = min(row.crest_y for row in rows)
    crest_high = max(row.crest_y for row in rows) + 1
    # Support statistics for every candidate row at once.  Closing bridges
    # small gaps and opening with the minimum run length keeps only runs at
    # least that long, exactly as the per-row helper does; the running-count
    # trick yields the longest run per row without a Python loop.  This block
    # replaced per-row loops that cost about 0.6 s per frame in method 1.
    support_matrix = np.stack([row.supported for row in rows], axis=0).astype(np.uint8)
    close_width = max(3, int(round(roi_width * 0.018)) | 1)
    minimum_run = max(5, int(round(roi_width * 0.035)))
    closed = cv2.morphologyEx(support_matrix, cv2.MORPH_CLOSE, np.ones((1, close_width), dtype=np.uint8))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, np.ones((1, minimum_run), dtype=np.uint8)).astype(bool)
    coverage_all = opened.mean(axis=1)
    running = np.cumsum(opened, axis=1)
    resets = np.maximum.accumulate(np.where(opened, 0, running), axis=1)
    longest_all = (running - resets).max(axis=1) / max(roi_width, 1)
    evidence_matrix = np.stack([row.evidence for row in rows], axis=0).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # rows without support
        strength_all = np.nan_to_num(
            np.nanmedian(np.where(opened, evidence_matrix, np.nan), axis=1), nan=0.0
        )
    any_support = opened.any(axis=1)
    diagnostic_max_coverage = float(coverage_all.max()) if len(rows) else 0.0
    diagnostic_max_run = float(longest_all.max()) if len(rows) else 0.0
    diagnostic_max_strength = float(strength_all.max()) if len(rows) else 0.0
    for index, row in enumerate(rows):
        coverage = float(coverage_all[index])
        longest = float(longest_all[index])
        if not any_support[index]:
            rejected["weak_pair"] += 1
            continue
        strength = float(strength_all[index])
        if coverage < settings.segmentation_min_paired_support_fraction:
            rejected["low_coverage"] += 1
            continue
        if longest < settings.segmentation_min_contiguous_support_fraction:
            rejected["short_support"] += 1
            continue
        toe_y = row.crest_y + row.height
        surface_prior = 1.0
        if surface is not None:
            if toe_y > surface[1] + surface_margin:
                rejected["below_operating_surface"] += 1
                continue
            if row.crest_y > surface[0] + surface_far_slack:
                # The berm body stands behind the surface the machinery drives
                # on: a crest below the farthest contact row is a stripe on the
                # surface itself (tracks, shadows, a road), not a berm.
                rejected["inside_operating_surface"] += 1
                continue
            if row.crest_y + 0.005 * target_height < surface[0] < toe_y - 0.005 * target_height:
                # Machinery touches the ground inside this body: it is ground
                # (a shadow or a stripe), so the berm must be another candidate.
                rejected["contains_far_contact"] += 1
                continue
            if row.bright_fraction > 0.5 and toe_y < surface[0] - 0.04 * target_height:
                # A bright body whose toe stops well above the farthest contact is
                # the sunlit ground beyond a dark berm, a bench or the horizon.
                # Dark bodies are exempt: a berm nobody drives next to (video 01,
                # far bank) is legitimately far from the machinery.
                surface_prior = 0.35
            excess_below_far = toe_y - (surface[0] + 0.02 * target_height)
            if excess_below_far > 0.0:
                # The toe meets the surface where the farthest machinery touches
                # it; a toe deeper into the surface is more likely a shadow or a
                # stripe on the ground.  Soft, because a dozer may sit on the berm.
                surface_prior /= 1.0 + excess_below_far / (0.03 * target_height)
        # Weak preference for the lower of two otherwise comparable bodies: the
        # berm is the last raised body before the operating surface, and the
        # ground beyond it is higher in the image.  Road stripes are already
        # excluded by the operating-surface rules, so this cannot pick them.
        # Only bright bodies get the "lower is likelier" preference: the sunlit
        # ground beyond a berm is always above it.  Dark bodies compete on
        # evidence alone, so a far bank nobody drives next to can still win.
        position_prior = 1.0
        if row.bright_fraction > 0.5:
            position_prior = 0.7 + 0.3 * (row.crest_y - crest_low) / max(crest_high - crest_low, 1)
        quality = (
            strength
            * np.sqrt(max(coverage, 1e-6))
            * np.sqrt(max(longest, 1e-6))
            * surface_prior
            * position_prior
        )
        qualified_candidates.append(
            {
                "crest_ratio": row.crest_y / max(target_height, 1),
                "toe_ratio": (row.crest_y + row.height) / max(target_height, 1),
                "height_ratio": row.height / max(target_height, 1),
                "quality": float(quality),
                "coverage": coverage,
                "run_ratio": longest,
                "strength": strength,
                "bright_fraction": float(row.bright_fraction),
                "surface_prior": float(surface_prior),
            }
        )
        qualified_rows.append((row, strength, coverage, longest))
        if quality > best_quality:
            best = row
            best_quality = float(quality)
            best_coverage = coverage
            best_run = longest
            best_strength = strength
    if best is None:
        return _unknown(
            context,
            "no_joint_crest_toe_support",
            diagnostics={
                "profile_source": source,
                "candidate_count": len(rows),
                "candidate_rejections": rejected,
                "maximum_candidate_coverage": diagnostic_max_coverage,
                "maximum_candidate_run_ratio": diagnostic_max_run,
                "maximum_supported_strength": diagnostic_max_strength,
                **source_diagnostics,
            },
        )

    # The toe is the first sufficient transition below the crest, not the
    # strongest one further down: loose material spilled in front of a berm and
    # the shadow it casts both end at a stronger edge with the road, but the
    # compacted body ends earlier.  Among candidates sharing the best crest,
    # take the shortest whose evidence still reaches the configured fraction.
    nearest_ratio = nearest_toe_ratio

    def _shorten(anchor, anchor_strength, anchor_coverage, anchor_run):
        """Apply the nearest-toe rule to one anchor; return the chosen row."""

        if nearest_ratio <= 0.0:
            return anchor, anchor_strength, anchor_coverage, anchor_run
        shorter = sorted(
            (
                item for item in qualified_rows
                if abs(item[0].crest_y - anchor.crest_y) <= 2 and item[0].height < anchor.height
            ),
            key=lambda item: item[0].height,
        )
        for row, strength, coverage, longest in shorter:
            if (
                strength >= nearest_ratio * anchor_strength
                and coverage >= settings.segmentation_min_paired_support_fraction
                and longest >= settings.segmentation_min_contiguous_support_fraction
            ):
                return row, strength, coverage, longest
        return anchor, anchor_strength, anchor_coverage, anchor_run

    # Anchors in quality order.  After the toe is shortened to its first
    # sufficient transition, the body must still meet the operating surface;
    # a bright ground band above the berm shortens to a toe that stops at the
    # berm crest, far from the surface, and is then discarded in favour of the
    # next anchor.
    # One anchor per crest row bucket: a bright band beyond the berm spawns
    # dozens of near-identical anchors that would otherwise fill the short list
    # and keep a genuinely different body from ever being scored.
    best_per_bucket: dict[int, tuple[tuple[_CandidateRow, float, float, float], float]] = {}
    for item, candidate in zip(qualified_rows, qualified_candidates):
        bucket = int(item[0].crest_y) // 3
        if bucket not in best_per_bucket or candidate["quality"] > best_per_bucket[bucket][1]:
            best_per_bucket[bucket] = (item, float(candidate["quality"]))
    ranked = sorted(best_per_bucket.values(), key=lambda entry: entry[1], reverse=True)
    # Relative, not absolute: the machinery contact does not always mark the
    # toe depth (a dozer box may end on the road in front of the berm), so a
    # shortened body whose toe stops far above the surface is only disfavoured
    # against bodies that reach it.  Quadratic in the excess above the surface.
    chosen = None
    best_score = -1.0
    for (row, strength, coverage, longest), quality in ranked[:12]:
        candidate = _shorten(row, strength, coverage, longest)
        score = quality
        if surface is not None and candidate[0].bright_fraction > 0.5:
            toe_final = candidate[0].crest_y + candidate[0].height
            excess = (surface[0] - 0.02 * target_height) - toe_final
            if excess > 0.0:
                score = quality / (1.0 + (excess / (0.04 * target_height)) ** 2)
        if score > best_score:
            chosen = (candidate, quality)
            best_score = score
    if chosen is None:
        return _unknown(
            context,
            "no_body_meets_operating_surface",
            diagnostics={
                "profile_source": source,
                "candidate_count": len(rows),
                "candidate_rejections": rejected,
                **source_diagnostics,
            },
        )
    (best, best_strength, best_coverage, best_run), best_quality = chosen
    local_y_radius = max(5, int(round(target_height * 0.045)))
    local_height_radius = max(3, int(round(best.height * 0.45)))
    local_rows = [
        row
        for row in rows
        if abs(row.crest_y - best.crest_y) <= local_y_radius
        and abs(row.height - best.height) <= local_height_radius
    ]
    score_matrix = np.stack([row.score for row in local_rows], axis=0)
    selected = np.argmax(score_matrix, axis=0)
    crest = np.asarray([local_rows[i].crest_y for i in selected], dtype=np.float32)
    heights = np.asarray([local_rows[i].height for i in selected], dtype=np.float32)
    evidence = np.asarray(
        [local_rows[i].evidence[column] for column, i in enumerate(selected)],
        dtype=np.float32,
    )
    supported = np.asarray(
        [local_rows[i].supported[column] for column, i in enumerate(selected)],
        dtype=bool,
    )
    crest = _median_smooth(crest, roi_width)
    heights = _median_smooth(heights, roi_width)
    toe = crest + heights
    supported = _clean_support(
        np.logical_and(supported, evidence >= evidence_threshold), roi_width
    )
    coverage = float(np.mean(supported))
    longest = _largest_true_run(supported) / max(roi_width, 1)
    if (
        coverage < settings.segmentation_min_paired_support_fraction
        or longest < settings.segmentation_min_contiguous_support_fraction
    ):
        return _unknown(
            context,
            "local_pair_support_failed",
            diagnostics={
                "profile_source": source,
                "candidate_count": len(rows),
                "paired_support_fraction": coverage,
                "largest_support_run_ratio": longest,
                "candidate_rejections": rejected,
                **source_diagnostics,
            },
        )

    xs_small = np.arange(x0, x1, dtype=np.float32)
    raw_crest = crest.copy()
    raw_toe = toe.copy()
    crest, crest_fit_residual, crest_slope = _robust_linear_profile(
        xs_small,
        raw_crest,
        supported,
    )
    toe, toe_fit_residual, toe_slope = _robust_linear_profile(
        xs_small,
        raw_toe,
        supported,
    )
    heights = toe - crest
    supported_heights = heights[supported]
    median_height_small = float(np.median(supported_heights))
    height_iqr = float(
        np.percentile(supported_heights, 75)
        - np.percentile(supported_heights, 25)
    )
    height_iqr_ratio = height_iqr / max(median_height_small, 1.0)
    fit_residual_ratio = max(crest_fit_residual, toe_fit_residual) / max(
        median_height_small,
        1.0,
    )
    if (
        height_iqr_ratio > settings.segmentation_max_height_iqr_ratio
        or fit_residual_ratio > 0.55
    ):
        return _unknown(
            context,
            "paired_geometry_dispersion_failed",
            diagnostics={
                "profile_source": source,
                "paired_support_fraction": coverage,
                "height_iqr_ratio": height_iqr_ratio,
                "linear_fit_residual_ratio": fit_residual_ratio,
                **source_diagnostics,
            },
        )

    scale_x = target_width / context.width
    scale_y = target_height / context.height
    xs = xs_small / scale_x
    crest_original = crest / scale_y
    toe_original = toe / scale_y
    small_mask = _mask_from_supported_profiles(
        target_height,
        target_width,
        xs_small,
        crest,
        toe,
        supported,
    )
    mask = cv2.resize(
        small_mask,
        (context.width, context.height),
        interpolation=cv2.INTER_NEAREST,
    )
    confidence = float(
        np.clip(
            (best_strength / max(evidence_threshold, 1e-6) - 0.75) / 2.25,
            0.0,
            1.0,
        )
        * np.sqrt(coverage)
        * np.sqrt(longest)
    )
    if confidence < confidence_threshold:
        return _unknown(
            context,
            "paired_profile_confidence_failed",
            confidence=confidence,
            diagnostics={
                "profile_source": source,
                "paired_support_fraction": coverage,
                "largest_support_run_ratio": longest,
                "pair_strength": best_strength,
                "candidate_rejections": rejected,
                **source_diagnostics,
            },
        )

    median_height = median_height_small / scale_y
    observation = BermObservation(
        context=context,
        status=ObservationStatus.OBSERVED,
        crest=tuple((float(x), float(y)) for x, y in zip(xs, crest_original)),
        base=tuple((float(x), float(y)) for x, y in zip(xs, toe_original)),
        height_px=median_height,
        height_m_estimated=median_height / settings.pixels_per_meter,
        diagnostics={
            "confidence": confidence,
            "spatially_valid": True,
            "candidate_kind": "berm",
            "candidate_count": len(rows),
            "candidate_rejections": rejected,
            "paired_support_fraction": coverage,
            "valid_column_fraction": coverage,
            "largest_support_run_ratio": longest,
            "pair_strength": best_strength,
            "global_pair_quality": best_quality,
            "global_pair_coverage": best_coverage,
            "global_pair_run_ratio": best_run,
            "global_anchor_crest_ratio": best.crest_y / max(target_height, 1),
            "global_anchor_toe_ratio": (best.crest_y + best.height) / max(target_height, 1),
            "global_anchor_height_ratio": best.height / max(target_height, 1),
            "top_spatial_candidates": sorted(
                qualified_candidates,
                key=lambda item: item["quality"],
                reverse=True,
            )[:8],
            "height_iqr_ratio": height_iqr_ratio,
            "profile_regularization": "robust_linear_irls",
            "linear_fit_residual_ratio": fit_residual_ratio,
            "crest_slope_px_per_px": crest_slope,
            "toe_slope_px_per_px": toe_slope,
            "profile_source": source,
            "base_source": "joint_toe_on_operational_grade_transition",
            "horizon_or_bank_selected": False,
            "operating_surface_used": surface is not None,
            **source_diagnostics,
        },
    )
    return SegmentationResult(observation, mask, confidence)


def extract_classical_paired_profile(
    frame: np.ndarray,
    context: FrameContext,
    settings: G2Settings,
    operating_surface: tuple[float, float] | None = None,
) -> SegmentationResult:
    original_height, original_width = frame.shape[:2]
    target_width = min(settings.segmentation_process_width, original_width)
    scale = target_width / original_width
    target_height = max(1, int(round(original_height * scale)))
    small = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
    lab_lightness = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)[:, :, 0]
    raw_lightness = lab_lightness.astype(np.float32) / 255.0
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(lab_lightness)
    signal = 0.72 * raw_lightness + 0.28 * (clahe.astype(np.float32) / 255.0)
    signal = cv2.GaussianBlur(signal, (5, 5), 0)
    signed_gradient = cv2.Sobel(signal, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    value = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[:, :, 2]
    saturated = np.logical_or(value <= 5, value >= 250).astype(np.float32)

    x0 = int(round(settings.segmentation_roi_x[0] * target_width))
    x1 = int(round(settings.segmentation_roi_x[1] * target_width))
    crest_low = max(4, int(round(settings.segmentation_crest_y[0] * target_height)))
    crest_high = min(
        target_height - 5,
        int(round(settings.segmentation_crest_y[1] * target_height)),
    )
    minimum_height = max(
        4, int(round(settings.segmentation_base_offset_y[0] * target_height))
    )
    maximum_height = max(
        minimum_height + 2,
        int(round(settings.segmentation_base_offset_y[1] * target_height)),
    )
    if x1 - x0 < 20 or crest_high - crest_low < 8:
        return _unknown(context, "empty_paired_profile_roi")

    height_values = np.unique(
        np.linspace(minimum_height, maximum_height, 13).round().astype(int)
    )
    horizontal_kernel = max(9, int(round((x1 - x0) * 0.045)) | 1)
    rows: list[_CandidateRow] = []
    polarities = (1.0, -1.0) if settings.segmentation_ridge_polarity == "both" else (1.0,)
    for candidate_height in height_values:
        radius = max(2, candidate_height // 7)
        local_mean = cv2.blur(signal, (1, radius * 2 + 1))
        saturation_fraction = cv2.blur(
            saturated,
            (1, max(3, candidate_height + radius * 2 + 1)),
        )
        upper_limit = min(crest_high, target_height - candidate_height - radius - 2)
        crest_start = max(crest_low, radius + 2)
        if upper_limit <= crest_start:
            continue
        # All crest rows of this height at once: row-shifted slices of the
        # smoothed image replace the former per-row Python loop.
        ys = np.arange(crest_start, upper_limit)
        above = local_mean[ys - radius - 1, x0:x1]
        body = local_mean[ys + candidate_height // 2, x0:x1]
        below = local_mean[ys + candidate_height + radius, x0:x1]
        gradient_crest = signed_gradient[ys, x0:x1]
        gradient_toe = signed_gradient[ys + candidate_height, x0:x1]
        # A compacted-earth berm may be darker than its surroundings (the
        # reddish body of videos 02-04) or a sunlit ridge brighter than the
        # ground beyond and the shadow it casts (the distant berm of video
        # 01).  Both polarities are ridge evidence; take the stronger one.
        evidence = np.zeros(above.shape, dtype=np.float32)
        bright_wins = np.zeros(above.shape, dtype=bool)
        for sign in polarities:
            upper_contrast = np.maximum(sign * (above - body), 0.0)
            lower_contrast = np.maximum(sign * (below - body), 0.0)
            contrast_pair = np.minimum(upper_contrast, lower_contrast)
            contrast_balance = np.sqrt(upper_contrast * lower_contrast)
            crest_edge = np.maximum(-sign * gradient_crest, 0.0)
            toe_edge = np.maximum(sign * gradient_toe, 0.0)
            edge_pair = np.minimum(crest_edge, toe_edge)
            polarity_evidence = (
                0.56 * contrast_pair
                + 0.28 * contrast_balance
                + 0.16 * edge_pair
            ).astype(np.float32)
            if sign < 0:
                bright_wins = polarity_evidence > evidence
            evidence = np.maximum(evidence, polarity_evidence)
        coherent = cv2.blur(evidence, (horizontal_kernel, 1))
        local_saturation = saturation_fraction[ys + candidate_height // 2, x0:x1]
        supported = np.logical_and(
            evidence >= settings.segmentation_min_pair_contrast,
            local_saturation <= settings.segmentation_max_saturated_fraction,
        )
        # Polarity of the body over its supported columns only.
        bright_fraction = (
            (bright_wins & supported).sum(axis=1) / np.maximum(supported.sum(axis=1), 1)
            if len(polarities) > 1
            else np.zeros(len(ys))
        )
        for index, crest_y in enumerate(ys.tolist()):
            rows.append(
                _CandidateRow(
                    int(crest_y),
                    int(candidate_height),
                    coherent[index],
                    evidence[index],
                    supported[index],
                    float(bright_fraction[index]),
                )
            )

    return _select_and_pack(
        rows,
        context,
        settings,
        target_width=target_width,
        target_height=target_height,
        x0=x0,
        x1=x1,
        evidence_threshold=settings.segmentation_min_pair_contrast,
        confidence_threshold=settings.segmentation_min_confidence,
        source="signed_photometric_joint_crest_body_toe",
        source_diagnostics={
            "mean_lightness": float(np.mean(lab_lightness)),
            "evidence_family": "classical_signed_photometric",
        },
        operating_surface=operating_surface,
        nearest_toe_ratio=settings.segmentation_toe_nearest_evidence_ratio,
    )


def extract_depth_paired_profile(
    depth: np.ndarray,
    context: FrameContext,
    settings: G2Settings,
    diagnostics: dict[str, Any] | None = None,
    operating_surface: tuple[float, float] | None = None,
) -> SegmentationResult:
    if depth.ndim != 2 or depth.size == 0 or not np.all(np.isfinite(depth)):
        return _unknown(context, "invalid_depth_map", diagnostics=diagnostics)
    target_height, target_width = depth.shape
    lower, upper = np.percentile(depth, (2.0, 98.0))
    dynamic_range = float(upper - lower)
    if dynamic_range <= 1e-8:
        return _unknown(context, "flat_depth_map", diagnostics=diagnostics)
    signal = np.clip(
        (depth.astype(np.float32) - lower) / dynamic_range, 0.0, 1.0
    ).astype(np.float32)
    signal = cv2.GaussianBlur(signal, (5, 5), 0)
    signed_gradient = cv2.Sobel(signal, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    gradient_magnitude = np.abs(signed_gradient)

    x0 = int(round(settings.segmentation_roi_x[0] * target_width))
    x1 = int(round(settings.segmentation_roi_x[1] * target_width))
    crest_low = max(4, int(round(settings.segmentation_crest_y[0] * target_height)))
    crest_high = min(
        target_height - 5,
        int(round(settings.segmentation_crest_y[1] * target_height)),
    )
    minimum_height = max(
        4, int(round(settings.segmentation_base_offset_y[0] * target_height))
    )
    maximum_height = max(
        minimum_height + 2,
        int(round(settings.segmentation_base_offset_y[1] * target_height)),
    )
    if x1 - x0 < 20 or crest_high - crest_low < 8:
        return _unknown(context, "empty_depth_paired_profile_roi", diagnostics=diagnostics)

    evidence_threshold = max(0.010, settings.segmentation_min_pair_contrast * 0.34)
    height_values = np.unique(
        np.linspace(minimum_height, maximum_height, 13).round().astype(int)
    )
    horizontal_kernel = max(9, int(round((x1 - x0) * 0.042)) | 1)
    rows: list[_CandidateRow] = []
    for candidate_height in height_values:
        radius = max(2, candidate_height // 7)
        local_mean = cv2.blur(signal, (1, radius * 2 + 1))
        gradient_mean = cv2.blur(
            gradient_magnitude,
            (1, max(3, candidate_height // 3 | 1)),
        )
        upper_limit = min(crest_high, target_height - candidate_height - radius - 2)
        for crest_y in range(max(crest_low, radius + 2), upper_limit):
            upper_body_y = crest_y + candidate_height // 4
            body_y = crest_y + candidate_height // 2
            lower_body_y = crest_y + (candidate_height * 3) // 4
            toe_y = crest_y + candidate_height
            above = local_mean[crest_y - radius - 1, x0:x1]
            upper_body = local_mean[upper_body_y, x0:x1]
            lower_body = local_mean[lower_body_y, x0:x1]
            below = local_mean[toe_y + radius, x0:x1]
            bump_top = np.maximum(upper_body - above, 0.0)
            # Depth Anything's relative-depth output grows toward the camera in
            # these scenes.  A real berm therefore does not form a symmetric
            # intensity "bump": depth normally increases from the far bench,
            # through the front slope, and again onto the nearer operating
            # grade.  The previous sign expected the signal to fall below the
            # toe, which made virtually every real berm impossible to support.
            # The model is relative rather than metrically oriented.  Most
            # official frames grow toward the camera, while some model/test
            # outputs encode a foreground ridge as a signed local bump.  The
            # geometry depends on a transition, not on that arbitrary sign.
            grade_transition = np.abs(below - lower_body)
            body_progression = np.abs(lower_body - upper_body)
            transition_pair = np.minimum(bump_top, grade_transition)
            interior_gradient = gradient_mean[body_y, x0:x1]
            crest_break = np.maximum(
                gradient_magnitude[crest_y, x0:x1] - interior_gradient,
                0.0,
            )
            toe_break = np.maximum(
                gradient_magnitude[toe_y, x0:x1] - interior_gradient,
                0.0,
            )
            # A crest discontinuity is mandatory; the toe may be expressed as
            # either a slope break or a smooth transition onto the grade.
            # Prefer an explicit slope break when present.  The weaker grade
            # term is only a fallback for visually smooth feet and must not
            # reward an arbitrarily long band below the real toe.
            toe_evidence = np.maximum(toe_break, 0.20 * grade_transition)
            slope_break_pair = np.sqrt(crest_break * toe_evidence)
            evidence = (
                0.58 * slope_break_pair
                + 0.27 * np.minimum(crest_break, transition_pair)
                + 0.15 * np.minimum(crest_break, body_progression)
            ).astype(np.float32)
            coherent = cv2.blur(
                evidence.reshape(1, -1), (horizontal_kernel, 1)
            ).reshape(-1)
            supported = evidence >= evidence_threshold
            rows.append(
                _CandidateRow(
                    crest_y,
                    candidate_height,
                    coherent,
                    evidence,
                    supported,
                )
            )

    return _select_and_pack(
        rows,
        context,
        settings,
        target_width=target_width,
        target_height=target_height,
        x0=x0,
        x1=x1,
        evidence_threshold=evidence_threshold,
        confidence_threshold=settings.learned_min_confidence,
        source="depth_relative_slope_break_joint_crest_toe",
        source_diagnostics={
            "relative_depth_dynamic_range": dynamic_range,
            "evidence_family": "learned_relative_depth",
            **(diagnostics or {}),
        },
        operating_surface=operating_surface,
    )
