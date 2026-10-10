"""V8: the V7 fixed-camera anchor, with the berm size actually measured.

V7 bought line stability and paid for it with the height series: it published a
height only when the extractor independently re-detected the whole berm in
daylight and that detection matched the anchor.  Across the corpus that left 21,
25, 6 and 18 frames with a height, while the assessment asks for a *continuous*
height profile.

V8 changes only measurement, never geometry.  The anchor stays immutable, so the
drawn lines are identical to V7's:

* current support is judged per column against the anchor, with the honest
  denominator being the columns the anchor itself supports, not the full width;
* when no current support survives, the anchored geometry is still a real
  daylight measurement, so it is published with its age and an explicit basis
  instead of leaving the curve empty;
* the profile is summarised as p10/median/p90 rather than a single number,
  because the assessment asks for a height profile;
* metres come from a ground-plane row regression over machinery, which works at
  rows where no vehicle happens to stand, and carry a declared interval.
"""

from dataclasses import replace

import cv2
import numpy as np

from bermguard.contracts import BermObservation, ObservationStatus
from bermguard.segmentation.result import SegmentationResult
from bermguard.segmentation.validation import TemporalBermValidator, _Profile
from bermguard.rendering import render_frame as original_render, _draw_dashed_polyline

import berm_metrics
import normativa_ds132 as normativa
from berm_metrics import GroundScaleModel, profile_statistics


# Absolute floor of matched columns, and the fraction of the anchor's own
# supported columns that must agree.  V7 required 20 % of the *full* width,
# which a couple of haul trucks in front of the berm is enough to defeat.
MIN_MATCHED_COLUMNS = 8
MIN_MATCH_FRACTION = 0.15
ANCHOR_TOLERANCE_RATIO = 0.01


class MeasuringFixedCameraValidator(TemporalBermValidator):
    def __init__(self, settings):
        super().__init__(settings)
        self.anchor = None
        self.anchor_context = None
        self.suppressed_resets = 0
        # Calibrated on the largest machine class, because DS 132 sizes a berm
        # against "el camión de mayor envergadura"; the configured 5 m is a
        # fleet average and understates the metres it produces.
        self.scale = GroundScaleModel(normativa.TRUCK_HEIGHT_M)

    def reset(self):
        super().reset()
        if self.anchor is not None:
            self.suppressed_resets += 1

    def refine(self, result, detections, frame=None, exclusion_mask=None,
               operating_surface=None):
        ctx = result.observation.context
        if self.anchor_context is not None and (
            ctx.video_id != self.anchor_context.video_id
            or (ctx.width, ctx.height) != (self.anchor_context.width, self.anchor_context.height)
            or ctx.frame_index <= self.anchor_context.frame_index
        ):
            self.anchor = self.anchor_context = None
            self.suppressed_resets = 0
            self.scale.reset()
            berm_metrics.clear_confirmed()
            super().reset()

        # Confirmed tracks from the previous frame, not raw proposals: see
        # ``berm_metrics.publish_confirmed``.
        self.scale.observe(berm_metrics.peek_confirmed())

        if self.anchor is None:
            refined = super().refine(result, detections, frame, exclusion_mask, operating_surface)
            if refined.observation.diagnostics.get("reference_initialized"):
                profile, _ = self._profile_from_result(refined)
                if profile is not None:
                    self.anchor = _Profile(*(a.copy() for a in (
                        profile.xs, profile.crest, profile.toe, profile.support)))
                    self.anchor_context = ctx
            diag = dict(refined.observation.diagnostics)
            diag.update(experiment="fixed_camera_measure_v8", camera_stationarity_assumed=True)
            if self.anchor is not None:
                diag["anchor_frame"] = ctx.frame_index
            berm_metrics.record({
                "video_id": ctx.video_id, "method": "1", "frame_index": ctx.frame_index,
                "timestamp_s": ctx.timestamp_s, "lighting": diag.get("lighting"),
                "state": refined.observation.status.value,
                "measurement_basis": diag.get("measurement_basis"),
                "raw_status": result.observation.status.value,
            })
            return replace(refined, observation=replace(refined.observation, diagnostics=diag))

        anchor = self.anchor
        light = float(np.median(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))) if frame is not None else None
        lighting = ("unknown" if light is None else "night" if light <= self.settings.scene_night_below_mean
                    else "transition" if light < self.settings.scene_day_above_mean else "day")
        anchor_heights = anchor.toe - anchor.crest
        anchored_height = float(np.median(anchor_heights))
        toe_row = float(np.median(anchor.toe))
        crest_row = float(np.median(anchor.crest))

        diag = dict(result.observation.diagnostics)
        diag.update(experiment="fixed_camera_measure_v8", camera_stationarity_assumed=True,
                    camera_stationarity_verified=False, registration_applied=False,
                    reference_updated=False, anchor_frame=self.anchor_context.frame_index,
                    profile_age_frames=ctx.frame_index - self.anchor_context.frame_index,
                    lighting=lighting, suppressed_berm_resets=self.suppressed_resets,
                    profile_source="immutable_daylight_anchor", historical_only=True,
                    measurement_basis="anchored_daylight_geometry_carried",
                    reason="fixed_camera_anchored_measurement", current_anchor_support=False,
                    anchor_height_px=anchored_height)

        height = None
        confidence = 0.0
        stats = None
        berm_metrics.clear_measured()
        matched_columns = 0
        profile_columns = int(np.count_nonzero(anchor.support)) or len(anchor.xs)
        match_fraction = None
        residual_median = None

        if lighting == "day" and diag.get("spatially_valid", True) is True:
            candidate, _ = self._profile_from_result(result)
            if candidate is not None:
                candidate, exclusions = self._exclude_occlusions(
                    candidate, detections, ctx.width, ctx.height, exclusion_mask)
                diag.update(exclusions)
                reference = self._resample(anchor, candidate.xs)
                supported = candidate.support & reference.support
                residual = np.maximum(np.abs(candidate.crest - reference.crest),
                                      np.abs(candidate.toe - reference.toe))
                matched = supported & (residual <= ANCHOR_TOLERANCE_RATIO * ctx.height)
                matched_columns = int(np.count_nonzero(matched))
                anchor_supported = max(int(np.count_nonzero(reference.support)), 1)
                profile_columns = anchor_supported
                match_fraction = matched_columns / anchor_supported
                if supported.any():
                    residual_median = float(np.median(residual[supported]))
                diag["anchor_matching_fraction"] = float(match_fraction)
                diag["candidate_anchor_residual_px"] = residual_median
                measured, stats, failure = self._measure(
                    candidate, matched, anchor_supported, ctx.height)
                if measured is not None:
                    height = measured
                    confidence = result.confidence
                    trusted = matched & (candidate.toe - candidate.crest > 0)
                    berm_metrics.publish_measured(
                        list(zip(candidate.xs[trusted].tolist(),
                                 candidate.crest[trusted].tolist())),
                        list(zip(candidate.xs[trusted].tolist(),
                                 candidate.toe[trusted].tolist())),
                    )
                    diag.update(current_anchor_support=True, historical_only=False,
                                measurement_basis="current_supported_columns_fixed_overlay",
                                reason="fixed_camera_current_support")
                elif failure:
                    diag["current_support_rejected"] = failure

        if height is None:
            # The anchor is itself a validated daylight measurement of a static
            # berm seen by a camera assumed fixed.  Publishing it with its age
            # keeps the required curve continuous; the basis says what it is.
            height = anchored_height
            stats = profile_statistics(anchor_heights[anchor.support] if anchor.support.any()
                                       else anchor_heights, len(anchor.xs))

        scale = self.scale.estimate(toe_row, ctx.height)
        metres = low = high = None
        compliance = None
        if scale is not None:
            metres, low, high = scale.to_meters(height)
            compliance = normativa.evaluate(metres, low, high)
            diag.update(ground_plane_pixels_per_meter=scale.pixels_per_meter,
                        ground_plane_scale_basis=scale.basis,
                        scale_reference_height_m=self.scale.reference_height_m,
                        berm_height_m_ground_plane=metres,
                        berm_height_m_interval=(low, high),
                        berm_height_m_required=compliance.required_m,
                        berm_norm_article=compliance.article,
                        berm_norm_verdict=compliance.verdict,
                        berm_height_wheel_ratio=compliance.wheel_ratio)
        if stats is not None:
            diag.update(height_profile_p10_px=stats.p10_px,
                        height_profile_p90_px=stats.p90_px,
                        height_profile_columns=stats.columns,
                        height_profile_coverage=stats.coverage)

        status = (ObservationStatus.TEMPORAL_ESTIMATE
                  if diag.get("current_anchor_support") else ObservationStatus.HISTORICAL_REFERENCE)
        diag["state"] = status.value

        berm_metrics.record({
            "video_id": ctx.video_id, "method": "1", "frame_index": ctx.frame_index,
            "timestamp_s": ctx.timestamp_s, "lighting": lighting, "state": status.value,
            "measurement_basis": diag["measurement_basis"],
            "anchor_frame": self.anchor_context.frame_index,
            "profile_age_frames": diag["profile_age_frames"],
            "raw_status": result.observation.status.value,
            "search_mode": diag.get("search_mode"),
            "matched_columns": matched_columns, "profile_columns": profile_columns,
            "match_fraction": match_fraction, "residual_median_px": residual_median,
            "height_px": height,
            "height_p10_px": None if stats is None else stats.p10_px,
            "height_p90_px": None if stats is None else stats.p90_px,
            "height_m": metres, "height_m_low": low, "height_m_high": high,
            "height_m_required": None if compliance is None else compliance.required_m,
            "norm_article": None if compliance is None else compliance.article,
            "norm_verdict": None if compliance is None else compliance.verdict,
            "wheel_ratio": None if compliance is None else compliance.wheel_ratio,
            "scale_reference_height_m": self.scale.reference_height_m,
            "pixels_per_meter": None if scale is None else scale.pixels_per_meter,
            "scale_basis": None if scale is None else scale.basis,
            "scale_samples": None if scale is None else scale.samples,
            "scale_horizon_row": None if scale is None else scale.horizon_row,
            "crest_y_median": crest_row, "toe_y_median": toe_row,
        })

        obs = BermObservation(
            ctx, status,
            tuple(zip(anchor.xs.tolist(), anchor.crest.tolist())),
            tuple(zip(anchor.xs.tolist(), anchor.toe.tolist())),
            height, metres, diag)
        return SegmentationResult(obs, np.zeros_like(result.mask), confidence)

    def _measure(self, candidate, matched, anchor_supported, frame_height):
        """Median height over the columns that still agree with the anchor."""

        heights = candidate.toe - candidate.crest
        plausible = np.logical_and(heights > 0, heights < 0.30 * frame_height)
        trusted = np.logical_and(matched, plausible)
        count = int(np.count_nonzero(trusted))
        if count < max(MIN_MATCHED_COLUMNS, int(MIN_MATCH_FRACTION * anchor_supported)):
            return None, None, "insufficient_matched_columns"
        values = heights[trusted]
        height_px = float(np.median(values))
        iqr_ratio = float((np.percentile(values, 75) - np.percentile(values, 25))
                          / max(height_px, 1.0))
        if iqr_ratio > self.settings.segmentation_max_height_iqr_ratio:
            return None, None, "matched_height_dispersion_failed"
        return height_px, profile_statistics(values, anchor_supported), ""


def render_experiment(frame, segmentation, *args, **kwargs):
    render_args = list(args)
    if len(render_args) >= 5:
        render_args[4] = ()
    else:
        kwargs = dict(kwargs)
        kwargs["raw_detections"] = ()
    rendered = original_render(frame, segmentation, *render_args, **kwargs)
    obs = segmentation.observation
    d = obs.diagnostics
    if d.get("profile_source") == "immutable_daylight_anchor":
        _draw_dashed_polyline(rendered, obs.crest, (30, 30, 240), 2)
        _draw_dashed_polyline(rendered, obs.base, (30, 190, 255), 2)
        # The measured columns, when they exist, so the reported number is
        # something the reviewer can see rather than a claim about the anchor.
        measured_crest, measured_toe = berm_metrics.peek_measured()
        if d.get("current_anchor_support") and measured_crest and measured_toe:
            for points in (measured_crest, measured_toe):
                cv2.polylines(
                    rendered,
                    [np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))],
                    False, (235, 235, 235), 1, cv2.LINE_AA,
                )
            # Labelled at the right end: the base renderer already writes the
            # crest and toe labels at the left one.
            last = measured_crest[-1]
            anchor_x = min(int(last[0]) + 6, rendered.shape[1] - 140)
            cv2.putText(rendered, "medicion actual",
                        (anchor_x, max(16, int(last[1]) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.rectangle(rendered, (8, 8), (min(frame.shape[1] - 8, 1180), 150), (8, 12, 8), -1)
    method = args[3] if len(args) > 3 else kwargs.get("method_label", "")
    locked = "anchor_frame" in d
    if not locked:
        lines = ["CAMARA FIJA SUPUESTA | " + str(method),
                 "Esperando referencia diurna validada (sin marcas manuales)",
                 str(d.get("reason", "N/D")),
                 ""]
    else:
        current = bool(d.get("current_anchor_support"))
        # Metres first: the norm is written in metres once the wheel of the
        # largest truck is declared.  Pixels and the wheel ratio stay as the
        # supporting reading underneath.
        metres = d.get("berm_height_m_ground_plane")
        interval = d.get("berm_height_m_interval")
        if metres is not None and interval is not None:
            size = normativa.summary_line(normativa.Compliance(
                height_m=metres, height_m_low=interval[0], height_m_high=interval[1],
                required_m=d.get("berm_height_m_required", normativa.DUMP_EDGE_MINIMUM_M),
                article=d.get("berm_norm_article", normativa.ARTICLE_DUMP),
                verdict=d.get("berm_norm_verdict", ""),
                wheel_ratio=d.get("berm_height_wheel_ratio", 0.0)))
        else:
            size = "h=N/D m | sin escala metrica en este frame"
        detail = f"h={obs.height_px:.1f} px" if obs.height_px is not None else "h=N/D px"
        p10, p90 = d.get("height_profile_p10_px"), d.get("height_profile_p90_px")
        # A flat anchored profile has no dispersion to report; printing
        # "p10-p90 42-42" would only look like noise.
        if p10 is not None and p90 is not None and p90 - p10 >= 1.0:
            detail += f" (p10-p90 {p10:.0f}-{p90:.0f})"
        scale = d.get("ground_plane_pixels_per_meter")
        if scale:
            detail += (f" | escala {scale:.1f} px/m sobre camion de "
                       f"{d.get('scale_reference_height_m', normativa.TRUCK_HEIGHT_M):.1f} m")
        lines = [
            f"CAMARA FIJA SUPUESTA | {method} | escena {obs.context.scene_id}",
            f"Pretil anclado en frame {d['anchor_frame']} | " + (
                "apoyo actual en columnas" if current
                else f"geometria anclada, sin apoyo actual ({d.get('profile_age_frames', 0)} frames)"),
            size,
            detail,
        ]
    for y, text in zip((32, 63, 94, 128), lines):
        if not text:
            continue
        weight = 2 if y < 100 else 1
        cv2.putText(rendered, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, .56,
                    (230, 230, 230), weight, cv2.LINE_AA)
    return rendered
