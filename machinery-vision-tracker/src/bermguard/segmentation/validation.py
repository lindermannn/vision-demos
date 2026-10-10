"""Causal G5.1/G5.3 validation and persistence for paired berm geometry.

G5.3 adds the low-light policy: a berm that was observed by day with a static
camera is *remembered*, not re-detected, at night.  The remembered profile is
published as ``TEMPORAL_ESTIMATE`` with its age while phase-correlation
registration keeps confirming that the camera did not move, is never updated
from low-light evidence, and must be reconfirmed by compatible daytime evidence
once light returns or it is dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import cv2
import numpy as np

from ..contracts import BermObservation, Detection, ObservationStatus
from ..g2_config import G2Settings
from .result import SegmentationResult


LOW_LIGHT_ESTIMATE_REASON = "low_light_reference_estimate"


@dataclass
class _Profile:
    xs: np.ndarray
    crest: np.ndarray
    toe: np.ndarray
    support: np.ndarray


@dataclass
class _TemporalState:
    scene_id: int | None = None
    reference: _Profile | None = None
    bootstrap: list[_Profile] = field(default_factory=list)
    last_observed_frame: int | None = None
    last_frame_index: int | None = None
    recovery_count: int = 0
    previous_structure: np.ndarray | None = None
    previous_static_mask: np.ndarray | None = None
    low_light_frames: int = 0
    unreliable_registration_frames: int = 0
    pending_reconfirmation: bool = False
    reconfirmation_failures: int = 0
    last_verified_frame: int | None = None
    day_frames_since_low_light: int = 10**6  # settle applies only after low light
    incompatible_day_frames: int = 0


class TemporalBermValidator:
    """Validate spatial evidence before allowing a causal temporal reference."""

    def __init__(self, settings: G2Settings) -> None:
        self.settings = settings
        self._state = _TemporalState()

    def reset(self) -> None:
        self._state = _TemporalState()

    def refine(
        self,
        result: SegmentationResult,
        detections: Sequence[Detection],
        frame: np.ndarray | None = None,
        exclusion_mask: np.ndarray | None = None,
        operating_surface: tuple[float, float] | None = None,
    ) -> SegmentationResult:
        observation = result.observation
        context = observation.context
        if self._state.scene_id != context.scene_id:
            self._state = _TemporalState(scene_id=context.scene_id)

        registration = self._register_static_structure(frame, detections, exclusion_mask)
        diagnostics = dict(observation.diagnostics)
        diagnostics.update(registration)
        self._state.last_frame_index = context.frame_index
        if operating_surface is not None:
            diagnostics["operating_surface_far_y"] = float(operating_surface[0])
            diagnostics["operating_surface_near_y"] = float(operating_surface[1])
            reference = self._state.reference
            if reference is not None and np.any(reference.support):
                toe_median = float(np.median(reference.toe[reference.support]))
                crest_median = float(np.median(reference.crest[reference.support]))
                if (
                    toe_median > operating_surface[1] + 0.03 * context.height
                    or crest_median > operating_surface[0] + 0.06 * context.height
                ):
                    # A remembered profile below the surface the machinery
                    # drives on is road, not berm: forget it and start over.
                    self._state.reference = None
                    self._state.bootstrap.clear()
                    self._state.recovery_count = 0
                    self._state.last_observed_frame = None
                    self._state.pending_reconfirmation = False
                    diagnostics["reference_dropped_below_operating_surface"] = True

        lighting = "day"
        if frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            median_lightness = float(np.median(gray))
            diagnostics.update(
                {
                    "mean_lightness": float(np.mean(gray)),
                    "median_lightness": median_lightness,
                }
            )
            if median_lightness <= self.settings.scene_night_below_mean:
                lighting = "night"
            elif median_lightness < self.settings.scene_day_above_mean:
                lighting = "transition"
        diagnostics["lighting"] = lighting

        if lighting != "day":
            return self._refine_low_light(result, lighting, diagnostics, registration)

        state = self._state
        if state.low_light_frames > 0:
            # Light came back.  A reference that survived the dark must earn its
            # place again with compatible current evidence, otherwise a scene
            # that changed in the dark would inherit the old geometry.
            state.low_light_frames = 0
            state.unreliable_registration_frames = 0
            state.day_frames_since_low_light = 0
            if state.reference is not None:
                state.pending_reconfirmation = True
                state.reconfirmation_failures = 0
        state.day_frames_since_low_light += 1
        if (
            state.reference is None
            and state.day_frames_since_low_light <= self.settings.segmentation_bootstrap_settle_frames
        ):
            # The first daylight frames after a dusk/dawn ramp are still dim and
            # the extractor is unreliable there; never seed a reference from them.
            state.bootstrap.clear()
            diagnostics["bootstrap_settling"] = True
            return self._unknown(result, "settling_after_low_light", diagnostics)

        refined = self._refine_day(result, detections, diagnostics, exclusion_mask)
        refined = self._apply_reconfirmation(result, refined)
        return self._challenge_reference(result, refined)

    # ------------------------------------------------------------------ day --

    def _refine_day(
        self,
        result: SegmentationResult,
        detections: Sequence[Detection],
        diagnostics: dict[str, object],
        exclusion_mask: np.ndarray | None,
    ) -> SegmentationResult:
        observation = result.observation
        context = observation.context

        if observation.status not in {
            ObservationStatus.OBSERVED,
            ObservationStatus.VALID,
        }:
            return self._degrade(
                result,
                str(diagnostics.get("reason", "no_spatial_candidate")),
                diagnostics,
            )
        if diagnostics.get("spatially_valid", True) is not True:
            self._state.bootstrap.clear()
            return self._degrade(
                result,
                "spatial_identity_not_verified",
                diagnostics,
            )

        profile, profile_reason = self._profile_from_result(result)
        if profile is None:
            return self._degrade(result, profile_reason, diagnostics)
        profile, exclusion_details = self._exclude_occlusions(
            profile,
            detections,
            context.width,
            context.height,
            exclusion_mask,
        )
        diagnostics.update(exclusion_details)
        coverage = float(np.mean(profile.support))
        largest_gap = self._maximum_false_run(profile.support) / len(profile.support)
        diagnostics.update(
            {
                "free_profile_fraction": coverage,
                "paired_support_fraction": coverage,
                "maximum_occlusion_gap_ratio": largest_gap,
                "profile_age_frames": 0,
            }
        )
        if coverage < self.settings.segmentation_min_current_support_for_estimate:
            return self._degrade(
                result,
                "insufficient_current_paired_support",
                diagnostics,
            )

        geometry_reason, height_px, height_iqr_ratio = self._validate_local_geometry(
            profile,
            context.height,
        )
        diagnostics["height_iqr_ratio"] = height_iqr_ratio
        if geometry_reason:
            return self._degrade(result, geometry_reason, diagnostics)
        assert height_px is not None

        if self._state.reference is None:
            if coverage < self.settings.segmentation_min_paired_support_fraction:
                self._state.bootstrap.clear()
                return self._unknown(
                    result,
                    "insufficient_support_to_bootstrap_reference",
                    diagnostics,
                )
            return self._bootstrap_reference(
                result,
                profile,
                height_px,
                diagnostics,
            )

        compatible, compatibility = self._compatible(
            profile,
            self._state.reference,
            context.height,
        )
        diagnostics.update(compatibility)
        if not compatible:
            self._state.recovery_count = 0
            return self._historical(
                result,
                "current_geometry_incompatible_with_reference",
                diagnostics,
            )

        aligned_reference = self._resample(self._state.reference, profile.xs)
        observed = profile.support
        occlusion_fraction = float(exclusion_details["profile_in_occlusion_fraction"])
        # A berm does not move when machinery passes in front of it.  The more
        # of the profile a vehicle covers, the less the current fit may pull the
        # reference; beyond a quarter of the width the reference is frozen.
        alpha = self.settings.segmentation_profile_ema_alpha * float(
            np.clip(1.0 - 4.0 * occlusion_fraction, 0.0, 1.0)
        )
        updated_crest = aligned_reference.crest.copy()
        updated_toe = aligned_reference.toe.copy()
        updated_crest[observed] = (
            alpha * profile.crest[observed]
            + (1.0 - alpha) * aligned_reference.crest[observed]
        )
        updated_toe[observed] = (
            alpha * profile.toe[observed]
            + (1.0 - alpha) * aligned_reference.toe[observed]
        )
        self._state.reference = _Profile(
            profile.xs.copy(),
            updated_crest,
            updated_toe,
            np.logical_or(aligned_reference.support, observed),
        )
        self._state.last_observed_frame = context.frame_index
        self._state.last_verified_frame = context.frame_index
        self._state.recovery_count += 1

        # Lines follow the causal reference geometry so a passing vehicle cannot
        # wobble them; the height is still measured on current supported columns.
        output_profile = _Profile(profile.xs, updated_crest, updated_toe, observed)
        has_occlusion = bool(exclusion_details["occluding_vehicle_count"]) or bool(
            exclusion_details["motion_excluded_columns"]
        )
        fully_observed = (
            coverage >= self.settings.segmentation_observed_coverage_fraction
            and not has_occlusion
            and self._state.recovery_count >= self.settings.segmentation_recovery_frames
        )
        status = (
            ObservationStatus.OBSERVED
            if fully_observed
            else ObservationStatus.TEMPORAL_ESTIMATE
        )
        diagnostics.update(
            {
                "validation": "g5_1_spatial_then_causal_profile",
                "reason": "" if fully_observed else "partial_current_support_completed_from_reference",
                "state": status.value,
                "measurement_basis": "current_supported_columns",
                "reference_updated": alpha > 0.0,
                "reference_compatible": True,
                "reference_update_alpha": alpha,
                "observed_column_fraction": coverage,
                "temporal_completion_fraction": 1.0 - coverage,
                "historical_only": False,
            }
        )
        return self._measured_result(
            result,
            output_profile,
            status,
            height_px,
            diagnostics,
        )

    def _apply_reconfirmation(
        self,
        original: SegmentationResult,
        refined: SegmentationResult,
    ) -> SegmentationResult:
        state = self._state
        if not state.pending_reconfirmation:
            return refined
        diagnostics = dict(refined.observation.diagnostics)
        if diagnostics.get("reference_compatible") is True or diagnostics.get(
            "reference_initialized"
        ) is True:
            state.pending_reconfirmation = False
            state.reconfirmation_failures = 0
            diagnostics["reference_reconfirmed_after_low_light"] = True
            return SegmentationResult(
                BermObservation(
                    context=refined.observation.context,
                    status=refined.observation.status,
                    crest=refined.observation.crest,
                    base=refined.observation.base,
                    height_px=refined.observation.height_px,
                    height_m_estimated=refined.observation.height_m_estimated,
                    diagnostics=diagnostics,
                ),
                refined.mask,
                refined.confidence,
            )
        if state.reference is None:
            state.pending_reconfirmation = False
            return refined
        state.reconfirmation_failures += 1
        diagnostics["reconfirmation_failures"] = state.reconfirmation_failures
        limit = self.settings.segmentation_reconfirmation_failure_frames
        if state.reconfirmation_failures < limit:
            return SegmentationResult(
                BermObservation(
                    context=refined.observation.context,
                    status=refined.observation.status,
                    crest=refined.observation.crest,
                    base=refined.observation.base,
                    height_px=refined.observation.height_px,
                    height_m_estimated=refined.observation.height_m_estimated,
                    diagnostics=diagnostics,
                ),
                refined.mask,
                refined.confidence,
            )
        state.reference = None
        state.bootstrap.clear()
        state.recovery_count = 0
        state.last_observed_frame = None
        state.pending_reconfirmation = False
        state.reconfirmation_failures = 0
        return self._unknown(
            original,
            "reference_not_reconfirmed_after_low_light",
            diagnostics,
        )

    def _challenge_reference(
        self,
        original: SegmentationResult,
        refined: SegmentationResult,
    ) -> SegmentationResult:
        """Drop a daylight reference contradicted by consistent evidence.

        A berm does not move, but a reference can be wrong from the start.  When
        the current spatial candidate is incompatible with the reference for
        ``reconfirmation_failure_frames`` consecutive daylight frames, the
        reference is discarded so the consistent evidence can bootstrap anew.
        Frames without a candidate neither count nor reset the streak.
        """

        state = self._state
        diagnostics = dict(refined.observation.diagnostics)
        reason = str(diagnostics.get("reason", ""))
        if state.reference is None:
            state.incompatible_day_frames = 0
            return refined
        if diagnostics.get("reference_compatible") is True or diagnostics.get("reference_initialized") is True:
            state.incompatible_day_frames = 0
            return refined
        if reason != "current_geometry_incompatible_with_reference":
            return refined
        state.incompatible_day_frames += 1
        diagnostics["incompatible_day_frames"] = state.incompatible_day_frames
        if state.incompatible_day_frames < self.settings.segmentation_reconfirmation_failure_frames:
            return SegmentationResult(
                BermObservation(
                    context=refined.observation.context,
                    status=refined.observation.status,
                    crest=refined.observation.crest,
                    base=refined.observation.base,
                    height_px=refined.observation.height_px,
                    height_m_estimated=refined.observation.height_m_estimated,
                    diagnostics=diagnostics,
                ),
                refined.mask,
                refined.confidence,
            )
        state.reference = None
        state.bootstrap.clear()
        state.recovery_count = 0
        state.last_observed_frame = None
        state.last_verified_frame = None
        state.pending_reconfirmation = False
        state.incompatible_day_frames = 0
        return self._unknown(original, "reference_replaced_by_consistent_evidence", diagnostics)

    # ------------------------------------------------------------ low light --

    def _refine_low_light(
        self,
        result: SegmentationResult,
        lighting: str,
        diagnostics: dict[str, object],
        registration: dict[str, float | bool | str],
    ) -> SegmentationResult:
        """Remember the daytime berm; never create or update it in the dark."""

        state = self._state
        context = result.observation.context
        state.low_light_frames += 1
        # Frames inside a dusk/dawn ramp can carry half-formed evidence; they
        # must not seed a reference.
        state.bootstrap.clear()
        diagnostics["reference_updated"] = False

        if state.reference is None or state.last_observed_frame is None:
            reason = (
                "night_without_reference"
                if lighting == "night"
                else "low_light_without_reference"
            )
            return self._unknown(result, reason, diagnostics)

        age = max(0, context.frame_index - state.last_observed_frame)
        verified_gap = context.frame_index - (
            state.last_verified_frame if state.last_verified_frame is not None else state.last_observed_frame
        )
        if verified_gap > self.settings.segmentation_reference_expiry_frames:
            # Expiry counts frames without verification: an observation or a
            # low-light carry with reliable registration keeps the memory alive.
            state.reference = None
            state.recovery_count = 0
            state.last_observed_frame = None
            state.last_verified_frame = None
            return self._unknown(result, "historical_reference_expired", diagnostics)

        if registration.get("registration_reliable") is True:
            state.unreliable_registration_frames = 0
        else:
            state.unreliable_registration_frames += 1
        tolerance = self.settings.segmentation_low_light_registration_tolerance_frames
        if state.unreliable_registration_frames > tolerance:
            return self._historical(
                result,
                "low_light_camera_stationarity_unverified",
                diagnostics,
            )

        profile = state.reference
        heights = profile.toe - profile.crest
        trusted = np.logical_and(profile.support, heights > 0)
        if np.count_nonzero(trusted) < 8:
            return self._historical(
                result,
                "low_light_reference_without_support",
                diagnostics,
            )
        height_px = float(np.median(heights[trusted]))
        state.last_verified_frame = context.frame_index
        state.incompatible_day_frames = 0
        diagnostics.update(
            {
                "validation": "g5_3_low_light_reference_estimate",
                "reason": LOW_LIGHT_ESTIMATE_REASON,
                "state": ObservationStatus.TEMPORAL_ESTIMATE.value,
                "measurement_basis": "day_reference_carried",
                "profile_age_frames": age,
                "observed_column_fraction": 0.0,
                "temporal_completion_fraction": 1.0,
                "historical_only": False,
                "camera_stationarity_verified": True,
                "spatially_valid": True,
            }
        )
        return self._measured_result(
            result,
            profile,
            ObservationStatus.TEMPORAL_ESTIMATE,
            height_px,
            diagnostics,
        )

    # --------------------------------------------------------------- shared --

    def _bootstrap_reference(
        self,
        result: SegmentationResult,
        profile: _Profile,
        height_px: float,
        diagnostics: dict[str, object],
    ) -> SegmentationResult:
        context = result.observation.context
        if self._state.bootstrap:
            median_reference = self._median_profiles(self._state.bootstrap)
            compatible, details = self._compatible(profile, median_reference, context.height)
            diagnostics.update({f"bootstrap_{key}": value for key, value in details.items()})
            if not compatible:
                self._state.bootstrap = [profile]
            else:
                self._state.bootstrap.append(profile)
        else:
            self._state.bootstrap.append(profile)
        required = self.settings.segmentation_minimum_valid_frames
        diagnostics.update(
            {
                "validation": "g5_1_spatial_then_causal_profile",
                "reason": "reference_bootstrap",
                "state": ObservationStatus.UNKNOWN.value,
                "bootstrap_candidates": len(self._state.bootstrap),
                "bootstrap_required": required,
                "historical_only": False,
            }
        )
        if len(self._state.bootstrap) < required:
            return self._unknown(result, "reference_bootstrap", diagnostics)

        self._state.reference = self._median_profiles(self._state.bootstrap[-required:])
        self._state.bootstrap.clear()
        self._state.last_observed_frame = context.frame_index
        self._state.last_verified_frame = context.frame_index
        self._state.recovery_count = self.settings.segmentation_recovery_frames
        diagnostics.update(
            {
                "reason": "",
                "state": ObservationStatus.OBSERVED.value,
                "reference_initialized": True,
                "measurement_basis": "current_supported_columns",
                "observed_column_fraction": float(np.mean(profile.support)),
                "temporal_completion_fraction": 0.0,
            }
        )
        return self._measured_result(
            result,
            profile,
            ObservationStatus.OBSERVED,
            height_px,
            diagnostics,
        )

    def _profile_from_result(
        self,
        result: SegmentationResult,
    ) -> tuple[_Profile | None, str]:
        observation = result.observation
        crest = np.asarray(observation.crest, dtype=np.float64)
        toe = np.asarray(observation.base, dtype=np.float64)
        if (
            crest.ndim != 2
            or toe.ndim != 2
            or crest.shape != toe.shape
            or crest.shape[0] < 8
            or crest.shape[1] != 2
            or not np.all(np.isfinite(crest))
            or not np.all(np.isfinite(toe))
        ):
            return None, "invalid_profile_geometry"
        order = np.argsort(crest[:, 0])
        crest = crest[order]
        toe = toe[order]
        xs = crest[:, 0]
        if np.any(np.diff(xs) <= 0) or not np.allclose(xs, toe[:, 0], atol=1.0):
            return None, "non_monotonic_or_unpaired_profile"

        if np.any(result.mask):
            support = np.zeros(len(xs), dtype=bool)
            for index, x_value in enumerate(xs):
                x = int(np.clip(round(x_value), 0, observation.context.width - 1))
                upper = int(np.clip(np.floor(crest[index, 1]), 0, observation.context.height - 1))
                lower = int(np.clip(np.ceil(toe[index, 1]) + 1, 0, observation.context.height))
                support[index] = lower > upper and np.any(result.mask[upper:lower, x] > 0)
        elif "spatially_valid" not in observation.diagnostics:
            # Compatibility for the pre-G5.1 synthetic contract tests only.
            support = np.ones(len(xs), dtype=bool)
        else:
            support = np.zeros(len(xs), dtype=bool)
        return _Profile(xs, crest[:, 1], toe[:, 1], support), ""

    def _exclude_occlusions(
        self,
        profile: _Profile,
        detections: Sequence[Detection],
        width: int,
        height: int,
        exclusion_mask: np.ndarray | None,
    ) -> tuple[_Profile, dict[str, float | int]]:
        support = profile.support.copy()
        margin = width * self.settings.segmentation_occlusion_margin_ratio
        vehicle_columns = np.zeros(len(profile.xs), dtype=bool)
        vehicle_count = 0
        for detection in detections:
            box = detection.bbox
            affected = np.logical_and(
                profile.xs >= box.x1 - margin,
                profile.xs <= box.x2 + margin,
            )
            if not np.any(affected):
                continue
            local = np.flatnonzero(affected)
            local_crest = float(np.median(profile.crest[local]))
            local_toe = float(np.median(profile.toe[local]))
            if box.y2 < local_crest - margin or box.y1 > local_toe + margin:
                continue
            vehicle_columns |= affected
            vehicle_count += 1
        support[vehicle_columns] = False

        motion_columns = np.zeros(len(profile.xs), dtype=bool)
        if exclusion_mask is not None and exclusion_mask.shape == (height, width):
            vertical_margin = max(2, int(round(height * 0.012)))
            for index, x_value in enumerate(profile.xs):
                x = int(np.clip(round(x_value), 0, width - 1))
                upper = int(np.clip(round(profile.crest[index]) - vertical_margin, 0, height))
                lower = int(np.clip(round(profile.toe[index]) + vertical_margin, 0, height))
                if lower > upper and np.any(exclusion_mask[upper:lower, x] > 0):
                    motion_columns[index] = True
            support[motion_columns] = False
        excluded = np.logical_or(vehicle_columns, motion_columns)
        return (
            _Profile(profile.xs, profile.crest, profile.toe, support),
            {
                "occluding_vehicle_count": vehicle_count,
                "vehicle_excluded_columns": int(np.count_nonzero(vehicle_columns)),
                "motion_excluded_columns": int(np.count_nonzero(motion_columns)),
                "profile_in_occlusion_fraction": float(np.mean(excluded)),
            },
        )

    def _validate_local_geometry(
        self,
        profile: _Profile,
        frame_height: int,
    ) -> tuple[str, float | None, float]:
        heights = profile.toe - profile.crest
        plausible = np.logical_and(heights > 0, heights < frame_height * 0.30)
        trusted = np.logical_and(profile.support, plausible)
        if np.count_nonzero(trusted) < max(
            8,
            int(len(profile.xs) * self.settings.segmentation_min_current_support_for_estimate),
        ):
            return "insufficient_associated_local_heights", None, 1.0
        values = heights[trusted]
        height_px = float(np.median(values))
        iqr_ratio = float(
            (np.percentile(values, 75) - np.percentile(values, 25))
            / max(height_px, 1.0)
        )
        if iqr_ratio > self.settings.segmentation_max_height_iqr_ratio:
            return "local_height_dispersion_failed", None, iqr_ratio
        expected = self.settings.segmentation_expected_height_y * frame_height
        deviation = abs(height_px - expected) / max(expected, 1.0)
        if deviation > self.settings.segmentation_max_expected_height_deviation_ratio:
            return "expected_height_geometry_failed", None, iqr_ratio
        return "", height_px, iqr_ratio

    def _compatible(
        self,
        current: _Profile,
        reference: _Profile,
        frame_height: int,
    ) -> tuple[bool, dict[str, float]]:
        ref = self._resample(reference, current.xs)
        common = np.logical_and(current.support, ref.support)
        if np.count_nonzero(common) < max(8, int(len(common) * 0.16)):
            return False, {
                "profile_common_support_fraction": float(np.mean(common)),
                "profile_vertical_shift_px": 0.0,
                "profile_residual_ratio": 1.0,
                "temporal_height_change_ratio": 1.0,
            }
        centre_current = (current.crest + current.toe) * 0.5
        centre_reference = (ref.crest + ref.toe) * 0.5
        shift = float(np.median(centre_current[common] - centre_reference[common]))
        reference_height = ref.toe - ref.crest
        current_height = current.toe - current.crest
        scale = max(float(np.median(reference_height[common])), 1.0)
        crest_residual = np.abs(current.crest[common] - shift - ref.crest[common])
        toe_residual = np.abs(current.toe[common] - shift - ref.toe[common])
        residual_ratio = float(np.median(np.maximum(crest_residual, toe_residual)) / scale)
        height_change = float(
            abs(float(np.median(current_height[common])) - float(np.median(reference_height[common])))
            / scale
        )
        shift_ratio = abs(shift) / frame_height
        compatible = (
            shift_ratio <= self.settings.segmentation_max_profile_shift_ratio
            and residual_ratio <= self.settings.segmentation_max_profile_residual_ratio
            and height_change <= self.settings.segmentation_max_temporal_height_change_ratio
        )
        return compatible, {
            "profile_common_support_fraction": float(np.mean(common)),
            "profile_vertical_shift_px": shift,
            "profile_residual_ratio": residual_ratio,
            "temporal_height_change_ratio": height_change,
        }

    @staticmethod
    def _resample(profile: _Profile, xs: np.ndarray) -> _Profile:
        crest = np.interp(xs, profile.xs, profile.crest)
        toe = np.interp(xs, profile.xs, profile.toe)
        support = np.interp(xs, profile.xs, profile.support.astype(np.float32)) >= 0.5
        return _Profile(xs.copy(), crest, toe, support)

    def _median_profiles(self, profiles: Sequence[_Profile]) -> _Profile:
        xs = profiles[-1].xs.copy()
        aligned = [self._resample(profile, xs) for profile in profiles]
        crest = np.median(np.stack([profile.crest for profile in aligned]), axis=0)
        toe = np.median(np.stack([profile.toe for profile in aligned]), axis=0)
        support = np.mean(
            np.stack([profile.support for profile in aligned]).astype(np.float32),
            axis=0,
        ) >= 0.5
        return _Profile(xs, crest, toe, support)

    def _degrade(
        self,
        result: SegmentationResult,
        reason: str,
        diagnostics: dict[str, object],
    ) -> SegmentationResult:
        if self._state.reference is None:
            return self._unknown(result, reason, diagnostics)
        return self._historical(result, reason, diagnostics)

    def _historical(
        self,
        result: SegmentationResult,
        reason: str,
        diagnostics: dict[str, object],
    ) -> SegmentationResult:
        context = result.observation.context
        reference = self._state.reference
        if reference is None or self._state.last_observed_frame is None:
            return self._unknown(result, reason, diagnostics)
        age = max(0, context.frame_index - self._state.last_observed_frame)
        last_verified = (
            self._state.last_verified_frame
            if self._state.last_verified_frame is not None
            else self._state.last_observed_frame
        )
        if context.frame_index - last_verified > self.settings.segmentation_reference_expiry_frames:
            # Same rule as in low light: expiry counts frames since the memory
            # was last verified, by observation or by a registered night carry.
            self._state.reference = None
            self._state.bootstrap.clear()
            self._state.recovery_count = 0
            self._state.last_observed_frame = None
            self._state.last_verified_frame = None
            return self._unknown(result, "historical_reference_expired", diagnostics)
        diagnostics.update(
            {
                "validation": "g5_1_spatial_then_causal_profile",
                "reason": reason,
                "state": ObservationStatus.HISTORICAL_REFERENCE.value,
                "profile_age_frames": age,
                "measurement_basis": "none_historical_only",
                "historical_only": True,
                "reference_updated": False,
                "observed_column_fraction": 0.0,
                "temporal_completion_fraction": 0.0,
            }
        )
        observation = BermObservation(
            context=context,
            status=ObservationStatus.HISTORICAL_REFERENCE,
            crest=tuple(
                (float(x), float(y)) for x, y in zip(reference.xs, reference.crest)
            ),
            base=tuple(
                (float(x), float(y)) for x, y in zip(reference.xs, reference.toe)
            ),
            height_px=None,
            height_m_estimated=None,
            diagnostics=diagnostics,
        )
        return SegmentationResult(
            observation,
            np.zeros_like(result.mask),
            max(0.0, result.confidence * (1.0 - age / (self.settings.segmentation_reference_expiry_frames + 1))),
        )

    @staticmethod
    def _unknown(
        result: SegmentationResult,
        reason: str,
        diagnostics: dict[str, object],
    ) -> SegmentationResult:
        context = result.observation.context
        diagnostics.update(
            {
                "validation": diagnostics.get(
                    "validation", "g5_1_spatial_then_causal_profile"
                ),
                "reason": reason,
                "state": ObservationStatus.UNKNOWN.value,
                "measurement_basis": "none",
                "historical_only": False,
            }
        )
        observation = BermObservation(
            context=context,
            status=ObservationStatus.UNKNOWN,
            diagnostics=diagnostics,
        )
        return SegmentationResult(
            observation,
            np.zeros_like(result.mask),
            result.confidence,
        )

    def _measured_result(
        self,
        result: SegmentationResult,
        profile: _Profile,
        status: ObservationStatus,
        height_px: float,
        diagnostics: dict[str, object],
    ) -> SegmentationResult:
        context = result.observation.context
        mask = self._mask(profile, context.height, context.width)
        observation = BermObservation(
            context=context,
            status=status,
            crest=tuple((float(x), float(y)) for x, y in zip(profile.xs, profile.crest)),
            base=tuple((float(x), float(y)) for x, y in zip(profile.xs, profile.toe)),
            height_px=height_px,
            height_m_estimated=height_px / self.settings.pixels_per_meter,
            diagnostics=diagnostics,
        )
        return SegmentationResult(observation, mask, result.confidence)

    @staticmethod
    def _mask(profile: _Profile, height: int, width: int) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.uint8)
        for start, end in TemporalBermValidator._true_runs(profile.support):
            if end - start < 2:
                continue
            xs = profile.xs[start:end]
            polygon = np.vstack(
                [
                    np.column_stack([xs, profile.crest[start:end]]),
                    np.column_stack([xs[::-1], profile.toe[start:end][::-1]]),
                ]
            )
            cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32)], 255)
        return mask

    @staticmethod
    def _true_runs(values: np.ndarray) -> list[tuple[int, int]]:
        padded = np.pad(values.astype(np.int8), (1, 1))
        changes = np.diff(padded)
        return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))

    @staticmethod
    def _maximum_false_run(values: np.ndarray) -> int:
        maximum = current = 0
        for value in values:
            if bool(value):
                current = 0
            else:
                current += 1
                maximum = max(maximum, current)
        return maximum

    def _register_static_structure(
        self,
        frame: np.ndarray | None,
        detections: Sequence[Detection],
        exclusion_mask: np.ndarray | None,
    ) -> dict[str, float | bool | str]:
        """Estimate camera motion from static structure only.

        Machinery boxes and the detector exclusion mask are removed from both
        structure maps before phase correlation, so a passing truck cannot be
        mistaken for a camera move.  Per-frame camera motion is capped by
        ``segmentation_max_registration_shift_ratio``; larger shifts are
        treated as unreliable and never applied to the remembered profile.
        """

        if frame is None:
            return {"registration_applied": False, "registration_reason": "frame_unavailable"}
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        equalized = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 6)).apply(gray)
        structure = cv2.Sobel(equalized, cv2.CV_32F, 1, 0, ksize=3) ** 2
        structure += cv2.Sobel(equalized, cv2.CV_32F, 0, 1, ksize=3) ** 2
        structure = np.sqrt(structure).astype(np.float32)
        static = np.ones((90, 160), dtype=np.float32)
        for detection in detections:
            box = detection.bbox
            x1 = int(np.floor(box.x1 * 160.0 / width)) - 3
            x2 = int(np.ceil(box.x2 * 160.0 / width)) + 3
            y1 = int(np.floor(box.y1 * 90.0 / height)) - 3
            y2 = int(np.ceil(box.y2 * 90.0 / height)) + 3
            static[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = 0.0
        if exclusion_mask is not None and exclusion_mask.shape == (height, width):
            moving = cv2.resize(exclusion_mask.astype(np.float32), (160, 90), interpolation=cv2.INTER_AREA)
            static[moving > 0.0] = 0.0
        previous = self._state.previous_structure
        previous_static = self._state.previous_static_mask
        self._state.previous_structure = structure
        self._state.previous_static_mask = static
        unreliable = {
            "registration_applied": False, "registration_reliable": False,
            "registration_response": 0.0, "registration_dx_px": 0.0, "registration_dy_px": 0.0,
        }
        if previous is None or previous_static is None:
            return {**unreliable, "registration_reason": "first_frame"}
        combined = previous_static * static
        static_fraction = float(np.mean(combined))
        if static_fraction < 0.5:
            return {**unreliable, "registration_reason": "insufficient_static_structure",
                    "registration_static_fraction": static_fraction}
        masked_previous = previous * combined
        masked_current = structure * combined
        if float(np.std(masked_previous)) < 1e-5 or float(np.std(masked_current)) < 1e-5:
            return {**unreliable, "registration_reason": "structureless_frame"}
        window = cv2.createHanningWindow((160, 90), cv2.CV_32F)
        # OpenCV applies the window in place when the inputs already have the
        # exact type and size; the stored maps must not be mutated.
        shift, response = cv2.phaseCorrelate(masked_previous.copy(), masked_current.copy(), window)
        dx_small, dy_small = float(shift[0]), float(shift[1])
        if not (np.isfinite(response) and np.isfinite(dx_small) and np.isfinite(dy_small)):
            response, dx_small, dy_small = 0.0, 0.0, 0.0
        # Sub-pixel jitter on the 160x90 map scales by 8x in a 720p frame and,
        # accumulated every frame, becomes a random walk: ignore the dead band.
        if abs(dx_small) < 0.3 and abs(dy_small) < 0.3:
            dx_small, dy_small = 0.0, 0.0
        dx = dx_small * width / 160.0
        dy = dy_small * height / 90.0
        limit = self.settings.segmentation_max_registration_shift_ratio
        reliable = response >= 0.08 and abs(dx) <= width * limit and abs(dy) <= height * limit
        if reliable and self._state.reference is not None and (dx or dy):
            reference = self._state.reference
            self._state.reference = _Profile(reference.xs + dx, reference.crest + dy, reference.toe + dy, reference.support.copy())
        return {
            "registration_applied": bool(reliable and self._state.reference is not None),
            "registration_reliable": bool(reliable),
            "registration_response": float(response),
            "registration_dx_px": dx,
            "registration_dy_px": dy,
            "registration_static_fraction": static_fraction,
            "registration_reason": "ok" if reliable else "unreliable_or_large_transform",
        }
