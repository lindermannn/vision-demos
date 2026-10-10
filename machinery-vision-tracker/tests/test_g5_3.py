"""G5.3 contracts: the daytime berm is remembered at night, never re-detected,
and a place change that morphs without a cut is still a scene change."""

from pathlib import Path
import sys
import unittest

import cv2
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bermguard.contracts import BoundingBox, Detection, ObservationStatus  # noqa: E402
from bermguard.g2_config import G2Settings  # noqa: E402
from bermguard.scene import SceneCutDetector  # noqa: E402
from bermguard.segmentation.classical import ClassicalBermSegmenter  # noqa: E402
from bermguard.segmentation.result import SegmentationResult  # noqa: E402
from bermguard.segmentation.validation import TemporalBermValidator  # noqa: E402
from test_g5_1 import (  # noqa: E402
    context,
    day_frame,
    raw_profile,
    settings,
    synthetic_berm_frame,
    unknown_profile,
)


def night_frame_with_lights() -> np.ndarray:
    """Crushed-black night frame where only lamps and a lit road carry structure."""

    frame = np.full((360, 640, 3), 18, dtype=np.uint8)
    for x in (90, 210, 330, 450, 560):
        cv2.circle(frame, (x, 120), 5, (235, 235, 235), -1)
    cv2.ellipse(frame, (320, 300), (260, 30), 0, 0, 360, (120, 120, 120), -1)
    return frame


def structured_day_frame(variant: str) -> np.ndarray:
    frame = day_frame(185)
    if variant == "a":
        cv2.rectangle(frame, (70, 135), (570, 220), (55, 65, 78), -1)
        cv2.line(frame, (30, 285), (610, 250), (230, 230, 230), 8)
    else:
        cv2.rectangle(frame, (90, 30), (300, 110), (55, 65, 78), -1)
        cv2.rectangle(frame, (380, 240), (600, 340), (90, 60, 40), -1)
        cv2.line(frame, (20, 40), (620, 330), (230, 230, 230), 8)
    return frame


def crest_median_y(result: SegmentationResult) -> float:
    return float(np.median([y for _, y in result.observation.crest]))


class LowLightAndDriftContractTests(unittest.TestCase):
    def configured(self, **changes: object) -> G2Settings:
        defaults = {
            "segmentation_minimum_valid_frames": 3,
            "segmentation_reference_expiry_frames": 30,
            "segmentation_recovery_frames": 1,
            "segmentation_reconfirmation_failure_frames": 5,
        }
        defaults.update(changes)
        return settings(**defaults)

    def bootstrap(self, validator: TemporalBermValidator) -> SegmentationResult:
        result = unknown_profile(0)
        for index in range(3):
            result = validator.refine(raw_profile(index), (), day_frame())
        self.assertEqual(ObservationStatus.OBSERVED, result.observation.status)
        return result

    def test_25_night_with_day_reference_is_estimated_never_observed(self) -> None:
        validator = TemporalBermValidator(self.configured())
        reference = self.bootstrap(validator)
        for index in range(3, 9):
            # The spatial extractor returns a shifted, wrong profile at night;
            # it must be ignored in favour of the remembered daytime geometry.
            result = validator.refine(
                raw_profile(index, vertical_shift=40.0), (), night_frame_with_lights()
            )
            observation = result.observation
            self.assertEqual(ObservationStatus.TEMPORAL_ESTIMATE, observation.status)
            self.assertEqual("low_light_reference_estimate", observation.diagnostics["reason"])
            self.assertEqual("day_reference_carried", observation.diagnostics["measurement_basis"])
            self.assertEqual(index - 2, observation.diagnostics["profile_age_frames"])
            self.assertAlmostEqual(
                float(reference.observation.height_px), float(observation.height_px), delta=4.0
            )
            self.assertAlmostEqual(crest_median_y(reference), crest_median_y(result), delta=4.0)

    def test_26_night_without_reference_states_the_reason(self) -> None:
        validator = TemporalBermValidator(self.configured())
        result = validator.refine(raw_profile(0), (), night_frame_with_lights())
        self.assertEqual(ObservationStatus.UNKNOWN, result.observation.status)
        self.assertEqual("night_without_reference", result.observation.diagnostics["reason"])
        self.assertEqual((), result.observation.crest)

    def test_27_low_light_never_seeds_a_reference(self) -> None:
        validator = TemporalBermValidator(self.configured(segmentation_bootstrap_settle_frames=2))
        for index in range(6):
            result = validator.refine(raw_profile(index), (), night_frame_with_lights())
            self.assertEqual(ObservationStatus.UNKNOWN, result.observation.status)
        # The first daylight frames after the ramp are a settle period: still no
        # reference and no bootstrap candidates are accumulated.
        for index in (6, 7):
            settling = validator.refine(raw_profile(index), (), day_frame())
            self.assertEqual(ObservationStatus.UNKNOWN, settling.observation.status)
            self.assertEqual("settling_after_low_light", settling.observation.diagnostics["reason"])
        first_day = validator.refine(raw_profile(8), (), day_frame())
        self.assertEqual(ObservationStatus.UNKNOWN, first_day.observation.status)
        self.assertEqual(1, first_day.observation.diagnostics["bootstrap_candidates"])

    def test_28_reference_must_be_reconfirmed_when_light_returns(self) -> None:
        validator = TemporalBermValidator(self.configured(segmentation_bootstrap_settle_frames=1))
        self.bootstrap(validator)
        for index in (3, 4):
            validator.refine(raw_profile(index), (), night_frame_with_lights())
        statuses = []
        result = unknown_profile(5)
        for index in range(5, 10):
            result = validator.refine(raw_profile(index, vertical_shift=60.0), (), day_frame())
            statuses.append(result.observation.status)
        self.assertEqual([ObservationStatus.HISTORICAL_REFERENCE] * 4, statuses[:4])
        self.assertEqual(ObservationStatus.UNKNOWN, statuses[4])
        self.assertEqual(
            "reference_not_reconfirmed_after_low_light",
            result.observation.diagnostics["reason"],
        )
        for index in range(10, 13):
            result = validator.refine(raw_profile(index, vertical_shift=60.0), (), day_frame())
        self.assertEqual(ObservationStatus.OBSERVED, result.observation.status)
        self.assertAlmostEqual(238.0, crest_median_y(result), delta=5.0)

    def test_29_compatible_daylight_reconfirms_and_clears_pending(self) -> None:
        validator = TemporalBermValidator(self.configured())
        self.bootstrap(validator)
        for index in (3, 4):
            validator.refine(raw_profile(index), (), night_frame_with_lights())
        result = validator.refine(raw_profile(5), (), day_frame())
        self.assertEqual(ObservationStatus.OBSERVED, result.observation.status)
        self.assertTrue(result.observation.diagnostics["reference_reconfirmed_after_low_light"])

    def test_30_structural_drift_is_a_deferred_cut(self) -> None:
        detector = SceneCutDetector(
            self.configured(scene_drift_lag_frames=8, scene_drift_frames=3, scene_cooldown_frames=2)
        )
        first = structured_day_frame("a").astype(np.float32)
        second = structured_day_frame("b").astype(np.float32)
        frames = [structured_day_frame("a")] * 12
        frames += [
            np.clip(first * (1 - k / 8) + second * (k / 8), 0, 255).astype(np.uint8)
            for k in range(1, 8)
        ]
        frames += [structured_day_frame("b")] * 24
        decisions = [detector.update(frame) for frame in frames]
        cuts = [index for index, decision in enumerate(decisions) if decision.is_cut]
        self.assertEqual(1, len(cuts), cuts)
        self.assertGreaterEqual(cuts[0], 12)
        self.assertTrue(decisions[cuts[0]].structural_drift)

    def test_31_day_night_ramp_is_not_structural_drift(self) -> None:
        detector = SceneCutDetector(
            self.configured(scene_drift_lag_frames=8, scene_drift_frames=3, scene_cooldown_frames=2)
        )
        bright = structured_day_frame("a")
        dark = np.clip(bright.astype(np.float32) * 0.22 + 6.0, 0, 255).astype(np.uint8)
        frames = [bright] * 12 + [dark] * 20 + [bright] * 20
        decisions = [detector.update(frame) for frame in frames]
        self.assertFalse(any(decision.is_cut for decision in decisions))
        self.assertFalse(any(decision.structural_drift for decision in decisions))

    def test_32_passing_vehicle_does_not_register_as_camera_motion(self) -> None:
        validator = TemporalBermValidator(self.configured())
        self.bootstrap(validator)
        reference = crest_median_y(validator.refine(raw_profile(3), (), structured_day_frame("a")))
        for index, x in enumerate((60, 140, 220, 300, 380), start=4):
            frame = structured_day_frame("a")
            cv2.rectangle(frame, (x, 150), (x + 160, 290), (240, 210, 40), -1)
            truck = Detection("haul truck", 0.9, BoundingBox(x, 150, x + 160, 290), 1)
            result = validator.refine(raw_profile(index), (truck,), frame)
            diagnostics = result.observation.diagnostics
            self.assertEqual(0.0, diagnostics["registration_dx_px"], diagnostics)
            self.assertEqual(0.0, diagnostics["registration_dy_px"], diagnostics)
        self.assertAlmostEqual(reference, crest_median_y(result), delta=1.0)

    def test_33_heavy_occlusion_freezes_the_reference(self) -> None:
        validator = TemporalBermValidator(self.configured())
        reference = crest_median_y(self.bootstrap(validator))
        truck = Detection("haul truck", 0.9, BoundingBox(120, 140, 400, 280), 1)
        for index in range(3, 9):
            result = validator.refine(raw_profile(index, vertical_shift=10.0), (truck,), day_frame())
            self.assertEqual(ObservationStatus.TEMPORAL_ESTIMATE, result.observation.status)
            self.assertFalse(result.observation.diagnostics["reference_updated"])
        self.assertAlmostEqual(reference, crest_median_y(result), delta=0.5)

    def test_34_light_occlusion_still_updates_slowly(self) -> None:
        validator = TemporalBermValidator(self.configured())
        reference = crest_median_y(self.bootstrap(validator))
        small = Detection("haul truck", 0.9, BoundingBox(300, 150, 340, 260), 1)
        for index in range(3, 9):
            result = validator.refine(raw_profile(index, vertical_shift=10.0), (small,), day_frame())
        self.assertTrue(result.observation.diagnostics["reference_updated"])
        moved = crest_median_y(result) - reference
        self.assertGreater(moved, 1.0)
        self.assertLess(moved, 10.0)

    def test_37_verified_night_carry_does_not_expire(self) -> None:
        # Expiry counts frames without verification; a night carry with reliable
        # registration keeps the memory alive beyond the plain expiry window.
        validator = TemporalBermValidator(self.configured(segmentation_reference_expiry_frames=10))
        self.bootstrap(validator)
        result = unknown_profile(3)
        for index in range(3, 30):
            result = validator.refine(raw_profile(index, vertical_shift=40.0), (), night_frame_with_lights())
        self.assertEqual(ObservationStatus.TEMPORAL_ESTIMATE, result.observation.status)
        self.assertEqual(27, result.observation.diagnostics["profile_age_frames"])

    def test_38_consistent_contrary_daylight_evidence_replaces_reference(self) -> None:
        # A daylight reference that was wrong from the start is challenged: after
        # the configured streak of incompatible candidates it is dropped and the
        # consistent evidence bootstraps a new one.
        validator = TemporalBermValidator(self.configured(segmentation_reconfirmation_failure_frames=4))
        wrong = self.bootstrap(validator)
        statuses = []
        result = unknown_profile(3)
        for index in range(3, 10):
            result = validator.refine(raw_profile(index, vertical_shift=60.0), (), day_frame())
            statuses.append(result.observation.status)
        self.assertEqual([ObservationStatus.HISTORICAL_REFERENCE] * 3, statuses[:3])
        self.assertEqual(ObservationStatus.UNKNOWN, statuses[3])
        self.assertEqual(ObservationStatus.OBSERVED, statuses[-1])
        self.assertGreater(crest_median_y(result) - crest_median_y(wrong), 50.0)

    def test_39_daylight_after_verified_night_does_not_expire_the_memory(self) -> None:
        validator = TemporalBermValidator(self.configured(segmentation_reference_expiry_frames=10))
        self.bootstrap(validator)
        for index in range(3, 25):
            validator.refine(raw_profile(index, vertical_shift=40.0), (), night_frame_with_lights())
        # Light returns with incompatible evidence: the memory must stay historical,
        # not expire because the last daylight observation is far behind.
        for index in range(25, 28):
            result = validator.refine(raw_profile(index, vertical_shift=60.0), (), day_frame())
            self.assertEqual(ObservationStatus.HISTORICAL_REFERENCE, result.observation.status)
            self.assertEqual("current_geometry_incompatible_with_reference", result.observation.diagnostics["reason"])

    def test_35_candidate_below_operating_surface_is_road_not_berm(self) -> None:
        # A strong dark band low in the image (a road stripe) competes with the
        # real berm.  Machinery touches the ground between rows 200 and 260, so
        # anything with its toe below the near contact cannot be the berm.
        frame = synthetic_berm_frame()
        frame[245:285, :] = (30, 40, 52)
        frame[285:293, :] = (150, 160, 170)
        segmenter = ClassicalBermSegmenter(settings())
        segmenter.set_operating_surface((200.0, 260.0))
        result = segmenter.segment(frame, context())
        self.assertEqual(ObservationStatus.OBSERVED, result.observation.status)
        self.assertLess(crest_median_y(result), 260.0)
        rejections = result.observation.diagnostics["candidate_rejections"]
        self.assertGreater(
            rejections["below_operating_surface"] + rejections["inside_operating_surface"], 0
        )
        self.assertTrue(result.observation.diagnostics["operating_surface_used"])

    def test_36_remembered_berm_below_operating_surface_is_dropped(self) -> None:
        validator = TemporalBermValidator(self.configured())
        self.bootstrap(validator)
        # The remembered toe sits near row 211; the surface reveals machinery
        # touching the ground far above it, so the memory was road.
        result = validator.refine(
            raw_profile(3), (), day_frame(), operating_surface=(100.0, 150.0)
        )
        self.assertTrue(result.observation.diagnostics["reference_dropped_below_operating_surface"])
        self.assertEqual(ObservationStatus.UNKNOWN, result.observation.status)
        self.assertEqual("reference_bootstrap", result.observation.diagnostics["reason"])


if __name__ == "__main__":
    unittest.main()
