"""Adversarial G5.1 contracts for spatial identity and causal persistence."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest

import cv2
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bermguard.config import load_configuration  # noqa: E402
from bermguard.contracts import (  # noqa: E402
    BermObservation,
    BoundingBox,
    Detection,
    FrameContext,
    ObservationStatus,
)
from bermguard.g2_config import G2Settings  # noqa: E402
from bermguard.scene import SceneCutDetector  # noqa: E402
from bermguard.segmentation.classical import ClassicalBermSegmenter  # noqa: E402
from bermguard.segmentation.result import SegmentationResult  # noqa: E402
from bermguard.segmentation.validation import TemporalBermValidator  # noqa: E402


def settings(**changes: object) -> G2Settings:
    base = G2Settings.from_mapping(load_configuration(None).data)
    return replace(base, **changes)


def context(frame_index: int = 0, scene_id: int = 0) -> FrameContext:
    return FrameContext("g5_1", scene_id, frame_index, frame_index / 24.0, 640, 360)


def day_frame(value: int = 150) -> np.ndarray:
    return np.full((360, 640, 3), value, dtype=np.uint8)


def synthetic_berm_frame() -> np.ndarray:
    """A dark compacted-earth ridge bounded by a brighter far field and road."""

    frame = np.empty((360, 640, 3), dtype=np.uint8)
    for y in range(360):
        value = int(np.clip(154 + 0.055 * y, 0, 255))
        frame[y, :] = (value - 12, value - 2, value + 7)
    for x in range(640):
        crest = 177 + int(round(5.0 * np.sin(x / 83.0)))
        toe = crest + 45 + int(round(3.0 * np.sin(x / 51.0)))
        frame[crest:toe, x] = (48, 73, 100)
        frame[toe : min(360, toe + 7), x] = (118, 132, 145)
    return frame


def raw_profile(
    frame_index: int,
    *,
    scene_id: int = 0,
    height: float = 33.0,
    vertical_shift: float = 0.0,
    spatially_valid: bool = True,
    support_start: int = 0,
    support_end: int = 96,
) -> SegmentationResult:
    ctx = context(frame_index, scene_id)
    xs = np.linspace(48.0, 592.0, 96)
    crest_y = 178.0 + vertical_shift + 2.5 * np.sin(xs / 71.0)
    local_height = height + 3.0 * np.sin(xs / 59.0)
    base_y = crest_y + local_height
    crest = tuple((float(x), float(y)) for x, y in zip(xs, crest_y))
    base = tuple((float(x), float(y)) for x, y in zip(xs, base_y))
    mask = np.zeros((ctx.height, ctx.width), dtype=np.uint8)
    for index in range(max(0, support_start), min(len(xs), support_end - 1)):
        polygon = np.rint(
            np.asarray(
                [
                    [xs[index], crest_y[index]],
                    [xs[index + 1], crest_y[index + 1]],
                    [xs[index + 1], base_y[index + 1]],
                    [xs[index], base_y[index]],
                ]
            )
        ).astype(np.int32)
        cv2.fillPoly(mask, [polygon], 255)
    support_fraction = max(0.0, min(1.0, (support_end - support_start) / len(xs)))
    return SegmentationResult(
        BermObservation(
            context=ctx,
            status=ObservationStatus.OBSERVED,
            crest=crest,
            base=base,
            height_px=float(np.median(local_height)),
            height_m_estimated=float(np.median(local_height)) / 5.0,
            diagnostics={
                "spatially_valid": spatially_valid,
                "valid_column_fraction": support_fraction,
                "paired_support_fraction": support_fraction,
                "height_iqr_ratio": 0.08,
                "candidate_kind": "berm" if spatially_valid else "horizon",
            },
        ),
        mask,
        0.86,
    )


def unknown_profile(frame_index: int, scene_id: int = 0, reason: str = "no_candidate") -> SegmentationResult:
    ctx = context(frame_index, scene_id)
    return SegmentationResult(
        BermObservation(
            context=ctx,
            status=ObservationStatus.UNKNOWN,
            diagnostics={"reason": reason, "spatially_valid": False},
        ),
        np.zeros((ctx.height, ctx.width), dtype=np.uint8),
        0.0,
    )


class TemporalStateContractTests(unittest.TestCase):
    """The twelve causal temporal tests retained by the G5.1 gate."""

    def configured(self, **changes: object) -> G2Settings:
        defaults = {
            "segmentation_minimum_valid_frames": 3,
            "segmentation_reference_expiry_frames": 8,
            "segmentation_recovery_frames": 1,
        }
        defaults.update(changes)
        return settings(**defaults)

    def bootstrap(self, validator: TemporalBermValidator) -> SegmentationResult:
        result = unknown_profile(0)
        for index in range(3):
            result = validator.refine(raw_profile(index), (), day_frame())
        return result

    def test_01_bootstrap_requires_multiple_compatible_observations(self) -> None:
        validator = TemporalBermValidator(self.configured())
        first = validator.refine(raw_profile(0), (), day_frame())
        second = validator.refine(raw_profile(1), (), day_frame())
        third = validator.refine(raw_profile(2), (), day_frame())
        self.assertEqual(ObservationStatus.UNKNOWN, first.observation.status)
        self.assertEqual(ObservationStatus.UNKNOWN, second.observation.status)
        self.assertEqual(ObservationStatus.OBSERVED, third.observation.status)

    def test_02_one_bad_frame_preserves_only_historical_reference(self) -> None:
        validator = TemporalBermValidator(self.configured())
        self.bootstrap(validator)
        degraded = validator.refine(unknown_profile(3), (), day_frame())
        self.assertEqual(ObservationStatus.HISTORICAL_REFERENCE, degraded.observation.status)
        self.assertIsNone(degraded.observation.height_px)

    def test_03_historical_reference_is_never_current_measurement(self) -> None:
        validator = TemporalBermValidator(self.configured())
        self.bootstrap(validator)
        degraded = validator.refine(unknown_profile(3), (), day_frame())
        self.assertIsNone(degraded.observation.height_m_estimated)
        self.assertGreater(len(degraded.observation.crest), 0)

    def test_04_partial_current_support_uses_temporal_estimate(self) -> None:
        validator = TemporalBermValidator(self.configured())
        self.bootstrap(validator)
        vehicle = Detection("haul truck", 0.9, BoundingBox(245, 145, 395, 270), 1)
        partial = validator.refine(raw_profile(3), (vehicle,), day_frame())
        self.assertEqual(ObservationStatus.TEMPORAL_ESTIMATE, partial.observation.status)
        self.assertIsNotNone(partial.observation.height_px)

    def test_05_reference_expires(self) -> None:
        validator = TemporalBermValidator(self.configured(segmentation_reference_expiry_frames=3))
        self.bootstrap(validator)
        state = unknown_profile(3)
        for index in range(3, 7):
            state = validator.refine(unknown_profile(index), (), day_frame())
        self.assertEqual(ObservationStatus.UNKNOWN, state.observation.status)
        self.assertEqual((), state.observation.crest)

    def test_06_compatible_evidence_recovers_observed(self) -> None:
        validator = TemporalBermValidator(self.configured())
        self.bootstrap(validator)
        validator.refine(unknown_profile(3), (), day_frame())
        recovered = validator.refine(raw_profile(4), (), day_frame())
        self.assertEqual(ObservationStatus.OBSERVED, recovered.observation.status)

    def test_07_isolated_geometric_jump_does_not_update_reference(self) -> None:
        validator = TemporalBermValidator(self.configured())
        baseline = self.bootstrap(validator)
        jumped = validator.refine(raw_profile(3, height=82.0), (), day_frame())
        recovered = validator.refine(raw_profile(4), (), day_frame())
        self.assertEqual(ObservationStatus.HISTORICAL_REFERENCE, jumped.observation.status)
        self.assertAlmostEqual(
            float(baseline.observation.height_px),
            float(recovered.observation.height_px),
            delta=4.0,
        )

    def test_08_scene_change_invalidates_reference_immediately(self) -> None:
        validator = TemporalBermValidator(self.configured())
        self.bootstrap(validator)
        changed = validator.refine(unknown_profile(3, scene_id=1), (), day_frame())
        self.assertEqual(ObservationStatus.UNKNOWN, changed.observation.status)
        self.assertEqual((), changed.observation.crest)

    def test_09_state_is_not_shared_between_methods_or_instances(self) -> None:
        first = TemporalBermValidator(self.configured())
        second = TemporalBermValidator(self.configured())
        self.bootstrap(first)
        uninitialized = second.refine(unknown_profile(3), (), day_frame())
        self.assertEqual(ObservationStatus.UNKNOWN, uninitialized.observation.status)

    def test_10_small_camera_translation_remains_compatible(self) -> None:
        validator = TemporalBermValidator(self.configured())
        self.bootstrap(validator)
        shifted = validator.refine(raw_profile(3, vertical_shift=3.0), (), day_frame())
        self.assertIn(
            shifted.observation.status,
            {ObservationStatus.OBSERVED, ObservationStatus.TEMPORAL_ESTIMATE},
        )

    def test_11_short_gap_does_not_destroy_reference(self) -> None:
        validator = TemporalBermValidator(self.configured())
        self.bootstrap(validator)
        for index in range(3, 6):
            result = validator.refine(unknown_profile(index), (), day_frame())
            self.assertEqual(ObservationStatus.HISTORICAL_REFERENCE, result.observation.status)

    def test_12_low_support_never_bootstraps(self) -> None:
        validator = TemporalBermValidator(self.configured())
        result = unknown_profile(0)
        for index in range(6):
            result = validator.refine(
                raw_profile(index, support_start=35, support_end=58), (), day_frame()
            )
        self.assertEqual(ObservationStatus.UNKNOWN, result.observation.status)

    def test_12b_day_night_change_with_same_structure_is_not_a_scene_cut(self) -> None:
        detector = SceneCutDetector(self.configured())
        bright = day_frame(185)
        cv2.rectangle(bright, (70, 135), (570, 220), (55, 65, 78), -1)
        cv2.line(bright, (30, 285), (610, 250), (230, 230, 230), 8)
        dark = np.clip(bright.astype(np.float32) * 0.22 + 6.0, 0, 255).astype(np.uint8)
        detector.update(bright)
        decision = detector.update(dark)
        self.assertTrue(decision.illumination_transition)
        self.assertTrue(decision.structural_match)
        self.assertFalse(decision.is_cut)


class SpatialTopographicContractTests(unittest.TestCase):
    def test_13_distant_horizon_is_not_a_berm(self) -> None:
        frame = np.full((360, 640, 3), (185, 180, 170), dtype=np.uint8)
        frame[:132] = (180, 145, 105)
        contour = np.asarray([[0, 132], [120, 118], [260, 136], [410, 112], [639, 130], [639, 205], [0, 205]], dtype=np.int32)
        cv2.fillPoly(frame, [contour], (48, 52, 58))
        result = ClassicalBermSegmenter(settings()).segment(frame, context())
        self.assertEqual(ObservationStatus.UNKNOWN, result.observation.status)

    def test_14_operational_road_without_berm_is_unknown(self) -> None:
        frame = day_frame()
        for y in range(360):
            frame[y, :] = np.uint8(np.clip(105 + y * 0.22, 0, 255))
        result = ClassicalBermSegmenter(settings()).segment(frame, context())
        self.assertEqual(ObservationStatus.UNKNOWN, result.observation.status)

    def test_15_crest_without_reliable_toe_has_no_height(self) -> None:
        frame = day_frame(175)
        frame[180:] = (65, 70, 75)
        result = ClassicalBermSegmenter(settings()).segment(frame, context())
        self.assertEqual(ObservationStatus.UNKNOWN, result.observation.status)
        self.assertIsNone(result.observation.height_px)

    def test_16_toe_without_reliable_crest_has_no_height(self) -> None:
        frame = day_frame(65)
        frame[230:] = (175, 180, 185)
        result = ClassicalBermSegmenter(settings()).segment(frame, context())
        self.assertEqual(ObservationStatus.UNKNOWN, result.observation.status)
        self.assertIsNone(result.observation.height_px)

    def test_17_boundaries_from_different_objects_are_rejected(self) -> None:
        frame = day_frame(65)
        frame[:175, :285] = (170, 170, 170)  # crown-like edge only on the left
        frame[230:, 355:] = (170, 170, 170)  # toe-like edge only on the right
        result = ClassicalBermSegmenter(settings()).segment(frame, context())
        self.assertEqual(ObservationStatus.UNKNOWN, result.observation.status)

    def test_18_vehicle_columns_are_excluded_without_deforming_reference(self) -> None:
        validator = TemporalBermValidator(settings(segmentation_minimum_valid_frames=1))
        baseline = validator.refine(raw_profile(0), (), day_frame())
        vehicle = Detection("haul truck", 0.95, BoundingBox(250, 140, 390, 275), 1)
        partial = validator.refine(raw_profile(1), (vehicle,), day_frame())
        self.assertEqual(ObservationStatus.TEMPORAL_ESTIMATE, partial.observation.status)
        self.assertEqual(0, int(np.count_nonzero(partial.mask[:, 250:390])))
        self.assertAlmostEqual(
            float(baseline.observation.height_px),
            float(partial.observation.height_px),
            delta=4.0,
        )

    def test_19_dust_or_light_beam_cannot_create_a_profile(self) -> None:
        frame = day_frame(105)
        cv2.ellipse(frame, (330, 210), (90, 135), 0, 0, 360, (220, 220, 220), -1)
        cv2.circle(frame, (505, 188), 28, (255, 255, 255), -1)
        result = ClassicalBermSegmenter(settings()).segment(frame, context())
        self.assertEqual(ObservationStatus.UNKNOWN, result.observation.status)

    def test_20_mask_is_contained_between_paired_profiles(self) -> None:
        result = ClassicalBermSegmenter(settings()).segment(synthetic_berm_frame(), context())
        self.assertEqual(ObservationStatus.OBSERVED, result.observation.status)
        crest = np.asarray(result.observation.crest, dtype=np.float64)
        base = np.asarray(result.observation.base, dtype=np.float64)
        ys, xs = np.nonzero(result.mask)
        self.assertGreater(len(xs), 0)
        upper = np.interp(xs, crest[:, 0], crest[:, 1])
        lower = np.interp(xs, base[:, 0], base[:, 1])
        self.assertTrue(np.all(ys >= np.floor(upper) - 1))
        self.assertTrue(np.all(ys <= np.ceil(lower) + 1))

    def test_21_height_uses_only_associated_supported_columns(self) -> None:
        configured = settings(segmentation_minimum_valid_frames=1)
        validator = TemporalBermValidator(configured)
        result = validator.refine(raw_profile(0, support_start=8, support_end=82), (), day_frame())
        self.assertEqual(ObservationStatus.OBSERVED, result.observation.status)
        local = np.asarray([b[1] - c[1] for c, b in zip(result.observation.crest, result.observation.base)])
        self.assertAlmostEqual(float(np.median(local)), float(result.observation.height_px), delta=2.0)

    def test_22_night_without_day_bootstrap_remains_unknown(self) -> None:
        validator = TemporalBermValidator(settings(segmentation_minimum_valid_frames=1))
        result = validator.refine(raw_profile(0), (), day_frame(20))
        self.assertEqual(ObservationStatus.UNKNOWN, result.observation.status)

    def test_23_stable_false_profile_is_rejected(self) -> None:
        validator = TemporalBermValidator(settings(segmentation_minimum_valid_frames=2))
        for index in range(8):
            result = validator.refine(
                raw_profile(index, spatially_valid=False), (), day_frame()
            )
        self.assertEqual(ObservationStatus.UNKNOWN, result.observation.status)

    def test_24_blind_run_requires_freeze_and_is_single_use(self) -> None:
        from bermguard.blind_gate import BlindRunLedger

        with TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = BlindRunLedger(root / "ledger.json")
            with self.assertRaises(RuntimeError):
                ledger.claim("source-sha", "code-sha", "config-sha")
            ledger.freeze("code-sha", "config-sha")
            ledger.claim("source-sha", "code-sha", "config-sha")
            with self.assertRaises(RuntimeError):
                ledger.claim("source-sha", "code-sha", "config-sha")


if __name__ == "__main__":
    unittest.main()
