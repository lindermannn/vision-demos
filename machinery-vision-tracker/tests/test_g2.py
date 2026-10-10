from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest

import cv2
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bermguard.charts import (  # noqa: E402
    write_height_curve,
    write_minimum_distance_matrix,
    write_vehicle_dispersion,
)
from bermguard.config import load_configuration  # noqa: E402
from bermguard.contracts import (  # noqa: E402
    BermObservation,
    BoundingBox,
    Detection,
    FrameContext,
    ObservationStatus,
    ProximityLevel,
)
from bermguard.detection import ClassicalVehicleDetector  # noqa: E402
from bermguard.g2_config import G2Settings  # noqa: E402
from bermguard.proximity import ProximityEngine  # noqa: E402
from bermguard.scene import SceneCutDetector  # noqa: E402
from bermguard.segmentation.classical import ClassicalBermSegmenter  # noqa: E402
from bermguard.segmentation.pretrained import profile_from_relative_depth  # noqa: E402
from bermguard.segmentation.result import SegmentationResult  # noqa: E402
from bermguard.segmentation.validation import TemporalBermValidator  # noqa: E402
from bermguard.tracking import CausalTracker  # noqa: E402


def settings() -> G2Settings:
    return G2Settings.from_mapping(load_configuration(None).data)


def context(frame_index: int = 0, scene_id: int = 0) -> FrameContext:
    return FrameContext("test", scene_id, frame_index, frame_index / 10.0, 640, 360)


class G2ConfigurationTests(unittest.TestCase):
    def test_default_g2_configuration_is_valid(self) -> None:
        parsed = settings()
        self.assertEqual(5.0, parsed.pixels_per_meter)
        self.assertLess(parsed.red_below_m, parsed.yellow_at_or_below_m)


class SegmentationTests(unittest.TestCase):
    def test_synthetic_horizontal_berm_produces_height(self) -> None:
        frame = np.full((360, 640, 3), (150, 175, 190), dtype=np.uint8)
        frame[178:230, :] = (45, 90, 135)
        frame[230:, :] = (145, 150, 155)
        result = ClassicalBermSegmenter(settings()).segment(frame, context())
        self.assertEqual("observed", result.observation.status.value)
        self.assertIsNotNone(result.observation.height_px)
        self.assertGreater(float(result.observation.height_px), 30.0)
        self.assertGreater(np.count_nonzero(result.mask), 0)

    def test_relative_depth_profile_is_a_distinct_valid_signal(self) -> None:
        depth = np.full((360, 640), 0.2, dtype=np.float32)
        depth[178:230, :] = 0.8
        depth[230:, :] = 0.35
        result = profile_from_relative_depth(depth, context(), settings())
        self.assertEqual("observed", result.observation.status.value)
        # The G5.1 depth extractor measures the toe as a slope break after
        # smoothing; on this 52 px step it reports about 58 px.  The bias is
        # documented in the benchmark report rather than tuned away here.
        self.assertAlmostEqual(52.0, float(result.observation.height_px), delta=8.0)
        self.assertEqual(
            "depth_relative_slope_break_joint_crest_toe",
            result.observation.diagnostics["profile_source"],
        )

    def test_flat_relative_depth_is_unknown(self) -> None:
        result = profile_from_relative_depth(
            np.ones((360, 640), dtype=np.float32), context(), settings()
        )
        self.assertEqual("unknown", result.observation.status.value)
        self.assertEqual("flat_depth_map", result.observation.diagnostics["reason"])


class SceneAndDetectionTests(unittest.TestCase):
    def test_abrupt_visual_change_starts_new_scene(self) -> None:
        detector = SceneCutDetector(settings())
        self.assertFalse(detector.update(np.zeros((360, 640, 3), dtype=np.uint8)).is_cut)
        self.assertTrue(detector.update(np.full((360, 640, 3), 255, dtype=np.uint8)).is_cut)

    def test_moving_yellow_object_is_a_candidate(self) -> None:
        configured = replace(settings(), detection_warmup_frames=1)
        detector = ClassicalVehicleDetector(configured)
        first = np.full((360, 640, 3), 90, dtype=np.uint8)
        second = first.copy()
        cv2.rectangle(second, (210, 210), (320, 285), (0, 190, 245), -1)
        detector.detect(first, context(0))
        detections = detector.detect(second, context(1))
        self.assertGreaterEqual(len(detections), 1)

    def test_low_ground_fragment_is_not_a_candidate(self) -> None:
        configured = replace(settings(), detection_warmup_frames=1)
        detector = ClassicalVehicleDetector(configured)
        first = np.full((360, 640, 3), 90, dtype=np.uint8)
        second = first.copy()
        cv2.rectangle(second, (210, 310), (340, 355), (0, 190, 245), -1)
        detector.detect(first, context(0))
        detections = detector.detect(second, context(1))
        self.assertEqual((), detections)


class TrackingAndProximityTests(unittest.TestCase):
    def test_tracker_confirms_stable_identity_and_resets_per_scene(self) -> None:
        tracker = CausalTracker(settings())
        first = Detection("candidate", 0.8, BoundingBox(100, 100, 180, 180))
        second = Detection("candidate", 0.8, BoundingBox(105, 102, 185, 182))
        third = Detection("candidate", 0.8, BoundingBox(110, 104, 190, 184))
        fourth = Detection("candidate", 0.8, BoundingBox(115, 106, 195, 186))
        fifth = Detection("candidate", 0.8, BoundingBox(120, 108, 200, 188))
        self.assertEqual((), tracker.update((first,), context(0)))
        self.assertEqual((), tracker.update((second,), context(1)))
        self.assertEqual((), tracker.update((third,), context(2)))
        self.assertEqual((), tracker.update((fourth,), context(3)))
        confirmed = tracker.update((fifth,), context(4))
        self.assertEqual(1, confirmed[0].track_id)
        tracker.reset()
        self.assertEqual((), tracker.update((first,), context(5, scene_id=1)))

    def test_single_track_has_no_proximity_claim(self) -> None:
        engine = ProximityEngine(settings())
        detection = Detection(
            "candidate", 0.8, BoundingBox(100, 100, 150, 160), 1
        )
        samples, colours = engine.update((detection,), context())
        self.assertEqual((), samples)
        self.assertEqual(ProximityLevel.UNKNOWN, colours[1])

    def test_border_truncated_track_is_not_used_for_proximity(self) -> None:
        engine = ProximityEngine(settings())
        detections = (
            Detection("candidate", 0.8, BoundingBox(0, 100, 150, 160), 1),
            Detection("candidate", 0.8, BoundingBox(130, 100, 180, 160), 2),
        )
        for frame_index in range(4):
            samples, colours = engine.update(detections, context(frame_index))
            self.assertEqual((), samples)
            self.assertEqual(ProximityLevel.UNKNOWN, colours[1])
            self.assertEqual(ProximityLevel.UNKNOWN, colours[2])

    def test_red_event_requires_persistence_and_minimum_is_retained(self) -> None:
        engine = ProximityEngine(settings())
        detections = (
            Detection("candidate", 0.8, BoundingBox(100, 100, 150, 160), 1),
            Detection("candidate", 0.8, BoundingBox(130, 100, 180, 160), 2),
        )
        for frame_index in range(2):
            _, colours = engine.update(detections, context(frame_index))
            self.assertEqual(ProximityLevel.UNKNOWN, colours[1])
        self.assertEqual([], engine.event_rows())
        _, colours = engine.update(detections, context(2))
        self.assertEqual(ProximityLevel.RED, colours[1])
        events = engine.event_rows()
        self.assertEqual(1, len(events))
        self.assertEqual(2, events[0]["start_frame"])
        self.assertEqual("red", events[0]["level"])
        self.assertEqual(1, len(engine.minimum_rows()))
        engine.reset_scene_state()
        self.assertEqual(1, len(engine.event_rows()))

    def test_vehicle_reference_scale_adjusts_for_apparent_size(self) -> None:
        configured = replace(
            settings(),
            event_persistence_frames=1,
            proximity_scale_mode="vehicle_reference",
            proximity_reference_vehicle_height_m=5.0,
        )
        engine = ProximityEngine(configured)
        detections = (
            Detection("candidate", 0.8, BoundingBox(100, 100, 150, 150), 1),
            Detection("candidate", 0.8, BoundingBox(200, 100, 250, 150), 2),
        )
        samples, _ = engine.update(detections, context())
        self.assertEqual(1, len(samples))
        self.assertAlmostEqual(10.0, samples[0].distance_m_estimated)


class BermValidationTests(unittest.TestCase):
    @staticmethod
    def result(height_px: float, frame_index: int = 0) -> SegmentationResult:
        frame_context = context(frame_index)
        xs = np.linspace(64.0, 576.0, 100)
        crest = tuple((float(x), 150.0) for x in xs)
        base = tuple((float(x), 150.0 + height_px) for x in xs)
        mask = np.zeros((360, 640), dtype=np.uint8)
        return SegmentationResult(
            BermObservation(
                context=frame_context,
                status=ObservationStatus.VALID,
                crest=crest,
                base=base,
                height_px=height_px,
                height_m_estimated=height_px / 5.0,
            ),
            mask,
            0.8,
        )

    def test_wide_vehicle_occlusion_makes_height_unknown(self) -> None:
        validator = TemporalBermValidator(settings())
        vehicle = Detection(
            "candidate", 0.8, BoundingBox(190, 120, 450, 270), 1
        )
        refined = validator.refine(self.result(50.0), (vehicle,))
        self.assertEqual(ObservationStatus.UNKNOWN, refined.observation.status)
        self.assertIsNone(refined.observation.height_px)

    def test_implausible_temporal_height_jump_is_unknown(self) -> None:
        configured = replace(
            settings(),
            segmentation_max_expected_height_deviation_ratio=1.0,
            segmentation_minimum_valid_frames=1,
        )
        validator = TemporalBermValidator(configured)
        first = validator.refine(self.result(18.0, 0), ())
        second = validator.refine(self.result(30.0, 1), ())
        self.assertEqual(ObservationStatus.VALID, first.observation.status)
        self.assertEqual(ObservationStatus.HISTORICAL_REFERENCE, second.observation.status)
        self.assertEqual(
            "current_geometry_incompatible_with_reference",
            second.observation.diagnostics["reason"],
        )

    def test_implausible_absolute_height_is_unknown(self) -> None:
        # 0.25 of the frame height exceeds the configured berm range
        # [0.018, 0.162] but stays under the 0.30 plausibility cap.
        validator = TemporalBermValidator(settings())
        refined = validator.refine(self.result(90.0), ())
        self.assertEqual(ObservationStatus.UNKNOWN, refined.observation.status)
        self.assertEqual(
            "expected_height_geometry_failed",
            refined.observation.diagnostics["reason"],
        )

    def test_height_requires_consecutive_valid_observations(self) -> None:
        validator = TemporalBermValidator(settings())
        first = validator.refine(self.result(34.0, 0), ())
        second = validator.refine(self.result(34.0, 1), ())
        third = validator.refine(self.result(34.0, 2), ())
        self.assertEqual(ObservationStatus.UNKNOWN, first.observation.status)
        self.assertEqual(ObservationStatus.UNKNOWN, second.observation.status)
        self.assertEqual("reference_bootstrap", first.observation.diagnostics["reason"])
        self.assertEqual(ObservationStatus.VALID, third.observation.status)


class ChartTests(unittest.TestCase):
    def test_all_charts_are_valid_pngs_even_without_pairs(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_height_curve(
                root / "height.png",
                [
                    {"scene_id": 0, "timestamp_s": 0.0, "status": "observed", "height_m_estimated": 8.0},
                    {"scene_id": 0, "timestamp_s": 1.0, "status": "observed", "height_m_estimated": 9.0},
                ],
            )
            write_vehicle_dispersion(
                root / "vehicles.png",
                [{"track_id": 1, "x_px": 100.0, "y_px": 200.0}],
                640,
                360,
            )
            write_minimum_distance_matrix(root / "matrix.png", [])
            for name in ("height.png", "vehicles.png", "matrix.png"):
                image = cv2.imread(str(root / name))
                self.assertIsNotNone(image)
                self.assertGreater((root / name).stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
