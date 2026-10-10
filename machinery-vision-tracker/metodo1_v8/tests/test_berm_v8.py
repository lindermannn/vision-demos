"""V8 tests: the anchor still never moves, and the berm size is now measured."""

import sys
from pathlib import Path

# Package layout: codigo/src holds bermguard, codigo/metodo1_v8 holds V8.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest

import numpy as np

import anchored_search
import berm_metrics
import normativa_ds132 as normativa
from berm_metrics import GroundScaleModel, profile_statistics
from fixed_camera_v8 import ANCHOR_TOLERANCE_RATIO, MeasuringFixedCameraValidator, render_experiment
from bermguard.config import load_configuration
from bermguard.g2_config import G2Settings
from bermguard.contracts import BermObservation, BoundingBox, Detection, FrameContext
from bermguard.contracts import ObservationStatus as S
from bermguard.segmentation.result import SegmentationResult


class _Box:
    """Minimal stand-in for a detector proposal."""

    def __init__(self, y_bottom: float, height: float, confidence: float = 0.9) -> None:
        self.bbox = BoundingBox(100.0, y_bottom - height, 100.0 + height, y_bottom)
        self.confidence = confidence
        self.track_id = None


def _pinhole_samples(rows, camera_height_m=10.0, vehicle_height_m=5.0, horizon=100.0):
    """Apparent height of a fixed-size object standing on a flat grade."""

    slope = vehicle_height_m / camera_height_m
    return [_Box(row, slope * (row - horizon)) for row in rows]


class GroundScaleTests(unittest.TestCase):
    def _model(self, rows=np.linspace(200, 600, 24)):
        model = GroundScaleModel(5.0)
        model.observe(_pinhole_samples(rows))
        return model, np.asarray(rows), np.array([0.5 * (row - 100.0) for row in rows])

    def test_regression_recovers_a_known_scale_and_horizon(self):
        model, rows, heights = self._model()
        estimate = model._regression(400.0, rows, heights, 720)
        self.assertIsNotNone(estimate)
        # h(400) = 0.5 * (400 - 100) = 150 px for a 5 m vehicle -> 30 px/m.
        self.assertAlmostEqual(30.0, estimate.pixels_per_meter, delta=0.6)
        self.assertAlmostEqual(100.0, estimate.horizon_row, delta=8.0)
        self.assertLess(estimate.low, estimate.pixels_per_meter)
        self.assertGreater(estimate.high, estimate.pixels_per_meter)

    def test_scale_is_available_where_no_vehicle_stands(self):
        """The case the earlier rule could not serve at all."""

        model, rows, heights = self._model()
        self.assertIsNone(model._nearest_row_fallback(700.0, rows, heights, 720))
        estimate = model.estimate(700.0, 720)
        self.assertIsNotNone(estimate)
        self.assertEqual("ground_plane_row_regression", estimate.basis)
        self.assertAlmostEqual(60.0, estimate.pixels_per_meter, delta=1.2)

    def test_direct_rule_wins_where_machinery_actually_stands(self):
        model, _, _ = self._model()
        estimate = model.estimate(400.0, 720)
        self.assertEqual("nearest_contact_rows", estimate.basis)
        self.assertAlmostEqual(1.0, estimate.disagreement_ratio, delta=0.25)

    def test_regression_refused_close_to_its_own_horizon(self):
        """Near the horizon the intercept error dominates and metres diverge."""

        model, rows, heights = self._model()
        self.assertIsNone(model._regression(130.0, rows, heights, 720))

    def test_too_few_samples_yield_no_scale(self):
        model = GroundScaleModel(5.0)
        model.observe(_pinhole_samples(np.linspace(300, 320, 4)))
        self.assertIsNone(model.estimate(400.0, 720))

    def test_metres_interval_brackets_the_estimate(self):
        model, _, _ = self._model()
        metres, low, high = model.estimate(400.0, 720).to_meters(60.0)
        self.assertLess(low, metres)
        self.assertLess(metres, high)


class ProfileStatisticsTests(unittest.TestCase):
    def test_percentiles_and_coverage(self):
        stats = profile_statistics([10, 12, 14, 16, 18], 10)
        self.assertEqual(5, stats.columns)
        self.assertAlmostEqual(0.5, stats.coverage)
        self.assertLessEqual(stats.p10_px, stats.median_px)
        self.assertLessEqual(stats.median_px, stats.p90_px)

    def test_empty_profile_has_no_statistics(self):
        self.assertIsNone(profile_statistics([], 10))


class MeasuringValidatorTests(unittest.TestCase):
    def setUp(self):
        berm_metrics.RECORDS.clear()
        self.validator = MeasuringFixedCameraValidator(
            G2Settings.from_mapping(load_configuration(None).data)
        )
        self.day = np.full((240, 480, 3), 160, np.uint8)
        self.night = np.zeros_like(self.day)

    def candidate(self, index, scene=0, crest=110, video="synthetic"):
        xs = np.linspace(20, 460, 80)
        observation = BermObservation(
            FrameContext(video, scene, index, index / 30, 480, 240), S.OBSERVED,
            tuple((x, crest) for x in xs), tuple((x, crest + 20) for x in xs),
            diagnostics={"spatially_valid": True},
        )
        mask = np.zeros((240, 480), np.uint8)
        mask[crest:crest + 21, 20:461] = 255
        return SegmentationResult(observation, mask, 0.9)

    def seed(self):
        for index in range(5):
            output = self.validator.refine(self.candidate(index), (), self.day)
        self.assertIsNotNone(self.validator.anchor)
        return output

    def test_night_still_cannot_seed_an_anchor(self):
        for index in range(10):
            output = self.validator.refine(self.candidate(index), (), self.night)
            self.assertEqual(S.UNKNOWN, output.observation.status)
        self.assertIsNone(self.validator.anchor)

    def test_anchor_geometry_never_moves(self):
        before = self.seed()
        self.validator.reset()
        after = self.validator.refine(
            self.candidate(100, scene=1, crest=140), (), self.night
        )
        self.assertEqual(before.observation.crest, after.observation.crest)
        self.assertEqual(before.observation.base, after.observation.base)
        self.assertFalse(after.mask.any())

    def test_night_publishes_the_anchored_size_instead_of_nothing(self):
        """V7 left the height empty here; the curve has to stay continuous."""

        self.seed()
        after = self.validator.refine(self.candidate(100), (), self.night)
        self.assertEqual(S.HISTORICAL_REFERENCE, after.observation.status)
        self.assertAlmostEqual(20.0, after.observation.height_px, places=6)
        diagnostics = after.observation.diagnostics
        self.assertEqual(
            "anchored_daylight_geometry_carried", diagnostics["measurement_basis"]
        )
        self.assertFalse(diagnostics["current_anchor_support"])
        self.assertEqual(
            100 - self.validator.anchor_context.frame_index,
            diagnostics["profile_age_frames"],
        )

    def test_matching_daylight_evidence_is_a_current_measurement(self):
        self.seed()
        output = self.validator.refine(self.candidate(6), (), self.day)
        self.assertEqual(S.TEMPORAL_ESTIMATE, output.observation.status)
        self.assertAlmostEqual(20.0, output.observation.height_px, places=6)
        diagnostics = output.observation.diagnostics
        self.assertTrue(diagnostics["current_anchor_support"])
        self.assertEqual(
            "current_supported_columns_fixed_overlay", diagnostics["measurement_basis"]
        )
        self.assertIn("height_profile_p10_px", diagnostics)

    def test_occluded_evidence_falls_back_to_the_anchored_size(self):
        self.seed()
        output = self.validator.refine(
            self.candidate(6), (), self.day, np.full((240, 480), 255, np.uint8)
        )
        self.assertFalse(output.observation.diagnostics["current_anchor_support"])
        self.assertAlmostEqual(20.0, output.observation.height_px, places=6)

    def test_incompatible_candidate_neither_moves_nor_measures(self):
        before = self.seed()
        for index in range(5, 40):
            output = self.validator.refine(
                self.candidate(index, crest=140 + index % 5), (), self.day
            )
            self.assertEqual(before.observation.crest, output.observation.crest)
            self.assertFalse(output.observation.diagnostics["current_anchor_support"])

    def test_metric_size_appears_once_machinery_gives_a_scale(self):
        self.seed()
        confirmed = tuple(
            Detection(label="maquinaria_pesada", confidence=0.9,
                      bbox=box.bbox, track_id=index + 1)
            for index, box in enumerate(
                _pinhole_samples(np.linspace(120, 220, 24),
                                 camera_height_m=10.0, horizon=60.0)
            )
        )
        # The tracker publishes its confirmed population one step earlier.
        berm_metrics.publish_confirmed(confirmed)
        output = self.validator.refine(self.candidate(6), (), self.day)
        diagnostics = output.observation.diagnostics
        self.assertIn("berm_height_m_ground_plane", diagnostics)
        low, high = diagnostics["berm_height_m_interval"]
        self.assertLess(low, diagnostics["berm_height_m_ground_plane"])
        self.assertLess(diagnostics["berm_height_m_ground_plane"], high)

    def test_sidecar_records_every_frame_after_the_anchor(self):
        self.seed()
        for index in range(5, 15):
            self.validator.refine(self.candidate(index), (), self.day)
        rows = [
            row for row in berm_metrics.RECORDS
            if row["height_px"] is not None and row["frame_index"] >= 5
        ]
        self.assertEqual(10, len(rows))
        self.assertTrue(all(row["crest_y_median"] == 110.0 for row in rows))

    def test_overlay_reports_the_size_without_mutating_the_frame(self):
        output = self.seed()
        original = self.day.copy()
        rendered = render_experiment(self.day, output, (), {}, 0, "metodo 1")
        self.assertTrue(np.array_equal(original, self.day))
        self.assertFalse(np.array_equal(original, rendered))


if __name__ == "__main__":
    unittest.main(verbosity=2)


class MeasuredOverlayTests(unittest.TestCase):
    """The reported number must correspond to something drawn on the frame."""

    def setUp(self):
        berm_metrics.RECORDS.clear()
        berm_metrics.clear_measured()
        self.validator = MeasuringFixedCameraValidator(
            G2Settings.from_mapping(load_configuration(None).data)
        )
        self.day = np.full((240, 480, 3), 160, np.uint8)

    def candidate(self, index, crest=110):
        xs = np.linspace(20, 460, 80)
        observation = BermObservation(
            FrameContext("synthetic", 0, index, index / 30, 480, 240), S.OBSERVED,
            tuple((x, crest) for x in xs), tuple((x, crest + 20) for x in xs),
            diagnostics={"spatially_valid": True},
        )
        mask = np.zeros((240, 480), np.uint8)
        mask[crest:crest + 21, 20:461] = 255
        return SegmentationResult(observation, mask, 0.9)

    def test_measured_columns_are_published_only_with_current_support(self):
        for index in range(5):
            self.validator.refine(self.candidate(index), (), self.day)
        self.validator.refine(self.candidate(6), (), self.day)
        crest, toe = berm_metrics.peek_measured()
        self.assertTrue(crest and toe)
        self.assertEqual(len(crest), len(toe))

        # An occluded frame has no current support and must publish nothing.
        self.validator.refine(
            self.candidate(7), (), self.day, np.full((240, 480), 255, np.uint8)
        )
        self.assertEqual((None, None), berm_metrics.peek_measured())


class NormativaTests(unittest.TestCase):
    """DS 132 sizes the berm against the wheel of the largest truck."""

    def test_thresholds_resolve_to_metres(self):
        self.assertAlmostEqual(1.798, normativa.DUMP_EDGE_MINIMUM_M, places=3)
        self.assertAlmostEqual(2.397, normativa.STEEP_ROAD_MINIMUM_M, places=3)

    def test_clear_pass_and_clear_fail(self):
        passing = normativa.evaluate(2.60, 2.20, 3.10)
        self.assertEqual("cumple", passing.verdict)
        self.assertGreater(passing.margin_m, 0.0)
        failing = normativa.evaluate(1.20, 1.00, 1.45)
        self.assertEqual("por debajo", failing.verdict)

    def test_interval_straddling_the_threshold_is_not_a_verdict(self):
        straddling = normativa.evaluate(1.85, 1.50, 2.20)
        self.assertEqual("no concluyente", straddling.verdict)

    def test_wheel_ratio_is_secondary_and_consistent(self):
        compliance = normativa.evaluate(1.798, 1.60, 2.00)
        self.assertAlmostEqual(0.50, compliance.wheel_ratio, places=2)

    def test_panel_line_leads_with_metres(self):
        line = normativa.summary_line(normativa.evaluate(2.26, 1.85, 2.76))
        self.assertTrue(line.startswith("h=2.26 m"))
        self.assertIn("min 1.80 m", line)
        self.assertIn("rueda", line)


class ScaleCalibrationTests(unittest.TestCase):
    """A mixed fleet must not drag the scale down to its smallest machines."""

    def test_largest_class_sets_the_scale(self):
        model = GroundScaleModel(normativa.TRUCK_HEIGHT_M)
        rows = np.linspace(200, 600, 24)
        trucks = _pinhole_samples(rows)
        # Dozers at the same rows, roughly half the apparent height.
        dozers = [_Box(box.bbox.y2, (box.bbox.y2 - box.bbox.y1) * 0.5) for box in trucks]
        model.observe(trucks + dozers)
        estimate = model.estimate(400.0, 720)
        # h(400) = 150 px for the truck class, declared 6.6 m tall.
        self.assertAlmostEqual(150.0 / normativa.TRUCK_HEIGHT_M,
                               estimate.pixels_per_meter, delta=2.0)


class DaylightGatedSearchTests(unittest.TestCase):
    """The sweep is skipped only where its answer would be discarded."""

    def setUp(self):
        anchored_search.reset_counts()
        self.settings = G2Settings.from_mapping(load_configuration(None).data)
        self.segmenter = anchored_search.DaylightOnlyClassicalSegmenter(self.settings)
        self.day = np.full((720, 1280, 3), 170, np.uint8)
        self.day[380:415, :, :] = 60
        self.night = np.full((720, 1280, 3), 20, np.uint8)

    def context(self, index):
        return FrameContext("synthetic", 0, index, index / 24.0, 1280, 720)

    def test_daylight_runs_the_full_unconstrained_grid(self):
        result = self.segmenter.segment(self.day, self.context(7))
        self.assertEqual("full_grid", result.observation.diagnostics["search_mode"])
        self.assertEqual(1, anchored_search.COUNTS["searched"])
        self.assertEqual(0, anchored_search.COUNTS["skipped"])

    def test_low_light_is_skipped_and_says_so(self):
        result = self.segmenter.segment(self.night, self.context(7))
        self.assertEqual(S.UNKNOWN, result.observation.status)
        self.assertEqual(anchored_search.SKIP_REASON,
                         result.observation.diagnostics["reason"])
        self.assertEqual(1, anchored_search.COUNTS["skipped"])
        self.assertFalse(result.mask.any())

    def test_a_skipped_frame_never_claims_geometry(self):
        result = self.segmenter.segment(self.night, self.context(7))
        self.assertEqual((), result.observation.crest)
        self.assertEqual((), result.observation.base)
        self.assertIsNone(result.observation.height_px)

    def test_the_threshold_is_the_one_the_validator_uses(self):
        threshold = self.settings.scene_day_above_mean
        self.assertTrue(anchored_search.frame_is_daylight(self.day, threshold))
        self.assertFalse(anchored_search.frame_is_daylight(self.night, threshold))
