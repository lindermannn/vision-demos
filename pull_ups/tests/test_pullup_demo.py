import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from main import clears_bar
from pullup_demo import render
from pullup_demo.extract import write_track
from pullup_demo.pose_contract import LANDMARK_COUNT, load_pose_track
from pullup_demo.pullup_calibration import apply_calibration, build_calibration, grade, parse_reps


def events_from(heights):
    return [{"height_above_bar": h} for h in heights]


class CalibrationTests(unittest.TestCase):
    def test_parse_reps(self):
        self.assertEqual(parse_reps("1-3, 5", 6), [1, 2, 3, 5])
        for bad in ("0-2", "4-2", "7", "a"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_reps(bad, 6)

    def test_thresholds_follow_the_athlete_rules(self):
        heights = [.5, .5, .48, .3, .2, .12, .13, .12]
        calibration = build_calibration([{"events": events_from(heights), "alto": [1, 2, 3],
                                          "minimo": [6, 7, 8], "validas": range(1, 9)}],
                                        tolerance=.02, semi_fraction=.2)
        thresholds = calibration["thresholds"]
        self.assertAlmostEqual(thresholds["rom_full_alto"], .46)
        self.assertAlmostEqual(thresholds["rom_full"], .10)
        self.assertEqual(grade(.47, calibration), "rom_full_alto")
        self.assertEqual(grade(.3, calibration), "rom_semi_alto")
        self.assertEqual(grade(.11, calibration), "rom_full")
        self.assertEqual(grade(.05, calibration), "bajo_minimo")
        self.assertEqual(grade(None, calibration), "sin_medida")
        events = events_from([.5, None])
        apply_calibration(events, calibration)
        self.assertEqual([e["grade_label"] for e in events], ["ROM full alto", "Sin medida"])

    def test_other_verified_reps_lower_the_minimum(self):
        refs = [{"events": events_from([.5, .5, .5, .12, .12, .12]), "alto": [1, 2, 3], "minimo": [4, 5, 6]},
                {"events": events_from([.06]), "validas": [1]}]
        self.assertAlmostEqual(build_calibration(refs, tolerance=0)["thresholds"]["rom_full"], .06)

    def test_too_few_or_inverted_references_rejected(self):
        with self.assertRaises(ValueError):
            build_calibration([{"events": events_from([.5, .1]), "alto": [1], "minimo": [2]}])
        with self.assertRaises(ValueError):
            build_calibration([{"events": events_from([.1, .1, .1, .5, .5, .5]),
                                "alto": [1, 2, 3], "minimo": [4, 5, 6]}])


class LayoutTests(unittest.TestCase):
    def series(self, calibrated=True):
        events = [{"start": i * 2.0, "peak": i * 2.0 + .4, "end": i * 2.0 + 1.0,
                   "height_above_bar": .1 + .02 * i, "grade": "rom_full"} for i in range(20)]
        times = np.arange(0, 42, .05)
        curve = {"times": times, "values": np.sin(times * np.pi) * .5, "bar_y": .35}
        calibration = {"thresholds": {"rom_full_alto": .45, "rom_semi_alto": .2, "rom_full": .08}}
        return render.build_series(events, curve, calibration if calibrated else None)

    def test_reel_keeps_side_column_free(self):
        canvas = Image.new("RGB", render.LAYOUTS["reel"], (255, 255, 255))
        render._reel_frame(canvas, self.series(), 12.5)
        middle = np.asarray(canvas)[render.REEL_SAFE_TOP:render.REEL_SAFE_BOTTOM]
        self.assertTrue((middle[:, render.REEL_SAFE_RIGHT:] == 255).all())

    def test_summary_stays_in_reel_safe_zone(self):
        canvas = Image.new("RGB", render.LAYOUTS["reel"], (255, 255, 255))
        render._summary(canvas, self.series(), "reel")
        middle = np.asarray(canvas)[render.REEL_SAFE_TOP:render.REEL_SAFE_BOTTOM]
        self.assertTrue((middle[:, render.REEL_SAFE_RIGHT:] == 255).all())

    def test_timer_pauses_and_stats(self):
        series = self.series()
        self.assertIsNone(series.reps[0].pause)
        self.assertAlmostEqual(series.reps[1].pause, 1.0)
        self.assertEqual(render._clock(series, -3), "00:00")
        self.assertEqual(render._clock(series, 12.5), "00:12")
        stats = render._stats(series)
        self.assertEqual(stats["best"].number, 20)
        self.assertEqual(len(render._fatigue_lines(stats)), 2)

    def test_uncalibrated_series_has_no_grades(self):
        series = self.series(calibrated=False)
        self.assertTrue(all(rep.grade is None for rep in series.reps))


class TrackFormatTests(unittest.TestCase):
    def test_written_track_loads_with_the_detector_reader(self):
        frames = 40
        image = np.random.default_rng(0).random((frames, LANDMARK_COUNT, 3))
        visibility = np.full((frames, LANDMARK_COUNT), .9)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "track.npz"
            write_track(path, np.arange(frames) / 20, image, visibility, image,
                        np.ones(frames, dtype=bool), {"fps": 20.0})
            track = load_pose_track(path)
        self.assertEqual(track.image_landmarks.shape, (frames, LANDMARK_COUNT, 3))
        self.assertTrue(np.allclose(track.timestamps, np.arange(frames) / 20))
        self.assertEqual(track.metadata["fps"], 20.0)

    def test_head_at_or_above_hands_is_flagged(self):
        self.assertTrue(clears_bar(.1))
        self.assertTrue(clears_bar(0.0))
        self.assertFalse(clears_bar(-.2))
        self.assertIsNone(clears_bar(None))


if __name__ == "__main__":
    unittest.main()
