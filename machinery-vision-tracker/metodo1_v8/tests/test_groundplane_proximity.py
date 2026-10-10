from dataclasses import replace
from pathlib import Path
import sys
import unittest

# Package layout: codigo/src holds bermguard, codigo/metodo1_v8 holds V8.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bermguard.contracts import BoundingBox, Detection, FrameContext, ProximityLevel
from bermguard.g2_config import G2Settings
from bermguard.config import load_configuration
from groundplane_proximity import GroundPlaneProximityEngine


def detection(track_id, center_x, height=200.0):
    return Detection("haul truck", 0.9, BoundingBox(center_x-100, 500-height, center_x+100, 500), track_id)


class GroundPlaneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = G2Settings.from_mapping(load_configuration(None).data)
        cls.context = FrameContext("test", 0, 0, 0.0, 1920, 1080)

    def level_for_gap(self, gap):
        engine = GroundPlaneProximityEngine(self.settings)
        samples, _ = engine.update((detection(1, 600), detection(2, 600+gap)), self.context)
        return samples[0].level

    def test_literal_threshold_bands(self):
        self.assertIs(self.level_for_gap(300), ProximityLevel.RED)
        self.assertIs(self.level_for_gap(500), ProximityLevel.YELLOW)
        self.assertIs(self.level_for_gap(900), ProximityLevel.GREEN)

    def test_safety_envelope_moves_near_boundary_without_changing_thresholds(self):
        # 420 px = 10.5 m centre distance; 1.5 m combined envelope => 9 m.
        self.assertIs(self.level_for_gap(420), ProximityLevel.RED)

    def test_depth_difference_is_not_false_red(self):
        engine = GroundPlaneProximityEngine(self.settings)
        samples, _ = engine.update((detection(1, 900, 100), detection(2, 900, 200)), self.context)
        self.assertIs(samples[0].level, ProximityLevel.GREEN)

    def test_unpaired_confirmed_detection_is_green(self):
        engine = GroundPlaneProximityEngine(self.settings)
        _, levels = engine.update((detection(1, 900),), self.context)
        self.assertIs(levels[1], ProximityLevel.GREEN)


if __name__ == "__main__":
    unittest.main()
