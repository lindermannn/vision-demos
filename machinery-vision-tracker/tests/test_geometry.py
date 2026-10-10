from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bermguard.contracts import BoundingBox, ProximityLevel  # noqa: E402
from bermguard.geometry import PixelScale, bbox_bottom_center, classify_proximity  # noqa: E402


class GeometryTests(unittest.TestCase):
    def test_five_pixels_equal_one_estimated_meter(self) -> None:
        self.assertEqual(PixelScale(5.0).to_estimated_meters(25.0), 5.0)

    def test_scale_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            PixelScale(0.0)

    def test_proximity_boundaries_are_yellow(self) -> None:
        self.assertEqual(classify_proximity(10.0), ProximityLevel.YELLOW)
        self.assertEqual(classify_proximity(20.0), ProximityLevel.YELLOW)

    def test_proximity_red_green_and_unknown(self) -> None:
        self.assertEqual(classify_proximity(9.99), ProximityLevel.RED)
        self.assertEqual(classify_proximity(20.01), ProximityLevel.GREEN)
        self.assertEqual(classify_proximity(None), ProximityLevel.UNKNOWN)

    def test_proximity_accepts_configurable_thresholds(self) -> None:
        self.assertEqual(
            classify_proximity(11.0, red_below_m=12.0, yellow_at_or_below_m=24.0),
            ProximityLevel.RED,
        )

    def test_bbox_bottom_center(self) -> None:
        self.assertEqual(bbox_bottom_center(BoundingBox(2, 4, 10, 20)), (6.0, 20))


if __name__ == "__main__":
    unittest.main()
