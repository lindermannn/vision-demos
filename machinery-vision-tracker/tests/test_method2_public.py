"""Small deterministic checks for the published Method 2 geometry contracts."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "metodo2"))
from method2_berm_height.height import height_metres, verdict


class Method2PublicTests(unittest.TestCase):
    def test_height_conversion(self):
        value, reason = height_metres(40, 500, 300, 10)
        self.assertEqual("ok", reason)
        self.assertEqual(2.0, value)

    def test_near_horizon_is_unknown(self):
        value, reason = height_metres(40, 310, 300, 10)
        self.assertIsNone(value)
        self.assertEqual("pretil_cerca_del_horizonte", reason)

    def test_interval_crossing_threshold_is_inconclusive(self):
        self.assertEqual("no_concluyente", verdict(1.5, 2.0, 1.8))


if __name__ == "__main__":
    unittest.main()
