from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bermguard.contracts import BoundingBox, Detection, FrameContext  # noqa: E402


class ContractTests(unittest.TestCase):
    def test_frame_context_rejects_invalid_dimensions(self) -> None:
        with self.assertRaises(ValueError):
            FrameContext("video", 0, 0, 0.0, 0, 720)

    def test_detection_rejects_confidence_over_one(self) -> None:
        with self.assertRaises(ValueError):
            Detection("truck", 1.1, BoundingBox(0, 0, 10, 10))

    def test_bounding_box_rejects_reversed_coordinates(self) -> None:
        with self.assertRaises(ValueError):
            BoundingBox(10, 0, 5, 10)


if __name__ == "__main__":
    unittest.main()
