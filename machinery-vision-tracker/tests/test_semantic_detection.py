from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bermguard.contracts import BoundingBox, Detection  # noqa: E402
from bermguard.semantic_detection import suppress_semantic_duplicates  # noqa: E402


class SemanticDuplicateSuppressionTests(unittest.TestCase):
    def test_overlapping_prompt_duplicates_collapse_to_best_box(self) -> None:
        detections = (
            Detection("maquinaria_pesada", 0.91, BoundingBox(10, 10, 110, 90)),
            Detection("maquinaria_pesada", 0.78, BoundingBox(14, 12, 108, 88)),
            Detection("maquinaria_pesada", 0.83, BoundingBox(220, 30, 310, 100)),
        )
        retained = suppress_semantic_duplicates(detections, 0.45)
        self.assertEqual(2, len(retained))
        self.assertEqual(0.91, retained[0].confidence)
        self.assertEqual(0.83, retained[1].confidence)

    def test_contained_box_is_suppressed_even_when_iou_is_small(self) -> None:
        detections = (
            Detection("maquinaria_pesada", 0.90, BoundingBox(10, 10, 210, 210)),
            Detection("maquinaria_pesada", 0.70, BoundingBox(70, 70, 120, 120)),
        )
        retained = suppress_semantic_duplicates(detections, 0.45)
        self.assertEqual(1, len(retained))


if __name__ == "__main__":
    unittest.main()
