from dataclasses import replace
from pathlib import Path
import sys
import unittest

# Package layout: codigo/src holds bermguard, codigo/metodo1_v8 holds V8.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bermguard.config import load_configuration
from bermguard.contracts import BoundingBox, Detection, FrameContext
from bermguard.g2_config import G2Settings
from fusion_tracking import FusionResistantTracker


def det(x1, y1, x2, y2, confidence=0.9):
    return Detection("maquinaria_pesada", confidence, BoundingBox(x1, y1, x2, y2))


class FusionTrackingTests(unittest.TestCase):
    def setUp(self):
        settings = G2Settings.from_mapping(load_configuration(None).data)
        self.tracker = FusionResistantTracker(settings)
        self.context = FrameContext("test", 0, 0, 0.0, 1920, 1080)

    def update(self, detections, frame):
        return self.tracker.update(
            detections, replace(self.context, frame_index=frame, timestamp_s=frame / 30)
        )

    def test_union_box_does_not_replace_two_confirmed_tracks(self):
        left = det(600, 500, 820, 800)
        right = det(880, 510, 1080, 790)
        self.update((left, right), 0)
        confirmed = self.update((left, right), 1)
        self.assertEqual(len(confirmed), 2)

        union = det(580, 480, 1110, 820)
        during_fusion = self.update((union,), 2)
        self.assertEqual(len(during_fusion), 2)
        self.assertEqual(
            {item.track_id for item in during_fusion},
            {item.track_id for item in confirmed},
        )
        self.assertEqual(self.tracker.diagnostics()["fusion_rejections"], 1)

    def test_short_missing_detection_remains_visible_then_expires(self):
        box = det(700, 500, 900, 800)
        self.update((box,), 0)
        self.update((box,), 1)
        for frame in range(2, 6):
            self.assertEqual(len(self.update((), frame)), 1)
        self.assertEqual(len(self.update((), 6)), 0)

    def test_normal_detection_is_not_split_or_duplicated(self):
        box = det(700, 500, 900, 800)
        self.update((box,), 0)
        result = self.update((box,), 1)
        self.assertEqual(len(result), 1)

    def test_nested_duplicate_track_keeps_tighter_box(self):
        boxes = (
            det(700, 500, 900, 800),
            det(680, 480, 930, 830),
        )
        result = FusionResistantTracker._deduplicate_visible(
            tuple(item for item in boxes)
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].bbox, boxes[0].bbox)

    def test_nested_duplicate_detections_do_not_create_two_tracks(self):
        outer = det(680, 480, 940, 840)
        inner = det(700, 500, 900, 800)
        self.update((outer, inner), 0)
        result = self.update((outer, inner), 1)
        self.assertEqual(len(result), 1)

    def test_adjacent_vehicles_remain_two_tracks(self):
        left = det(500, 520, 760, 820)
        right = det(770, 525, 1030, 825)
        self.update((left, right), 0)
        result = self.update((left, right), 1)
        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
