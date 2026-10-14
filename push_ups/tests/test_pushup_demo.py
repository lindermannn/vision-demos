from __future__ import annotations

import unittest
import numpy as np

from pushup_demo.pose_contract import PoseTrack, PoseFrame, PoseObservation, FrameTransform, PoseProvenance, LANDMARK_COUNT


class TestPushupDemo(unittest.TestCase):
    def test_counter_advances_only_after_return_to_plank(self):
        from pushup_demo.render import _state, build_series

        events = [
            {"start": 0.5, "peak": 1.0, "end": 1.5},
            {"start": 2.0, "peak": 2.5, "end": 3.0},
        ]
        curve = {"times": [0.0, 1.0, 2.0, 3.0], "values": [0.0] * 4,
                 "shoulder_y": None}
        series = build_series(events, curve, None)

        for seconds, expected in ((0.5, 0), (1.0, 0), (1.499, 0),
                                  (1.5, 1), (2.5, 1), (2.999, 1), (3.0, 2)):
            with self.subTest(seconds=seconds):
                count, last = _state(series, seconds)
                self.assertEqual(count, expected)
                self.assertEqual(last.number if last else None,
                                 expected if expected else None)

        self.assertAlmostEqual(series.reps[0].down, 0.5)
        self.assertAlmostEqual(series.reps[0].up, 0.5)

    def test_track_roundtrip(self):
        """A minimal track can be created and loaded."""
        timestamps = np.arange(0.0, 1.0, 0.05, dtype=np.float32)
        count = len(timestamps)
        image = np.full((count, LANDMARK_COUNT, 3), np.nan, dtype=np.float32)
        visibility = np.full((count, LANDMARK_COUNT), np.nan, dtype=np.float32)
        presence = np.full((count, LANDMARK_COUNT), np.nan, dtype=np.float32)
        world = np.full((count, LANDMARK_COUNT, 3), np.nan, dtype=np.float32)
        detected = np.zeros(count, dtype=bool)
        track = PoseTrack(timestamps=timestamps, image_landmarks=image,
                          visibility=visibility, presence=presence,
                          world_landmarks=world, detected=detected,
                          metadata={"schema_version": "pose-track/1.0", "topology": "blazepose-33"})
        payload = track.npz_payload()
        assert "timestamps" in payload
        assert "landmarks" in payload
        assert "image_landmarks" in payload
        assert "pose_metadata_json" in payload
        assert payload["timestamps"].shape == (count,)
        assert payload["image_landmarks"].shape == (count, LANDMARK_COUNT, 3)

    def test_empty_observation(self):
        obs = PoseObservation.empty()
        assert obs.detected is False
        assert np.isnan(obs.image_landmarks).all()

    def test_cycle_state_machine_basic(self):
        from pushup_demo.calisthenics import CycleStateMachine, POLICIES, SignalFrame
        policy = POLICIES["push_up"]
        machine = CycleStateMachine(policy)
        # Create a simple rest-effort-peak-return cycle
        signals = []
        for i, t in enumerate(np.linspace(0, 2, 40)):
            angle = 160 if i < 10 else (100 if i < 20 else (160 if i < 30 else 160))
            effort = 0.1 if angle > 140 else 0.8
            signals.append(SignalFrame(
                source_index=i, timestamp_s=t, elbow_angle=angle,
                effort_score=effort, landmark_score=0.9,
                shoulder_y=0.4, elbow_y=0.5, hip_y=0.6, head_y=0.3, wrist_y=0.5
            ))
        candidates = machine.detect(signals)
        # Should detect at least one cycle
        self.assertGreater(len(candidates), 0)
        complete = [c for c in candidates if c.cycle_status == "complete"]
        self.assertGreater(len(complete), 0)


if __name__ == "__main__":
    unittest.main()
