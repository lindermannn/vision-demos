import json
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bermguard.artifacts import reserve_run_files  # noqa: E402
from bermguard.contracts import MethodSelection  # noqa: E402
from bermguard.exceptions import ConfigurationError, OutputExistsError  # noqa: E402
from bermguard.pipeline import PipelineRequest, _benchmark_summary, run_pipeline  # noqa: E402
from test_video import HAS_CV, create_synthetic_video  # noqa: E402


@unittest.skipUnless(HAS_CV, "OpenCV/NumPy not installed")
class PipelineTests(unittest.TestCase):
    def test_benchmark_distinguishes_weighted_and_per_video_valid_rates(self) -> None:
        summary = _benchmark_summary(
            [
                {
                    "method": "2", "processed_frames": 100, "height_valid_frames": 50,
                    "height_valid_rate": 0.5, "wall_s": 10.0, "segmentation_s": 5.0,
                    "height_m_estimated_median": 1.0, "video_id": "long",
                },
                {
                    "method": "2", "processed_frames": 10, "height_valid_frames": 10,
                    "height_valid_rate": 1.0, "wall_s": 1.0, "segmentation_s": 0.5,
                    "height_m_estimated_median": 2.0, "video_id": "short",
                },
            ]
        )["methods"]["2"]
        self.assertAlmostEqual(60 / 110, summary["aggregate_height_valid_rate"])
        self.assertEqual(0.75, summary["mean_height_valid_rate"])

    def test_method_one_completes_without_learned_dependencies(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            input_root = root / "entrada con espacio"
            output_root = root / "salida"
            input_root.mkdir()
            # A mounted host directory may already contain files; that must
            # never block a run (official command reuses one output root).
            output_root.mkdir()
            (output_root / ".gitkeep").write_text("", encoding="utf-8")
            create_synthetic_video(input_root / "Máquina.mp4", frames=6)

            request = PipelineRequest(
                input_directory=input_root,
                output_directory=output_root,
                method=MethodSelection.METHOD_1,
            )
            summary = run_pipeline(request)
            self.assertEqual("complete", summary.status)
            self.assertEqual(1, summary.completed_outputs)

            manifest = json.loads(
                (output_root / "run_manifest_method_1.json").read_text(encoding="utf-8")
            )
            self.assertEqual(["1"], manifest["methods"])
            video_id = manifest["inputs"][0]["video_id"]
            method_one = output_root / video_id / "method_1"
            self.assertTrue((method_one / "processed.mp4").is_file())
            metadata = json.loads(
                (method_one / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual("G2_classical_analytics", metadata["stage"])
            self.assertTrue(metadata["method"]["cv_inference"])
            self.assertFalse(metadata["proximity_alert"])
            self.assertEqual(0, metadata["proximity_alert_count"])
            evaluation = metadata["evaluation_summary"]
            self.assertGreater(evaluation["average_processing_fps"], 0.0)
            self.assertGreater(evaluation["processing_ms_per_frame"], 0.0)
            self.assertIn("width", evaluation["analyzed_resolution"])
            self.assertEqual("1", manifest["method_selection"])
            self.assertEqual(
                "run_manifest_method_1.json", manifest["run_files"]["run_manifest"]
            )
            self.assertTrue((method_one / "height_curve.png").is_file())
            self.assertTrue((method_one / "vehicle_dispersion.png").is_file())
            self.assertTrue((method_one / "minimum_distance_matrix.png").is_file())
            self.assertTrue((method_one / "proximity_events.csv").is_file())
            self.assertTrue((output_root / "benchmark_comparison_method_1.csv").is_file())
            self.assertTrue((output_root / "benchmark_summary_method_1.json").is_file())

            with self.assertRaises(OutputExistsError):
                run_pipeline(request)
            # The same root still accepts the other selection afterwards.
            reserve_run_files(output_root, MethodSelection.METHOD_2)

    def test_corrupt_video_produces_controlled_failed_manifest(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            input_root = root / "input"
            output_root = root / "output"
            input_root.mkdir()
            (input_root / "broken.mp4").write_bytes(b"not a video")
            summary = run_pipeline(
                PipelineRequest(
                    input_directory=input_root,
                    output_directory=output_root,
                    method=MethodSelection.METHOD_1,
                )
            )
            self.assertEqual("failed", summary.status)
            manifest = json.loads(
                (output_root / "run_manifest_method_1.json").read_text(encoding="utf-8")
            )
            self.assertEqual("inspect", manifest["failures"][0]["stage"])
            self.assertEqual([], manifest["results"])

    def test_method_two_requires_local_model_before_creating_output(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            input_root = root / "input"
            output_root = root / "output"
            input_root.mkdir()
            create_synthetic_video(input_root / "sample.mp4", frames=3)
            with self.assertRaises(ConfigurationError):
                run_pipeline(
                    PipelineRequest(
                        input_directory=input_root,
                        output_directory=output_root,
                        method=MethodSelection.METHOD_2,
                    )
                )
            self.assertFalse(output_root.exists())


if __name__ == "__main__":
    unittest.main()
