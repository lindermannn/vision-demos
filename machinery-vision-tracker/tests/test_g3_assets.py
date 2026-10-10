import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bermguard.config import load_configuration  # noqa: E402
from bermguard.exceptions import ConfigurationError  # noqa: E402
from bermguard.g2_config import G2Settings  # noqa: E402
from bermguard.segmentation.pretrained import DepthAnythingV2Segmenter  # noqa: E402
from bermguard.semantic_detection import GroundingDinoVehicleDetector  # noqa: E402


class G3AssetTests(unittest.TestCase):
    def test_manifest_config_and_fetch_script_use_the_same_immutable_model(self) -> None:
        manifest = json.loads(
            (ROOT / "models" / "MODEL_MANIFEST.json").read_text(encoding="utf-8")
        )
        settings = G2Settings.from_mapping(load_configuration(None).data)
        script_path = ROOT / "scripts" / "fetch_depth_model.py"
        spec = importlib.util.spec_from_file_location("fetch_depth_model", script_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        self.assertEqual(manifest["model_id"], settings.learned_model_id)
        self.assertEqual(manifest["revision"], settings.learned_revision)
        self.assertEqual(manifest["weight_sha256"], settings.learned_weight_sha256)
        self.assertEqual(manifest["model_id"], module.MODEL_ID)
        self.assertEqual(manifest["revision"], module.REVISION)
        self.assertEqual(manifest["weight_sha256"], module.WEIGHT_SHA256)
        self.assertEqual(
            manifest["weight_sha256"], manifest["files"][manifest["weight_file"]]
        )

    def test_incomplete_snapshot_is_rejected_before_optional_imports(self) -> None:
        settings = G2Settings.from_mapping(load_configuration(None).data)
        with TemporaryDirectory() as directory:
            model_directory = Path(directory)
            (model_directory / "config.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ConfigurationError, "snapshot is incomplete"):
                DepthAnythingV2Segmenter(settings, model_directory)

    def test_wrong_weight_checksum_is_rejected_before_model_loading(self) -> None:
        settings = G2Settings.from_mapping(load_configuration(None).data)
        with TemporaryDirectory() as directory:
            model_directory = Path(directory)
            for name in ("config.json", "preprocessor_config.json"):
                (model_directory / name).write_text("{}", encoding="utf-8")
            (model_directory / "model.safetensors").write_bytes(b"not-the-pinned-model")
            with self.assertRaisesRegex(ConfigurationError, "checksum"):
                DepthAnythingV2Segmenter(settings, model_directory)

    def test_semantic_manifest_config_and_fetch_script_are_pinned_together(self) -> None:
        manifest = json.loads(
            (ROOT / "models" / "SEMANTIC_DETECTOR_MANIFEST.json").read_text(
                encoding="utf-8"
            )
        )
        configured = G2Settings.from_mapping(load_configuration(None).data)
        script_path = ROOT / "scripts" / "fetch_semantic_detector.py"
        spec = importlib.util.spec_from_file_location(
            "fetch_semantic_detector", script_path
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        self.assertEqual(manifest["model_id"], configured.semantic_model_id)
        self.assertEqual(manifest["revision"], configured.semantic_revision)
        self.assertEqual(
            manifest["weight_sha256"], configured.semantic_weight_sha256
        )
        self.assertEqual(manifest["model_id"], module.MODEL_ID)
        self.assertEqual(manifest["revision"], module.REVISION)
        self.assertEqual(manifest["weight_sha256"], module.WEIGHT_SHA256)

    def test_incomplete_semantic_snapshot_is_rejected_before_optional_imports(self) -> None:
        configured = G2Settings.from_mapping(load_configuration(None).data)
        with TemporaryDirectory() as directory:
            model_directory = Path(directory)
            (model_directory / "config.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ConfigurationError, "snapshot is incomplete"):
                GroundingDinoVehicleDetector(configured, model_directory)


if __name__ == "__main__":
    unittest.main()
