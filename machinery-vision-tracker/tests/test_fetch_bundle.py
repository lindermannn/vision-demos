"""The fetch scripts must use a shipped snapshot when its SHA-256 matches."""

import hashlib
import importlib.util
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BundleFetchTests(unittest.TestCase):
    def _run(self, name: str) -> None:
        module = _load(name)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "bundle"
            bundle.mkdir()
            weight = bundle / "model.safetensors"
            weight.write_bytes(b"pinned-weight-fixture")
            (bundle / "config.json").write_text("{}", encoding="utf-8")
            module.WEIGHT_SHA256 = hashlib.sha256(weight.read_bytes()).hexdigest()
            captured = io.StringIO()
            with redirect_stdout(captured):
                code = module.main(
                    ["--destination", str(root / "dest"), "--bundle", str(bundle)]
                )
            self.assertEqual(0, code)
            report = json.loads(captured.getvalue())
            self.assertEqual("bundle", report["source"])
            self.assertEqual("verified", report["status"])
            self.assertTrue((root / "dest" / "model.safetensors").is_file())
            self.assertTrue((root / "dest" / "config.json").is_file())

    def test_depth_fetch_uses_matching_bundle_without_network(self) -> None:
        self._run("fetch_depth_model")

    def test_detector_fetch_uses_matching_bundle_without_network(self) -> None:
        self._run("fetch_semantic_detector")

    def test_mismatching_bundle_is_not_trusted(self) -> None:
        module = _load("fetch_depth_model")
        with TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "bundle"
            bundle.mkdir()
            (bundle / "model.safetensors").write_bytes(b"tampered")
            module.WEIGHT_SHA256 = "0" * 64

            def refuse(**_: object) -> None:
                raise RuntimeError("network path reached")

            module.snapshot_download = refuse  # type: ignore[attr-defined]
            import sys
            import types

            fake_hub = types.ModuleType("huggingface_hub")
            fake_hub.snapshot_download = refuse  # type: ignore[attr-defined]
            previous = sys.modules.get("huggingface_hub")
            sys.modules["huggingface_hub"] = fake_hub
            try:
                with self.assertRaisesRegex(RuntimeError, "network path reached"):
                    module.main(["--destination", str(root / "dest"), "--bundle", str(bundle)])
            finally:
                if previous is not None:
                    sys.modules["huggingface_hub"] = previous
                else:
                    del sys.modules["huggingface_hub"]


if __name__ == "__main__":
    unittest.main()
