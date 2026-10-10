"""The VRAM probe measures without changing what it measures."""

import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("vram_probe", ROOT / "scripts" / "vram_probe.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["vram_probe"] = module
    spec.loader.exec_module(module)
    return module


vram_probe = _load()

try:
    import torch

    CUDA = torch.cuda.is_available()
except Exception:  # pragma: no cover - torch is optional for the probe
    torch = None
    CUDA = False


class FakePredictor:
    """Shape of a library predictor: inference methods and a ``model`` attribute."""

    def __init__(self, device="cpu"):
        self.model = torch.nn.Linear(256, 256).to(device) if torch is not None else None
        self.inner_calls = 0

    def predict(self, value):
        return value * 2

    def outer(self, value):
        # Library entry points call each other; only the outermost may count.
        self.inner_calls += 1
        return self.predict(value) + 1

    def stream(self, count):
        for index in range(count):
            yield index

    def allocate(self, megabytes):
        block = torch.empty(int(megabytes * 1024 * 1024), dtype=torch.uint8, device="cuda")
        return int(block.numel())


def _fresh_class():
    return type("Predictor", (FakePredictor,), {})


class WrappingTests(unittest.TestCase):
    def test_calls_and_results_are_unchanged_without_cuda(self):
        probe = vram_probe.Probe(force_cpu=True)
        cls = _fresh_class()
        probe.wrap_method(cls, "predict", "fake")
        self.assertEqual(14, cls().predict(7))
        report = probe.report(label="cpu")
        self.assertEqual(1, report["components"]["fake"]["calls"])
        self.assertIsNone(report["components"]["fake"]["weights_mb"])
        self.assertEqual("not_applicable_without_cuda", report["vram_basis"])

    def test_wrapping_is_idempotent(self):
        probe = vram_probe.Probe(force_cpu=True)
        cls = _fresh_class()
        self.assertTrue(probe.wrap_method(cls, "predict", "fake"))
        self.assertFalse(probe.wrap_method(cls, "predict", "fake"))
        cls().predict(1)
        self.assertEqual(1, probe.components["fake"].calls)

    def test_nested_entry_points_count_once(self):
        probe = vram_probe.Probe(force_cpu=True)
        cls = _fresh_class()
        probe.wrap_method(cls, "predict", "fake")
        probe.wrap_method(cls, "outer", "fake")
        self.assertEqual(7, cls().outer(3))
        self.assertEqual(1, probe.components["fake"].calls)

    def test_generators_are_measured_per_item(self):
        probe = vram_probe.Probe(force_cpu=True)
        cls = _fresh_class()
        probe.wrap_method(cls, "stream", "fake")
        self.assertEqual([0, 1, 2, 3], list(cls().stream(4)))
        # One call for creating the generator plus one per produced item.
        self.assertEqual(5, probe.components["fake"].calls)

    def test_exceptions_propagate_and_still_count(self):
        probe = vram_probe.Probe(force_cpu=True)
        cls = _fresh_class()
        probe.wrap_method(cls, "predict", "fake")
        with self.assertRaises(TypeError):
            cls().predict(None)
        self.assertEqual(1, probe.components["fake"].calls)
        self.assertEqual(0, probe._depth())


class LauncherTests(unittest.TestCase):
    def test_report_is_written_and_exit_code_preserved(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.py"
            target.write_text("import sys\nprint('ok')\nsys.exit(3)\n", encoding="utf-8")
            report_path = root / "vram.json"
            probe = vram_probe.Probe(force_cpu=True)
            code = vram_probe.run_target(target, ["--flag"], report_path, "prueba", ["none"], probe)
            self.assertEqual(3, code)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(3, report["exit_code"])
            self.assertEqual("prueba", report["label"])
            self.assertTrue(report["command"][-1] == "--flag")
            self.assertIn("definitions", report)

    def test_target_sees_its_own_arguments(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.py"
            out = root / "argv.json"
            target.write_text(
                "import json, sys\n"
                f"with open(r'{out}', 'w', encoding='utf-8') as handle:\n"
                "    handle.write(json.dumps(sys.argv[1:]))\n",
                encoding="utf-8",
            )
            probe = vram_probe.Probe(force_cpu=True)
            vram_probe.run_target(target, ["--input", "a", "--method", "1"], root / "r.json", "", ["none"], probe)
            self.assertEqual(["--input", "a", "--method", "1"], json.loads(out.read_text()))

    def test_missing_separator_is_a_usage_error(self):
        self.assertEqual(2, vram_probe.main(["--report", "x.json"]))


@unittest.skipUnless(CUDA, "requires CUDA")
class CudaMeasurementTests(unittest.TestCase):
    def test_activation_peak_reflects_an_allocation(self):
        probe = vram_probe.Probe()
        cls = _fresh_class()
        probe.wrap_method(cls, "allocate", "fake")
        cls(device="cuda").allocate(64)
        stats = probe.report()["components"]["fake"]
        self.assertGreaterEqual(stats["activation_peak_mb"], 60.0)

    def test_weights_are_counted_once_per_model(self):
        probe = vram_probe.Probe()
        cls = _fresh_class()
        probe.wrap_method(cls, "predict", "fake")
        predictor = cls(device="cuda")
        for _ in range(3):
            predictor.predict(torch.ones(1, device="cuda"))
        stats = probe.components["fake"]
        expected = sum(p.numel() * p.element_size() for p in predictor.model.parameters())
        self.assertEqual(expected, stats.weights_bytes)
        self.assertEqual(1, stats.models)

    def test_shared_weights_are_not_double_counted(self):
        probe = vram_probe.Probe()
        cls = _fresh_class()
        probe.wrap_method(cls, "predict", "fake")
        first = cls(device="cuda")
        second = cls(device="cuda")
        second.model = first.model
        first.predict(torch.ones(1, device="cuda"))
        second.predict(torch.ones(1, device="cuda"))
        expected = sum(p.numel() * p.element_size() for p in first.model.parameters())
        self.assertEqual(expected, probe.components["fake"].weights_bytes)


class AdapterTargetTests(unittest.TestCase):
    """Catch API drift: every wrapped entry point must exist where installed."""

    def test_installed_libraries_expose_the_wrapped_entry_points(self):
        checked = 0
        for name, targets in vram_probe.ADAPTERS.items():
            library = vram_probe.LIBRARY_OF[name]
            if importlib.util.find_spec(library) is None:
                continue
            for module_name, class_name, attribute in targets:
                with self.subTest(target=f"{class_name}.{attribute}"):
                    owner = getattr(importlib.import_module(module_name), class_name)
                    self.assertTrue(callable(getattr(owner, attribute)))
                    checked += 1
        if checked == 0:
            self.skipTest("none of the adapter libraries is installed")


if __name__ == "__main__":
    unittest.main()
