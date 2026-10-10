from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bermguard.cli import parse_args  # noqa: E402
from bermguard.contracts import MethodSelection  # noqa: E402


class CliContractTests(unittest.TestCase):
    def test_all_method_is_parsed(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            options = parse_args(
                ["--input", "input", "--output", "output", "--method", "all"]
            )
        self.assertEqual(options.method, MethodSelection.ALL)
        self.assertEqual(options.input_directory, Path("input"))
        self.assertIsNone(options.model_directory)

    def test_model_directory_is_optional_and_parsed(self) -> None:
        options = parse_args(
            [
                "--input", "input", "--output", "output", "--method", "2",
                "--model-dir", "models/depth",
            ]
        )
        self.assertEqual(Path("models/depth"), options.model_directory)

    def test_model_directory_can_come_from_offline_container_environment(self) -> None:
        with patch.dict("os.environ", {"BERMGUARD_MODEL_DIR": "/opt/bermguard/model"}):
            options = parse_args(
                ["--input", "input", "--output", "output", "--method", "all"]
            )
        self.assertEqual(Path("/opt/bermguard/model"), options.model_directory)

    def test_detector_directory_can_come_from_offline_container_environment(self) -> None:
        with patch.dict(
            "os.environ",
            {"BERMGUARD_DETECTOR_MODEL_DIR": "/opt/bermguard/detector-model"},
        ):
            options = parse_args(
                ["--input", "input", "--output", "output", "--method", "1"]
            )
        self.assertEqual(
            Path("/opt/bermguard/detector-model"),
            options.detector_model_directory,
        )

    def test_invalid_method_is_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(["--input", "input", "--output", "output", "--method", "3"])


class OfficialEntryPointTests(unittest.TestCase):
    """`python main.py ...` is the invocation fixed by the assessment brief."""

    ROOT = Path(__file__).resolve().parents[1]

    def _run(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "main.py", *arguments],
            cwd=self.ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )

    def test_python_main_py_exposes_the_official_flags(self) -> None:
        completed = self._run("--help")
        self.assertEqual(0, completed.returncode, completed.stderr)
        for flag in ("--input", "--output", "--method"):
            self.assertIn(flag, completed.stdout)
        self.assertIn("{1,2,all}", completed.stdout)

    def test_python_main_py_reports_controlled_error_without_traceback(self) -> None:
        with TemporaryDirectory() as directory:
            missing = Path(directory) / "missing"
            completed = self._run(
                "--input", str(missing), "--output", str(Path(directory) / "out"),
                "--method", "1",
            )
        self.assertEqual(2, completed.returncode)
        self.assertIn("bermguard:", completed.stderr)
        self.assertNotIn("Traceback", completed.stderr)


if __name__ == "__main__":
    unittest.main()
