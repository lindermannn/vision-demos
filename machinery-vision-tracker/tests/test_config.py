from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bermguard.config import load_configuration  # noqa: E402
from bermguard.exceptions import ConfigurationError  # noqa: E402


class ConfigurationTests(unittest.TestCase):
    def test_built_in_configuration_is_versioned_and_hashed(self) -> None:
        configuration = load_configuration(None)
        self.assertEqual(1, configuration.data["schema_version"])
        self.assertEqual(64, len(configuration.sha256))
        self.assertFalse(configuration.fail_fast)

    def test_overwrite_cannot_be_enabled(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "unsafe.yaml"
            path.write_text(
                "schema_version: 1\nruntime:\n  fail_fast: false\n"
                "output:\n  overwrite: true\n",
                encoding="utf-8",
            )
            with self.assertRaises(ConfigurationError):
                load_configuration(path)

    def test_repository_example_matches_packaged_default(self) -> None:
        repository_default = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
        self.assertEqual(
            load_configuration(None).sha256,
            load_configuration(repository_default).sha256,
        )


if __name__ == "__main__":
    unittest.main()
