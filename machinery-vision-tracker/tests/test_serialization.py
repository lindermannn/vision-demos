from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bermguard.serialization import encode_json, write_json_exclusive  # noqa: E402


class SerializationTests(unittest.TestCase):
    def test_non_finite_values_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            encode_json({"fps": float("nan")})

    def test_unicode_is_preserved(self) -> None:
        self.assertIn("pretil", encode_json({"categoría": "pretil"}))

    def test_existing_file_is_not_overwritten(self) -> None:
        with TemporaryDirectory() as directory:
            target = Path(directory) / "metadata.json"
            write_json_exclusive(target, {"run": 1})
            with self.assertRaises(FileExistsError):
                write_json_exclusive(target, {"run": 2})
            self.assertIn('"run": 1', target.read_text(encoding="utf-8"))
            self.assertEqual([], list(target.parent.glob(".metadata.json.*.tmp")))


if __name__ == "__main__":
    unittest.main()
