from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bermguard.artifacts import (  # noqa: E402
    method_output_transaction,
    prepare_output_root,
    reserve_run_files,
)
from bermguard.contracts import MethodSelection  # noqa: E402
from bermguard.exceptions import OutputExistsError  # noqa: E402


class ArtifactTests(unittest.TestCase):
    def test_transaction_publishes_complete_directory(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            input_root = root / "input"
            input_root.mkdir()
            output_root = prepare_output_root(input_root, root / "output")
            with method_output_transaction(
                output_root, "video-abcd1234", MethodSelection.METHOD_1
            ) as work:
                (work / "complete.txt").write_text("ok", encoding="utf-8")
            final = output_root / "video-abcd1234" / "method_1" / "complete.txt"
            self.assertEqual("ok", final.read_text(encoding="utf-8"))
            self.assertFalse((output_root / ".work").exists())

    def test_failed_transaction_is_not_published(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            input_root = root / "input"
            input_root.mkdir()
            output_root = prepare_output_root(input_root, root / "output")
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                with method_output_transaction(
                    output_root, "video-abcd1234", MethodSelection.METHOD_1
                ) as work:
                    (work / "partial.txt").write_text("partial", encoding="utf-8")
                    raise RuntimeError("interrupted")
            self.assertFalse((output_root / "video-abcd1234").exists())
            self.assertFalse((output_root / ".work").exists())

    def test_non_empty_output_root_is_accepted_and_preserved(self) -> None:
        # The official command mounts one host directory for every method run;
        # a previous run or a stray file must not block the next selection.
        with TemporaryDirectory() as directory:
            root = Path(directory)
            input_root = root / "input"
            output_root = root / "output"
            input_root.mkdir()
            output_root.mkdir()
            (output_root / "existing.txt").write_text("keep", encoding="utf-8")
            resolved = prepare_output_root(input_root, output_root)
            self.assertEqual(output_root.resolve(), resolved)
            self.assertEqual(
                "keep", (output_root / "existing.txt").read_text(encoding="utf-8")
            )

    def test_output_path_that_is_a_file_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            input_root = root / "input"
            input_root.mkdir()
            target = root / "output"
            target.write_text("not a directory", encoding="utf-8")
            with self.assertRaises(OutputExistsError):
                prepare_output_root(input_root, target)

    def test_run_files_are_reserved_once_per_method_selection(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = reserve_run_files(root, MethodSelection.METHOD_1)
            self.assertEqual("run_manifest_method_1.json", first.manifest.name)
            self.assertEqual(
                "benchmark_comparison_method_1.csv", first.benchmark_table.name
            )
            self.assertEqual(
                "benchmark_summary_method_1.json", first.benchmark_summary.name
            )
            first.manifest.write_text("{}", encoding="utf-8")

            # Sequential method 1 -> method 2 -> all into one root is allowed.
            reserve_run_files(root, MethodSelection.METHOD_2)
            reserve_run_files(root, MethodSelection.ALL)
            with self.assertRaisesRegex(OutputExistsError, "run_manifest_method_1.json"):
                reserve_run_files(root, MethodSelection.METHOD_1)

    def test_existing_method_directory_is_still_refused(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            input_root = root / "input"
            input_root.mkdir()
            output_root = prepare_output_root(input_root, root / "output")
            (output_root / "video-abcd1234" / "method_1").mkdir(parents=True)
            with self.assertRaises(OutputExistsError):
                with method_output_transaction(
                    output_root, "video-abcd1234", MethodSelection.METHOD_1
                ):
                    pass

    def test_output_cannot_be_nested_inside_input(self) -> None:
        with TemporaryDirectory() as directory:
            input_root = Path(directory)
            with self.assertRaises(OutputExistsError):
                prepare_output_root(input_root, input_root / "generated")


if __name__ == "__main__":
    unittest.main()
