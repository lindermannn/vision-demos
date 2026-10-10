import importlib.util
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_release.py"
SPEC = importlib.util.spec_from_file_location("build_release", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
BUILD_RELEASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD_RELEASE)


def git(repository: Path, *arguments: str) -> None:
    subprocess.check_call(
        ["git", "-C", str(repository), *arguments],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


class ReleaseBuilderTests(unittest.TestCase):
    def test_clean_commit_and_artifacts_create_hashed_non_overwriting_zip(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "repo"
            artifacts = root / "artifacts"
            destination = root / "release"
            repository.mkdir()
            artifacts.mkdir()
            (repository / "README.md").write_text("assessment\n", encoding="utf-8")
            git(repository, "init", "--quiet")
            git(repository, "add", "README.md")
            git(
                repository,
                "-c", "user.name=Test",
                "-c", "user.email=test@invalid.local",
                "commit", "--quiet", "-m", "fixture",
            )
            (artifacts / "run_manifest_method_1.json").write_text(
                '{"status":"complete"}\n', encoding="utf-8"
            )
            (artifacts / "processed.mp4").write_bytes(b"fixture")
            bundle = root / "bundle"
            (bundle / "depth-fixture").mkdir(parents=True)
            (bundle / "depth-fixture" / "model.safetensors").write_bytes(b"weights")
            (bundle / "depth-fixture" / "config.json").write_text("{}", encoding="utf-8")
            (bundle / "not-a-snapshot").mkdir()

            arguments = [
                "--repository", str(repository),
                "--artifacts", str(artifacts),
                "--destination", str(destination),
                "--bundle-models", str(bundle),
            ]
            self.assertEqual(0, BUILD_RELEASE.main(arguments))
            release_json = next(destination.glob("*.release.json"))
            release = json.loads(release_json.read_text(encoding="utf-8"))
            archive = destination / release["archive"]
            self.assertEqual(BUILD_RELEASE._sha256(archive), release["archive_sha256"])
            with zipfile.ZipFile(archive) as packaged:
                names = packaged.namelist()
                manifest_name = next(n for n in names if n.endswith("/RELEASE_MANIFEST.json"))
                internal = json.loads(packaged.read(manifest_name).decode("utf-8"))
            self.assertTrue(any(name.endswith("/README.md") for name in names))
            self.assertTrue(any(name.endswith("/output/processed.mp4") for name in names))
            self.assertTrue(any(name.endswith("/MANIFEST.sha256") for name in names))
            self.assertTrue(
                any(name.endswith("/models/depth-fixture/model.safetensors") for name in names)
            )
            self.assertFalse(any("/models/not-a-snapshot" in name for name in names))
            self.assertTrue(internal["model_weight_included"])
            self.assertEqual(["depth-fixture"], internal["bundled_snapshots"])

            with self.assertRaisesRegex(SystemExit, "already exists"):
                BUILD_RELEASE.main(arguments)

    def test_dirty_repository_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "repo"
            artifacts = root / "artifacts"
            repository.mkdir()
            artifacts.mkdir()
            git(repository, "init", "--quiet")
            (repository / "untracked.txt").write_text("dirty", encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "clean Git worktree"):
                BUILD_RELEASE.main(
                    [
                        "--repository", str(repository),
                        "--artifacts", str(artifacts),
                        "--destination", str(root / "release"),
                    ]
                )


if __name__ == "__main__":
    unittest.main()
