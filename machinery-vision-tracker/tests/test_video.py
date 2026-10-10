from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bermguard.exceptions import InputDiscoveryError  # noqa: E402
from bermguard.video import discover_videos, inspect_video, transcode_passthrough  # noqa: E402

try:
    import cv2
    import numpy

    HAS_CV = True
except ImportError:
    HAS_CV = False


def create_synthetic_video(path: Path, fps: float = 12.0, frames: int = 12) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48)
    )
    if not writer.isOpened():
        raise RuntimeError("test environment cannot encode MP4")
    try:
        for index in range(frames):
            frame = numpy.full((48, 64, 3), index * 10, dtype=numpy.uint8)
            writer.write(frame)
    finally:
        writer.release()


class DiscoveryTests(unittest.TestCase):
    def test_unicode_and_nested_video_names_are_discovered(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "día con espacio"
            nested.mkdir()
            video = nested / "Vídeo 01.MP4"
            video.write_bytes(b"placeholder")
            (root / "notes.txt").write_text("ignore", encoding="utf-8")
            self.assertEqual((video,), discover_videos(root))

    def test_empty_corpus_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            with self.assertRaises(InputDiscoveryError):
                discover_videos(Path(directory))


@unittest.skipUnless(HAS_CV, "OpenCV/NumPy not installed")
class VideoProcessingTests(unittest.TestCase):
    def test_passthrough_preserves_geometry_fps_and_frames(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "Vídeo prueba.mp4"
            output = root / "result" / "processed.mp4"
            create_synthetic_video(source)
            info = inspect_video(root, source)
            stats = transcode_passthrough(info, output)
            encoded = inspect_video(output.parent, output)
            self.assertEqual((64, 48), (encoded.width, encoded.height))
            self.assertAlmostEqual(12.0, encoded.fps, places=2)
            self.assertEqual(12, stats.frame_count)
            self.assertEqual(64, len(stats.output_sha256))


if __name__ == "__main__":
    unittest.main()
