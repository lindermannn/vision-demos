"""Deterministic discovery, inspection and G1 passthrough processing."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re
from time import perf_counter
import unicodedata

from .exceptions import InputDiscoveryError, VideoProcessingError


SUPPORTED_VIDEO_EXTENSIONS = frozenset({".mp4", ".avi", ".mov", ".mkv"})


@dataclass(frozen=True)
class VideoInfo:
    source_path: Path
    relative_path: str
    video_id: str
    sha256: str
    size_bytes: int
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float
    codec_fourcc: str

    def public_metadata(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "video_id": self.video_id,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "duration_s": self.duration_s,
            "codec_fourcc": self.codec_fourcc,
        }


@dataclass(frozen=True)
class ProcessingStats:
    frame_count: int
    decode_s: float
    encode_s: float
    finalize_s: float
    wall_s: float
    observed_fps: float
    output_sha256: str
    output_size_bytes: int
    output_codec_fourcc: str
    scene_s: float = 0.0
    segmentation_s: float = 0.0
    detection_s: float = 0.0
    tracking_s: float = 0.0
    analytics_s: float = 0.0
    rendering_s: float = 0.0


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def discover_videos(input_directory: Path) -> tuple[Path, ...]:
    try:
        root = input_directory.expanduser().resolve(strict=True)
    except OSError as exc:
        raise InputDiscoveryError(
            f"input directory not found: {input_directory.name or input_directory}"
        ) from exc
    if not root.is_dir():
        raise InputDiscoveryError("input path must be a directory")

    candidates = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and path.suffix.casefold() in SUPPORTED_VIDEO_EXTENSIONS
    ]
    candidates.sort(key=lambda path: path.relative_to(root).as_posix().casefold())
    if not candidates:
        raise InputDiscoveryError("input directory contains no supported video files")
    return tuple(candidates)


def _slug(stem: str) -> str:
    ascii_stem = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_stem).strip("-").lower()
    return slug or "video"


def _fourcc_text(value: float) -> str:
    code = int(value)
    text = "".join(chr((code >> (8 * index)) & 0xFF) for index in range(4))
    return "".join(character for character in text if character.isprintable()).strip()


def inspect_video(input_root: Path, source_path: Path) -> VideoInfo:
    import cv2

    root = input_root.expanduser().resolve(strict=True)
    source = source_path.expanduser().resolve(strict=True)
    if root != source and root not in source.parents:
        raise VideoProcessingError("video resolves outside the input directory")

    capture = cv2.VideoCapture(str(source))
    try:
        if not capture.isOpened():
            raise VideoProcessingError(f"cannot open video: {source.name}")
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        codec = _fourcc_text(capture.get(cv2.CAP_PROP_FOURCC))
    finally:
        capture.release()

    if width <= 0 or height <= 0:
        raise VideoProcessingError(f"invalid frame dimensions: {source.name}")
    if not math.isfinite(fps) or fps <= 0:
        raise VideoProcessingError(f"invalid frame rate: {source.name}")
    if frame_count <= 0:
        raise VideoProcessingError(f"video declares no frames: {source.name}")

    digest = sha256_file(source)
    relative = source.relative_to(root).as_posix()
    return VideoInfo(
        source_path=source,
        relative_path=relative,
        video_id=f"{_slug(source.stem)}-{digest[:8]}",
        sha256=digest,
        size_bytes=source.stat().st_size,
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_s=frame_count / fps,
        codec_fourcc=codec,
    )


def transcode_passthrough(video: VideoInfo, output_path: Path) -> ProcessingStats:
    """Decode and encode every frame while making no visual modification."""

    import cv2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise VideoProcessingError(f"output already exists: {output_path.name}")

    capture = cv2.VideoCapture(str(video.source_path))
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        video.fps,
        (video.width, video.height),
    )
    if not capture.isOpened():
        capture.release()
        writer.release()
        raise VideoProcessingError(f"cannot decode video: {video.source_path.name}")
    if not writer.isOpened():
        capture.release()
        writer.release()
        raise VideoProcessingError("local OpenCV build cannot create MP4 output")

    frame_count = 0
    decode_s = 0.0
    encode_s = 0.0
    started = perf_counter()
    try:
        while True:
            stage_started = perf_counter()
            readable, frame = capture.read()
            decode_s += perf_counter() - stage_started
            if not readable:
                break
            if frame is None or frame.shape[:2] != (video.height, video.width):
                raise VideoProcessingError(
                    f"unexpected frame geometry at frame {frame_count}"
                )
            stage_started = perf_counter()
            writer.write(frame)
            encode_s += perf_counter() - stage_started
            frame_count += 1
    finally:
        capture.release()
        finalize_started = perf_counter()
        writer.release()
        finalize_s = perf_counter() - finalize_started
    wall_s = perf_counter() - started

    if frame_count <= 0 or not output_path.is_file() or output_path.stat().st_size <= 0:
        raise VideoProcessingError("video processing produced no frames")
    if abs(video.frame_count - frame_count) > 1:
        raise VideoProcessingError(
            "decoded frame count differs from the source container declaration"
        )

    output = inspect_video(output_path.parent, output_path)
    if (output.width, output.height) != (video.width, video.height):
        raise VideoProcessingError("encoded video changed frame dimensions")
    if abs(output.fps - video.fps) > max(0.01, video.fps * 0.001):
        raise VideoProcessingError("encoded video changed frame rate")
    if abs(output.frame_count - frame_count) > 1:
        raise VideoProcessingError("encoded video frame count failed validation")

    return ProcessingStats(
        frame_count=frame_count,
        decode_s=decode_s,
        encode_s=encode_s,
        finalize_s=finalize_s,
        wall_s=wall_s,
        observed_fps=frame_count / wall_s if wall_s > 0 else 0.0,
        output_sha256=output.sha256,
        output_size_bytes=output.size_bytes,
        output_codec_fourcc=output.codec_fourcc,
    )
