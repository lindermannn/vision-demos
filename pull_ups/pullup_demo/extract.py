"""Pose extraction: video -> PoseTrack .npz, using MediaPipe Pose locally.

Frames are decoded with FFmpeg so rotation metadata is applied the same way the
analysis renderer applies it. Landmarks are normalized to the displayed frame.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

LANDMARK_COUNT = 33
MAX_SIDE = 720  # Longest side used for pose estimation; coordinates stay normalized.
# The detector's timing thresholds were tuned at 20 frames per second. Sampling
# at the native rate changes rep durations and counts, so frames are sampled here.
SAMPLE_FPS = 20


def probe(video: Path) -> dict[str, Any]:
    output = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,avg_frame_rate,nb_frames:stream_side_data=rotation:format=duration",
         "-of", "json", str(video)],
        capture_output=True, text=True, check=True,
    ).stdout
    data = json.loads(output)
    stream = data["streams"][0]
    rotation = next((abs(int(item.get("rotation", 0))) for item in stream.get("side_data_list", [])), 0)
    num, _, den = stream["avg_frame_rate"].partition("/")
    fps = float(num) / float(den or 1)
    width, height = int(stream["width"]), int(stream["height"])
    if rotation in (90, 270):
        width, height = height, width
    return {"width": width, "height": height, "fps": fps,
            "duration_s": float(data["format"]["duration"])}


def write_track(path: Path, timestamps, image, visibility, world, detected, metadata: dict) -> None:
    """Write the PoseTrack v1 layout that ``pose_contract.load_pose_track`` reads."""
    timestamps = np.asarray(timestamps, dtype=np.float64)
    count = len(timestamps)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        timestamps=timestamps,
        image_landmarks=np.asarray(image, dtype=np.float32).reshape(count, LANDMARK_COUNT, 3),
        visibility=np.asarray(visibility, dtype=np.float32).reshape(count, LANDMARK_COUNT),
        presence=np.full((count, LANDMARK_COUNT), np.nan, dtype=np.float32),
        world_landmarks=np.asarray(world, dtype=np.float32).reshape(count, LANDMARK_COUNT, 3),
        detected=np.asarray(detected, dtype=bool),
        pose_metadata_json=np.array([json.dumps(metadata)]),
        effective_sample_fps=np.array([metadata.get("fps", np.nan)], dtype=np.float64),
    )


def extract(video: Path, output: Path) -> Path:
    """Run MediaPipe Pose on every frame of ``video`` and save the track to ``output``."""
    import mediapipe as mp

    info = probe(video)
    width, height = info["width"], info["height"]
    scale = min(1.0, MAX_SIDE / max(width, height))
    work_w, work_h = max(2, round(width * scale) // 2 * 2), max(2, round(height * scale) // 2 * 2)
    decode = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"fps={SAMPLE_FPS},scale={work_w}:{work_h}",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    frame_bytes = work_w * work_h * 3
    timestamps, image, visibility, world, detected = [], [], [], [], []
    empty_pose = np.full((LANDMARK_COUNT, 3), np.nan, dtype=np.float32)
    empty_vis = np.full(LANDMARK_COUNT, np.nan, dtype=np.float32)
    index = 0
    try:
        with mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1,
                                    smooth_landmarks=True, min_detection_confidence=.5,
                                    min_tracking_confidence=.5) as pose:
            assert decode.stdout is not None
            while True:
                raw = decode.stdout.read(frame_bytes)
                if len(raw) != frame_bytes:
                    break
                frame = np.frombuffer(raw, dtype=np.uint8).reshape(work_h, work_w, 3)
                result = pose.process(frame)
                timestamps.append(index / SAMPLE_FPS)
                if result.pose_landmarks:
                    points = result.pose_landmarks.landmark
                    image.append([[p.x, p.y, p.z] for p in points])
                    visibility.append([p.visibility for p in points])
                    world_points = result.pose_world_landmarks.landmark if result.pose_world_landmarks else points
                    world.append([[p.x, p.y, p.z] for p in world_points])
                    detected.append(True)
                else:
                    image.append(empty_pose)
                    visibility.append(empty_vis)
                    world.append(empty_pose)
                    detected.append(False)
                index += 1
    finally:
        if decode.poll() is None:
            decode.kill()
        if decode.stdout:
            decode.stdout.close()
        decode.wait()
    if not index:
        raise RuntimeError(f"No se pudieron leer fotogramas de {video}")
    metadata = {
        "backend": "mediapipe_legacy_pose",
        "mediapipe_version": mp.__version__,
        "model_complexity": 1,
        "fps": SAMPLE_FPS,
        "source_fps": info["fps"],
        "source_width": width,
        "source_height": height,
        "frames": index,
    }
    write_track(output, timestamps, np.asarray(image), np.asarray(visibility),
                np.asarray(world), detected, metadata)
    return output


def file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cached_track(video: Path, cache_dir: Path) -> Path:
    """Return the cached pose track for ``video``, extracting it on first use."""
    target = cache_dir / f"{file_sha256(video)[:16]}-{video.stem}.npz"
    if not target.exists():
        extract(video, target)
    return target
