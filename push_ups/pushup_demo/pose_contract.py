"""Contrato neutral de pose para backends intercambiables.

``PoseFrame v1`` separa deliberadamente conceptos que el NPZ historico
mezclaba en una matriz ``(33, 4)``:

* coordenadas normalizadas de imagen ``x/y/z``;
* visibilidad (ocluido o fuera de cuadro);
* presencia (el punto existe en la escena);
* coordenadas world del modelo, que no son el mismo ``z`` de imagen.

El contrato no importa MediaPipe. Los adaptadores de cada motor son quienes
traducen su topologia a estos 33 nombres canonicos. Los arrays densos se
mantienen en NPZ y nunca se incorporan al estado de LangGraph.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


POSE_FRAME_SCHEMA = "pose-frame/1.0"
POSE_TRACK_SCHEMA = "pose-track/1.0"
TOPOLOGY = "blazepose-33"

CANONICAL_LANDMARKS: tuple[str, ...] = (
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
)
LANDMARK_INDEX = {name: index for index, name in enumerate(CANONICAL_LANDMARKS)}
LANDMARK_COUNT = len(CANONICAL_LANDMARKS)


def landmark_index(name: str) -> int:
    try:
        return LANDMARK_INDEX[name.lower()]
    except KeyError as exc:
        raise KeyError(f"landmark canonico desconocido: {name}") from exc


def _float_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != shape:
        raise ValueError(f"{name} debe tener shape {shape}, recibido {result.shape}")
    return result


def _score_array(value: Any, name: str) -> np.ndarray:
    result = _float_array(value, (LANDMARK_COUNT,), name)
    finite = result[np.isfinite(result)]
    if finite.size and ((finite < 0).any() or (finite > 1).any()):
        raise ValueError(f"{name} debe estar entre 0 y 1 o ser NaN")
    return result


def _size(value: Sequence[int], name: str) -> tuple[int, int]:
    if len(value) != 2:
        raise ValueError(f"{name} debe ser (width, height)")
    width, height = int(value[0]), int(value[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"{name} debe contener dimensiones positivas")
    return width, height


@dataclass(frozen=True)
class FrameTransform:
    """Transformacion aplicada antes de inferir pose."""

    source_size: tuple[int, int]
    frame_size: tuple[int, int]
    rotation_degrees: int = 0
    # ROI normalizado x, y, width, height respecto de la fuente ya rotada.
    roi: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_size", _size(self.source_size, "source_size"))
        object.__setattr__(self, "frame_size", _size(self.frame_size, "frame_size"))
        rotation = int(self.rotation_degrees) % 360
        if rotation not in (0, 90, 180, 270):
            raise ValueError("rotation_degrees debe ser 0, 90, 180 o 270")
        object.__setattr__(self, "rotation_degrees", rotation)
        if self.roi is not None:
            if len(self.roi) != 4:
                raise ValueError("roi debe ser (x, y, width, height)")
            x, y, width, height = (float(item) for item in self.roi)
            if not (0 <= x <= 1 and 0 <= y <= 1 and width > 0 and height > 0):
                raise ValueError("roi normalizado invalido")
            if x + width > 1.000001 or y + height > 1.000001:
                raise ValueError("roi sale de la imagen")
            object.__setattr__(self, "roi", (x, y, width, height))

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_size": list(self.source_size),
            "frame_size": list(self.frame_size),
            "rotation_degrees": self.rotation_degrees,
            "roi": list(self.roi) if self.roi is not None else None,
        }


@dataclass(frozen=True)
class PoseProvenance:
    backend: str
    backend_version: str
    model_name: str | None = None
    model_sha256: str | None = None
    delegate: str = "cpu"

    def __post_init__(self) -> None:
        if not self.backend.strip():
            raise ValueError("backend es obligatorio")
        if not self.backend_version.strip():
            raise ValueError("backend_version es obligatorio")
        if self.model_sha256 is not None:
            digest = self.model_sha256.lower()
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError("model_sha256 debe ser SHA-256 hexadecimal")
            object.__setattr__(self, "model_sha256", digest)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "backend_version": self.backend_version,
            "model_name": self.model_name,
            "model_sha256": self.model_sha256,
            "delegate": self.delegate,
        }


@dataclass(frozen=True)
class PoseObservation:
    image_landmarks: np.ndarray
    visibility: np.ndarray
    presence: np.ndarray
    world_landmarks: np.ndarray
    detected: bool = True

    def __post_init__(self) -> None:
        image = _float_array(self.image_landmarks, (LANDMARK_COUNT, 3), "image_landmarks")
        visibility = _score_array(self.visibility, "visibility")
        presence = _score_array(self.presence, "presence")
        world = _float_array(self.world_landmarks, (LANDMARK_COUNT, 3), "world_landmarks")
        object.__setattr__(self, "image_landmarks", image)
        object.__setattr__(self, "visibility", visibility)
        object.__setattr__(self, "presence", presence)
        object.__setattr__(self, "world_landmarks", world)
        if not self.detected and (
            np.isfinite(image).any() or np.isfinite(visibility).any()
            or np.isfinite(presence).any() or np.isfinite(world).any()
        ):
            raise ValueError("una observacion no detectada debe contener solo NaN")

    @classmethod
    def empty(cls) -> "PoseObservation":
        return cls(
            image_landmarks=np.full((LANDMARK_COUNT, 3), np.nan, dtype=np.float32),
            visibility=np.full(LANDMARK_COUNT, np.nan, dtype=np.float32),
            presence=np.full(LANDMARK_COUNT, np.nan, dtype=np.float32),
            world_landmarks=np.full((LANDMARK_COUNT, 3), np.nan, dtype=np.float32),
            detected=False,
        )

    def legacy_array(self) -> np.ndarray:
        """Proyeccion compatible con el NPZ historico ``(33, 4)``."""
        return np.column_stack((self.image_landmarks, self.visibility)).astype(np.float32)


@dataclass(frozen=True)
class PoseFrame:
    timestamp_ms: int
    transform: FrameTransform
    provenance: PoseProvenance
    observation: PoseObservation
    schema_version: str = POSE_FRAME_SCHEMA

    def __post_init__(self) -> None:
        timestamp = int(self.timestamp_ms)
        if timestamp < 0:
            raise ValueError("timestamp_ms no puede ser negativo")
        object.__setattr__(self, "timestamp_ms", timestamp)
        if self.schema_version != POSE_FRAME_SCHEMA:
            raise ValueError(f"schema PoseFrame no soportado: {self.schema_version}")

    @property
    def timestamp_s(self) -> float:
        return self.timestamp_ms / 1000.0

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "timestamp_ms": self.timestamp_ms,
            "topology": TOPOLOGY,
            "detected": self.observation.detected,
            "transform": self.transform.to_dict(),
            "provenance": self.provenance.to_dict(),
        }


@dataclass
class PoseTrack:
    """Track denso apilado, apto para persistencia NPZ."""

    timestamps: np.ndarray
    image_landmarks: np.ndarray
    visibility: np.ndarray
    presence: np.ndarray
    world_landmarks: np.ndarray
    detected: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.timestamps = np.asarray(self.timestamps, dtype=np.float32)
        count = len(self.timestamps)
        self.image_landmarks = _float_array(
            self.image_landmarks, (count, LANDMARK_COUNT, 3), "image_landmarks",
        )
        self.visibility = _float_array(self.visibility, (count, LANDMARK_COUNT), "visibility")
        self.presence = _float_array(self.presence, (count, LANDMARK_COUNT), "presence")
        self.world_landmarks = _float_array(
            self.world_landmarks, (count, LANDMARK_COUNT, 3), "world_landmarks",
        )
        self.detected = np.asarray(self.detected, dtype=bool)
        if self.detected.shape != (count,):
            raise ValueError(f"detected debe tener shape {(count,)}")
        if count > 1 and (np.diff(self.timestamps) <= 0).any():
            raise ValueError("timestamps deben ser estrictamente crecientes")

    @classmethod
    def from_frames(cls, frames: Sequence[PoseFrame]) -> "PoseTrack":
        if not frames:
            empty = np.empty
            return cls(
                timestamps=empty(0, dtype=np.float32),
                image_landmarks=empty((0, LANDMARK_COUNT, 3), dtype=np.float32),
                visibility=empty((0, LANDMARK_COUNT), dtype=np.float32),
                presence=empty((0, LANDMARK_COUNT), dtype=np.float32),
                world_landmarks=empty((0, LANDMARK_COUNT, 3), dtype=np.float32),
                detected=empty(0, dtype=bool),
                metadata={"schema_version": POSE_TRACK_SCHEMA, "topology": TOPOLOGY},
            )
        provenance = frames[0].provenance
        previous = -1
        for frame in frames:
            if frame.timestamp_ms <= previous:
                raise ValueError("PoseFrame timestamps deben ser estrictamente crecientes")
            if frame.provenance != provenance:
                raise ValueError("un PoseTrack no puede mezclar backends/modelos")
            previous = frame.timestamp_ms
        observations = [frame.observation for frame in frames]
        return cls(
            timestamps=np.asarray([frame.timestamp_s for frame in frames], dtype=np.float32),
            image_landmarks=np.stack([item.image_landmarks for item in observations]),
            visibility=np.stack([item.visibility for item in observations]),
            presence=np.stack([item.presence for item in observations]),
            world_landmarks=np.stack([item.world_landmarks for item in observations]),
            detected=np.asarray([item.detected for item in observations], dtype=bool),
            metadata={
                "schema_version": POSE_TRACK_SCHEMA,
                "frame_schema_version": POSE_FRAME_SCHEMA,
                "topology": TOPOLOGY,
                "provenance": provenance.to_dict(),
                "transform": frames[0].transform.to_dict(),
            },
        )

    @property
    def legacy_landmarks(self) -> np.ndarray:
        return np.concatenate((self.image_landmarks, self.visibility[..., None]), axis=2).astype(np.float32)

    def npz_payload(self, **extras: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            # Claves nuevas.
            "pose_schema": np.asarray([POSE_TRACK_SCHEMA]),
            "pose_metadata_json": np.asarray([
                json.dumps(self.metadata, ensure_ascii=False, sort_keys=True),
            ]),
            "image_landmarks": self.image_landmarks,
            "presence": self.presence,
            "world_landmarks": self.world_landmarks,
            "detected": self.detected.astype(np.uint8),
            # Claves Legacy que consumidores actuales siguen leyendo.
            "timestamps": self.timestamps,
            "landmarks": self.legacy_landmarks,
            "visibility": self.visibility,
        }
        payload.update(extras)
        return payload


def load_pose_track(path: Path) -> PoseTrack:
    """Lee PoseTrack v1 o migra en memoria un NPZ Legacy."""
    with np.load(path, allow_pickle=False) as data:
        timestamps = data["timestamps"].astype(np.float32)
        stored_fps = None
        for key in ("effective_sample_fps", "sample_fps"):
            if key in data and len(data[key]):
                stored_fps = float(data[key][0])
                break
        if "image_landmarks" in data:
            image = data["image_landmarks"].astype(np.float32)
            visibility = data["visibility"].astype(np.float32)
            presence = data["presence"].astype(np.float32)
            world = data["world_landmarks"].astype(np.float32)
            detected = data["detected"].astype(bool)
            raw_metadata = str(data["pose_metadata_json"][0]) if "pose_metadata_json" in data else "{}"
            metadata = json.loads(raw_metadata)
        else:
            legacy = data["landmarks"].astype(np.float32)
            if legacy.ndim != 3 or legacy.shape[1:] != (LANDMARK_COUNT, 4):
                raise ValueError("track Legacy debe tener shape (N, 33, 4)")
            image = legacy[:, :, :3]
            visibility = legacy[:, :, 3]
            presence = np.full_like(visibility, np.nan)
            world = np.full_like(image, np.nan)
            detected = np.isfinite(image[:, :, :2]).any(axis=(1, 2))
            metadata = {
                "schema_version": "legacy",
                "topology": TOPOLOGY,
                "migrated_in_memory": True,
            }
        if stored_fps is not None:
            metadata.setdefault("sampling", {})["effective_sample_fps"] = stored_fps
    return PoseTrack(
        timestamps=timestamps,
        image_landmarks=image,
        visibility=visibility,
        presence=presence,
        world_landmarks=world,
        detected=detected,
        metadata=metadata,
    )