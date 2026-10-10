"""Pinned zero-shot heavy-equipment detector with causal box propagation."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import cv2
import numpy as np

from .contracts import BoundingBox, Detection, FrameContext
from .exceptions import ConfigurationError
from .g2_config import G2Settings
from .video import sha256_file


def _area(box: BoundingBox) -> float:
    return max(0.0, box.x2 - box.x1) * max(0.0, box.y2 - box.y1)


def _intersection(first: BoundingBox, second: BoundingBox) -> float:
    return max(0.0, min(first.x2, second.x2) - max(first.x1, second.x1)) * max(
        0.0, min(first.y2, second.y2) - max(first.y1, second.y1)
    )


def _overlap(first: BoundingBox, second: BoundingBox) -> tuple[float, float]:
    intersection = _intersection(first, second)
    first_area = _area(first)
    second_area = _area(second)
    union = first_area + second_area - intersection
    iou = intersection / union if union > 0 else 0.0
    containment = intersection / max(min(first_area, second_area), 1.0)
    return iou, containment


def suppress_semantic_duplicates(
    detections: Sequence[Detection], iou_threshold: float
) -> tuple[Detection, ...]:
    """Class-agnostic NMS because all prompt labels represent heavy equipment."""

    retained: list[Detection] = []
    for candidate in sorted(detections, key=lambda item: item.confidence, reverse=True):
        duplicate = False
        for existing in retained:
            iou, containment = _overlap(candidate.bbox, existing.bbox)
            if iou >= iou_threshold or containment >= 0.82:
                duplicate = True
                break
        if not duplicate:
            retained.append(candidate)
    return tuple(retained)


class GroundingDinoVehicleDetector:
    """Semantic keyframe detector; intermediate boxes move using optical flow."""

    name = "grounding_dino_tiny_semantic_heavy_equipment"

    def __init__(self, settings: G2Settings, model_directory: Path) -> None:
        self.settings = settings
        try:
            self.model_directory = model_directory.expanduser().resolve(strict=True)
        except OSError as exc:
            raise ConfigurationError(
                "local semantic detector model directory was not found"
            ) from exc
        required = (
            "config.json",
            "model.safetensors",
            "preprocessor_config.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "vocab.txt",
        )
        missing = [
            name for name in required if not (self.model_directory / name).is_file()
        ]
        if missing:
            raise ConfigurationError(
                "local semantic detector snapshot is incomplete: "
                + ", ".join(missing)
            )
        actual_hash = sha256_file(self.model_directory / "model.safetensors")
        if actual_hash != settings.semantic_weight_sha256:
            raise ConfigurationError(
                "semantic detector weight checksum does not match configuration"
            )
        try:
            import torch
            import transformers
            from PIL import Image
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:
            raise ConfigurationError(
                "semantic detection requires the optional learned dependency set"
            ) from exc

        if settings.semantic_device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = settings.semantic_device
        if device == "cuda" and not torch.cuda.is_available():
            raise ConfigurationError(
                "CUDA was requested for semantic detection but is unavailable"
            )
        self._torch = torch
        self._image_type = Image
        self._transformers_version = transformers.__version__
        self._device = torch.device(device)
        if self._device.type == "cuda":
            torch.set_float32_matmul_precision("high")
        self._processor = AutoProcessor.from_pretrained(
            self.model_directory,
            local_files_only=True,
        )
        self._processor.image_processor.size = {
            "shortest_edge": settings.semantic_input_short_edge,
            "longest_edge": settings.semantic_input_long_edge,
        }
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_directory,
            local_files_only=True,
            dtype=torch.float32,
        ).to(self._device).eval()
        self._warmup_ms = 0.0
        self.reset_run()

    def reset_scene(self) -> None:
        """Discard temporal state without erasing run-level evidence."""

        self._previous_gray: np.ndarray | None = None
        self._cached: tuple[Detection, ...] = ()
        self.last_diagnostics: dict[str, int | float | str] = {}
        self.last_exclusion_mask: np.ndarray | None = None

    def reset_run(self) -> None:
        """Start a new video run and clear cumulative diagnostics."""

        self._semantic_inferences = 0
        self._propagated_frames = 0
        self._raw_label_counts: Counter[str] = Counter()
        self.reset_scene()

    def reset(self) -> None:
        """Compatibility alias used by the detector protocol."""

        self.reset_run()

    def _valid_geometry(
        self, box: BoundingBox, context: FrameContext
    ) -> bool:
        width = box.x2 - box.x1
        height = box.y2 - box.y1
        area_ratio = width * height / float(context.width * context.height)
        return (
            self.settings.semantic_min_area_ratio <= area_ratio
            <= self.settings.semantic_max_area_ratio
            and width / context.width >= self.settings.semantic_min_width_ratio
            and height / context.height >= self.settings.semantic_min_height_ratio
        )

    @staticmethod
    def _clip(box: BoundingBox, width: int, height: int) -> BoundingBox:
        return BoundingBox(
            float(np.clip(box.x1, 0, width - 1)),
            float(np.clip(box.y1, 0, height - 1)),
            float(np.clip(box.x2, 1, width)),
            float(np.clip(box.y2, 1, height)),
        )

    def _infer(
        self, frame: np.ndarray, context: FrameContext
    ) -> tuple[Detection, ...]:
        torch = self._torch
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = self._image_type.fromarray(rgb)
        preprocess_started = perf_counter()
        labels = list(self.settings.semantic_labels)
        inputs = self._processor(images=image, text=[labels], return_tensors="pt")
        inputs = {
            key: value.to(self._device) if torch.is_tensor(value) else value
            for key, value in inputs.items()
        }
        preprocess_ms = (perf_counter() - preprocess_started) * 1000.0
        inference_started = perf_counter()
        with torch.inference_mode():
            outputs = self._model(**inputs)
        if self._device.type == "cuda":
            torch.cuda.synchronize()
        inference_ms = (perf_counter() - inference_started) * 1000.0
        post_started = perf_counter()
        processed = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=self.settings.semantic_box_threshold,
            text_threshold=self.settings.semantic_text_threshold,
            target_sizes=[image.size[::-1]],
            text_labels=[labels],
        )[0]
        raw: list[Detection] = []
        geometry_rejections = 0
        for score, label, coordinates in zip(
            processed["scores"],
            processed["text_labels"],
            processed["boxes"],
        ):
            raw_label = str(label)
            self._raw_label_counts[raw_label] += 1
            values = [float(value) for value in coordinates.tolist()]
            box = self._clip(BoundingBox(*values), context.width, context.height)
            if not self._valid_geometry(box, context):
                geometry_rejections += 1
                continue
            raw.append(
                Detection(
                    label="maquinaria_pesada",
                    confidence=float(score.item()),
                    bbox=box,
                )
            )
        retained = suppress_semantic_duplicates(
            raw, self.settings.semantic_nms_iou_threshold
        )[: self.settings.semantic_max_detections]
        self._semantic_inferences += 1
        self.last_diagnostics = {
            "backend": self.name,
            "semantic_inference": 1,
            "raw_semantic_candidates": len(processed["scores"]),
            "rejected_semantic_geometry": geometry_rejections,
            "accepted_candidates": len(retained),
            "semantic_preprocess_ms": preprocess_ms,
            "semantic_inference_ms": inference_ms,
            "semantic_postprocess_ms": (perf_counter() - post_started) * 1000.0,
        }
        return tuple(retained)

    def _propagate(
        self,
        previous_gray: np.ndarray,
        current_gray: np.ndarray,
        context: FrameContext,
    ) -> tuple[Detection, ...]:
        propagated: list[Detection] = []
        for detection in self._cached:
            box = detection.bbox
            x1 = max(0, int(np.floor(box.x1)))
            y1 = max(0, int(np.floor(box.y1)))
            x2 = min(context.width, int(np.ceil(box.x2)))
            y2 = min(context.height, int(np.ceil(box.y2)))
            shift_x = shift_y = 0.0
            if x2 - x1 >= 8 and y2 - y1 >= 8:
                features = cv2.goodFeaturesToTrack(
                    previous_gray[y1:y2, x1:x2],
                    maxCorners=48,
                    qualityLevel=0.012,
                    minDistance=5,
                    blockSize=5,
                )
                if features is not None and len(features) >= 4:
                    features[:, 0, 0] += x1
                    features[:, 0, 1] += y1
                    moved, status, error = cv2.calcOpticalFlowPyrLK(
                        previous_gray,
                        current_gray,
                        features,
                        None,
                        winSize=(21, 21),
                        maxLevel=3,
                        criteria=(
                            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                            24,
                            0.01,
                        ),
                    )
                    if moved is not None and status is not None:
                        valid = status.reshape(-1).astype(bool)
                        if error is not None:
                            valid &= error.reshape(-1) < 35.0
                        if np.count_nonzero(valid) >= 3:
                            delta = moved.reshape(-1, 2)[valid] - features.reshape(-1, 2)[valid]
                            shift_x, shift_y = np.median(delta, axis=0).tolist()
            moved_box = self._clip(
                BoundingBox(
                    box.x1 + shift_x,
                    box.y1 + shift_y,
                    box.x2 + shift_x,
                    box.y2 + shift_y,
                ),
                context.width,
                context.height,
            )
            if self._valid_geometry(moved_box, context):
                propagated.append(
                    Detection(
                        label=detection.label,
                        confidence=max(0.0, detection.confidence * 0.985),
                        bbox=moved_box,
                    )
                )
        self._propagated_frames += 1
        self.last_diagnostics = {
            "backend": self.name,
            "semantic_inference": 0,
            "propagated_candidates": len(propagated),
            "accepted_candidates": len(propagated),
        }
        return tuple(propagated)

    def _update_exclusion_mask(
        self, detections: Sequence[Detection], context: FrameContext
    ) -> None:
        mask = np.zeros((context.height, context.width), dtype=np.uint8)
        margin = max(3, int(round(context.width * 0.008)))
        for detection in detections:
            box = detection.bbox
            x1 = max(0, int(round(box.x1)) - margin)
            y1 = max(0, int(round(box.y1)) - margin)
            x2 = min(context.width, int(round(box.x2)) + margin)
            y2 = min(context.height, int(round(box.y2)) + margin)
            mask[y1:y2, x1:x2] = 255
        self.last_exclusion_mask = mask

    def detect(
        self, frame: Any, context: FrameContext
    ) -> Sequence[Detection]:
        if not isinstance(frame, np.ndarray) or frame.ndim != 3:
            raise ValueError("frame must be a BGR image")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keyframe = (
            not self._cached
            or context.frame_index % self.settings.semantic_inference_interval == 0
            or self._previous_gray is None
        )
        if keyframe:
            detections = self._infer(frame, context)
        else:
            detections = self._propagate(self._previous_gray, gray, context)
        self._cached = tuple(detections)
        self._previous_gray = gray
        self._update_exclusion_mask(detections, context)
        return detections

    def runtime_metadata(self) -> dict[str, Any]:
        torch = self._torch
        metadata: dict[str, Any] = {
            "backend": self.name,
            "model_id": self.settings.semantic_model_id,
            "revision": self.settings.semantic_revision,
            "weight_sha256": self.settings.semantic_weight_sha256,
            "weight_size_bytes": (
                self.model_directory / "model.safetensors"
            ).stat().st_size,
            "labels": list(self.settings.semantic_labels),
            "box_threshold": self.settings.semantic_box_threshold,
            "text_threshold": self.settings.semantic_text_threshold,
            "nms_iou_threshold": self.settings.semantic_nms_iou_threshold,
            "inference_interval": self.settings.semantic_inference_interval,
            "input_size": dict(self._processor.image_processor.size),
            "device": self._device.type,
            "torch": torch.__version__,
            "transformers": self._transformers_version,
            "semantic_inferences": self._semantic_inferences,
            "propagated_frames": self._propagated_frames,
            "raw_label_counts": dict(self._raw_label_counts),
            "local_files_only": True,
        }
        if self._device.type == "cuda":
            metadata.update(
                {
                    "cuda": torch.version.cuda,
                    "gpu_name": torch.cuda.get_device_name(self._device),
                    "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                    "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
                }
            )
        return metadata
