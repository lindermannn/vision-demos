"""Method 2: local-only Depth Anything V2 evidence for paired berm geometry."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any, Optional

import cv2
import numpy as np

from ..contracts import BermObservation, FrameContext
from ..exceptions import ConfigurationError
from ..g2_config import G2Settings
from ..video import sha256_file
from .result import SegmentationResult
from .spatial import extract_depth_paired_profile


def profile_from_relative_depth(
    depth: np.ndarray,
    context: FrameContext,
    settings: G2Settings,
    diagnostics: Optional[dict[str, Any]] = None,
    operating_surface: Optional[tuple[float, float]] = None,
) -> SegmentationResult:
    """Convert relative depth into a jointly supported crest/body/toe hypothesis."""

    return extract_depth_paired_profile(
        depth, context, settings, diagnostics, operating_surface
    )


class DepthAnythingV2Segmenter:
    name = "depth_anything_v2_relative_slope_break_joint_crest_toe"

    def __init__(self, settings: G2Settings, model_directory: Path) -> None:
        self.settings = settings
        self.operating_surface: Optional[tuple[float, float]] = None
        try:
            self.model_directory = model_directory.expanduser().resolve(strict=True)
        except OSError as exc:
            raise ConfigurationError("local learned model directory was not found") from exc
        if not self.model_directory.is_dir():
            raise ConfigurationError("local learned model path must be a directory")
        required = ("config.json", "model.safetensors", "preprocessor_config.json")
        missing = [name for name in required if not (self.model_directory / name).is_file()]
        if missing:
            raise ConfigurationError(
                "local learned model snapshot is incomplete: " + ", ".join(missing)
            )
        actual_hash = sha256_file(self.model_directory / "model.safetensors")
        if actual_hash != settings.learned_weight_sha256:
            raise ConfigurationError("learned model weight checksum does not match configuration")
        try:
            import torch
            import transformers
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as exc:
            raise ConfigurationError(
                "method 2 requires the optional learned dependency set"
            ) from exc

        if settings.learned_device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = settings.learned_device
        if device == "cuda" and not torch.cuda.is_available():
            raise ConfigurationError("CUDA was requested for method 2 but is unavailable")
        self._torch = torch
        self._transformers_version = transformers.__version__
        self._device = torch.device(device)
        self._processor = AutoImageProcessor.from_pretrained(
            self.model_directory,
            local_files_only=True,
            use_fast=False,
        )
        self._model = AutoModelForDepthEstimation.from_pretrained(
            self.model_directory,
            local_files_only=True,
        ).to(self._device).eval()
        warmup_width = settings.learned_input_size * 25 // 14
        warmup_width = max(14, int(round(warmup_width / 14)) * 14)
        dummy = torch.zeros(
            (1, 3, settings.learned_input_size, warmup_width),
            dtype=torch.float32,
            device=self._device,
        )
        started = perf_counter()
        with torch.inference_mode():
            self._model(pixel_values=dummy)
        if self._device.type == "cuda":
            torch.cuda.synchronize()
        self._warmup_ms = (perf_counter() - started) * 1000.0

    def set_operating_surface(self, band: Optional[tuple[float, float]]) -> None:
        self.operating_surface = band

    def segment(self, frame: Any, context: FrameContext) -> SegmentationResult:
        if not isinstance(frame, np.ndarray) or frame.ndim != 3:
            raise ValueError("frame must be a BGR image")
        torch = self._torch
        preprocess_started = perf_counter()
        target_height = self.settings.learned_input_size
        target_width = int(
            round((frame.shape[1] / frame.shape[0] * target_height) / 14)
        ) * 14
        target_width = max(14, target_width)
        resized_input = cv2.resize(
            frame,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA,
        )
        rgb = cv2.cvtColor(resized_input, cv2.COLOR_BGR2RGB)
        inputs = self._processor(
            images=rgb,
            return_tensors="pt",
            do_resize=False,
        )
        inputs = {name: value.to(self._device) for name, value in inputs.items()}
        preprocess_ms = (perf_counter() - preprocess_started) * 1000.0

        inference_started = perf_counter()
        with torch.inference_mode():
            predicted = self._model(**inputs).predicted_depth
        if self._device.type == "cuda":
            torch.cuda.synchronize()
        inference_ms = (perf_counter() - inference_started) * 1000.0

        postprocess_started = perf_counter()
        process_width = min(self.settings.segmentation_process_width, context.width)
        scale = process_width / context.width
        process_height = max(1, int(round(context.height * scale)))
        resized = torch.nn.functional.interpolate(
            predicted.unsqueeze(1),
            size=(process_height, process_width),
            mode="bicubic",
            align_corners=False,
        )[0, 0]
        depth = resized.float().cpu().numpy()
        transfer_ms = (perf_counter() - postprocess_started) * 1000.0
        result = profile_from_relative_depth(
            depth,
            context,
            self.settings,
            {
                "preprocess_ms": preprocess_ms,
                "inference_ms": inference_ms,
                "depth_transfer_ms": transfer_ms,
                "learned_device": self._device.type,
            },
            operating_surface=self.operating_surface,
        )
        total_postprocess_ms = (perf_counter() - postprocess_started) * 1000.0
        updated = dict(result.observation.diagnostics)
        updated["postprocess_ms"] = total_postprocess_ms
        observation = BermObservation(
            context=result.observation.context,
            status=result.observation.status,
            crest=result.observation.crest,
            base=result.observation.base,
            height_px=result.observation.height_px,
            height_m_estimated=result.observation.height_m_estimated,
            diagnostics=updated,
        )
        return SegmentationResult(observation, result.mask, result.confidence)

    def runtime_metadata(self) -> dict[str, Any]:
        torch = self._torch
        metadata: dict[str, Any] = {
            "model_id": self.settings.learned_model_id,
            "revision": self.settings.learned_revision,
            "weight_sha256": self.settings.learned_weight_sha256,
            "weight_size_bytes": (self.model_directory / "model.safetensors").stat().st_size,
            "input_size": self.settings.learned_input_size,
            "device": self._device.type,
            "torch": torch.__version__,
            "transformers": self._transformers_version,
            "warmup_ms": self._warmup_ms,
            "local_files_only": True,
        }
        if self._device.type == "cuda":
            metadata["cuda"] = torch.version.cuda
            metadata["gpu_name"] = torch.cuda.get_device_name(self._device)
            metadata["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 1024**2
            metadata["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 1024**2
        return metadata
