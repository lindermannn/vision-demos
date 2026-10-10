"""Method 1: conservative paired photometric berm extraction."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..contracts import FrameContext
from ..g2_config import G2Settings
from .result import SegmentationResult
from .spatial import extract_classical_paired_profile


ClassicalSegmentationResult = SegmentationResult


class ClassicalBermSegmenter:
    """Find a dark earth ridge with jointly supported crest and road-side toe."""

    name = "classical_signed_photometric_joint_crest_toe"

    def __init__(self, settings: G2Settings) -> None:
        self.settings = settings
        self.operating_surface: tuple[float, float] | None = None

    def set_operating_surface(self, band: tuple[float, float] | None) -> None:
        """Far/near machinery contact rows in source pixels, or None."""

        self.operating_surface = band

    def segment(self, frame: Any, context: FrameContext) -> SegmentationResult:
        if not isinstance(frame, np.ndarray) or frame.ndim != 3:
            raise ValueError("frame must be a BGR image")
        return extract_classical_paired_profile(
            frame, context, self.settings, self.operating_surface
        )
