"""Shared detailed result returned by both berm segmenters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..contracts import BermObservation


@dataclass(frozen=True)
class SegmentationResult:
    observation: BermObservation
    mask: np.ndarray
    confidence: float
