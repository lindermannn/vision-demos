"""Stop paying for a berm search whose answer will be thrown away.

The starting idea was to restrict the classical sweep to a band around the
anchor: once the geometry is fixed, why look anywhere else?  That was built and
measured, and it did cut the sweep in half.  It also did two things that made it
unusable as written:

1. **It biased the agreement.**  A search that may only propose bodies near the
   anchor will, unsurprisingly, agree with the anchor.  The median match
   fraction of video 04 went from 0.01 to 0.57 without a single pixel of new
   evidence.  V8 measures the berm on the columns where the current frame agrees
   with the anchor, so that agreement has to come from the image, not from where
   the search was told to look.
2. **It leaked between videos.**  A fresh validator is built per video but the
   band lived in this module, so videos 02 to 04 bootstrapped their anchor while
   still constrained by the previous video's band, and their anchors moved.

What survives is the useful half of the observation.  The extractor runs on every
frame, but its result is only ever used when the light allows a measurement: the
validator refuses to measure outside daylight and simply carries the anchored
geometry.  So below the daylight threshold the sweep is pure waste, and skipping
it costs nothing that anyone reads.

Daylight frames keep the full, unconstrained grid.  That is what preserves the
system's ability to say "the current evidence does not match the anchor", which
is the whole point of the per-column support test.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import cv2
import numpy as np

from bermguard.contracts import FrameContext
from bermguard.segmentation.classical import ClassicalBermSegmenter
from bermguard.segmentation.result import SegmentationResult
from bermguard.segmentation.spatial import extract_classical_paired_profile

SKIP_REASON = "low_light_search_skipped"

COUNTS: dict[str, int] = {"searched": 0, "skipped": 0}


def reset_counts() -> None:
    COUNTS["searched"] = 0
    COUNTS["skipped"] = 0


def frame_is_daylight(frame: np.ndarray, day_above_mean: float) -> bool:
    """Same median-grey test the validator uses, on a cheap thumbnail."""

    small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
    grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return float(np.median(grey)) >= day_above_mean


class DaylightOnlyClassicalSegmenter(ClassicalBermSegmenter):
    """Full grid in daylight, no search at all below the daylight threshold."""

    name = "classical_signed_photometric_joint_crest_toe_daylight_gated"

    def segment(self, frame: Any, context: FrameContext) -> SegmentationResult:
        if not isinstance(frame, np.ndarray) or frame.ndim != 3:
            raise ValueError("frame must be a BGR image")

        if not frame_is_daylight(frame, self.settings.scene_day_above_mean):
            COUNTS["skipped"] += 1
            return _skipped(context)

        COUNTS["searched"] += 1
        result = extract_classical_paired_profile(
            frame, context, self.settings, self.operating_surface
        )
        diagnostics = dict(result.observation.diagnostics)
        diagnostics["search_mode"] = "full_grid"
        observation = replace(result.observation, diagnostics=diagnostics)
        return SegmentationResult(observation, result.mask, result.confidence)


def _skipped(context: FrameContext) -> SegmentationResult:
    """An explicit "not looked at" result, never a claim about the berm."""

    from bermguard.contracts import BermObservation, ObservationStatus

    observation = BermObservation(
        context,
        ObservationStatus.UNKNOWN,
        (),
        (),
        None,
        None,
        {"reason": SKIP_REASON, "search_mode": "skipped_low_light",
         "spatially_valid": False},
    )
    return SegmentationResult(
        observation, np.zeros((context.height, context.width), dtype=np.uint8), 0.0
    )
