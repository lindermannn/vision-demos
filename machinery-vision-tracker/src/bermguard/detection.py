"""Conservative motion/appearance detector for visual equipment candidates."""

from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

from .contracts import BoundingBox, Detection, FrameContext
from .g2_config import G2Settings


def _intersection_over_union(first: BoundingBox, second: BoundingBox) -> float:
    x1 = max(first.x1, second.x1)
    y1 = max(first.y1, second.y1)
    x2 = min(first.x2, second.x2)
    y2 = min(first.y2, second.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first.x2 - first.x1) * max(0.0, first.y2 - first.y1)
    second_area = max(0.0, second.x2 - second.x1) * max(0.0, second.y2 - second.y1)
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _suppress_overlaps(
    candidates: Sequence[tuple[BoundingBox, float]], threshold: float = 0.35
) -> list[tuple[BoundingBox, float]]:
    """Keep distinct candidates; never create a larger union box."""

    retained: list[tuple[BoundingBox, float]] = []
    for candidate in sorted(candidates, key=lambda item: item[1], reverse=True):
        if any(
            _intersection_over_union(candidate[0], existing[0]) >= threshold
            for existing in retained
        ):
            continue
        retained.append(candidate)
    return retained


def _suppress_adjacent_fragments(
    candidates: Sequence[tuple[BoundingBox, float]],
    frame_width: int,
    frame_height: int,
) -> list[tuple[BoundingBox, float]]:
    """Drop small pieces adjacent to a larger region; never join their boxes."""

    retained: list[tuple[BoundingBox, float]] = []
    by_area = sorted(
        candidates,
        key=lambda item: (item[0].x2 - item[0].x1) * (item[0].y2 - item[0].y1),
        reverse=True,
    )
    for candidate in by_area:
        box = candidate[0]
        area = (box.x2 - box.x1) * (box.y2 - box.y1)
        fragment = False
        for larger, _ in retained:
            larger_area = (larger.x2 - larger.x1) * (larger.y2 - larger.y1)
            if area >= larger_area * 0.55:
                continue
            horizontal_gap = max(larger.x1 - box.x2, box.x1 - larger.x2, 0.0)
            overlap_y = max(0.0, min(larger.y2, box.y2) - max(larger.y1, box.y1))
            minimum_height = max(1.0, min(larger.y2 - larger.y1, box.y2 - box.y1))
            vertical_gap = max(larger.y1 - box.y2, box.y1 - larger.y2, 0.0)
            overlap_x = max(0.0, min(larger.x2, box.x2) - max(larger.x1, box.x1))
            minimum_width = max(1.0, min(larger.x2 - larger.x1, box.x2 - box.x1))
            horizontally_adjacent = (
                horizontal_gap <= frame_width * 0.025
                and overlap_y / minimum_height >= 0.25
            )
            vertically_adjacent = (
                vertical_gap <= frame_height * 0.025
                and overlap_x / minimum_width >= 0.25
            )
            if horizontally_adjacent or vertically_adjacent:
                fragment = True
                break
        if not fragment:
            retained.append(candidate)
    return retained


class ClassicalVehicleDetector:
    """Precision-first classical detector; outputs candidates, not semantic classes."""

    name = "conservative_motion_appearance_visual_candidates"

    def __init__(self, settings: G2Settings) -> None:
        self.settings = settings
        self.last_diagnostics: dict[str, int | float] = {}
        self.last_exclusion_mask: Optional[np.ndarray] = None
        self.reset()

    def reset(self) -> None:
        self._background = cv2.createBackgroundSubtractorMOG2(
            history=100, varThreshold=32, detectShadows=False
        )
        self._previous_gray: Optional[np.ndarray] = None
        self._frames_seen = 0
        self.last_diagnostics = {}
        self.last_exclusion_mask = None

    def detect(self, frame: np.ndarray, context: FrameContext) -> Sequence[Detection]:
        original_height, original_width = frame.shape[:2]
        process_width = min(self.settings.segmentation_process_width, original_width)
        scale = process_width / original_width
        process_height = max(1, int(round(original_height * scale)))
        working = (
            frame
            if scale == 1.0
            else cv2.resize(
                frame, (process_width, process_height), interpolation=cv2.INTER_AREA
            )
        )
        height, width = working.shape[:2]
        gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        foreground = self._background.apply(working, learningRate=0.018)

        if self._previous_gray is None:
            difference = np.zeros_like(gray)
        else:
            difference = cv2.absdiff(gray, self._previous_gray)
            _, difference = cv2.threshold(difference, 20, 255, cv2.THRESH_BINARY)
        self._previous_gray = gray
        self._frames_seen += 1

        _, foreground = cv2.threshold(foreground, 180, 255, cv2.THRESH_BINARY)
        difference_support = cv2.dilate(
            difference,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 13)),
            iterations=1,
        )
        motion = cv2.bitwise_and(foreground, difference_support)

        hsv = cv2.cvtColor(working, cv2.COLOR_BGR2HSV)
        mean_lightness = float(np.mean(gray))
        median_lightness = float(np.median(gray))
        night = median_lightness < self.settings.scene_day_above_mean
        if night:
            appearance = cv2.inRange(hsv, (0, 0, 210), (179, 135, 255))
        else:
            appearance = cv2.inRange(hsv, (4, 45, 35), (44, 255, 255))
        local_motion_support = cv2.dilate(
            motion,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 11)),
            iterations=1,
        )
        supported_appearance = cv2.bitwise_and(appearance, local_motion_support)
        combined = cv2.bitwise_or(motion, supported_appearance)
        combined[: int(round(height * 0.28)), :] = 0
        combined[int(round(height * 0.97)) :, :] = 0
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                max(5, int(round(width * 0.010))) | 1,
                max(5, int(round(height * 0.012))) | 1,
            ),
        )
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, close_kernel)
        combined = cv2.morphologyEx(
            combined,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )
        exclusion = cv2.dilate(
            combined,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 21)),
            iterations=1,
        )
        self.last_exclusion_mask = (
            exclusion
            if scale == 1.0
            else cv2.resize(
                exclusion,
                (original_width, original_height),
                interpolation=cv2.INTER_NEAREST,
            )
        )

        if self._frames_seen <= self.settings.detection_warmup_frames:
            self.last_diagnostics = {
                "raw_contours": 0,
                "accepted_candidates": 0,
                "warmup": 1,
            }
            return ()

        frame_area = float(width * height)
        minimum_area = self.settings.detection_min_area_ratio * frame_area
        maximum_area = self.settings.detection_max_area_ratio * frame_area
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[tuple[BoundingBox, float]] = []
        rejection_counts: dict[str, int] = {
            "area": 0,
            "geometry": 0,
            "vertical_position": 0,
            "motion": 0,
            "texture": 0,
            "night_glare": 0,
            "confidence": 0,
        }
        for contour in contours:
            contour_area = float(cv2.contourArea(contour))
            if not minimum_area <= contour_area <= maximum_area:
                rejection_counts["area"] += 1
                continue
            x, y, box_width, box_height = cv2.boundingRect(contour)
            box_area = float(box_width * box_height)
            box_area_ratio = box_area / frame_area
            width_ratio = box_width / width
            height_ratio = box_height / height
            aspect = box_width / max(box_height, 1)
            if (
                width_ratio < self.settings.detection_min_width_ratio
                or height_ratio < self.settings.detection_min_height_ratio
                or width_ratio > self.settings.detection_max_width_ratio
                or height_ratio > self.settings.detection_max_height_ratio
                or not self.settings.detection_min_aspect_ratio
                <= aspect
                <= self.settings.detection_max_aspect_ratio
                or box_area_ratio > self.settings.detection_max_area_ratio
            ):
                rejection_counts["geometry"] += 1
                continue
            center_y_ratio = (y + box_height / 2.0) / height
            if not (
                self.settings.detection_min_vertical_center_ratio
                <= center_y_ratio
                <= self.settings.detection_max_vertical_center_ratio
            ):
                rejection_counts["vertical_position"] += 1
                continue

            contour_extent = contour_area / max(box_area, 1.0)
            motion_patch = motion[y : y + box_height, x : x + box_width]
            motion_fill = float(np.count_nonzero(motion_patch)) / max(box_area, 1.0)
            if motion_fill < self.settings.detection_min_motion_fill:
                rejection_counts["motion"] += 1
                continue
            gray_patch = gray[y : y + box_height, x : x + box_width]
            texture_std = float(np.std(gray_patch))
            if texture_std < self.settings.detection_min_texture_std:
                rejection_counts["texture"] += 1
                continue
            hsv_patch = hsv[y : y + box_height, x : x + box_width]
            bright_fraction = float(
                np.mean(
                    np.logical_and(
                        hsv_patch[:, :, 2] >= 215,
                        hsv_patch[:, :, 1] <= 120,
                    )
                )
            )
            if (
                night
                and bright_fraction
                > self.settings.detection_max_night_bright_fraction
            ):
                rejection_counts["night_glare"] += 1
                continue

            motion_score = min(1.0, motion_fill / 0.30)
            extent_score = min(1.0, contour_extent / 0.55)
            texture_score = min(1.0, texture_std / 38.0)
            aspect_score = max(0.0, 1.0 - abs(aspect - 2.0) / 3.0)
            confidence = float(
                np.clip(
                    0.12
                    + 0.34 * motion_score
                    + 0.22 * extent_score
                    + 0.20 * texture_score
                    + 0.12 * aspect_score
                    - (0.18 * bright_fraction if night else 0.0),
                    0.0,
                    0.95,
                )
            )
            if confidence < self.settings.detection_min_confidence:
                rejection_counts["confidence"] += 1
                continue
            candidates.append(
                (
                    BoundingBox(
                        float(x),
                        float(y),
                        float(x + box_width),
                        float(y + box_height),
                    ),
                    confidence,
                )
            )

        retained = _suppress_adjacent_fragments(
            _suppress_overlaps(candidates), width, height
        )
        retained.sort(key=lambda item: item[1], reverse=True)
        detections: list[Detection] = []
        for box, confidence in retained[: self.settings.detection_max_detections]:
            # Revalidate after every post-processing operation. NMS never enlarges a
            # box, but this invariant guards future changes from repeating G4's bug.
            box_width = box.x2 - box.x1
            box_height = box.y2 - box.y1
            if (
                box_width * box_height > maximum_area
                or box_width > width * self.settings.detection_max_width_ratio
                or box_height > height * self.settings.detection_max_height_ratio
                or box_width / max(box_height, 1.0)
                > self.settings.detection_max_aspect_ratio
            ):
                rejection_counts["geometry"] += 1
                continue
            detections.append(
                Detection(
                    label="visual_equipment_candidate",
                    confidence=confidence,
                    bbox=BoundingBox(
                        box.x1 / scale,
                        box.y1 / scale,
                        box.x2 / scale,
                        box.y2 / scale,
                    ),
                )
            )
        self.last_diagnostics = {
            "raw_contours": len(contours),
            "pre_nms_candidates": len(candidates),
            "accepted_candidates": len(detections),
            "mean_lightness": mean_lightness,
            "median_lightness": median_lightness,
            "night_mode": int(night),
            **{f"rejected_{key}": value for key, value in rejection_counts.items()},
        }
        return tuple(detections)
