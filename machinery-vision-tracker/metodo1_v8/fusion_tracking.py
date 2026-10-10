"""Tracker experimental que conserva identidades durante fusiones visuales breves."""
from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from bermguard.contracts import BoundingBox, Detection, FrameContext
from bermguard.tracking import CausalTracker

import berm_metrics


def _area(box: BoundingBox) -> float:
    return max(0.0, box.x2 - box.x1) * max(0.0, box.y2 - box.y1)


def _center(box: BoundingBox) -> tuple[float, float]:
    return ((box.x1 + box.x2) * 0.5, (box.y1 + box.y2) * 0.5)


def _contains(box: BoundingBox, point: tuple[float, float], margin: float = 12.0) -> bool:
    return (
        box.x1 - margin <= point[0] <= box.x2 + margin
        and box.y1 - margin <= point[1] <= box.y2 + margin
    )


def _intersection(first: BoundingBox, second: BoundingBox) -> float:
    return max(0.0, min(first.x2, second.x2) - max(first.x1, second.x1)) * max(
        0.0, min(first.y2, second.y2) - max(first.y1, second.y1)
    )


def _translated(box: BoundingBox, dx: float, dy: float) -> BoundingBox:
    return BoundingBox(box.x1 + dx, box.y1 + dy, box.x2 + dx, box.y2 + dy)


class FusionResistantTracker(CausalTracker):
    """No permite que una caja compuesta sustituya varios tracks confirmados.

    La protección es estrictamente causal: sólo conserva objetos que ya fueron
    observados por separado. No inventa ni divide detecciones usando posiciones
    específicas de un video.
    """

    MAX_PREDICTED_MISSES = 4
    UNION_AREA_RATIO = 1.30
    MAX_SINGLE_EXPANSION = 1.42
    NESTED_IOU = 0.72
    NESTED_INTERSECTION = 0.88

    def __init__(self, settings):
        super().__init__(
            replace(
                settings,
                tracking_minimum_hits=2,
                tracking_max_size_change_ratio=min(
                    settings.tracking_max_size_change_ratio, 1.55
                ),
            )
        )
        self._metadata: dict[int, tuple[str, float]] = {}
        self._fusion_rejections = 0

    def reset(self) -> None:
        super().reset()
        self._metadata = {}
        self._fusion_rejections = 0

    def diagnostics(self):
        values = super().diagnostics()
        values["fusion_rejections"] = self._fusion_rejections
        values["predicted_tracks_visible"] = sum(
            1
            for track in self._tracks.values()
            if track.confirmed and 0 < track.missed <= self.MAX_PREDICTED_MISSES
        )
        return values

    def _is_union_detection(self, detection: Detection) -> bool:
        covered = []
        detection_area = max(_area(detection.bbox), 1.0)
        for track in self._tracks.values():
            if not track.confirmed or track.missed > self.MAX_PREDICTED_MISSES:
                continue
            horizon = min(track.missed + 1, 3)
            predicted_box = _translated(
                track.bbox, track.velocity_x * horizon, track.velocity_y * horizon
            )
            if _contains(detection.bbox, _center(predicted_box)):
                covered.append(track)
        return (
            len(covered) >= 2
            and detection_area
            >= self.UNION_AREA_RATIO * max(_area(track.bbox) for track in covered)
        )

    def _is_oversized_replacement(self, detection: Detection) -> bool:
        """Reject a sudden broad proposal that would replace one confirmed track."""
        da = max(_area(detection.bbox), 1.0)
        for track in self._tracks.values():
            if not track.confirmed or track.missed > self.MAX_PREDICTED_MISSES:
                continue
            predicted = _translated(track.bbox, track.velocity_x, track.velocity_y)
            overlap = _intersection(detection.bbox, predicted) / da
            ratio = da / max(_area(track.bbox), 1.0)
            if overlap >= 0.48 and ratio > self.MAX_SINGLE_EXPANSION:
                return True
        return False

    def update(
        self, detections: Sequence[Detection], context: FrameContext
    ) -> tuple[Detection, ...]:
        # Remove duplicate proposals before association; otherwise each proposal
        # can create a new identity in the same frame.
        detections = self._deduplicate_detections(detections)
        filtered = []
        for detection in detections:
            if self._is_union_detection(detection) or self._is_oversized_replacement(detection):
                self._fusion_rejections += 1
            else:
                filtered.append(detection)

        tracked = list(super().update(tuple(filtered), context))
        visible_ids = {detection.track_id for detection in tracked}
        for detection in tracked:
            if detection.track_id is not None:
                self._metadata[detection.track_id] = (
                    detection.label,
                    detection.confidence,
                )

        # Durante una oclusión/fusión breve, exponer la predicción causal de cada
        # track confirmado para que proximidad siga recibiendo objetos separados.
        for track_id, track in self._tracks.items():
            if (
                track_id in visible_ids
                or not track.confirmed
                or not 0 < track.missed <= self.MAX_PREDICTED_MISSES
            ):
                continue
            label, confidence = self._metadata.get(
                track_id, ("maquinaria_pesada", 0.50)
            )
            predicted_box = _translated(
                track.bbox,
                track.velocity_x * min(track.missed, 3),
                track.velocity_y * min(track.missed, 3),
            )
            tracked.append(
                Detection(
                    label=label,
                    confidence=max(0.35, confidence * (0.88 ** track.missed)),
                    bbox=predicted_box,
                    track_id=track_id,
                )
            )
        visible = self._deduplicate_visible(tracked)
        # Share the confirmed population with the berm scale model, which runs
        # one step earlier in the frame and would otherwise only see the
        # detector's raw proposals.
        berm_metrics.publish_confirmed(visible)
        return visible

    @classmethod
    def _deduplicate_detections(cls, detections: Sequence[Detection]) -> tuple[Detection, ...]:
        kept: list[Detection] = []
        for candidate in sorted(detections, key=lambda d: _area(d.bbox)):
            ca = max(_area(candidate.bbox), 1.0)
            duplicate = False
            for existing in kept:
                ea = max(_area(existing.bbox), 1.0)
                overlap = _intersection(candidate.bbox, existing.bbox)
                inter_min = overlap / min(ca, ea)
                union = ca + ea - overlap
                iou = overlap / union if union > 0 else 0.0
                c1, c2 = _center(candidate.bbox), _center(existing.bbox)
                gap = ((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2) ** 0.5
                scale = max(min(candidate.bbox.x2-candidate.bbox.x1, candidate.bbox.y2-candidate.bbox.y1), 1.0)
                # Strong nesting/near-identical geometry only; side-by-side
                # vehicles remain separate even when close.
                if (inter_min >= cls.NESTED_INTERSECTION or iou >= cls.NESTED_IOU) and gap <= 0.28 * scale:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(candidate)
        return tuple(kept)

    @staticmethod
    def _deduplicate_visible(detections: Sequence[Detection]) -> tuple[Detection, ...]:
        """Collapse nested boxes from duplicate detector tracks.

        Only a containment-like match is removed; nearby side-by-side machines
        remain independent even when their boxes touch or partially overlap.
        The tighter (smaller) box is retained because the larger one is the
        characteristic union/duplicate proposal seen in the approved run.
        """
        kept: list[Detection] = []
        for candidate in sorted(detections, key=lambda item: _area(item.bbox)):
            candidate_area = max(_area(candidate.bbox), 1.0)
            duplicate = False
            for existing in kept:
                overlap = _intersection(candidate.bbox, existing.bbox)
                existing_area = max(_area(existing.bbox), 1.0)
                intersection_fraction = overlap / min(candidate_area, existing_area)
                center_a, center_b = _center(candidate.bbox), _center(existing.bbox)
                center_gap = ((center_a[0] - center_b[0]) ** 2 + (center_a[1] - center_b[1]) ** 2) ** 0.5
                scale = max(candidate.bbox.x2 - candidate.bbox.x1, candidate.bbox.y2 - candidate.bbox.y1, 1.0)
                if intersection_fraction >= 0.82 and center_gap <= 0.20 * scale:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(candidate)
        return tuple(kept)
