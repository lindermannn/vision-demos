"""Safe temporal fusion; invalid or stale geometry is never published."""
from __future__ import annotations
from .contracts import DistanceRecord, FusionResult, ObservationStatus

def _status(value):
    try:return ObservationStatus(value)
    except (ValueError,TypeError):return ObservationStatus.UNKNOWN

def fuse(frame_index,tracks,berm,distance_by_track,calibration_status=None,timestamp=None,distance_ttl_s=.15):
    timestamp=float(frame_index if timestamp is None else timestamp)
    results=[]
    for track in tracks:
        record=distance_by_track.get(track.track_id)
        reasons=[];px=None;metres=None;cal="unavailable";dist_conf=0.0
        if not isinstance(record,DistanceRecord):
            reasons.append("distance_missing_or_untyped")
        elif record.track_id!=track.track_id:
            reasons.append("track_id_mismatch")
        elif record.frame_index!=frame_index:
            reasons.append("distance_frame_mismatch")
        elif timestamp-record.timestamp>distance_ttl_s or timestamp<record.timestamp:
            reasons.append("distance_stale_or_future")
        elif _status(record.observation_status) is not ObservationStatus.OBSERVED:
            reasons.append("distance_not_current_observation")
        elif record.source=="historical_reference":
            reasons.append("distance_from_historical_reference")
        else:
            cal=record.calibration_status
            if cal not in {"calibrated","estimated","pixel_only","unavailable"}:
                reasons.append("calibration_status_invalid")
            else:
                px=record.distance_px
                metres=record.distance_m if cal in {"calibrated","estimated"} else None
                if record.distance_m is not None and metres is None:reasons.append("metric_distance_suppressed_without_metric_calibration")
                dist_conf=record.confidence
        berm_status=_status(berm.observation_status)
        if berm_status is not ObservationStatus.OBSERVED:
            px=None;metres=None;reasons.append("berm_not_current_observation")
        reason="accepted_current_distance" if not reasons else ";".join(dict.fromkeys(reasons))
        results.append(FusionResult(frame_index,timestamp,track.track_id,track.state,track.position_source,track.bbox,
            track.observed_mask if track.position_source.value=="observed" else track.propagated_mask,
            berm_status,float(berm.age_since_observation_s or 0.0),px,metres,cal,
            min(float(track.confidence),float(berm.confidence),float(dist_conf)) if px is not None else 0.0,
            reason,list(dict.fromkeys(reasons))))
    return results
