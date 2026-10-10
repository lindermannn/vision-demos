"""Bounded deterministic multi-object tracker for Method 2."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from .assignment import hungarian_max_cardinality
from .contracts import ObservationStatus, PositionSource, TrackLifecycle, TrackResult, VehicleDetection
from .cost import CostConfig, association_cost
from .motion import MotionState

@dataclass
class _Track:
    track_id: int
    motion: MotionState
    class_name: str
    confidence: float
    hits: int = 1
    misses: int = 0
    age_s: float = 0.0
    age_since_observation_s: float = 0.0
    last_observed_bbox: tuple | None = None
    last_observed_mask: np.ndarray | None = None
    current_bbox: tuple | None = None
    position_source: PositionSource = PositionSource.OBSERVED
    last_diagnostics: dict[str, Any] = field(default_factory=dict)

class VehicleTracker:
    def __init__(self, fps: float, confirm_s: float=.2, max_lost_s: float=1.0,
                 max_cost: float=.88, max_speed_px_s: float=900.0,
                 reid_window_s: float|None=None, cost_config: CostConfig|None=None):
        if not np.isfinite(fps) or fps <= 0: raise ValueError("fps must be finite and > 0")
        self.fps=float(fps)
        self.confirm_hits=max(1, int(np.ceil(confirm_s*self.fps)))
        self.max_lost_s=float(max_lost_s)
        self.reid_window_s=float(max_lost_s if reid_window_s is None else min(reid_window_s,max_lost_s))
        self.max_speed_px_s=float(max_speed_px_s)
        base=cost_config or CostConfig()
        self.cost_config=CostConfig(**{**base.__dict__,"max_cost":float(max_cost)})
        self.tracks: list[_Track]=[]
        self.next_id=1

    def _new_track(self,d:VehicleDetection)->_Track:
        t=_Track(self.next_id,MotionState.from_bbox(d.bbox),d.class_name,float(d.confidence),
                 last_observed_bbox=d.bbox,last_observed_mask=d.mask,current_bbox=d.bbox)
        self.next_id+=1
        return t

    def reset(self)->None:
        self.tracks.clear()

    def update(self,detections:list[VehicleDetection],dt:float|None=None,camera_discontinuity:bool=False)->list[TrackResult]:
        dt=1.0/self.fps if dt is None else float(dt)
        if not np.isfinite(dt) or dt<=0: raise ValueError("dt must be finite and > 0")
        if camera_discontinuity:
            self.reset()
        detections=sorted(detections,key=lambda d:((d.bbox[0]+d.bbox[2])/2,(d.bbox[1]+d.bbox[3])/2,d.class_name,-d.confidence))
        for t in self.tracks:
            t.age_s+=dt; t.age_since_observation_s+=dt
            t.current_bbox=t.motion.predict(dt)
            t.position_source=PositionSource.PREDICTED
            t.misses+=1
            t.last_diagnostics={"reason":"prediction_without_observation"}
        n,m=len(self.tracks),len(detections)
        costs=np.zeros((n,m),float); valid=np.zeros((n,m),bool); details={}
        for i,t in enumerate(self.tracks):
            for j,d in enumerate(detections):
                c,ok,diag=association_cost(t,d,self.cost_config)
                costs[i,j]=c; valid[i,j]=ok and t.age_since_observation_s<=self.reid_window_s+dt
                if not valid[i,j] and ok: diag={**diag,"reason":"reidentification_window_expired"}
                details[(i,j)]=diag
        matches,_,unmatched_dets=hungarian_max_cardinality(costs,valid)
        for i,j in matches:
            t=self.tracks[i]; d=detections[j]
            t.motion.update(d.bbox,dt,self.max_speed_px_s)
            t.current_bbox=d.bbox
            t.last_observed_bbox=d.bbox
            t.last_observed_mask=d.mask
            t.position_source=PositionSource.OBSERVED
            t.confidence=float(d.confidence)
            if d.class_name==t.class_name or t.hits<2: t.class_name=d.class_name
            t.hits+=1; t.misses=0; t.age_since_observation_s=0.0
            t.last_diagnostics={**details[(i,j)],"reason":"accepted_assignment"}
        for j in unmatched_dets:
            self.tracks.append(self._new_track(detections[j]))
        # Physical deletion prevents unbounded memory; removed IDs can never reappear.
        self.tracks=[t for t in self.tracks if t.age_since_observation_s<=self.max_lost_s+1e-12]
        out=[]
        for t in sorted(self.tracks,key=lambda x:x.track_id):
            if t.misses==0:
                lifecycle=TrackLifecycle.CONFIRMED if t.hits>=self.confirm_hits else TrackLifecycle.TENTATIVE
                observed_bbox=t.last_observed_bbox; predicted_bbox=None
                observed_mask=t.last_observed_mask; status=ObservationStatus.OBSERVED
            else:
                lifecycle=TrackLifecycle.LOST
                observed_bbox=None; predicted_bbox=t.current_bbox
                observed_mask=None; status=ObservationStatus.TEMPORAL_ESTIMATE
            out.append(TrackResult(t.track_id,lifecycle,t.current_bbox,t.position_source,t.confidence,t.class_name,
                t.age_s,t.age_since_observation_s,observed_bbox,predicted_bbox,observed_mask,None,status,
                {**t.last_diagnostics,"hits":t.hits,"misses":t.misses,"uncertainty":t.motion.uncertainty}))
        return out
