"""Finite-TTL temporal berm memory; provenance is never promoted."""
from __future__ import annotations
import numpy as np
from .contracts import ObservationStatus, TemporalBermState

class BermMemory:
    def __init__(self,ttl_s=2.0,min_confidence=.7,temporal_fraction=.5):
        if ttl_s<=0:raise ValueError("ttl_s must be > 0")
        self.ttl_s=float(ttl_s);self.min_confidence=float(min_confidence);self.temporal_fraction=float(temporal_fraction);self.last=None
    def invalidate(self,reason="invalidated"):
        self.last=None
        return TemporalBermState(ObservationStatus.UNKNOWN,None,0.0,None,reason,{})
    def update(self,mask,confidence,status,dt,compatible=True,scene_changed=False,camera_changed=False):
        if not np.isfinite(dt) or dt<=0:raise ValueError("dt must be finite and > 0")
        if scene_changed or camera_changed:return self.invalidate("camera_or_scene_invalidated")
        status=ObservationStatus(status)
        if status is ObservationStatus.OBSERVED and mask is not None and confidence>=self.min_confidence:
            candidate=np.asarray(mask)
            if candidate.ndim!=2 or candidate.size==0 or not candidate.astype(bool).any():
                return self.invalidate("invalid_observed_mask")
            if compatible:
                self.last={"mask":candidate.copy(),"confidence":float(confidence),"age":0.0}
                return TemporalBermState(ObservationStatus.OBSERVED,candidate.copy(),float(confidence),0.0,"validated_observation",{})
            if self.last is None:return self.invalidate("incompatible_candidate_rejected")
        if self.last is None:return TemporalBermState(ObservationStatus.UNKNOWN,None,0.0,None,"no_valid_reference",{})
        self.last["age"]+=dt;age=self.last["age"]
        if age>self.ttl_s:return self.invalidate("ttl_expired")
        conf=self.last["confidence"]*max(0.0,1.0-age/self.ttl_s)
        if age<=self.ttl_s*self.temporal_fraction:
            return TemporalBermState(ObservationStatus.TEMPORAL_ESTIMATE,self.last["mask"].copy(),conf,age,"propagated_not_current_measurement",{})
        return TemporalBermState(ObservationStatus.HISTORICAL_REFERENCE,self.last["mask"].copy(),conf,age,"historical_not_current_measurement",{})
