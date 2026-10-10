"""Configurable multicriterion association costs and gates."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional
import numpy as np

from .contracts import BBox, VehicleDetection


def bbox_iou(a: BBox, b: BBox) -> float:
    ix1, iy1, ix2, iy2 = max(a[0],b[0]), max(a[1],b[1]), min(a[2],b[2]), min(a[3],b[3])
    inter = max(0.0, ix2-ix1)*max(0.0, iy2-iy1)
    aa=max(0.0,a[2]-a[0])*max(0.0,a[3]-a[1]); bb=max(0.0,b[2]-b[0])*max(0.0,b[3]-b[1])
    union=aa+bb-inter
    return inter/union if union>0 else 0.0


def mask_iou(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None:
        return None
    aa=np.asarray(a); bb=np.asarray(b)
    if aa.ndim!=2 or bb.ndim!=2 or aa.shape!=bb.shape or aa.size==0 or bb.size==0:
        return None
    aa=aa.astype(bool); bb=bb.astype(bool)
    if not aa.any() or not bb.any():
        return None
    union=np.logical_or(aa,bb).sum()
    return float(np.logical_and(aa,bb).sum()/union) if union else None


def _center(b: BBox) -> tuple[float,float]:
    return ((b[0]+b[2])/2.0,(b[1]+b[3])/2.0)


@dataclass(frozen=True)
class CostConfig:
    bbox_iou_weight: float = 0.25
    mask_iou_weight: float = 0.25
    motion_weight: float = 0.32
    size_weight: float = 0.08
    class_weight: float = 0.04
    age_weight: float = 0.03
    confidence_weight: float = 0.03
    base_gate_px: float = 45.0
    uncertainty_gate_factor: float = 2.5
    max_size_log_ratio: float = 1.4
    max_cost: float = 0.92


def association_cost(track, detection: VehicleDetection, cfg: CostConfig) -> tuple[float,bool,dict]:
    pb=track.motion.bbox
    biou=bbox_iou(pb,detection.bbox)
    miou=mask_iou(track.last_observed_mask,detection.mask)
    pc,dc=_center(pb),_center(detection.bbox)
    center_distance=math.hypot(pc[0]-dc[0],pc[1]-dc[1])
    gate_radius=cfg.base_gate_px+cfg.uncertainty_gate_factor*math.sqrt(max(0.0,track.motion.uncertainty))
    motion_cost=min(2.0,center_distance/max(1e-6,gate_radius))
    pw=max(1.0,pb[2]-pb[0]); ph=max(1.0,pb[3]-pb[1])
    dw=max(1.0,detection.bbox[2]-detection.bbox[0]); dh=max(1.0,detection.bbox[3]-detection.bbox[1])
    size_delta=max(abs(math.log(dw/pw)),abs(math.log(dh/ph)))
    size_cost=min(1.0,size_delta/max(cfg.max_size_log_ratio,1e-6))
    class_cost=0.0 if detection.class_name==track.class_name else 1.0
    age_cost=min(1.0,track.age_since_observation_s/max(getattr(track,"reid_window_s",1.0),1e-6))
    conf_cost=1.0-float(detection.confidence)
    visual_iou=miou if miou is not None else biou
    weights=(cfg.bbox_iou_weight,cfg.mask_iou_weight if miou is not None else 0.0,cfg.motion_weight,cfg.size_weight,cfg.class_weight,cfg.age_weight,cfg.confidence_weight)
    values=(1.0-biou,1.0-visual_iou,motion_cost,size_cost,class_cost,age_cost,conf_cost)
    denom=sum(weights)
    total=sum(w*v for w,v in zip(weights,values))/max(denom,1e-9)
    gates=[]
    if center_distance>gate_radius and biou<=0 and (miou is None or miou<=0): gates.append('motion_gate')
    if size_delta>cfg.max_size_log_ratio: gates.append('size_gate')
    valid=not gates and total<=cfg.max_cost
    diagnostics={
        'total_cost':float(total),'bbox_iou':biou,'mask_iou':miou,
        'center_distance_px':center_distance,'motion_gate_px':gate_radius,
        'motion_cost':motion_cost,'size_cost':size_cost,'class_penalty':class_cost,
        'age_cost':age_cost,'detection_confidence_cost':conf_cost,
        'gates_applied':gates,'decision':'accepted' if valid else 'rejected'
    }
    return float(total),valid,diagnostics
