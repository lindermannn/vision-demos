"""Explicit non-mutating adapters for Sprint 2 and Sprint 3 contracts."""
from __future__ import annotations
from collections.abc import Mapping
import numpy as np
from .contracts import ObservationStatus,VehicleDetection

def _get(item,name,default=None):return item.get(name,default) if isinstance(item,Mapping) else getattr(item,name,default)

def vehicle_from_sprint2(item):
    bbox=_get(item,"bbox")
    if bbox is None:raise ValueError("sprint2_adapter:missing_bbox")
    return VehicleDetection(tuple(float(v) for v in bbox),_get(item,"mask"),str(_get(item,"class_name","vehicle")),float(_get(item,"confidence",0.0)),"sprint2")

def berm_from_sprint3(item):
    mask=_get(item,"mask")
    status=_get(item,"observation_status","unknown")
    try:status=ObservationStatus(status)
    except ValueError:status=ObservationStatus.UNKNOWN
    return {"mask":None if mask is None else np.asarray(mask),"crest_points":list(_get(item,"crest_points",[])),"toe_points":list(_get(item,"toe_points",[])),"visible_columns":set(_get(item,"visible_columns",[])),"occluded_columns":set(_get(item,"occluded_columns",[])),"ignore_columns":set(_get(item,"ignore_columns",[])),"observation_status":status,"confidence":float(_get(item,"confidence",0.0)),"source":"sprint3"}
