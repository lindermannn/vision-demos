"""Strict controlled JSON/YAML calibration loader."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import yaml

_VALID={".json",".yaml",".yml"}

def _matrix(value,name,shape):
    try:a=np.asarray(value,dtype=float)
    except (TypeError,ValueError) as exc:raise ValueError(f"calibration_schema_error:{name}_numeric") from exc
    if a.shape!=shape or not np.isfinite(a).all():raise ValueError(f"calibration_schema_error:{name}_{shape[0]}x{shape[1]}_finite")
    return a

def _points(value,name):
    try:a=np.asarray(value,dtype=float)
    except (TypeError,ValueError) as exc:raise ValueError(f"calibration_schema_error:{name}_numeric") from exc
    if a.ndim!=2 or a.shape[1]!=2:raise ValueError(f"calibration_schema_error:{name}_Nx2")
    if len(a)<4:raise ValueError("calibration_schema_error:four_points_required")
    if not np.isfinite(a).all():raise ValueError(f"calibration_schema_error:{name}_finite")
    centered=a-a.mean(axis=0)
    if np.linalg.matrix_rank(centered)<2:raise ValueError(f"calibration_schema_error:{name}_degenerate")
    return a

def load(path:Path|str)->dict:
    path=Path(path)
    suffix=path.suffix.lower()
    if suffix not in _VALID:raise ValueError("calibration_extension_error:only_json_yaml_yml")
    try:
        raw=path.read_text(encoding="utf8")
        data=json.loads(raw) if suffix==".json" else yaml.safe_load(raw)
    except (OSError,json.JSONDecodeError,yaml.YAMLError) as exc:
        raise ValueError(f"calibration_load_error:{exc}") from exc
    if not isinstance(data,dict):raise ValueError("calibration_schema_error:object_required")
    for name in ("image_points","ground_points","source"):
        if name not in data:raise ValueError(f"calibration_schema_error:missing_{name}")
    image=_points(data["image_points"],"image_points"); ground=_points(data["ground_points"],"ground_points")
    if len(image)!=len(ground):raise ValueError("calibration_schema_error:correspondence_count")
    confidence=data.get("confidence",0.0)
    if isinstance(confidence,bool):raise ValueError("calibration_schema_error:confidence_numeric")
    try:confidence=float(confidence)
    except (TypeError,ValueError) as exc:raise ValueError("calibration_schema_error:confidence_numeric") from exc
    if not np.isfinite(confidence) or not 0<=confidence<=1:raise ValueError("calibration_schema_error:confidence_range")
    if "intrinsic_matrix" in data:_matrix(data["intrinsic_matrix"],"intrinsic_matrix",(3,3))
    if "homography" in data:
        h=_matrix(data["homography"],"homography",(3,3))
        if abs(np.linalg.det(h))<1e-12:raise ValueError("calibration_schema_error:homography_degenerate")
    source=data["source"]
    if not isinstance(source,str) or not source.strip():raise ValueError("calibration_schema_error:source_string")
    source=source.strip()
    if source in {"operator_survey","measured_correspondences","surveyed_ground_control"} and confidence>0:
        status="calibrated"
    elif source in {"estimated_reference","approximate_reference","estimated"} and confidence>0:
        status="estimated"
    else:status="unavailable"
    return {**data,"image_points":image.tolist(),"ground_points":ground.tolist(),"confidence":confidence,"calibration_status":status}
