from __future__ import annotations
from dataclasses import dataclass
from statistics import median
import numpy as np

WHEEL_DIAMETER_M = 3.596
DS342B_MINIMUM_M = WHEEL_DIAMETER_M * 0.5
DS351_MINIMUM_M = WHEEL_DIAMETER_M * (2.0 / 3.0)

@dataclass(frozen=True)
class BandMeasurement:
    thickness_px: float | None
    base_row: float | None
    valid_columns: int
    area_px: int
    reason: str = "ok"

def measure_band(mask: np.ndarray, dynamic_mask: np.ndarray | None = None, dilation_px: int = 5) -> BandMeasurement:
    if mask.ndim != 2 or not np.any(mask):
        return BandMeasurement(None, None, 0, 0, "sin_mascara")
    binary = mask.astype(bool)
    blocked = np.zeros_like(binary)
    if dynamic_mask is not None:
        if dynamic_mask.shape != binary.shape:
            raise ValueError("dynamic_mask incompatible")
        if dilation_px:
            import cv2
            k = np.ones((2*dilation_px+1, 2*dilation_px+1), np.uint8)
            blocked = cv2.dilate(dynamic_mask.astype(np.uint8), k).astype(bool)
        else:
            blocked = dynamic_mask.astype(bool)
    thicknesses=[]; bases=[]
    for x in np.flatnonzero(binary.any(axis=0)):
        ys=np.flatnonzero(binary[:,x])
        if ys.size and not np.any(blocked[ys,x]):
            thicknesses.append(float(ys[-1]-ys[0]+1)); bases.append(float(ys[-1]))
    if not thicknesses:
        return BandMeasurement(None, None, 0, int(binary.sum()), "soporte_insuficiente")
    return BandMeasurement(float(median(thicknesses)), float(median(bases)), len(thicknesses), int(binary.sum()))

def height_metres(thickness_px: float, base_row: float, horizon_y: float, camera_height_m: float, margin_px: float=25.0):
    denominator=base_row-horizon_y
    if denominator < margin_px:
        return None, "pretil_cerca_del_horizonte"
    if camera_height_m <= 0 or not np.isfinite([thickness_px,base_row,horizon_y,camera_height_m]).all():
        return None, "sin_modelo_de_camara"
    return thickness_px*camera_height_m/denominator, "ok"

def height_interval(thickness_px, base_row, horizon_low, horizon_high, camera_height_m, camera_uncertainty_fraction):
    hs=[]
    for horizon in (horizon_low,horizon_high):
        for scale in (1-camera_uncertainty_fraction,1+camera_uncertainty_fraction):
            value,reason=height_metres(thickness_px,base_row,horizon,camera_height_m*scale,0)
            if value is not None: hs.append(value)
    return (min(hs),max(hs)) if hs else (None,None)

def verdict(low: float, high: float, minimum: float=DS342B_MINIMUM_M) -> str:
    if low >= minimum:return "cumple"
    if high < minimum:return "por_debajo"
    return "no_concluyente"

def classify_frame(measurement, lighting, anchor_support, anchor_area, history, cfg):
    if measurement.reason != "ok": return "N/D", measurement.reason
    if anchor_support and measurement.valid_columns < cfg["minimum_support_ratio"]*anchor_support:return "N/D","soporte_insuficiente"
    if anchor_area:
        ratio=measurement.area_px/anchor_area
        if not cfg["area_ratio_range"][0] <= ratio <= cfg["area_ratio_range"][1]:return "N/D","salto_de_area"
    if history:
        centre=float(np.median(history[-cfg["rolling_window"]:]))
        if abs(measurement.thickness_px-centre)/max(centre,1) > cfg["thickness_jump_ratio"]:return "N/D","salto_de_grosor"
    if lighting != "dia":return "memoria","mascara_propagada_sin_observacion_diurna"
    return "medida","ok"
