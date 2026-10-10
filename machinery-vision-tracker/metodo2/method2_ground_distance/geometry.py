from __future__ import annotations
from dataclasses import dataclass
from math import hypot,tan,radians
import numpy as np

@dataclass(frozen=True)
class Reference:
 height_m:float; length_m:float; width_m:float; uncertainty_fraction:float
@dataclass(frozen=True)
class Sample:
 track_id:int; class_name:str; contact_x:float; contact_y:float; apparent_height_px:float; complete:bool=True
@dataclass(frozen=True)
class CameraModel:
 horizon_y:float; camera_height_m:float; samples:int; row_spread:float; residual_ratio:float; valid:bool; reason:str="ok"
@dataclass(frozen=True)
class GroundPose:
 track_id:int; class_name:str; x:float; z:float; consistency:float; valid:bool; reason:str="ok"
@dataclass(frozen=True)
class Distance:
 a:int;b:int;nominal:float|None;low:float|None;high:float|None;state:str;uncertain:bool=False

def fit_camera(samples,refs,frame_height,min_samples=12,min_spread_ratio=.06,max_residual=.35):
 pts=[s for s in samples if s.complete and s.class_name in refs and s.apparent_height_px>0]
 if len(pts)<min_samples:return CameraModel(0,0,len(pts),0,1,False,"sin_modelo_de_camara")
 y=np.array([s.contact_y for s in pts]);q=np.array([s.apparent_height_px/refs[s.class_name].height_m for s in pts])
 spread=float(np.ptp(y))
 if spread<min_spread_ratio*frame_height:return CameraModel(0,0,len(pts),spread,1,False,"dispersion_de_filas_insuficiente")
 keep=np.ones(len(y),bool)
 for _ in range(3):
  a,b=np.polyfit(y[keep],q[keep],1);res=np.abs(q-(a*y+b));scale=max(float(np.median(q[keep])),1e-6);keep=res<=max(np.quantile(res,.8),.02*scale)
 if a<=0:return CameraModel(0,0,len(pts),spread,1,False,"pendiente_invalida")
 yh=-b/a; hc=1/a; rr=float(np.median(np.abs(q-(a*y+b)))/max(np.median(q),1e-6))
 if yh>=float(y.min())-.01*frame_height or rr>max_residual:return CameraModel(yh,hc,len(pts),spread,rr,False,"compuerta_de_plausibilidad")
 return CameraModel(float(yh),float(hc),len(pts),spread,rr,True)

def focal_px(width,fov_deg):return (width/2)/tan(radians(fov_deg)/2)

def ground_pose(sample,ref,model,width,fov_deg=70,max_discrepancy=.35,median_height_px=None):
 if not sample.complete:return GroundPose(sample.track_id,sample.class_name,0,0,0,False,"deteccion_incompleta")
 if not model.valid:return GroundPose(sample.track_id,sample.class_name,0,0,0,False,"sin_modelo_de_camara")
 den=sample.contact_y-model.horizon_y
 if den<=0:return GroundPose(sample.track_id,sample.class_name,0,0,0,False,"geometria_inconsistente")
 f=focal_px(width,fov_deg);z=f*model.camera_height_m/den;x=(sample.contact_x-width/2)*z/f
 hz=median_height_px or sample.apparent_height_px;z_size=f*ref.height_m/max(hz,1e-6);r=abs(z_size-z)/max(z,1e-6);cons=1-min(1,r/max_discrepancy)
 return GroundPose(sample.track_id,sample.class_name,x,z,cons,r<=max_discrepancy,"ok" if r<=max_discrepancy else "geometria_inconsistente")

def clearance(a,b,refs,uncertainty_ratio=1.7):
 if not a.valid or not b.valid:return Distance(a.track_id,b.track_id,None,None,None,a.reason if not a.valid else b.reason)
 # Heading-free conservative footprint: enclosing circle, contact is near edge.
 ra=refs[a.class_name].length_m/2;rb=refs[b.class_name].length_m/2
 nominal=max(0,hypot(a.x-b.x,a.z-b.z)-ra-rb)
 ua=refs[a.class_name].uncertainty_fraction;ub=refs[b.class_name].uncertainty_fraction;u=max(ua,ub)
 low=max(0,nominal*(1-u));high=nominal*(1+u);state="red" if low<10 else "orange" if low<=20 else "green"
 return Distance(a.track_id,b.track_id,nominal,low,high,state,(high/max(low,1e-6))>uncertainty_ratio)
