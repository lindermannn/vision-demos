from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
import cv2
import numpy as np

COLORS={"red":(0,0,255),"orange":(0,170,255),"green":(50,200,50),"unknown":(180,180,180)}

@dataclass(frozen=True)
class VehicleOverlay:
 track_id:int;class_name:str;bbox:tuple[int,int,int,int];state:str;label:str
@dataclass(frozen=True)
class PairOverlay:
 a:int;b:int;low:float;nominal:float;high:float;state:str

def finite(value):
 try:return np.isfinite(float(value))
 except (TypeError,ValueError):return False

def build_overlay_plan(track_rows,distance_rows,proximity_limit=20.0):
 observed=[r for r in track_rows if r.get("position_source")=="observed"]
 by_track={int(r["track_id"]):r for r in observed};nearest={};geometry_tracks=set()
 for r in distance_rows:
  if not all(finite(r.get(k)) for k in ("distance_m_estimated","distance_m_low","distance_m_high")):continue
  a=int(r["track_a"]);b=int(r["track_b"]);low=float(r["distance_m_low"]);geometry_tracks|={a,b}
  for own,other in ((a,b),(b,a)):
   if own not in nearest or low<nearest[own][0]:nearest[own]=(low,other,r)
 vehicles=[]
 for tid,r in sorted(by_track.items()):
  candidate=nearest.get(tid);state=candidate[2].get("state","unknown") if candidate else ("green" if tid in geometry_tracks else "unknown")
  label=f"{r['class_name']} ID {tid}"
  if candidate and candidate[0]<=proximity_limit:
   d=candidate[2];label+=f" ~{float(d['distance_m_estimated']):.0f} m [{float(d['distance_m_low']):.0f}–{float(d['distance_m_high']):.0f}]"
  bbox=tuple(int(round(float(r[k]))) for k in ("x1","y1","x2","y2"));vehicles.append(VehicleOverlay(tid,r["class_name"],bbox,state if state in COLORS else "unknown",label))
 pairs={}
 for tid,candidate in nearest.items():
  low,other,d=candidate
  if low<=proximity_limit and tid in by_track and other in by_track:
   key=tuple(sorted((tid,other)))
   pairs[key]=PairOverlay(key[0],key[1],low,float(d["distance_m_estimated"]),float(d["distance_m_high"]),d.get("state","unknown"))
 return vehicles,list(pairs.values())

def height_header(row,segment):
 if row.get("state")=="N/D" or not segment or segment.get("estimate_m") is None:return f"PRETIL N/D | {row.get('reason') or 'sin_dato'}"
 text=f"PRETIL ~{segment['estimate_m']:.1f} m [{segment['low_m']:.1f}–{segment['high_m']:.1f}] | DS 132 art. 342 b: {str(segment['verdict']).replace('_',' ')}"
 if row.get("state")=="memoria":text+=" | estimada por memoria"
 return text

def draw_berm(frame,mask,vehicle_rows,row,dilation_px=5):
 if mask is None or row.get("state")=="N/D":return 0
 h,w=frame.shape[:2]
 if mask.shape!=(h,w):mask=cv2.resize(mask,(w,h),interpolation=cv2.INTER_NEAREST)
 blocked=np.zeros((h,w),np.uint8)
 for r in vehicle_rows:
  if r.get("position_source")=="observed":
   x1,y1,x2,y2=[int(round(float(r[k]))) for k in ("x1","y1","x2","y2")];cv2.rectangle(blocked,(x1,y1),(x2,y2),1,-1)
 if dilation_px:blocked=cv2.dilate(blocked,np.ones((2*dilation_px+1,2*dilation_px+1),np.uint8))
 valid=(mask>127)&~blocked.astype(bool);top=[];bottom=[]
 for x in np.flatnonzero(valid.any(axis=0)):
  ys=np.flatnonzero(valid[:,x]);top.append((int(x),int(ys[0])));bottom.append((int(x),int(ys[-1])))
 if len(top)>1:
  cv2.polylines(frame,[np.asarray(top,np.int32)],False,(0,180,255),2,cv2.LINE_AA);cv2.polylines(frame,[np.asarray(bottom,np.int32)],False,(255,180,0),2,cv2.LINE_AA)
 return len(top)

def center(bbox):return ((bbox[0]+bbox[2])//2,(bbox[1]+bbox[3])//2)
def draw_overlay(frame,vehicle_rows,distance_rows,height_row,segment,mask):
 vehicles,pairs=build_overlay_plan(vehicle_rows,distance_rows);draw_berm(frame,mask,vehicle_rows,height_row)
 lookup={v.track_id:v for v in vehicles}
 for pair in pairs:
  cv2.line(frame,center(lookup[pair.a].bbox),center(lookup[pair.b].bbox),COLORS.get(pair.state,COLORS["unknown"]),2,cv2.LINE_AA)
 for v in vehicles:
  x1,y1,x2,y2=v.bbox;color=COLORS[v.state];cv2.rectangle(frame,(x1,y1),(x2,y2),color,3);cv2.putText(frame,v.label,(x1,max(105,y1-7)),cv2.FONT_HERSHEY_SIMPLEX,.52,color,2,cv2.LINE_AA)
 h,w=frame.shape[:2];cv2.rectangle(frame,(8,8),(min(w-8,1510),94),(5,10,8),-1);cv2.putText(frame,height_header(height_row,segment),(18,36),0,.58,(255,255,255),2,cv2.LINE_AA);cv2.putText(frame,"rojo <10 m | ambar 10-20 m | verde >20 m | sin geometria N/D",(18,62),0,.49,(80,220,220),1,cv2.LINE_AA);cv2.putText(frame,"metros estimados con modelo de camara y FOV supuesto",(18,84),0,.45,(140,200,150),1,cv2.LINE_AA)
 return {"vehicles":len(vehicles),"pairs":len(pairs),"header":height_header(height_row,segment)}
