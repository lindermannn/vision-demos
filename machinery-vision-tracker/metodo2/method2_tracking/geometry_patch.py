"""Sprint-4 corrections: segmented toe polyline and calibrated-domain clipping."""
from __future__ import annotations
import cv2
import numpy as np

def toe_segments(points,visible,occluded):
    pts=sorted((tuple(map(float,p)) for p in points if int(p[0]) in visible and int(p[0]) not in occluded),key=lambda p:p[0]);out=[];cur=[]
    for p in pts:
        if cur and p[0]-cur[-1][0]>1.01:out.append(cur);cur=[]
        cur.append(p)
    if cur:out.append(cur)
    return [s for s in out if len(s)>=2]

def point_segment_distance(p,a,b):
    p,a,b=np.asarray(p,float),np.asarray(a,float),np.asarray(b,float);v=b-a;den=float(v@v)
    t=0.0 if den==0 else max(0.0,min(1.0,float((p-a)@v)/den))
    return float(np.linalg.norm(p-(a+t*v)))

def distance_to_segments(point,segments):
    values=[point_segment_distance(point,a,b) for s in segments for a,b in zip(s,s[1:])]
    return min(values) if values else None

def inside(point,polygon):return cv2.pointPolygonTest(np.asarray(polygon,np.float32),tuple(map(float,point)),False)>=0

def _cross(a,b):return float(a[0]*b[1]-a[1]*b[0])
def _clip_to_polygon(a,b,polygon):
    a=np.asarray(a,float);b=np.asarray(b,float);direction=b-a;poly=[np.asarray(p,float) for p in polygon];ts=[0.0,1.0]
    for p,q in zip(poly,poly[1:]+poly[:1]):
        edge=q-p;den=_cross(direction,edge)
        if abs(den)<1e-12:continue
        delta=p-a;t=_cross(delta,edge)/den;u=_cross(delta,direction)/den
        if -1e-9<=t<=1+1e-9 and -1e-9<=u<=1+1e-9:ts.append(max(0.0,min(1.0,t)))
    ts=sorted(set(round(x,12) for x in ts));parts=[]
    for lo,hi in zip(ts,ts[1:]):
        mid=a+direction*((lo+hi)/2)
        if inside(mid,polygon):parts.append((tuple(a+direction*lo),tuple(a+direction*hi)))
    return parts

def metric_distance(point,boundary,segments,H,polygon):
    if H is None:return None,{"reason":"homography_unavailable"}
    if not inside(point,polygon):return None,{"reason":"vehicle_outside_calibrated_domain"}
    source_count=sum(max(0,len(s)-1) for s in segments);valid=[];clipped=False
    for s in segments:
        for a,b in zip(s,s[1:]):
            parts=_clip_to_polygon(a,b,polygon);valid.extend(parts)
            if len(parts)!=1 or (parts and (np.linalg.norm(np.asarray(parts[0][0])-a)>1e-7 or np.linalg.norm(np.asarray(parts[0][1])-b)>1e-7)):clipped=True
    if not valid:return None,{"reason":"no_toe_segment_inside_calibrated_domain","partial_domain":source_count>0}
    h=np.asarray(H,float)
    if h.shape!=(3,3) or not np.isfinite(h).all() or abs(np.linalg.det(h))<1e-12:return None,{"reason":"invalid_homography"}
    q=cv2.perspectiveTransform(np.array([[point]],np.float64),h)[0,0];best=float("inf")
    for a,b in valid:
        projected=cv2.perspectiveTransform(np.array([[a,b]],np.float64),h)[0]
        best=min(best,point_segment_distance(q,projected[0],projected[1]))
    return best,{"valid_segments":len(valid),"partial_domain":clipped or len(valid)<source_count,"reason":"partial_calibrated_domain" if clipped else "metric_distance_valid"}
