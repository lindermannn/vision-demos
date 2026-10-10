"""Background-only camera change detector."""
from __future__ import annotations
import cv2
import numpy as np

def _gray(frame):
    a=np.asarray(frame)
    if a.ndim==2:return a.astype(np.uint8)
    if a.ndim==3 and a.shape[2] in (3,4):return cv2.cvtColor(a[:,:,:3],cv2.COLOR_BGR2GRAY)
    raise ValueError("frame must be HxW or HxWx3/4")

def detect(previous,current,dynamic_mask=None):
    if previous is None:
        return {"change_type":"unreliable","scene_cut":False,"geometric_change":False,"illumination_change":False,"translation":False,"rotation":False,"zoom":False,"unreliable":True,"confidence":0.0}
    a,b=_gray(previous),_gray(current)
    if a.shape!=b.shape: raise ValueError("frame dimensions must match")
    static=np.full(a.shape,255,np.uint8)
    if dynamic_mask is not None:
        dm=np.asarray(dynamic_mask)
        if dm.ndim!=2 or dm.shape!=a.shape or dm.dtype.kind not in "buif":
            raise ValueError("dynamic_mask must be numeric/bool HxW matching the frame")
        static[np.asarray(dm,dtype=bool)]=0
    valid=static>0
    if valid.sum()<max(64,int(a.size*.05)):
        return {"change_type":"unreliable","scene_cut":False,"geometric_change":False,"illumination_change":False,"translation":False,"rotation":False,"zoom":False,"unreliable":True,"confidence":0.0,"diagnostics":["insufficient_static_background"]}
    delta=b.astype(np.float32)-a.astype(np.float32)
    mean_abs=float(np.mean(np.abs(delta[valid]))); delta_std=float(np.std(delta[valid]))
    illumination=mean_abs>25 and delta_std<15
    orb=cv2.ORB_create(700)
    ka,da=orb.detectAndCompute(a,static); kb,db=orb.detectAndCompute(b,static)
    if da is None or db is None:
        cut=mean_abs>75 and not illumination
        return {"change_type":"illumination_change" if illumination else ("scene_cut" if cut else "unreliable"),"scene_cut":cut,"geometric_change":False,"illumination_change":illumination,"translation":False,"rotation":False,"zoom":False,"unreliable":not(illumination or cut),"confidence":0.45 if illumination else 0.0}
    raw=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True).match(da,db)
    good=sorted((m for m in raw if m.distance<58),key=lambda match: match.distance)[:350]
    good_count=len(good)
    if good_count<8:
        cut=mean_abs>80 and not illumination
        return {"change_type":"illumination_change" if illumination else ("scene_cut" if cut else "unreliable"),"scene_cut":cut,"geometric_change":False,"illumination_change":illumination,"translation":False,"rotation":False,"zoom":False,"unreliable":not(illumination or cut),"confidence":min(0.45,good_count/16.0)}
    pa=np.float32([ka[m.queryIdx].pt for m in good]); pb=np.float32([kb[m.trainIdx].pt for m in good])
    M,inliers=cv2.estimateAffinePartial2D(pa,pb,method=cv2.RANSAC,ransacReprojThreshold=3.0)
    if M is None:
        return {"change_type":"unreliable","scene_cut":False,"geometric_change":False,"illumination_change":illumination,"translation":False,"rotation":False,"zoom":False,"unreliable":True,"confidence":0.0}
    inlier_ratio=float(inliers.mean()) if inliers is not None else 0.0
    scale=float(np.hypot(M[0,0],M[1,0])); angle=float(np.degrees(np.arctan2(M[1,0],M[0,0]))); shift=float(np.hypot(M[0,2],M[1,2]))
    translation=shift>6.0; rotation=abs(angle)>1.5; zoom=abs(scale-1.0)>.025
    geometric=(translation or rotation or zoom) and inlier_ratio>=.35
    cut=(inlier_ratio<.18 and mean_abs>65 and not illumination)
    kind="scene_cut" if cut else "zoom" if zoom and geometric else "rotation" if rotation and geometric else "translation" if translation and geometric else "illumination_change" if illumination else "static"
    return {"change_type":kind,"scene_cut":cut,"geometric_change":geometric,"illumination_change":illumination,"translation":translation and geometric,"rotation":rotation and geometric,"zoom":zoom and geometric,"unreliable":False,"confidence":inlier_ratio,"transform":M.tolist(),"scale":scale,"rotation_deg":angle,"translation_px":shift,"static_fraction":float(valid.mean())}

def apply_change_policy(change,tracker=None,berm_memory=None,compensated=False):
    """Apply invalidation policy and report whether a prior homography remains valid."""
    invalidate=bool(change.get("scene_cut")) or (bool(change.get("geometric_change")) and not compensated)
    if invalidate:
        if tracker is not None:tracker.reset()
        if berm_memory is not None:berm_memory.invalidate("camera_geometry_invalidated")
    return {"tracks_valid":not invalidate,"berm_reference_valid":not invalidate,"homography_valid":not invalidate,"reason":"camera_geometry_invalidated" if invalidate else "geometry_unchanged_or_compensated"}
