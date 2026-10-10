from __future__ import annotations
import cv2
import numpy as np

def structural_signature(frame, size=(160,90)):
    gray=cv2.cvtColor(cv2.resize(frame,size),cv2.COLOR_BGR2GRAY)
    gray=cv2.createCLAHE(2.0,(8,8)).apply(gray)
    gx=cv2.Sobel(gray,cv2.CV_32F,1,0);gy=cv2.Sobel(gray,cv2.CV_32F,0,1)
    value=cv2.GaussianBlur(cv2.magnitude(gx,gy),(9,9),0)
    return (value-value.mean())/(value.std()+1e-6)

def structural_distance(a,b):
    return max(0.0,1.0-float(np.mean(a*b)))

def detect_boundaries(frames, consecutive_threshold=.52, reference_threshold=.34, minimum_length=12):
    if not frames:return [0],[]
    signatures=[structural_signature(f) for f in frames]
    raw=[structural_distance(a,b) for a,b in zip(signatures,signatures[1:])]
    med=float(np.median(raw));mad=float(np.median(np.abs(np.asarray(raw)-med)))
    hard_threshold=max(consecutive_threshold,med+6.0*max(mad,1e-6))
    boundaries=[0]; diagnostics=[]; reference=signatures[0]
    previous=signatures[0]
    for i,current in enumerate(signatures[1:],1):
        dc=structural_distance(previous,current);dr=structural_distance(reference,current)
        cut=dc>=hard_threshold or (dr>=reference_threshold and i-boundaries[-1]>=minimum_length)
        if cut and i-boundaries[-1]>=minimum_length:
            boundaries.append(i);reference=current
        diagnostics.append({"frame_index":i,"consecutive":dc,"reference":dr,"hard_threshold":hard_threshold,"boundary":bool(cut and boundaries[-1]==i)})
        previous=current
    return boundaries,diagnostics

def segment_for(frame,boundaries):
    return max(i for i,b in enumerate(boundaries) if frame>=b)
