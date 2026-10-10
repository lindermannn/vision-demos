"""Polynomial global assignment with lexicographic max-cardinality objective."""
from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment

def hungarian_max_cardinality(cost,valid,unmatched_penalty=None):
    cost=np.asarray(cost,dtype=float);valid=np.asarray(valid,dtype=bool)
    if cost.shape!=valid.shape:raise ValueError("cost_valid_shape_mismatch")
    if cost.ndim!=2:raise ValueError("cost must be a matrix")
    n,m=cost.shape
    if n==0 or m==0:return [],list(range(n)),list(range(m))
    if np.any(valid & ~np.isfinite(cost)):raise ValueError("valid costs must be finite")
    finite=np.abs(cost[valid]);scale=float(finite.max())+1.0 if finite.size else 1.0
    penalty=float(unmatched_penalty) if unmatched_penalty is not None else scale*(min(n,m)+1)
    if penalty<=scale*min(n,m):penalty=scale*(min(n,m)+1)
    size=n+m;big=penalty*(size+2)
    matrix=np.full((size,size),big,dtype=float)
    matrix[:n,:m]=np.where(valid,cost,big)
    eps=np.finfo(float).eps*max(1.0,scale)*32
    for i in range(n):
        for j in range(m):
            if valid[i,j]:matrix[i,j]+=eps*((j+1)/(m+1))**(i+1)
        matrix[i,m+i]=penalty
    for j in range(m):matrix[n+j,j]=penalty
    matrix[n:,m:]=0.0
    rows,cols=linear_sum_assignment(matrix)
    matches=[(int(i),int(j)) for i,j in zip(rows,cols) if i<n and j<m and valid[i,j]]
    used_r={i for i,_ in matches};used_c={j for _,j in matches}
    return sorted(matches),[i for i in range(n) if i not in used_r],[j for j in range(m) if j not in used_c]
