"""Causal, explicitly qualified memory-pose writes in canonical c2w coordinates.

This module stages labels; hosts must rebuild their pose-dependent caches or
geometry before committing them. Camera alignment is the host's responsibility.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class MemoryWrite:
    frame_ids: np.ndarray
    poses: np.ndarray
    applied: bool
    reason: str


def stage_memory_write(frame_ids, commanded, measured, *, available_last_frame, quality):
    ids=np.asarray(frame_ids)
    old=np.asarray(commanded)
    if (ids.ndim!=1 or not len(ids) or not np.issubdtype(ids.dtype,np.integer)
        or np.any(ids<0) or np.any(ids[1:]<=ids[:-1])
        or not isinstance(available_last_frame,(int,np.integer))
        or ids[-1]>available_last_frame or old.shape!=(len(ids),4,4)):
        raise ValueError('Memory timestamps must match generated canonical poses')
    if not np.isfinite(old).all():raise ValueError('Invalid existing memory poses')
    def reject(reason):return MemoryWrite(ids.copy(),old.copy(),False,reason)
    if not quality.get('passed',False):return reject('observer_quality_rejected')
    new=np.asarray(measured,float)
    if new.shape!=old.shape or not np.isfinite(new).all():return reject('invalid_measured_poses')
    rotation=new[:,:3,:3]
    if (not np.allclose(new[:,3],[0,0,0,1],atol=1e-6)
        or not np.allclose(rotation.transpose(0,2,1)@rotation,np.eye(3),atol=1e-4)
        or not np.allclose(np.linalg.det(rotation),1,atol=1e-4)):
        return reject('invalid_measured_SE3')
    if 0 in ids and not np.allclose(new[ids==0],old[ids==0],atol=1e-7):
        return reject('fixed_anchor_changed')
    # Preserve exact existing dtype/bytes for the identity control.
    if np.array_equal(new,old):return MemoryWrite(ids.copy(),old.copy(),False,'identity')
    return MemoryWrite(ids.copy(),new.copy(),True,'measured')


def bridge_future(future, measured_boundary, stored_boundary):
    """inv(stored_boundary) @ command = inv(measured_boundary) @ desired."""
    future=np.asarray(future)
    if np.array_equal(measured_boundary,stored_boundary):return future.copy()
    return (np.asarray(stored_boundary)@np.linalg.inv(measured_boundary))@future
