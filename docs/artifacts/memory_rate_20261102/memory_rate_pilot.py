"""Experimental translation-rate continuation after a qualified memory write.

Inputs must already share a supported native geometry gauge. This module does
not identify units, estimate generator gain, or guarantee generated motion.
"""
from dataclasses import dataclass
import numpy as np
from .geometry import poses


@dataclass(frozen=True)
class RateContinuation:
    commands: np.ndarray
    receipt: dict


def bounded_memory_continuation(plan, measured, frame_ids, *, native_stride):
    """Replan the remaining translation from the actually committed boundary.

    Speed is capped by max(original plan speed, incoming measured speed).
    Acceleration permits stopping over one native camera interval. Residual is
    spread over the remaining chunk. Progress uses its planned net direction;
    chunks with reversing native target intervals abstain. Rotation targets are
    preserved. All values
    are in one native geometry gauge; no motion-response scale is fitted.
    """
    target = poses(plan).copy()
    history = poses(measured)
    ids = np.asarray(frame_ids)
    if (target.ndim != 3 or history.ndim != 3 or ids.shape != (len(history),)
            or not np.issubdtype(ids.dtype, np.integer) or len(ids) < 2
            or np.any(np.diff(ids) <= 0) or ids[0] < 0 or ids[-1] >= len(target)-1
            or isinstance(native_stride, bool) or not isinstance(native_stride, (int, np.integer))
            or native_stride < 1):
        raise ValueError('Ordered emitted poses and a positive native camera stride required')
    boundary = int(ids[-1]); horizon = len(target)-1-boundary
    prior = np.flatnonzero(ids <= boundary-native_stride)
    if not len(prior):
        raise ValueError('An emitted native interval is required for incoming velocity')
    previous_index = int(prior[-1]); duration = boundary-int(ids[previous_index])
    velocity = (history[-1, :3, 3]-history[previous_index, :3, 3])/duration
    planned_velocity = np.diff(target[:, :3, 3], axis=0)
    planned_max = float(np.linalg.norm(planned_velocity, axis=1).max())
    incoming_speed = float(np.linalg.norm(velocity))
    ceiling = max(planned_max, incoming_speed)
    acceleration = ceiling/native_stride
    receipt = dict(boundary_frame=boundary, horizon_frames=horizon,
                   incoming_interval_frames=duration, native_stride=int(native_stride),
                   planned_speed_max=planned_max, incoming_velocity=velocity.tolist(),
                   speed_ceiling=ceiling, acceleration_ceiling=acceleration,
                   history_unchanged=True, rotation_targets_unchanged=True,
                   physical_unit_calibration='required from host; not estimated here',
                   response_gain='not fitted', applied=False, reason=None, steps=[])
    if np.array_equal(history, target[ids]):
        receipt['reason'] = 'exact_identity_no_intervention'
        return RateContinuation(target, receipt)
    output = target.copy(); position = history[-1, :3, 3].copy()
    tolerance = max(ceiling*1e-10, np.finfo(float).tiny)
    heading = target[-1, :3, 3]-target[boundary, :3, 3]
    heading_norm = float(np.linalg.norm(heading))
    direction = heading/heading_norm if heading_norm > 0 else None
    native_ids = np.unique(np.r_[np.arange(boundary, len(target), native_stride), len(target)-1])
    native_steps = np.diff(target[native_ids, :3, 3], axis=0)
    if ((direction is None and np.any(np.linalg.norm(native_steps, axis=1) > tolerance))
            or (direction is not None and np.any(native_steps@direction < -tolerance))):
        receipt['reason'] = 'planned_native_chunk_not_forward_monotone'
        return RateContinuation(target, receipt)
    receipt['forward_axis'] = direction.tolist() if direction is not None else None

    for frame in range(boundary+1, len(target)):
        tangent = planned_velocity[frame-1]
        desired = tangent+(target[frame, :3, 3]-(position+tangent))/horizon
        if direction is not None:
            desired -= min(0., float(desired@direction))*direction
        norm = float(np.linalg.norm(desired))
        if norm > ceiling:
            desired *= ceiling/norm
        delta = desired-velocity
        change = float(np.linalg.norm(delta))
        weight = min(1., acceleration/change) if change > 0 else 1.
        following = velocity+weight*delta
        # A curved plan or already-reversed prefix may defeat this simple
        # transition. Do not silently violate acceleration or claim that the
        # generator is uncontrollable. Abstain from the whole candidate.
        if direction is not None and following@direction < -tolerance:
            receipt.update(reason='bounded_transition_cannot_preserve_forward_progress',
                           rejected_at_frame=frame, steps=[])
            return RateContinuation(target, receipt)
        position = position+following
        output[frame, :3, 3] = position
        receipt['steps'].append(dict(frame=frame, velocity=following.tolist(),
                                     speed=float(np.linalg.norm(following)),
                                     acceleration=float(np.linalg.norm(following-velocity)),
                                     forward_component=float(following@direction) if direction is not None else None,
                                     residual=(target[frame, :3, 3]-position).tolist()))
        velocity = following
    receipt.update(applied=True, reason='bounded_translation_continuation',
                   endpoint_residual=(target[-1, :3, 3]-output[-1, :3, 3]).tolist())
    np.testing.assert_array_equal(output[:boundary+1], target[:boundary+1])
    np.testing.assert_array_equal(output[:, :3, :3], target[:, :3, :3])
    return RateContinuation(output, receipt)
