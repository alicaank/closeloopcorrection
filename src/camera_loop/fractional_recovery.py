"""Experimental chunk-fraction recovery in an already qualified geometry gauge.

The requested fraction describes command geometry, not learned plant gain or
guaranteed generated recovery. Native adapters must commit/bridge memory and
qualify their encoded inputs before using this policy.
"""
import numpy as np
from .geometry import poses
from .memory_rate import RateContinuation


def fractional_memory_continuation(plan, measured, frame_ids, *, native_stride,
                                   recovery_fraction, speed_multiplier):
    """Request a fixed fraction of the boundary residual over this chunk.

    Initially desired increments are planned increments + fraction * residual / H.
    Recovery not yet served is redistributed over the remaining command frames.
No forward halfspace is imposed. The nominal speed budget is an explicit
multiple of the plan envelope; an already faster incoming velocity is slewed
down, never instantaneously truncated. Unserved recovery is reported once.
"""
    target = poses(plan).copy(); history = poses(measured); ids = np.asarray(frame_ids)
    if (target.ndim != 3 or history.ndim != 3 or ids.shape != (len(history),)
            or not np.issubdtype(ids.dtype, np.integer) or len(ids) < 2
            or np.any(ids[1:] <= ids[:-1]) or ids[0] < 0 or ids[-1] >= len(target)-1
            or isinstance(native_stride, (bool, np.bool_))
            or not isinstance(native_stride, (int, np.integer)) or native_stride < 1):
        raise ValueError('Ordered emitted poses and a positive native stride required')
    if (not np.isfinite(recovery_fraction) or not 0 <= recovery_fraction <= 1
            or not np.isfinite(speed_multiplier) or speed_multiplier < 1):
        raise ValueError('Fraction must be in [0,1] and speed multiplier at least one')
    boundary = int(ids[-1]); horizon = len(target)-1-boundary
    previous = np.flatnonzero(ids <= boundary-native_stride)
    if not len(previous):raise ValueError('An emitted native interval is required')
    j = int(previous[-1]); duration = boundary-int(ids[j])
    velocity = (history[-1,:3,3]-history[j,:3,3])/duration
    increments = np.diff(target[:,:3,3], axis=0)
    plan_max = float(np.linalg.norm(increments, axis=1).max())
    budget = float(speed_multiplier)*plan_max
    ceiling = max(budget, float(np.linalg.norm(velocity)))
    acceleration = ceiling/native_stride
    residual = target[boundary,:3,3]-history[-1,:3,3]
    requested = float(recovery_fraction)*residual
    offset = requested/horizon
    receipt = dict(policy='fractional_recovery_v1', boundary_frame=boundary,
                   horizon_frames=horizon, native_stride=int(native_stride),
                   recovery_fraction=float(recovery_fraction), speed_multiplier=float(speed_multiplier),
                   initial_residual=residual.tolist(), requested_recovery=requested.tolist(),
                   requested_offset_per_frame=offset.tolist(), incoming_velocity=velocity.tolist(),
                   planned_speed_max=plan_max, nominal_speed_budget=budget,
                   speed_ceiling=ceiling, acceleration_ceiling=acceleration,
                   response_gain='not fitted', forward_halfspace=False,
                   history_unchanged=True, rotation_targets_unchanged=True,
                   applied=False, reason=None, steps=[])
    if np.array_equal(history, target[ids]):
        receipt.update(reason='exact_identity_no_intervention', endpoint_residual=[0.,0.,0.],
                       realized_command_recovery=[0.,0.,0.], unserved_recovery=[0.,0.,0.],
                       realized_fraction=None)
        return RateContinuation(target, receipt)
    output = target.copy(); start = history[-1,:3,3].copy(); position = start.copy()
    for frame in range(boundary+1, len(target)):
        recovered = position-start-(target[frame-1,:3,3]-target[boundary,:3,3])
        desired = increments[frame-1]+(requested-recovered)/(len(target)-frame)
        norm = float(np.linalg.norm(desired))
        speed_limited = desired*min(1., budget/norm) if norm else desired.copy()
        delta = speed_limited-velocity; change = float(np.linalg.norm(delta))
        following = velocity+delta*min(1., acceleration/change) if change else velocity.copy()
        position += following; output[frame,:3,3] = position
        receipt['steps'].append(dict(frame=frame, velocity=following.tolist(),
            requested_velocity=desired.tolist(),
            speed=float(np.linalg.norm(following)), acceleration=float(np.linalg.norm(following-velocity)),
            speed_clipped=bool(norm>budget), slew_clipped=bool(change>acceleration)))
        velocity = following
    endpoint = target[-1,:3,3]-position
    realized = residual-endpoint
    magnitude2 = float(residual@residual)
    receipt.update(applied=True, reason='fractional_translation_recovery' if budget else 'no_plan_speed_budget',
                   endpoint_residual=endpoint.tolist(), realized_command_recovery=realized.tolist(),
                   unserved_recovery=(requested-realized).tolist(),
                   realized_fraction=float(realized@residual/magnitude2) if magnitude2 else None)
    np.testing.assert_array_equal(output[:boundary+1],target[:boundary+1])
    np.testing.assert_array_equal(output[:,:3,:3],target[:,:3,:3])
    return RateContinuation(output, receipt)


def issue_fractional_future(plan, issued_commands, measured, frame_ids, *, next_end,
                            available_frame, feedback_index, nominal_boundaries,
                            history_labels, native_stride, recovery_fraction,
                            speed_multiplier):
    """Issue a future chunk, with explicit nominal warmup and memory bridge.

For DreamX's proposed future-only arm, nominal_boundaries=1 leaves the first
feedback command exactly nominal. Later accepted observations drive recovery;
no inverse response is inferred here. The host owns the monotone feedback index
and must skip native re-encoding when the receipt reports no intervention.
"""
    target=poses(plan); issued=poses(issued_commands).copy(); history=poses(measured)
    ids=np.asarray(frame_ids)
    integers=(next_end,available_frame,feedback_index,nominal_boundaries)
    if (any(isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,np.integer)) for v in integers)
            or target.ndim!=3 or issued.shape!=target.shape or history.ndim!=3
            or ids.shape!=(len(history),) or len(ids)<2 or not np.issubdtype(ids.dtype,np.integer)
            or np.any(ids[1:]<=ids[:-1]) or ids[0]<0
            or ids[-1]!=available_frame or not 0<=available_frame<next_end-1<len(target)
            or feedback_index<0 or nominal_boundaries<0
            or history_labels not in ('commanded','measured')
            or not np.isfinite(recovery_fraction) or not 0<=recovery_fraction<=1
            or not np.isfinite(speed_multiplier) or speed_multiplier<1
            or isinstance(native_stride,(bool,np.bool_))
            or not isinstance(native_stride,(int,np.integer)) or native_stride<1):
        raise ValueError('Invalid causal issuance contract')
    b=int(available_frame)
    if feedback_index<nominal_boundaries:
        if not np.array_equal(issued[b+1:next_end],target[b+1:next_end]):
            raise ValueError("Nominal bootstrap requires untouched planned future commands")
        issued[b+1:next_end]=target[b+1:next_end]
        return RateContinuation(issued,dict(applied=False, reason='nominal_bootstrap',
            feedback_index=int(feedback_index), nominal_boundaries=int(nominal_boundaries),
            history_labels=history_labels, response_gain='not fitted',
            bootstrap_policy='preserve original plan amplitude; no rescaling'))
    result=fractional_memory_continuation(target[:next_end],history,ids,
        native_stride=native_stride,recovery_fraction=recovery_fraction,speed_multiplier=speed_multiplier)
    future=result.commands[b+1:next_end]
    bridge=issued[b]@np.linalg.inv(history[-1]) if history_labels=='commanded' else np.eye(4)
    issued[b+1:next_end]=bridge@future
    changed=not np.array_equal(issued[b+1:next_end],np.asarray(issued_commands)[b+1:next_end])
    receipt=dict(result.receipt, applied=changed, feedback_index=int(feedback_index),
                 history_labels=history_labels, memory_bridge=bridge.tolist(),
                 reference_rotation_targets_unchanged=True)
    return RateContinuation(issued,receipt)
