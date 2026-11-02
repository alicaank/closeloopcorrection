"""Causal command-response adaptation; never a physical-unit calibration.

All observations remain in a fixed anchor gauge. Native command history is
immutable. No inverse is used until natural excitation and forward predictions
support a local response model with an explicit history term.
"""
from dataclasses import dataclass, asdict
import numpy as np
from .geometry import poses


@dataclass(frozen=True)
class ResponseConfig:
    forgetting: float = .9
    window: int = 32
    min_samples: int = 4
    validation_steps: int = 2
    max_condition: float = 100.
    min_excitation: float = .05
    max_relative_slope_error: float = .5
    prediction_tolerance: float = .25
    recovery_frames: float = 64.
    speed_multiplier: float = 2.
    acceleration_frames: float = 12.


class OnlineAffineResponse:
    """Per-axis y=h*x+beta*previous_y+b in declared normalized velocities."""
    def __init__(self, config=ResponseConfig()):
        self.config=config
        if not (0 < config.forgetting <= 1 and config.window >= config.min_samples >= 4
                and config.validation_steps >= 1 and config.max_condition > 1
                and config.min_excitation > 0 and config.prediction_tolerance > 0
                and config.max_relative_slope_error > 0):
            raise ValueError('Invalid response estimation configuration')
        self.rows=[];self.tick=0;self.candidate=[None]*3;self.streak=np.zeros(3,int)
        self.active=np.zeros(3,bool);self.last_frame=-1;self.events=[]

    def update(self, x, y, history, *, frame_id, noise, confidence=1., accepted=True, rejection_reason="observer_rejected"):
        if isinstance(frame_id,(bool,np.bool_)) or not isinstance(frame_id,(int,np.integer)) or frame_id <= self.last_frame:
            raise ValueError('Response observations must have fresh increasing frame IDs')
        self.last_frame=int(frame_id);self.tick+=1;self.active[:]=False
        event=dict(frame_id=int(frame_id),accepted=False,axes=[],active=[False]*3)
        if not accepted:
            self.streak[:]=0;event['reason']=rejection_reason;self.events.append(event);return event
        x,y,history=[np.asarray(v,float) for v in (x,y,history)]
        if any(v.shape!=(3,) or not np.isfinite(v).all() for v in (x,y,history)):
            raise ValueError('Expected finite three-axis velocities')
        if not np.isfinite(noise) or noise < 0 or not np.isfinite(confidence) or not 0 < confidence <= 1:
            raise ValueError('Invalid response evidence')
        noise=max(float(noise),1e-6)
        self.rows.append((self.tick,x.copy(),y.copy(),history.copy(),noise,float(confidence)))
        self.rows=self.rows[-self.config.window:];c=self.config
        weight=np.array([r[5]*c.forgetting**(self.tick-r[0]) for r in self.rows]);weight/=weight.sum()
        for axis in range(3):
            old=self.candidate[axis];innovation=None
            if old is not None:
                innovation=float(y[axis]-np.array([x[axis],history[axis],1.])@old)
                good=abs(innovation)<=max(c.prediction_tolerance,3*noise)
                self.streak[axis]=self.streak[axis]+1 if good else 0
            else:self.streak[axis]=0
            row=dict(axis=axis,samples=len(self.rows),innovation=innovation,validated_steps=int(self.streak[axis]),reason='insufficient_samples')
            self.candidate[axis]=None
            if len(self.rows)>=c.min_samples:
                X=np.array([[r[1][axis],r[3][axis],1.] for r in self.rows]);Y=np.array([r[2][axis] for r in self.rows])
                excitation=float(np.sqrt(np.sum(weight*(X[:,0]-np.sum(weight*X[:,0]))**2)))
                W=X*np.sqrt(weight[:,None]);condition=float(np.linalg.cond(W))
                row.update(excitation=excitation,condition=condition if np.isfinite(condition) else None)
                if excitation<c.min_excitation:row['reason']='insufficient_command_variation'
                elif np.linalg.matrix_rank(W)<3 or condition>c.max_condition:row['reason']='command_history_not_separable'
                else:
                    fit=np.linalg.lstsq(W,Y*np.sqrt(weight),rcond=None)[0]
                    residual=Y-X@fit
                    variance=max(float(np.sum(weight*residual**2))*len(Y)/max(1,len(Y)-3),float(np.sum(weight*np.array([r[4]**2 for r in self.rows]))))
                    cov=np.linalg.inv(X.T@(weight[:,None]*X))*variance/len(Y)
                    se=float(np.sqrt(max(0,cov[0,0])))
                    row.update(coefficients=fit.tolist(),slope_error=se,weighted_residual=float(np.sqrt(np.sum(weight*residual**2))))
                    if fit[0]<=max(3*se,1e-6) or se/fit[0]>c.max_relative_slope_error:row['reason']='slope_not_supported'
                    elif abs(fit[1])>=.95:row['reason']='unstable_history_response'
                    elif row['weighted_residual']>max(c.prediction_tolerance,3*noise):row['reason']='forward_model_mismatch'
                    else:
                        self.candidate[axis]=fit
                        self.active[axis]=self.streak[axis]>=c.validation_steps
                        row['reason']='identified' if self.active[axis] else 'awaiting_forward_validation'
            if self.candidate[axis] is None:self.streak[axis]=0
            event['axes'].append(row)
        event.update(accepted=True,active=self.active.tolist(),noise_proxy=noise)
        self.events.append(event);return event


class OnlineCommandController:
    """Bootstrap native commands, learn at each emitted boundary, alter future only."""
    def __init__(self, plan, frame_ids, *, native_step, arm='C', config=ResponseConfig()):
        self.plan=poses(plan);self.ids=np.asarray(frame_ids)
        if self.plan.ndim!=3 or self.ids.shape!=(len(self.plan),) or len(self.plan)<2:
            raise ValueError('One frame ID per target pose required')
        if not np.issubdtype(self.ids.dtype,np.integer) or self.ids[0]!=0 or np.any(np.diff(self.ids)<=0):
            raise ValueError('Invalid target timestamps')
        if not np.allclose(self.plan[0],np.eye(4),atol=1e-6):raise ValueError('Plan must use initial-camera reference coordinates')
        if arm not in ('A','S','C') or not np.isfinite(native_step) or native_step<=0:
            raise ValueError('Invalid arm or native navigation contract')
        if config.recovery_frames<=0 or config.acceleration_frames<=0 or config.speed_multiplier<1:
            raise ValueError('Invalid command bounds')
        self.arm=arm;self.config=config;self.dt=np.diff(self.ids)
        steps=np.linalg.norm(np.diff(self.plan[:,:3,3],axis=0),axis=1)
        # Bound EVERY bootstrap increment by the released native navigation step.
        # Future target commands are known; future recorded RGB is never used.
        self.native_step=float(native_step)
        self.bootstrap=float(native_step/steps.max()) if steps.max()>1e-9 else 0.
        self.base=self.plan.copy();self.base[:,:3,3]*=self.bootstrap;self.commands=self.base.copy()
        self.reference_speed=float(np.max(steps/self.dt));self.native_speed=self.reference_speed*self.bootstrap
        self.response=OnlineAffineResponse(config)
        self.previous_index=0;self.previous_pose=np.eye(4);self.previous_velocity=None
        self.last_boundary=0;self.events=[];self.last_issued_end=1
        self.previous_duration=None;self.response_duration=None;self.control_engaged=False

    def boundary(self, index, measured, evidence, *, next_end, available_frame):
        if not isinstance(index,(int,np.integer)) or not self.last_boundary<index<len(self.plan):raise ValueError('Stale observation boundary')
        if not isinstance(next_end,(int,np.integer)) or not index<next_end<=len(self.plan):raise ValueError('Invalid next chunk')
        if not isinstance(available_frame,(int,np.integer)) or available_frame!=self.ids[index] or evidence.get('frame_id')!=available_frame:
            raise ValueError('Observation and available output timestamps disagree')
        if self.last_issued_end>1 and index!=self.last_issued_end-1:raise ValueError('Observation skips the issued chunk boundary')
        old=self.commands[:index+1].copy();self.last_boundary=int(index)
        valid=bool(evidence.get('passed',False)) and float(evidence.get('confidence',0))>0
        fit=None;velocity=None;pose=None
        future_duration=float(self.ids[next_end-1]-self.ids[index])
        if valid:
            pose=poses(measured)
            if pose.shape!=(4,4):raise ValueError('Expected one endpoint')
            duration=None
            reason='history_not_ready'
            if self.previous_index is not None:
                duration=float(self.ids[index]-self.ids[self.previous_index])
                velocity=(pose[:3,3]-self.previous_pose[:3,3])/duration
                command_velocity=(self.commands[index,:3,3]-self.commands[self.previous_index,:3,3])/duration
                ready=(self.previous_velocity is not None and self.native_speed>0
                       and duration==self.previous_duration and not evidence.get('bootstrap',False))
                if ready:
                    # A discrete history coefficient belongs to one chunk duration.
                    if self.response_duration != duration:
                        if self.response_duration is not None:self.response=OnlineAffineResponse(self.config)
                        self.response_duration=duration
                    depth_unit=float(evidence.get('reference_depth_unit',1.))
                    noise=2*float(evidence.get('shared_position_p90_depth_fraction',0))*depth_unit/duration/self.reference_speed
                    fit=self.response.update(command_velocity/self.native_speed,velocity/self.reference_speed,
                        self.previous_velocity/self.reference_speed,frame_id=int(available_frame),noise=noise,
                        confidence=float(evidence['confidence']))
                elif self.previous_velocity is not None and duration!=self.previous_duration:
                    reason='chunk_duration_changed'
            if fit is None:
                fit=self.response.update(None,None,None,frame_id=int(available_frame),noise=0,
                                         accepted=False,rejection_reason=reason)
            self.previous_index=int(index);self.previous_pose=pose.copy();self.previous_velocity=velocity
            self.previous_duration=duration
        else:
            self.previous_index=None;self.previous_velocity=None;self.previous_duration=None
            fit=self.response.update(None,None,None,frame_id=int(available_frame),noise=0,accepted=False)
        same_duration=(self.response_duration is not None and future_duration==self.response_duration)
        active=self.response.active.copy() if self.arm=='C' and valid and velocity is not None and same_duration else np.zeros(3,bool)
        ceiling=self.config.speed_multiplier*self.native_speed
        previous=(self.commands[index,:3,3]-self.commands[index-1,:3,3])/float(self.ids[index]-self.ids[index-1])
        residual=self.plan[index,:3,3]-pose[:3,3] if valid else np.zeros(3)
        history=velocity/self.reference_speed if velocity is not None and self.reference_speed>0 else np.zeros(3)
        base_average=(self.base[next_end-1,:3,3]-self.base[index,:3,3])/future_duration if future_duration>0 else np.zeros(3)
        requested_average=base_average.copy()
        desired_average=None
        if np.any(active):
            # Invert ONCE for the next chunk average. Spread that offset over its
            # native profile; do not iterate a chunk recurrence at latent cadence.
            desired_average=((self.plan[next_end-1,:3,3]-self.plan[index,:3,3])/future_duration
                             +residual/max(self.config.recovery_frames,future_duration))/self.reference_speed
            for axis in np.flatnonzero(active):
                h,beta,offset=self.response.candidate[axis]
                requested_average[axis]=(desired_average[axis]-beta*history[axis]-offset)/h*self.native_speed
        offset_velocity=requested_average-base_average
        # Keep the transition feasible even when identification or observation is
        # lost. Baseline/sham and never-intervened abstention remain exact no-ops.
        engaged=self.arm=='C' and (self.control_engaged or np.any(active)
                                  or not np.array_equal(self.commands[:index+1],self.base[:index+1]))
        clips=[];max_slew_ratio=0.
        for j in range(index+1,next_end):
            dt=float(self.ids[j]-self.ids[j-1])
            raw=(self.base[j,:3,3]-self.base[j-1,:3,3])/dt+offset_velocity
            if engaged:
                limited=raw*min(1.,ceiling/max(np.linalg.norm(raw),1e-15))
                delta=limited-previous;limit=ceiling*dt/self.config.acceleration_frames
                limited=previous+delta*min(1.,limit/max(np.linalg.norm(delta),1e-15))
                max_slew_ratio=max(max_slew_ratio,float(np.linalg.norm(limited-previous)/max(limit,1e-15)))
                clips.append(bool(np.linalg.norm(raw-limited)>1e-10));raw=limited
            else:clips.append(False)
            self.commands[j,:3,3]=self.commands[j-1,:3,3]+raw*dt
            previous=raw
        self.control_engaged=bool(engaged)
        if not engaged:self.commands[index+1:next_end]=self.base[index+1:next_end]
        issued_average=(self.commands[next_end-1,:3,3]-self.commands[index,:3,3])/future_duration if future_duration>0 else np.zeros(3)
        predicted_average=[None]*3
        for axis in np.flatnonzero(active):
            h,beta,bias=self.response.candidate[axis]
            predicted_average[axis]=float((h*issued_average[axis]/self.native_speed+beta*history[axis]+bias)*self.reference_speed)
        commands_changed=not np.allclose(self.commands[index+1:next_end],self.base[index+1:next_end],atol=1e-12,rtol=0)
        np.testing.assert_array_equal(old,self.commands[:index+1]);self.last_issued_end=int(next_end)
        event=dict(frame_id=int(available_frame),accepted=valid,response=fit,active_axes=active.tolist(),
            applied=bool(np.any(active) and next_end>index+1),commands_differ_from_bootstrap=commands_changed,
            future_frame_ids=self.ids[index+1:next_end].tolist(),clipped=clips,
            response_duration_frames=self.response_duration,future_duration_frames=future_duration,
            inverse_abstention_reason=None if np.any(active) else ('chunk_duration_mismatch' if any(self.response.active) and not same_duration else 'response_not_active'),
            requested_chunk_native_velocity=requested_average.tolist(),issued_chunk_native_velocity=issued_average.tolist(),
            predicted_chunk_reference_velocity=predicted_average,control_engaged=self.control_engaged,
            max_slew_ratio=max_slew_ratio,native_speed_ceiling=ceiling,
            reference_position_error=float(np.linalg.norm(residual)) if valid else None,
            measured=None if pose is None else pose.tolist(),target=self.plan[index].tolist(),
            semantic='online chunk response v2; physical unit mapping unidentified',memory_policy='native commanded labels unchanged')
        self.events.append(event)
        return self.commands[index+1:next_end].copy(),event

    def metadata(self):
        return dict(protocol='chunk_response_v2',config=asdict(self.config),bootstrap_command_mapping=self.bootstrap,
            bootstrap_policy='maximum planned native increment equals released navigation step',native_navigation_step=self.native_step,
            physical_units_identified=False,plan=self.plan.tolist(),native_commands=self.commands.tolist(),
            pose_frame_ids=self.ids.tolist(),events=self.events)
