"""Causal geometry-unit calibration, independent of execution-response learning.

This module is also shipped verbatim to native adapter environments. It never
loads a calibration cache, fits a commanded trajectory, or imports a generator.
"""
import hashlib
import importlib
import json
import os
from pathlib import Path
import numpy as np


class CalibrationUnavailable(ValueError):
    """The current native context has no supported reference-unit conversion."""


def array(value):
    if hasattr(value, 'detach'):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value)


def fingerprint(value):
    a = np.ascontiguousarray(array(value))
    return hashlib.sha256(str((a.shape, a.dtype)).encode()+a.tobytes()).hexdigest()


def fit_depth_units(reference, native, mask=None):
    r, n = np.asarray(reference, float), np.asarray(native, float)
    if r.ndim != 2 or r.shape != n.shape:
        raise CalibrationUnavailable('Depths must represent the same pixels')
    good = np.isfinite(r) & np.isfinite(n) & (r > 1e-6) & (n > 1e-6)
    if mask is not None:
        m = np.asarray(mask, bool)
        if m.shape != r.shape:
            raise CalibrationUnavailable('Depth mask shape mismatch')
        good &= m
    if good.sum() < 256 or good.mean() < .2:
        raise CalibrationUnavailable('Insufficient shared finite depth support')
    ratio = n[good]/r[good]
    k = float(np.median(ratio)); p10, p90 = np.quantile(ratio, [.1, .9])
    if not np.isfinite(k) or k <= 0 or p90/p10 > 4:
        raise CalibrationUnavailable('Shared geometry does not support a scalar unit mapping')
    return k, dict(valid_pixels=int(good.sum()), valid_fraction=float(good.mean()),
                   p90_over_p10=float(p90/p10),
                   median_relative_residual=float(np.median(np.abs(ratio/k-1))))


def _rgb(value):
    raw = array(value)
    a = raw.astype(float)
    if raw.dtype == np.uint8:
        a /= 255
    if a.ndim != 3 or a.shape[-1] != 3 or not np.isfinite(a).all() or a.min() < 0 or a.max() > 1:
        raise CalibrationUnavailable('Expected finite HWC RGB in [0,1] or uint8')
    return a


class OnlineDepthCalibration:
    """Carry a fixed reference gauge across overlapping generated-source windows.

    Each accepted window is converted using its OWN native geometry. The previous
    window contributes only depth on an explicitly shared, already emitted frame.
    Rejected windows invalidate the current scale; no stale-scale fallback exists.
    Receipts are write-only evidence and are never read to seed a new instance.
    """
    def __init__(self, reference_anchor_depth, anchor_rgb, *, model, receipt=None,
                 max_shared_rgb_mae=8/255):
        self.model = model
        self.receipt = Path(receipt) if receipt is not None else None
        self.max_rgb_mae = float(max_shared_rgb_mae)
        if not 0 <= self.max_rgb_mae <= 1:
            raise ValueError('Invalid shared-image tolerance')
        depth = np.asarray(reference_anchor_depth, float).copy()
        rgb = _rgb(anchor_rgb).copy()
        if depth.shape != rgb.shape[:2]:
            raise CalibrationUnavailable('Anchor depth and image must share pixels')
        fit_depth_units(depth, depth)
        self.anchors = {0: (depth, rgb)}
        self.events = []
        self.current = None
        self.available = -1
        self.contexts = set()
        if self.receipt is not None and self.receipt.exists():
            raise CalibrationUnavailable('Use a new rollout receipt; calibration is not loaded from disk')

    def update(self, *, first_depth, last_depth, first_rgb, last_rgb,
               first_frame, last_frame, available_frame, context, source='generated_prefix'):
        event = dict(model=self.model, context=context, first_frame=first_frame,
                     last_frame=last_frame, available_frame=available_frame,
                     source=source, status='unavailable')
        self.current = None
        try:
            if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)) for x in [first_frame,last_frame,available_frame]):
                raise CalibrationUnavailable('Decoded frame IDs must be integers')
            first_frame,last_frame,available_frame=map(int,(first_frame,last_frame,available_frame))
            event.update(first_frame=first_frame,last_frame=last_frame,available_frame=available_frame)
            if not 0 <= first_frame <= last_frame <= available_frame or available_frame < self.available:
                raise CalibrationUnavailable('Future or out-of-order calibration geometry')
            if source not in ['initial_input','generated_prefix'] or (source=='initial_input' and last_frame!=0):
                raise CalibrationUnavailable('Only initial input and emitted RGB may establish units')
            if not isinstance(context,str) or not context or context in self.contexts:
                raise CalibrationUnavailable('A fresh native normalization context is required')
            if first_frame not in self.anchors:
                raise CalibrationUnavailable('No calibrated shared frame in this source window')
            reference, rgb = self.anchors[first_frame]
            now = _rgb(first_rgb); tail = _rgb(last_rgb)
            if now.shape != rgb.shape:
                raise CalibrationUnavailable('Shared frames have different pixel grids')
            mae = float(np.abs(now-rgb).mean())
            if mae > self.max_rgb_mae:
                raise CalibrationUnavailable('Shared-frame image correspondence failed')
            k, quality = fit_depth_units(reference, first_depth)
            last = np.asarray(last_depth, float)
            if last.shape != tail.shape[:2]:
                raise CalibrationUnavailable('Last source depth and RGB must share pixels')
            fit_depth_units(last, last)
            event.update(status='identified',native_units_per_reference_unit=k,
                         shared_rgb_mae=mae,quality=quality,
                         reference_depth_sha256=fingerprint(reference),
                         native_depth_sha256=fingerprint(first_depth),
                         method='causal shared-frame depth correspondence; no trajectory fit')
            # Retain the original anchor and the last emitted source frame only.
            if last_frame != 0:
                self.anchors = {0:self.anchors[0], last_frame:(last.copy()/k,tail.copy())}
            self.available = available_frame
            self.contexts.add(context)
            self.current = event.copy()
            return event.copy()
        except (ValueError,TypeError) as exc:
            event['reason'] = str(exc)
            raise CalibrationUnavailable(str(exc)) from exc
        finally:
            self.events.append(event)
            if self.receipt is not None:
                self.receipt.parent.mkdir(parents=True,exist_ok=True)
                self.receipt.write_text(json.dumps(dict(model=self.model,events=self.events),indent=2)+'\n')

    def scale_for(self, context):
        if self.current is None or self.current['context'] != context:
            raise CalibrationUnavailable('No valid calibration for this native context')
        return self.current['native_units_per_reference_unit']


def remap_depth(reference, input_K, processed_K, output_shape):
    """Bilinear same-camera pixel mapping using the actual preprocessing intrinsics."""
    d=np.asarray(reference,float); a=np.asarray(input_K,float); b=np.asarray(processed_K,float)
    if d.ndim!=2 or min(d.shape)<2 or a.shape!=(3,3) or b.shape!=(3,3) or not np.isfinite(a).all() or not np.isfinite(b).all() or min(a[0,0],a[1,1],b[0,0],b[1,1])<=0:
        raise CalibrationUnavailable('Invalid depth/intrinsics pixel contract')
    h,w=d.shape;v,u=np.indices(output_shape,dtype=float)
    x=(u-b[0,2])/b[0,0]*a[0,0]+a[0,2];y=(v-b[1,2])/b[1,1]*a[1,1]+a[1,2]
    bad=(x<0)|(x>w-1)|(y<0)|(y>h-1)
    x=np.clip(x,0,w-1);y=np.clip(y,0,h-1)
    ix=np.minimum(x.astype(int),w-2);iy=np.minimum(y.astype(int),h-2)
    dx=x-ix;dy=y-iy;out=np.zeros(output_shape,float)
    for ox,oy,weight in [(0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)]:
        value=d[iy+oy,ix+ox];use=weight>0
        bad |= use & ~np.isfinite(value)
        out += np.where(use,weight*np.nan_to_num(value),0)
    out[bad]=np.nan
    return out


def metric_prediction_scale(observer, prediction, anchor, reference_depth, K, *, divisor=4.):
    """Memorizon: freshly estimated metric z-depth / native training divisor.

    Use the input processor's pixel transform, not the model's estimated focal
    length. The native divisor is an encoding contract, not a fitted gain.
    """
    metric_flag=np.asarray(prediction.is_metric)
    if metric_flag.size!=1 or metric_flag.item() not in (True, 1):
        raise CalibrationUnavailable('Metric depth unavailable; refusing depth=1 fallback')
    if not np.isfinite(divisor) or divisor<=0:
        raise CalibrationUnavailable('Invalid native metric divisor')
    _,_,intrinsics=observer.input_processor(image=[anchor],intrinsics=np.asarray(K,np.float32)[None],
        process_res=504,process_res_method='upper_bound_resize')
    if intrinsics is None:
        raise CalibrationUnavailable('Observer preprocessing did not return intrinsics')
    native=array(prediction.depth)[0]
    reference=remap_depth(reference_depth,K,array(intrinsics)[0],native.shape)
    metres,quality=fit_depth_units(reference,native)
    return metres/divisor,dict(status='identified',method='runtime metric anchor depth at corresponding pixels',
        native_units_per_reference_unit=metres/divisor,metres_per_reference_unit=metres,
        native_metric_divisor=float(divisor),is_metric=True,source='initial_input',
        frame_id=0,available_frame=0,anchor_sha256=fingerprint(anchor),quality=quality)


def require_runtime_provider(model):
    spec=os.environ.get('CAMERA_LOOP_UNIT_PROVIDER','')
    if not spec or ':' not in spec:
        raise CalibrationUnavailable(model+': native units are unidentified; provide a runtime native-depth provider (module:function). Offline factors and command/response ratios cannot establish geometry units.')
    module,name=spec.split(':',1)
    provider=getattr(importlib.import_module(module),name)
    if not callable(provider):
        raise CalibrationUnavailable('Runtime unit provider is not callable')
    return provider


def runtime_anchor_scale(model, anchor, reference_depth, K, *, receipt, provider=None):
    """Pose-only hosts must obtain native-unit geometry from an actual provider.

    The provider receives only the initial image, intrinsics and frame metadata;
    it must return depth in the geometry units consumed by the native model.
    It is called on every rollout. No numeric-scale or reference-motion input is
    accepted. No such provider is claimed for the released Astronex/WorldPlay
    encoders; without one, these hosts fail before changing commands or memory.
    """
    path=Path(receipt)
    if path.exists():raise CalibrationUnavailable('Refusing to reuse an earlier calibration receipt')
    try:
        fn=provider if provider is not None else require_runtime_provider(model)
        digest=fingerprint(anchor)
        result=fn(model=model,anchor=np.asarray(anchor).copy(),intrinsics=np.asarray(K).copy(),frame_id=0,available_frame=0,anchor_sha256=digest)
        if not isinstance(result,dict) or result.get('source')!='initial_input' or result.get('frame_id')!=0 or result.get('anchor_sha256')!=digest or not result.get('native_unit_evidence'):
            raise CalibrationUnavailable('Runtime provider must bind native depth to the current input and native unit contract')
        if 'scale' in result or 'native_units_per_reference_unit' in result:
            raise CalibrationUnavailable('Return runtime native geometry, not a prefitted scalar')
        k,quality=fit_depth_units(reference_depth,result['native_depth'],result.get('mask'))
        record=dict(model=model,status='identified',native_units_per_reference_unit=k,method='runtime native anchor geometry',
                    source='initial_input',frame_id=0,available_frame=0,anchor_sha256=digest,
                    native_unit_evidence=result['native_unit_evidence'],quality=quality)
    except (ValueError,TypeError,KeyError,ImportError,AttributeError) as exc:
        record=dict(model=model,status='unavailable',reason=str(exc),source='initial_input',frame_id=0)
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(record,indent=2)+'\n')
        raise CalibrationUnavailable(str(exc)) from exc
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(record,indent=2)+'\n')
    return k,record


def capture_native_normalizer(native_callback, record, available_frame):
    """Observe VMem's actual per-window factor without changing native tensors."""
    context=0
    def wrapped(cameras):
        nonlocal context
        factor,normalized=native_callback(cameras)
        value=float(array(factor))
        if not np.isfinite(value) or value<=0:
            raise CalibrationUnavailable('Native normalizer returned an invalid factor')
        record(dict(kind='native_window_scale',context=context,available_frame=int(available_frame()),
                    factor=value,cameras=len(cameras),source='native encoder at issuance',
                    scope='normalization contract, not physical-unit or execution-gain calibration'))
        context+=1
        return factor,normalized
    return wrapped
