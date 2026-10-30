"""Anchor-only unit conversion. No generated pose or evaluation score enters the fit."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np

VERSION = 'anchor_correspondence_v1_20261030'

def worldwarp_camera_window(target, is_first, source_frames=81):
    target = np.asarray(target)
    if target.shape != (81,4,4):
        raise ValueError('WorldWarp requires 81 target poses per emitted chunk')
    return target if is_first else np.concatenate([np.repeat(target[:1],source_frames,axis=0),target[1:]])

def point_cloud_depth(points, w2c, kn, size=(156,88)):
    """Nearest surface depth on anchor pixels; no biased point-median fallback."""
    points = np.asarray(points, dtype=np.float64)
    camera = points @ np.asarray(w2c)[:3,:3].T + np.asarray(w2c)[:3,3]
    camera = camera[np.isfinite(camera).all(axis=1) & (camera[:,2] > 1e-6)]
    w,h = size
    u = np.floor((kn[0]*camera[:,0]/camera[:,2]+kn[2])*w).astype(int)
    v = np.floor((kn[1]*camera[:,1]/camera[:,2]+kn[3])*h).astype(int)
    keep = (u>=0)&(u<w)&(v>=0)&(v<h)
    depth = np.full(w*h, np.inf)
    np.minimum.at(depth, v[keep]*w+u[keep], camera[keep,2])
    return depth.reshape(h,w)

def fit_scale(reference, native, mask=None):
    ref = np.asarray(reference, dtype=np.float64)
    nat = np.asarray(native, dtype=np.float64)
    if ref.shape != nat.shape or ref.ndim != 2:
        raise ValueError('Depths must be registered to the same anchor pixels')
    valid = np.isfinite(ref) & np.isfinite(nat) & (ref > 1e-6) & (nat > 1e-6)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    if valid.sum() < 256 or valid.mean() < .20:
        raise ValueError(f'Insufficient common anchor coverage: {valid.mean():.4f}')
    ratio = nat[valid] / ref[valid]
    scale = float(np.median(ratio))
    p10, p90 = np.quantile(ratio, [.1, .9])
    if not np.isfinite(scale) or scale <= 0 or p90 / p10 > 4:
        raise ValueError('Anchor geometry does not support a reliable scalar conversion')
    return scale, dict(version=VERSION, method='median_of_corresponding_native_over_reference_depth',
        native_units_per_reference_unit=scale, valid_pixels=int(valid.sum()), valid_fraction=float(valid.mean()),
        ratio_p10_p90=[float(p10), float(p90)], shape_residual_median=float(np.median(np.abs(ratio/scale-1))),
        reference_sha256=hashlib.sha256(ref.tobytes()).hexdigest(),
        native_sha256=hashlib.sha256(nat.tobytes()).hexdigest(),
        thresholds=dict(min_coverage=.20, min_pixels=256, max_p90_over_p10=4),
        scope='initial observed anchor only; evaluation gauge unchanged')

def freeze_scale(scale, receipt, model, scene):
    """All arms share the first anchor calibration; reject unexpected recomputation drift."""
    cache = os.environ.get('GV_SCALE_CACHE')
    if not cache:
        raise RuntimeError('GV_SCALE_CACHE is required for paired scale-controlled runs')
    root = Path(cache); root.mkdir(parents=True, exist_ok=True)
    path = root / f'{model}_{scene}.json'
    if path.exists():
        previous = json.loads(path.read_text())
        if previous['version'] != VERSION or previous['reference_sha256'] != receipt['reference_sha256']:
            raise RuntimeError('Cached scale belongs to a different reference or protocol')
        frozen = previous['native_units_per_reference_unit']
        if abs(scale/frozen-1) > .03:
            raise RuntimeError(f'Anchor calibration changed across arms: {scale} vs {frozen}')
        receipt['recomputed_scale'] = scale
        scale = frozen
    else:
        path.write_text(json.dumps(dict(receipt, model=model, scene=scene), indent=2))
    receipt.update(native_units_per_reference_unit=scale, frozen_calibration=str(path))
    return scale, receipt

def calibrate_lyra(w2c, depth, mask):
    import cv2
    path = Path(os.environ['LONG_LYRA_META'])
    meta = json.loads(path.read_text())
    native = depth.detach().float().cpu().numpy().squeeze()
    with np.load(meta['trajectory']) as z:
        reference = cv2.resize(z['anchor_depth'].astype(np.float32), (native.shape[1], native.shape[0]))
    scale, receipt = fit_scale(reference, native, mask.detach().cpu().numpy().squeeze() > .5)
    scale, receipt = freeze_scale(scale, receipt, 'lyra', Path(meta['trajectory']).stem)
    old = meta['pose_scale']
    corrected = w2c.clone()
    corrected[..., :3, 3] *= scale / old
    receipt.update(previous_pose_scale=old, camera_multiplier=scale/old, model='lyra',
                   native_depth_stage='DA3 after native MoGe alignment', observer_pose_scale=scale)
    meta.update(pose_scale=scale, original_pose_scale=old, scale_protocol=VERSION)
    path.write_text(json.dumps(meta, indent=2))
    (path.parent/'scale_calibration.json').write_text(json.dumps(receipt, indent=2))
    np.savez_compressed(path.parent/'scale_anchor.npz', reference=reference, native=native,
                        mask=mask.detach().cpu().numpy().squeeze())
    print('[scale] ' + json.dumps(receipt), flush=True)
    return corrected
