"""Two-chunk WorldWarp check using the same original anchor in both reconstructions.
This validates scale; it never rescales poses from generated trajectory measurements.
"""
import json
from pathlib import Path
import numpy as np
from geometry_quality import depth_quality,GeometryQualityError

def assess_continuation(reference_depth,native_depth,source_anchor,original_anchor,max_ratio=1.2,max_anchor_mae=8/255):
    if max_ratio<1 or max_anchor_mae<0:raise ValueError('Invalid continuation thresholds')
    a=np.asarray(source_anchor,dtype=float);b=np.asarray(original_anchor,dtype=float)
    if a.shape!=b.shape:raise ValueError('Anchor images do not share pixel coordinates')
    q=depth_quality(reference_depth,native_depth);reasons=list(q['reasons'])
    mae=float(np.mean(np.abs(a-b)))
    if not np.isfinite(mae) or mae>max_anchor_mae:reasons.append('source_anchor_correspondence_unverified')
    ratio=q.get('reference_units_per_native_unit')
    if ratio is None or not 1/max_ratio<=ratio<=max_ratio:reasons.append('continuation_scale_drift')
    return dict(q,passed=not reasons,reasons=reasons,anchor_rgb_mae=mae,max_anchor_rgb_mae=max_anchor_mae,max_scale_ratio=max_ratio,scope='second chunk only; source frame 0 corresponds to original anchor; raw scale multiplier is diagnostic, not automatically applied')

def check_native(depth_maps,reference,video_tensor,anchor,out,mode='enforce',max_ratio=1.2):
    native=depth_maps[0,0].detach().float().cpu().numpy()
    first=video_tensor[0,0].detach().float().cpu().permute(1,2,0).numpy()
    original=np.asarray(anchor,dtype=np.float32)/255
    q=assess_continuation(reference,native,first,original,max_ratio=max_ratio)
    path=Path(out);np.savez_compressed(path/'continuation_anchor_depths.npz',reference=reference,native=native)
    (path/'continuation_scale_check.json').write_text(json.dumps(dict(q,mode=mode),indent=2))
    if mode=='enforce' and not q['passed']:raise GeometryQualityError(q)
    return q
