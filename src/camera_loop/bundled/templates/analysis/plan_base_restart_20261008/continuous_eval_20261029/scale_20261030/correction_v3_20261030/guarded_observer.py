"""Opt-in wrapper; preserves native observer poses/scale exactly when diagnostics pass."""
import importlib.util,json,os
from pathlib import Path
import numpy as np
import torch
from geometry_quality import depth_quality,pose_quality,GeometryQualityError
R=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('v3_original_realized',R/'snapshots/realized_pose.py');native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)

def realized_poses(da3,anchor_rgb,anchor_depth,frames,K,max_views=16,tail=4):
    mode=os.environ.get('GV_OBSERVER_QUALITY','report')
    if mode not in ['report','enforce']:raise ValueError('Guarded observer requires report or enforce mode')
    diagnostics={}
    class Checked:
        def inference(self,*args,**kwargs):
            pred=da3.inference(*args,**kwargs)
            d=np.asarray(pred.depth[0],dtype=np.float64)
            if anchor_depth is None:
                q=dict(passed=False,reasons=['missing_fixed_anchor_depth'])
            else:
                reference=torch.nn.functional.interpolate(torch.as_tensor(anchor_depth)[None,None].float(),size=d.shape,mode='bilinear',align_corners=False)[0,0].numpy()
                q=depth_quality(reference,d)
            diagnostics['depth']=q
            if not q['passed'] and mode=='enforce':raise GeometryQualityError(q)
            return pred
    poses,info=native.realized_poses(Checked(),anchor_rgb,anchor_depth,frames,K,max_views=max_views,tail=tail)
    diagnostics['pose']=pose_quality(poses);diagnostics['mode']=mode
    diagnostics['passed']=diagnostics['depth']['passed'] and diagnostics['pose']['passed']
    if mode=='enforce' and not diagnostics['pose']['passed']:raise GeometryQualityError(diagnostics['pose'])
    info['quality']=diagnostics
    return poses,info
