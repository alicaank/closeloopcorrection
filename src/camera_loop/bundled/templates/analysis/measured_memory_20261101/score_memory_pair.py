"""Paired memory-only camera evaluation with held-out VIPE observations.

No pose from the online DA3 observer is used to compute these scores.
"""
import argparse
import os
from camera_loop.evaluation_cache import cached_estimate
from camera_loop.lossless_archive import read_rgb
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
from runtime import A

sys.path.insert(0,str(A/'vipe_eval_20261030'))
from camera import estimate,errors
sys.path.insert(0,str(A/'fair_common_20261025'))
import fair_common as fc


def load_frames(source,scene):
    source=Path(source)
    if (source/'frames.npz').exists():
        with np.load(source/'frames.npz') as z:return z['frames'].copy(),'lossless_npz'
    if (source/'frames.lossless.mkv').exists():return read_rgb(source/'frames.lossless.mkv'),'lossless_ffv1'
    files=sorted((source/(scene+'_frames')).glob('frame_*.png'))
    if files:return np.stack([np.asarray(Image.open(f).convert('RGB')) for f in files]),'lossless_png'
    if source.suffix!='.mp4':raise ValueError('No frames: '+str(source))
    import imageio.v2 as imageio
    with imageio.get_reader(source) as reader:frames=[f for f in reader]
    return np.stack(frames),'native_lossy_mp4'


def output_K(model,source,reference_K,reference_shape,output_shape):
    source=Path(source)
    if model in ['lingbot','sana','matrixgame','worldwarp']:
        return np.asarray(json.loads((source/'run.json').read_text())['K'],float)
    if model=='astronex':
        return np.asarray(json.loads(source.with_suffix('.json').read_text())['K_out'],float)
    # DreamX and Lyra use a full-frame stretch, without a crop.
    h,w=output_shape;sh,sw=reference_shape
    K=reference_K.copy();K[0]*=w/sw;K[1]*=h/sh
    return K


def metrics(measured,target,start,position_valid):
    pe,re=errors(measured,target)
    ids=np.arange(start,len(target))
    if not len(ids):raise ValueError('No post-intervention frames')
    return dict(first_frame=int(start),last_frame=len(target)-1,
        position_mean=float(pe[ids].mean()) if position_valid else None,
        position_endpoint=float(pe[-1]) if position_valid else None,
        rotation_mean_deg=float(re[ids].mean()),rotation_endpoint_deg=float(re[-1]),
        position_errors=pe[ids].tolist() if position_valid else [None]*len(ids),
        rotation_errors_deg=re[ids].tolist())


def pair_delta(control,measured):
    result={}
    for key in ['position_mean','position_endpoint','rotation_mean_deg','rotation_endpoint_deg']:
        c,m=control[key],measured[key]
        result[key]=dict(commanded=c,measured=m,measured_minus_commanded=None if c is None or m is None else m-c,
            reduction_percent=None if c is None or m is None or abs(c)<1e-12 else 100*(c-m)/c)
    return result


def evaluate(spec,out):
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    batch=Path(spec['batch']);scene=spec['scene'];model=spec['model']
    with np.load(batch/'trajectories'/f'{scene}.npz') as z:
        target=np.linalg.inv(z['w2c_overrun' if 'w2c_overrun' in z else 'w2c'].astype(float))
        depth=z['anchor_depth'].astype(np.float32);K=z['intrinsics'][0].astype(float)
    frames={};formats={};ks={}
    for arm in ['commanded','measured']:
        source=Path(spec[arm])
        receipt=source.with_suffix('.json') if source.suffix=='.mp4' else source/'run.json'
        meta=json.loads(receipt.read_text())
        if meta.get('model')!=model or meta.get('scene')!=scene or meta.get('seed')!=spec['seed']:
            raise ValueError('Mismatched model/scene/seed in '+str(receipt))
        frames[arm],formats[arm]=load_frames(spec[arm],scene)
        ks[arm]=output_K(model,spec[arm],K,depth.shape,frames[arm].shape[1:3])
    c,m=frames['commanded'],frames['measured']
    if c.shape!=m.shape or c.dtype!=np.uint8 or m.dtype!=np.uint8:raise ValueError('Pair output shape/dtype mismatch')
    if not np.allclose(ks['commanded'],ks['measured'],rtol=0,atol=1e-5):raise ValueError('Pair camera intrinsics mismatch')
    h=len(c)-1;first=spec['first_affected_frame'];target=target[:h+1]
    if len(target)!=h+1 or not 0<first<=h:raise ValueError('Invalid target horizon or intervention boundary')
    prefix_exact=bool(np.array_equal(c[:first],m[:first]))
    event_root=Path(spec['measured'])
    if event_root.is_file():event_root=event_root.parent.parent
    events=[json.loads(line) for p in event_root.rglob('memory_events.jsonl') for line in p.read_text().splitlines()]
    report=dict(protocol='measured_memory_vipe_pair_v1',spec=spec,horizon=h,first_affected_frame=first,
        accepted_memory_writes=sum(bool(e['applied']) for e in events),
        rejected_memory_writes=[e['reason'] for e in events if not e['applied']],
        sources=formats,prefix_rgb_exact=prefix_exact,
        prefix_pixel_mae=float(np.abs(c[:first].astype(np.float32)-m[:first].astype(np.float32)).mean()),
        comparison_valid=prefix_exact if all(v.startswith('lossless') for v in formats.values()) else None,
        comparison_note='Raw pre-intervention equality required. Lossy export cannot establish raw prefix parity.',
        reference_units='fixed reference anchor-depth units; not verified metric metres',
        reference_provenance='stored batch poses may themselves be VIPE-derived; not independent ground truth',
        inference_scope='paired development evaluation; active writes, scenes and post-intervention duration reported separately',
        observer='VIPE; distinct from online DA3; RGB and intrinsics only enter SLAM; no trajectory fitting',
        arms={},started=time.time())
    for arm in ['commanded','measured']:
        f=frames.pop(arm);ko=ks[arm]
        rd=fc.remap_depth(depth,K,ko,f.shape[1:3])
        poses,meta=cached_estimate(f,ko,rd,out/(arm+'.npz'),cache_dir=out.parent/'observer_cache',evaluator_id=os.environ['CAMERA_EVALUATOR_CONTRACT'],estimate=estimate)
        valid=bool(meta['registration']['valid'])
        report['arms'][arm]=dict(position_valid=valid,registration=meta['registration'],
            post_intervention=metrics(poses,target,first,valid),
            whole_rollout=metrics(poses,target,1,valid),
            frame_sha256=hashlib.sha256(f.tobytes()).hexdigest(),K=ko.tolist(),vipe=meta)
        (out/'partial.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    report['identical_observer_inputs']=report['arms']['commanded']['vipe']['evaluation_cache']['key']==report['arms']['measured']['vipe']['evaluation_cache']['key']
    report['delta']=pair_delta(*(report['arms'][a]['post_intervention'] for a in ['commanded','measured']))
    report['completed']=time.time()
    (out/'score.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(dict(model=model,prefix_exact=prefix_exact,delta=report['delta'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    evaluate(json.loads(a.spec.read_text()),a.out)
