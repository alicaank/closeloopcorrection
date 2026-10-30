"""Independent first-80-frame execution gate. Never fit scale to the trajectory."""
import argparse,json,sys,time,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
R=Path(__file__).resolve().parent;A=R.parent.parent
sys.path[:0]=[str(A/'fair_common_20261025'),str(A/'memorizon_plugins'),str(A/'new_models_loop_20261028'),
 '@WORKSPACE@/training_runs/joint_samplerfix_20261012',str(A/'fresh_teacher_20261021'),
 '@WORKSPACE@/training_runs/memory1500_resplat_20260905/python_deps']

def execution_metrics(measured,commanded):
    x=commanded[:,:3,3]-commanded[0,:3,3];y=measured[:,:3,3]-measured[0,:3,3]
    denom=float(np.linalg.norm(x[-1]));energy=float((x*x).sum())
    if denom<1e-4 or energy<1e-6:raise ValueError('Insufficient commanded motion for execution ratio')
    ratio=float(np.linalg.norm(y[-1])/denom)
    gain=float((x*y).sum()/energy)
    return dict(endpoint_execution_ratio=ratio,projected_gain=gain,
        path_length_ratio=float(np.linalg.norm(np.diff(y,axis=0),axis=1).sum()/np.linalg.norm(np.diff(x,axis=0),axis=1).sum()),
        endpoint_position_error=float(np.linalg.norm(y[-1]-x[-1])),
        mean_position_error=float(np.linalg.norm(y[1:]-x[1:],axis=1).mean()),
        passed=bool(np.isfinite(measured).all() and 0<ratio<=1.3 and gain>0),
        gate_definition='norm(t80-t0 measured) / norm(t80-t0 commanded) <= 1.3; finite poses and positive projected motion required',
        expected_ratio_hypothesis=1.05,upper_limit=1.3)

def read_views(source,scene,ids):
    if (source/f'{scene}_frames').exists():
        return [np.asarray(Image.open(source/f'{scene}_frames'/f'frame_{i:05d}.png').convert('RGB')) for i in ids]
    if (source/'frames.npz').exists():
        with np.load(source/'frames.npz') as z:return list(z['frames'][ids])
    import cv2
    path=source if source.is_file() else source/'review_full.mp4'
    cap=cv2.VideoCapture(str(path));assert cap.isOpened(),path;out=[]
    for i in range(max(ids)+1):
        ok,f=cap.read();assert ok,(path,i)
        if i in ids:out.append(cv2.cvtColor(f,cv2.COLOR_BGR2RGB))
    cap.release();return out

def main():
    p=argparse.ArgumentParser()
    for k in ['batch','model','scene','out']:p.add_argument('--'+k,required=True)
    p.add_argument('--runs',nargs='+',required=True);a=p.parse_args()
    from score_vggt import load_vggt
    from score_vggt_fair import vggt_relposes,vggt_size
    from score_pair import transform
    import fair_common as fc
    batch=Path(a.batch);out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    ids=list(range(0,81,4))
    with np.load(batch/'trajectories'/f'{a.scene}.npz') as z:
        key='w2c_overrun' if 'w2c_overrun' in z else 'w2c'
        plan=np.linalg.inv(z[key].astype(np.float64))[ids];depth=z['anchor_depth'].astype(np.float32);K=z['intrinsics'][0]
    model=load_vggt();result=dict(model=a.model,scene=a.scene,horizon=80,frame_ids=ids,
        units='fixed stored reference anchor-depth gauge; no pose_scale division and no fitted trajectory alignment',
        expected_ratio_is_hypothesis=True,arms={})
    for spec in a.runs:
        arm,path=spec.split('=',1);source=Path(path);views=read_views(source,a.scene,ids);h,w=views[0].shape[:2]
        if a.model=='lyra':dep=fc.remap_depth(depth,K,fc.scale_crop_K(K,w/864,h/480),(h,w))
        else:dep=transform(depth,a.model,True)
        measured,scale=vggt_relposes(model,views,dep,vggt_size((h,w)))
        metrics=execution_metrics(measured,plan)
        metrics.update(measured_c2w=measured.tolist(),commanded_c2w=plan.tolist(),vggt_anchor_depth_scale=scale,
            source=str(source),sampled_rgb_sha256=hashlib.sha256(np.stack(views).tobytes()).hexdigest())
        result['arms'][arm]=metrics;result['completed']=time.time()
        out.write_text(json.dumps(result,indent=2));print(arm,{k:metrics[k] for k in ['endpoint_execution_ratio','projected_gain','path_length_ratio','passed']},flush=True)

if __name__=='__main__':main()
