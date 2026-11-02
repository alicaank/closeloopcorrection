"""Persistent causal DA3 worker; receives selected RGB/K/depth, never commands."""
import os, sys, json, time, hashlib, random, traceback
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import map_coordinates

ROOT = Path(os.environ['CAMERA_LOOP_WORKSPACE'])
sys.path.insert(0,str(ROOT/'analysis/plan_base_restart_20261008/memorizon_plugins'))
from camera_loop.evidence import assess_window
from camera_loop.online_units import remap_depth, array
from realized_pose import load_da3, _pad44


def main():
    proto = os.fdopen(os.dup(1), 'w', buffering=1)
    os.dup2(2, 1);sys.stdout = sys.stderr
    torch.set_num_threads(2)
    da3 = load_da3('cpu')
    previous = None
    proto.write('READY '+json.dumps(dict(fake=False, pid=os.getpid(), protocol='overlap_evidence_v1'))+'\n')
    for line in sys.stdin:
        request = json.loads(line)
        if request['cmd'] == 'quit':break
        try:
            start = time.time()
            with np.load(request['req']) as z:
                views=z['views'].copy();ids=z['frame_ids'].copy();K=z['K'].copy();reference=z['anchor_depth'].copy()
            seed = int.from_bytes(hashlib.sha256(views.tobytes()+K.tobytes()+reference.tobytes()).digest()[:4], 'little') % (2**31)
            torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
            torch.cuda.reset_peak_memory_stats();da3.to('cuda')
            try:
                with torch.no_grad():
                    pred=da3.inference(image=list(views), intrinsics=np.repeat(K[None],len(views),0),
                        extrinsics=None,align_to_input_extrinsics=False,align_to_input_ext_scale=False,
                        infer_gs=False,process_res=504,export_dir=None)
            finally:da3.to('cpu');torch.cuda.empty_cache()
            native=np.linalg.inv(_pad44(pred.extrinsics))
            depths=np.asarray(pred.depth);intrinsics=np.asarray(pred.intrinsics)
            images=np.asarray(pred.processed_images)
            if images.dtype!=np.uint8 or images.shape!=(*depths.shape,3):
                raise ValueError('DA3 processed RGB/depth grid contract changed')
            _,_,processed_K=da3.input_processor(image=list(views),intrinsics=np.repeat(K[None],len(views),0),
                process_res=504,process_res_method='upper_bound_resize')
            ref=remap_depth(reference,K,array(processed_K)[0],depths.shape[1:])
            registered,evidence=assess_window(ids,native,depths,ref,intrinsics,images,previous)
            if evidence['passed']:previous={int(i):p.copy() for i,p in zip(ids,registered)}
            evidence.update(observer_seed=seed,seconds=time.time()-start,peak_gib=torch.cuda.max_memory_allocated()/2**30)
            np.savez_compressed(request['out'], poses=registered, frame_ids=ids)
            proto.write(json.dumps(dict(ok=True,quality=evidence,accepted_ids=list(previous or {})))+'\n')
        except Exception as exc:
            traceback.print_exc()
            proto.write(json.dumps(dict(ok=False,error=f'{type(exc).__name__}: {exc}'))+'\n')


if __name__=='__main__':main()
