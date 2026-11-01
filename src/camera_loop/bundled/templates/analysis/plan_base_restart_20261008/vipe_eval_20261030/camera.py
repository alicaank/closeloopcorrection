"""CPU geometry and serialized VIPE worker interface; no path-fitting registration."""
import json,os,subprocess,tempfile
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
R=Path(__file__).resolve().parent

def register(raw,native_depth,reference_depth):
 raw=np.asarray(raw,float);nd=np.asarray(native_depth,float);rd=np.asarray(reference_depth,float)
 if raw.ndim!=3 or raw.shape[1:]!=(4,4) or not np.isfinite(raw).all():raise ValueError('Invalid VIPE poses')
 if not np.allclose(raw[:,3],[0,0,0,1],atol=1e-5):raise ValueError('Invalid homogeneous poses')
 rotations=raw[:,:3,:3]
 if np.max(np.abs(rotations.transpose(0,2,1)@rotations-np.eye(3)))>1e-3 or np.max(np.abs(np.linalg.det(rotations)-1))>1e-3:raise ValueError('Invalid VIPE rotations')
 if nd.shape!=rd.shape or nd.ndim!=2:raise ValueError('Depth pixels must be registered')
 # The pinned VIPE retractor clamps inverse depth to [1e-3,10]. Clamp
 # endpoints are optimizer saturation, not supported surface measurements.
 saturated=(nd<=0.1*(1+1e-5))|(nd>=1000*(1-1e-5))
 valid=np.isfinite(nd)&np.isfinite(rd)&~saturated&(rd>1e-6)
 if valid.sum()<256 or valid.mean()<.2:raise ValueError('VIPE anchor depth coverage insufficient')
 ratios=rd[valid]/nd[valid];scale=float(np.median(ratios));p10,p90=np.quantile(ratios,[.1,.9])
 if not np.isfinite(scale) or scale<=0 or p90/p10>4:raise ValueError('VIPE anchor depth does not support a scalar gauge conversion')
 measured=np.linalg.inv(raw[0])@raw;measured[:,:3,3]*=scale
 return measured,dict(scale=scale,saturated_pixels=int(saturated.sum()),valid_pixels=int(valid.sum()),valid_fraction=float(valid.mean()),p90_over_p10=float(p90/p10),median_relative_residual=float(np.median(np.abs(ratios/scale-1))),method='reference/native optimized anchor z-depth at matching pixels, excluding native disparity saturation; first-pose SE3 reanchor; no trajectory fit')

def estimate(frames,K,reference_depth,output):
 import sys
 sys.path.insert(0,str(R.parent/'fair_common_20261025'));import fair_common as fc
 output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
 if output.exists():raise FileExistsError('Existing VIPE trajectory preserved: '+str(output))
 py=os.environ.get('CAMERA_VIPE_PYTHON','@WORKSPACE@/vipe/.venv/bin/python')
 if not Path(py).is_file():raise FileNotFoundError('VIPE interpreter missing: '+py)
 with tempfile.TemporaryDirectory(prefix='vipe-ipc-',dir=output.parent) as td:
  td=Path(td);np.savez(td/'input.npz',frames=frames,K=K)
  env=dict(os.environ,HF_HUB_OFFLINE='1',PYTHONUNBUFFERED='1');env.pop('LD_LIBRARY_PATH',None)
  with output.with_suffix('.log').open('x') as log:
   subprocess.run([py,str(R/'worker.py'),'--input',str(td/'input.npz'),'--out',str(td/'raw.npz')],stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
  with np.load(td/'raw.npz') as z:
   native=z['anchor_depth'];ref=fc.remap_depth(reference_depth,K,z['anchor_K'],native.shape)
   measured,quality=register(z['c2w'],native,ref);raw=z['c2w'].copy()
  metadata=json.loads((td/'raw.json').read_text());metadata['registration']=quality
  np.savez_compressed(output,c2w=measured,raw_c2w=raw,frame_ids=np.arange(len(frames)))
  output.with_suffix('.json').write_text(json.dumps(metadata,indent=2))
 return measured,metadata

def errors(measured,target):
 pe=np.linalg.norm(measured[:,:3,3]-target[:,:3,3],axis=-1)
 rot=np.degrees(Rotation.from_matrix(measured[:,:3,:3].transpose(0,2,1)@target[:,:3,:3]).magnitude())
 return pe,rot

def path_metrics(measured,target,minimum_target_length=1e-8):
 """Translation diagnostics in fixed anchor units, without a fabricated denominator.

 The threshold is numerical, not an estimator noise floor. Rotation-only targets
 still have meaningful absolute translation drift, but no translation path ratio.
 """
 measured=np.asarray(measured,float);target=np.asarray(target,float)
 if measured.ndim!=3 or measured.shape[1:]!=(4,4) or measured.shape!=target.shape or not len(measured):
  raise ValueError('Expected matching nonempty pose sequences')
 if not np.isfinite(measured).all() or not np.isfinite(target).all():raise ValueError('Nonfinite trajectory')
 if not np.isfinite(minimum_target_length) or minimum_target_length<0:raise ValueError('Invalid path-length threshold')
 actual=float(np.linalg.norm(np.diff(measured[:,:3,3],axis=0),axis=1).sum())
 planned=float(np.linalg.norm(np.diff(target[:,:3,3],axis=0),axis=1).sum())
 valid=planned>minimum_target_length
 return dict(measured_path_length=actual,target_path_length=planned,
  path_length_ratio=actual/planned if valid else None,
  path_length_ratio_valid=valid,
  path_length_ratio_reason=None if valid else 'insufficient_target_translation',
  minimum_target_path_length=minimum_target_length)
