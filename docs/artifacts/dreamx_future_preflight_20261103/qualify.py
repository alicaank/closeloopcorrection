"""CPU-only DreamX command timeline/bridge qualification using released encoding."""
import ast,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
W=Path('/home/hj/Alican');R=Path(__file__).resolve().parent
sys.path.insert(0,str(W/'camera-closed-loop/src'))
from camera_loop.fractional_recovery import issue_fractional_future,fractional_memory_continuation
repo=W/'analysis/plan_base_restart_20261008/model_readiness_20261027/repos/dreamx'
entry=repo/'inference_ar_forcing.py';camera=repo/'utils/trajectory_processor.py'
ns=dict(np=np,torch=torch)
def extract(path,names):
 nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
 assert {n.name for n in nodes}==set(names)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
extract(camera,['Camera']);extract(entry,['_invert_SE3','get_relative_pose','cam_params_to_prope_dict'])
def encode(poses,dtype):
 rows=np.zeros((len(poses),19));rows[:,1:5]=[.5,.9,.5,.5];rows[:,7:]=np.linalg.inv(poses)[:,:3].reshape(-1,12)
 return ns['cam_params_to_prope_dict']([ns['Camera'](r) for r in rows],device='cpu',dtype=dtype,chunk_relative=True)
def digest(t):return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
batch=W/'analysis/plan_base_restart_20261008/continuous_eval_20261029/batch_continuous1009'
results=[]
for scene in ['02267acf','c7418b7d','0238de14','924228bf']:
 path=batch/'trajectories'/f'{scene}.npz'
 with np.load(path) as z:plan=np.linalg.inv(z['w2c_overrun'].astype(float))[:69];K=z['intrinsics'][0]
 # Synthetic observations isolate arithmetic; these are not measured rollouts.
 ids=np.array([0,24,28,29,32]);history=plan[ids].copy()
 lag=np.array([.03,-.01,.02]);history[1:,:3,3]-=lag
 history[-2,:3,:3]=history[-2,:3,:3]@Rotation.from_euler('y',7,degrees=True).as_matrix()
 args=dict(next_end=42,available_frame=32,feedback_index=1,nominal_boundaries=1,history_labels='commanded',native_stride=4,recovery_fraction=.5,speed_multiplier=2.,memory_anchor_frame=29)
 issued=issue_fractional_future(plan,plan,history,ids,**args)
 desired=fractional_memory_continuation(plan[:42],history,ids,native_stride=4,recovery_fraction=.5,speed_multiplier=2.)
 native_ids=np.array([0]+[1+4*i for i in range(17)])
 future=np.array([33,37,41]);lo=9*880;hi=12*880
 expected=np.linalg.inv(history[-2])@desired.commands[future]
 encoded={}
 for dtype in [torch.float32,torch.bfloat16]:
  baseline=encode(plan,dtype);candidate=encode(issued.commands,dtype)
  actual=np.linalg.inv(candidate['viewmats'][0,lo:hi:880].double().numpy())
  error=float(np.max(np.abs(actual-expected)))
  assert error<(2e-6 if dtype==torch.float32 else .025),(scene,dtype,error)
  assert torch.equal(baseline['viewmats'][:,:lo],candidate['viewmats'][:,:lo])
  assert torch.equal(baseline['K'],candidate['K'])
  # Host must commit only the next chunk; full re-encoding may change a later
  # chunk's relative reference even though those future absolute targets did not change.
  committed={k:v.clone() for k,v in baseline.items()}
  committed['viewmats'][:,lo:hi].copy_(candidate['viewmats'][:,lo:hi])
  assert torch.equal(committed['viewmats'][:,hi:],baseline['viewmats'][:,hi:])
  encoded[str(dtype)]=dict(max_matrix_error=error,prefix_exact=True,tail_exact=True,intrinsics_unchanged=True,future_sha256=digest(committed['viewmats'][:,lo:hi]))
 np.testing.assert_array_equal(issued.commands[:33],plan[:33]);np.testing.assert_array_equal(issued.commands[42:],plan[42:])
 wrong=issue_fractional_future(plan,plan,history,ids,**dict(args,memory_anchor_frame=32))
 wrong_relative=np.linalg.inv(plan[29])@wrong.commands[future]
 wrong_error=float(np.max(np.abs(wrong_relative-expected)));assert wrong_error>.01
 first=issue_fractional_future(plan,plan,history,ids,**dict(args,feedback_index=0))
 assert np.array_equal(first.commands,plan)
 nativeK=encode(plan,torch.float32)['K'][0,0].numpy();native_pixel=nativeK.copy();native_pixel[0]*=864;native_pixel[1]*=480
 results.append(dict(scene=scene,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),last_emitted_rgb=32,last_native_camera=29,next_native_cameras=future.tolist(),nominal_first_issuance_exact=True,wrong_endpoint_bridge_error=wrong_error,encoding=encoded,source_K=K.tolist(),native_pixel_K=native_pixel.tolist(),focal_ratio=(np.diag(native_pixel)[:2]/np.diag(K)[:2]).tolist()))
out=dict(passed=True,scope='CPU arithmetic and actual released encoder; synthetic lag/rotation observations, no GPU generation or efficacy test',sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [entry,camera,Path(__file__),W/'camera-closed-loop/src/camera_loop/fractional_recovery.py']},rows=results,remaining=['Matched native GPU observer/no-op parity','Resolve native fixed-K versus source intrinsics contract before efficacy qualification','Freeze future-only pilot with commanded memory; audit active boundaries and repeated tracking'])
(R/'qualification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
