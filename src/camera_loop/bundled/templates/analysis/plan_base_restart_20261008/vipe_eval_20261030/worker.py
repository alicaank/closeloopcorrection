"""Isolated VIPE SLAM worker. Input contains only generated RGB and pixel K."""
import argparse,json,os,sys,time,hashlib,subprocess
from pathlib import Path
os.environ.setdefault('HF_HUB_OFFLINE','1')
p=argparse.ArgumentParser();p.add_argument('--input',type=Path);p.add_argument('--out',type=Path);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
if a.dry_run:os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,os.environ.get('CAMERA_VIPE_SOURCE','@WORKSPACE@/vipe'))
import numpy as np
import torch
from vipe.config import parse_typed_config
from vipe.slam.system import SLAMSystem
from vipe.streams.base import VideoStream,VideoFrame,FrameAttribute
from vipe.utils.cameras import CameraType
from vipe.utils.model_cache import ModelCache
OVERRIDES=['pipeline=default','pipeline.init.intrinsics=gt','pipeline.init.instance=null','pipeline.post.depth_align_model=null','pipeline.output.save_viz=false','pipeline.output.save_artifacts=false','pipeline.slam.buffer=512','pipeline.slam.keyframe_depth=unidepth-l']
cfg=parse_typed_config('default',hydra_args=OVERRIDES).pipeline.slam.to_dictconfig()
assert not cfg.optimize_intrinsics
if a.dry_run:
 print(json.dumps({'passed':True,'estimator':'VIPE','overrides':OVERRIDES,'cuda_initialized':torch.cuda.is_initialized()}));sys.exit(0)
if a.out.exists():raise FileExistsError(a.out)
with np.load(a.input,allow_pickle=False) as z:frames=z['frames'];K=z['K']
assert frames.ndim==4 and frames.shape[-1]==3 and frames.dtype==np.uint8 and len(frames)>1
assert K.shape==(3,3) and np.isfinite(K).all()
class Frames(VideoStream):
 def __init__(self):super().__init__();self.k=torch.tensor([K[0,0],K[1,1],K[0,2],K[1,2]],dtype=torch.float32)
 def frame_size(self):return frames.shape[1:3]
 def name(self):return 'generated_clip'
 def fps(self):return 16.
 def __len__(self):return len(frames)
 def attributes(self):return {FrameAttribute.INTRINSICS,FrameAttribute.CAMERA_TYPE}
 def __getitem__(self,i):return VideoFrame(raw_frame_idx=i,rgb=torch.from_numpy(frames[i].copy()).cuda().float()/255,intrinsics=self.k.clone(),camera_type=CameraType.PINHOLE)
 def __iter__(self):
  for i in range(len(self)):yield self[i]
class GuardedSLAM(SLAMSystem):
 def _add_keyframe(self,*args,**kwargs):
  phase=kwargs.get('phase',args[4] if len(args)>4 else 1)
  reserve=int(self.config.infill_chunk_size)+2 if phase==1 else 0
  if self.buffer.n_frames>=int(self.config.buffer)-reserve:raise RuntimeError('VIPE buffer capacity reached; truncated trajectories are rejected')
  return super()._add_keyframe(*args,**kwargs)
torch.manual_seed(7);torch.set_num_threads(4);start=time.time()
with torch.no_grad():
 slam=GuardedSLAM(torch.device('cuda'),cfg,ModelCache());result=slam.run([Frames()],camera_type=CameraType.PINHOLE)
 poses=result.trajectory.matrix().detach().cpu().double().numpy()
 if poses.shape!=(len(frames),4,4) or not np.isfinite(poses).all():raise ValueError('VIPE did not return every input pose')
 if int(slam.buffer.tstamp[0])!=0:raise ValueError('First VIPE keyframe is not the conditioning anchor')
 disp=slam.buffer.disps[0,0].detach().cpu().float().numpy();depth=np.full_like(disp,np.nan);valid=np.isfinite(disp)&(disp>0);depth[valid]=1/disp[valid]
 ki=slam.buffer.intrinsics[0,:4].detach().cpu().double().numpy()/8
 kg=np.array([[ki[0],0,ki[2]],[0,ki[1],ki[3]],[0,0,1.]])
 recovered=result.intrinsics[0,:4].detach().cpu().double().numpy()
 if not np.allclose(recovered,[K[0,0],K[1,1],K[0,2],K[1,2]],atol=.001):raise ValueError('VIPE intrinsics changed')
 np.savez_compressed(a.out,c2w=poses,anchor_depth=depth,anchor_K=kg)
 source=Path(sys.path[0]);rev=subprocess.run(['git','-C',str(source),'rev-parse','HEAD'],capture_output=True,text=True).stdout.strip()
 meta={'estimator':'VIPE','source_commit':rev,'worker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'frames':len(frames),'keyframes':int(slam.buffer.n_frames),'seconds':time.time()-start,'overrides':OVERRIDES,'anchor_depth':'optimized frame-0 depth in the same gauge as final SLAM poses','commanded_poses_supplied':False,'torch_peak_gib':torch.cuda.max_memory_allocated()/2**30}
 a.out.with_suffix('.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)
