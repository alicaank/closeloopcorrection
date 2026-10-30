"""WorldWarp external trajectory adapter; native 81-frame chunks with one-frame overlap."""
import argparse,sys,json,time,hashlib,os
from pathlib import Path
import numpy as np
from PIL import Image
R=Path(__file__).resolve().parent;P=R.parent/'model_readiness_20261027';sys.path[:0]=[str(P/'repos/worldwarp'),str(R)]
# This overlay imports torch/ninja from the Spatia environment. Python can see
# ninja's package via .pth, but cpp_extension invokes its executable via PATH.
os.environ['PATH']=os.pathsep.join([str(P/'envs/worldwarp312/bin'),'@WORKSPACE@/baselines/venv_spatia/bin',os.environ.get('PATH','')])
from common import rejoin,PoseWorker

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--batch',type=Path,required=True);ap.add_argument('--scene',required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--arm',choices=['native','baseline','sham','correction'],required=True);ap.add_argument('--chunks',type=int,default=2);ap.add_argument('--seed',type=int,default=7)
 ap.add_argument('--scale_mode',choices=['anchor','legacy'],default='anchor');ap.add_argument('--camera_window',choices=['fixed','legacy'],default='fixed')
 a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 if a.camera_window=='legacy' and a.chunks>1:raise ValueError('Legacy continuation indexing is structurally invalid; only allowed in one-chunk scale ablation')
 import torch,imageio.v2 as imageio
 import inspect,textwrap
 from omegaconf import OmegaConf
 from pose_control import CONFIG,WanVideoGenerator,seed_everything
 import models.pos_embed as pos_embed
 # The released TTT3R pose token uses position -1. The Python RoPE fallback
 # incorrectly indexes an embedding table with that negative index; CUDA RoPE
 # evaluates sin/cos at signed positions. Match that signed-angle contract.
 def signed_rope(self,tokens,positions):
  D=tokens.shape[-1]//2
  cos,sin=self.get_cos_sin(D,int(positions.abs().max())+1,tokens.device,tokens.dtype)
  def axis(x,pos):
   c=torch.nn.functional.embedding(pos.abs(),cos)[:,None]
   sn=torch.nn.functional.embedding(pos.abs(),sin)[:,None]*pos.sign()[:,None,:,None]
   return x*c+self.rotate_half(x)*sn
  y,x=tokens.chunk(2,dim=-1)
  return torch.cat([axis(y,positions[:,:,0]),axis(x,positions[:,:,1])],dim=-1)
 assert hasattr(pos_embed.RoPE2D,'get_cos_sin'),'Audit native compiled RoPE instead of replacing it'
 pos_embed.RoPE2D.forward=signed_rope
 row=next(x for x in json.load(open(a.batch/'index.json'))['scenes'] if x['name']==a.scene)
 with np.load(a.batch/'trajectories'/f'{a.scene}.npz') as z:
  plan=np.linalg.inv(z['w2c_overrun'].astype(np.float64))[:80*a.chunks+1];K=z['intrinsics'][0].copy();depth=z['anchor_depth'][:,72:792].copy()
 assert len(plan)==80*a.chunks+1;K[0,2]-=72
 anchor=np.asarray(Image.open(a.batch/row['image']).convert('RGB').crop((72,0,792,480)))
 cfg=OmegaConf.create(OmegaConf.to_container(CONFIG,resolve=True));cfg.experiment.seed=a.seed;cfg.experiment.output_root=str(a.out/'run');cfg.prompts.positive=row['caption']
 cfg.paths.base_model_path=str(P/'weights/worldwarp/Wan-AI/Wan2.1-T2V-1.3B-Diffusers');cfg.paths.finetuned_checkpoint_path=str(P/'weights/worldwarp/worldwarp_latest.ckpt')
 ck=list((Path.home()/'.cache/huggingface/hub/models--liguang0115--cut3r/snapshots').glob('*/cut3r_512_dpt_4_64.pth'));assert len(ck)==1;cfg.paths.ttt3r_model_path=str(ck[0])
 cfg.loop_params.n_chunks=a.chunks;cfg.loop_params.output_fps=16;cfg.inference_params.n_frames=81;cfg.paths.input_video_path=str(a.out/'anchor_only.mp4');cfg.camera_pose_control.enabled=True
 assert cfg.inference_params.context_frames==1 and cfg.inference_params.context_frames_2nd==1
 imageio.mimwrite(cfg.paths.input_video_path,[anchor]*81,fps=16,macro_block_size=1)
 worker=None;events=[];prefix=[]
 sys.path.insert(0,str(R.parent/'continuous_eval_20261029'))
 from anchor_scale import fit_scale,freeze_scale,worldwarp_camera_window
 from src.ttt3r import ttt3r as geometry
 scale_state={}
 def calibrate_native(camera_poses,depth_maps,is_first_chunk):
  if not is_first_chunk:
   assert scale_state, 'Initial anchor scale must be frozen before continuation'
   return camera_poses
  assert not scale_state, 'Anchor calibration must run only once'
  native=depth_maps[0,0].detach().float().cpu().numpy()
  scale,receipt=fit_scale(depth,native)
  scale,receipt=freeze_scale(scale,receipt,'worldwarp',a.scene)
  receipt['calibrated_native_units_per_reference_unit']=scale
  if a.scale_mode=='legacy':scale=1.0
  receipt.update(applied_native_units_per_reference_unit=scale,scale_mode=a.scale_mode,camera_window=a.camera_window)
  plan[:,:3,3]*=scale;depth[:]*=scale
  result=camera_poses.clone();result[...,:3,3]*=scale
  receipt.update(model='worldwarp',native_depth_stage='TTT3R initial anchor depth',observer_pose_scale=scale)
  scale_state.update(receipt)
  (a.out/'scale_calibration.json').write_text(json.dumps(receipt,indent=2))
  return result
 # Native reconstruction and commanded poses must meet in the same units.
 source=textwrap.dedent(inspect.getsource(geometry.GS3DWarper.warp_with_validity_mask))
 marker='    print(f"  Generated depth and poses for {T_video} frames")'
 assert source.count(marker)==1, 'WorldWarp geometry hook changed; audit required'
 source=source.replace(marker,'    camera_poses = _calibrate_native(camera_poses, depth_maps, is_first_chunk)\n'+marker)
 scope=dict(geometry.__dict__,_calibrate_native=calibrate_native)
 exec(compile(source,str(a.out/'geometry_scale_hook.py'),'exec'),scope)
 geometry.GS3DWarper.warp_with_validity_mask=scope['warp_with_validity_mask']
 (a.out/'geometry_scale_hook.py').write_text(source)
 class Generator(WanVideoGenerator):
  def _init_captioner(self,is_vl=True):self.caption_model=None
  def _load_models(self):
   super()._load_models()
   encode=self.text_pipe.encode_prompt
   def staged(*args,**kw):
    self.text_pipe.text_encoder.to(self.device)
    try:return encode(*args,**kw)
    finally:self.text_pipe.text_encoder.cpu();torch.cuda.empty_cache()
   self.text_pipe.encode_prompt=staged;self.text_pipe.text_encoder.cpu();torch.cuda.empty_cache()
  def _generate_controlled_poses(self):
   return torch.from_numpy(plan)[None].float(),torch.from_numpy(np.repeat(K[None],len(plan),axis=0))[None].float()
  def run_inference_chunk(self,i,input_video_path,unused_poses,unused_K,ctx,is_first,**kw):
   nonlocal worker,prefix
   lo=i*80;target=plan[lo:lo+81].copy()
   if i and a.arm!='native':
    if worker is None:worker=PoseWorker(a.out/'observer')
    with torch.random.fork_rng(devices=[0]):measured,info=worker.measure(np.asarray(prefix),K,depth)
    if a.arm in ['correction','sham']:
     start=measured if a.arm=='correction' else plan[lo]
     target[1:]=rejoin(start,plan[lo],plan[lo+1:lo+81]);target[0]=start
    events.append({'chunk':i,'observed_frame':lo,'measured':measured.tolist(),'conditioning_path':target.tolist(),'correction_applied':a.arm=='correction','measurement':info,'prefix_sha256':hashlib.sha256(np.asarray(prefix).tobytes()).hexdigest()});(a.out/'events.json').write_text(json.dumps(events,indent=2))
   # Feed actual output-time poses: native run() advances its source-array slice by ctx,
   # although stitched video advances by 81-ctx. Both arms use the same external adapter.
   # Native geometry indexes the 81 source frames followed by 80 new targets.
   # Source poses are replaced by native TTT3R estimates; their final slot fixes
   # the origin for the future command. An 81-pose array loses those targets.
   combined=target if a.camera_window=='legacy' else worldwarp_camera_window(target,is_first)
   assert len(combined)==(81 if is_first else 161)
   intr=np.repeat(K[None],len(combined),axis=0)
   path=super().run_inference_chunk(i,input_video_path,torch.from_numpy(combined)[None].float().to(self.device),torch.from_numpy(intr)[None].float().to(self.device),ctx,is_first,**kw)
   with imageio.get_reader(path) as reader:frames=[f for f in reader]
   assert len(frames)==81;prefix.extend(frames if not i else frames[ctx:]);return path
 seed_everything(a.seed);t=time.time()
 try:Generator(cfg).run()
 finally:
  if worker is not None:worker.close()
 np.savez_compressed(a.out/'frames.npz',frames=np.asarray(prefix,np.uint8))
 (a.out/'run.json').write_text(json.dumps({'model':'worldwarp','arm':a.arm,'scene':a.scene,'seed':a.seed,'seconds':time.time()-t,'events':events,'plan':plan.tolist(),'K':K.tolist(),'deviations':['Fixed supplied caption; only starting RGB repeated as native GUI source.','T5 offloaded after encoding, no dtype change. Python RoPE fallback fixed to handle the native -1 pose-token coordinate with signed sine/cosine, matching the released CUDA formula; GPU attention unchanged.','External path adapter advances by 80 emitted frames per chunk, preserving native 81/1 chunk/overlap.','Frames decoded from native mp4; identical encoding in all arms.']},indent=2))
if __name__=='__main__':main()
