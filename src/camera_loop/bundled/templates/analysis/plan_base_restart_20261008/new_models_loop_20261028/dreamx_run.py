"""Native DreamX schedule, optional causal feedback; CPU offload only during VAE decode."""
import argparse,inspect,textwrap,sys,json,time,hashlib,os
from pathlib import Path
import numpy as np
from PIL import Image
R=Path(__file__).resolve().parent;P=R.parent/'model_readiness_20261027';REPO=P/'repos/dreamx'
sys.path[:0]=[str(REPO),str(R)]
from common import PoseWorker,rejoin

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--batch',type=Path,required=True);ap.add_argument('--scene',required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--arm',choices=['native','baseline','correction','sham'],required=True);ap.add_argument('--latents',type=int,default=9);ap.add_argument('--seed',type=int,default=7);a=ap.parse_args()
 assert a.latents%3==0;a.out.mkdir(parents=True,exist_ok=False)
 import torch
 import inference_ar_forcing as entry
 import pipeline.pipeline_causal_camera as lib
 row=next(x for x in json.load(open(a.batch/'index.json'))['scenes'] if x['name']==a.scene)
 n=4*(a.latents-1)+1
 with np.load(a.batch/'trajectories'/f'{a.scene}.npz') as z:
  plan=np.linalg.inv(z['w2c_overrun'].astype(np.float64))[:n];depth=z['anchor_depth'].astype(np.float32);K=z['intrinsics'][0].copy()
 assert len(plan)==n
 K[0,:]*=1280/864;K[1,:]*=704/480
 depth=np.asarray(Image.fromarray(depth).resize((1280,704),Image.Resampling.BILINEAR))
 np.save(a.out/'commands.npy',plan);commands=plan.copy();ids=[0]+[1+4*i for i in range(a.latents-1)]
 (a.out/'input.json').write_text(json.dumps([{'image_path':str(a.batch/row['image']),'caption':row['caption'],'camera_c2w_path':str(a.out/'commands.npy'),'task_id':a.scene}]))
 events=[];worker=None
 def observe(pipe,output,cam,lo,total):
  nonlocal worker
  if lo>=total:return
  t=time.time();hi=min(lo+3,total);frame_id=ids[lo-1]
  # Decode only already generated latents. Native camera sampling is [0,1,5,...].
  # Measure at its matching image index, leaving the last three available frames unused.
  with torch.random.fork_rng(devices=[0]):
   pipe.generator.cpu();torch.cuda.empty_cache()
   decoded=pipe.vae.decode_to_pixel(output[:,:lo]);pipe.vae.model.clear_cache()
   frames=((decoded[0].float()+1)*127.5).round().clamp(0,255).byte().permute(0,2,3,1).cpu().numpy();del decoded
   torch.cuda.empty_cache()
   assert frame_id<len(frames)
   if worker is None:worker=PoseWorker(a.out/'observer')
   measured,info=worker.measure(frames[:frame_id+1],K,depth)
   pipe.generator.to(output.device)
  before=commands.copy();future=ids[lo:hi]
  if a.arm in ['correction','sham']:
   m=measured if a.arm=='correction' else commands[ids[lo-1]]
   commands[future]=rejoin(m,commands[ids[lo-1]],plan[future])
   # Reuse exactly the native per-chunk camera construction.
   conditioning_commands=commands.copy();conditioning_commands[ids[lo-1]]=m
   np.save(a.out/'commands.npy',conditioning_commands)
   from camera_input import load_camera_rows
   cams=[entry.Camera(v.tolist()) for v in load_camera_rows(a.out/'commands.npy',n)]
   updated=entry.cam_params_to_prope_dict(cams,device=output.device,chunk_relative=True)
   start,end=lo*pipe.frame_seq_length,hi*pipe.frame_seq_length
   cam['viewmats'][:,start:end].copy_(updated['viewmats'][:,start:end])
   assert np.array_equal(commands[:future[0]],before[:future[0]])
   assert np.allclose(commands[future[-1]],plan[future[-1]])
  events.append({'latent_boundary':lo,'observed_frame':frame_id,'available_last_frame':len(frames)-1,'future_camera_frame_ids':future,'measured':measured.tolist(),'correction_applied':a.arm=='correction','seconds':time.time()-t,'measurement':info,'prefix_sha256':hashlib.sha256(frames.tobytes()).hexdigest()})
  (a.out/'events.json').write_text(json.dumps(events,indent=2))
 src=textwrap.dedent(inspect.getsource(lib.CausalCameraInferencePipeline.inference))
 needle='\n    video = self.vae.decode_to_pixel(output)';assert src.count(needle)==1
 src=src.replace(needle,'\n    self.generator.cpu()\n    torch.cuda.empty_cache()'+needle)
 if a.arm!='native':
  needle='\n    if profile:\n        diffusion_end.record()';assert src.count(needle)==1
  src=src.replace(needle,'\n        _observe(self, output, y_camera, current_start_frame, num_output_frames)\n'+needle)
 scope=dict(lib.__dict__,_observe=observe);exec(compile(src,str(R/'dreamx_hook_generated.py'),'exec'),scope)
 lib.CausalCameraInferencePipeline.inference=scope['inference'];(a.out/'hook_source.py').write_text(src)
 write=entry.write_video
 def save(path,video,**kw):
  np.savez_compressed(a.out/'frames.npz',frames=video.cpu().numpy().astype(np.uint8));return write(path,video,**kw)
 entry.write_video=save
 sys.argv=['inference_ar_forcing.py','--config_path',str(REPO/'configs/dreamx-ar/causal_camera_forcing_5b.yaml'),'--model_name','@WORKSPACE@/FrameCrafter/models/Wan-AI/Wan2.2-TI2V-5B','--transformer_path',str(REPO/'configs/dreamx-ar'),'--base_checkpoint_path',str(P/'weights/dreamx/model.safetensors'),'--data_path',str(a.out/'input.json'),'--output_folder',str(a.out),'--num_output_frames',str(a.latents),'--fps','16','--seed',str(a.seed),'--color_correction_strength','1.0','--chunk_relative']
 t=time.time()
 try:entry.main()
 finally:
  if worker is not None:worker.close()
 (a.out/'run.json').write_text(json.dumps({'model':'dreamx','arm':a.arm,'scene':a.scene,'seed':a.seed,'seconds':time.time()-t,'events':events,'plan':plan.tolist(),'commands':commands.tolist(),'K':K.tolist(),'native_camera_frame_ids':ids,'deviations':['Generator offloaded only during VAE decode in every arm.','Native fixed camera intrinsics retained; observer uses resized input K.'],'feedback_delay_frames':3},indent=2))
if __name__=='__main__':main()
