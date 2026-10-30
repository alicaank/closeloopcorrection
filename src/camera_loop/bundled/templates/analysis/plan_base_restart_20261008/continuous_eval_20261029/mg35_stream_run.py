import argparse,sys,os,json,subprocess,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
R=Path(__file__).resolve().parent;REPO=Path('@WORKSPACE@/sota_repos/matrix_game/Matrix-Game-3.5');sys.path.insert(0,str(REPO));import infer as MG

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--batch',type=Path,required=True);ap.add_argument('--scene',required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--arm',choices=['native','baseline','sham','correction'],required=True);ap.add_argument('--seed',type=int,default=7);ap.add_argument('--blocks',type=int,default=6);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 row=next(s for s in json.load(open(a.batch/'index.json'))['scenes'] if s['name']==a.scene)
 with np.load(a.batch/'trajectories'/f'{a.scene}.npz') as z:
  plan=np.linalg.inv(z['w2c_overrun'].astype(np.float64))[:1+84*a.blocks];K=z['intrinsics'][0].astype(np.float32);depth=z['anchor_depth'].astype(np.float32);mpu=float(z['metres_per_anchor_unit']) if 'metres_per_anchor_unit' in z else float(row['metres_per_anchor_unit'])
 assert len(plan)==1+84*a.blocks;plan[:,:3,3]*=mpu;depth*=mpu
 np.savez(a.out/'camera.npz',extrinsics_c2w=plan.astype(np.float32),intrinsics=np.array([K[0,0],K[1,1],K[0,2],K[1,2]],np.float32))
 K[0,:]*=1280/864;K[1,:]*=704/480;np.save(a.out/'depth.npy',np.asarray(Image.fromarray(depth).resize((1280,704),Image.Resampling.BILINEAR)))
 meta={'out':str(a.out),'arm':a.arm,'plan':plan.tolist(),'K':K.tolist(),'depth':str(a.out/'depth.npy'),'seed':a.seed,'scene':a.scene,'metres_per_anchor_unit':mpu};(a.out/'input_meta.json').write_text(json.dumps(meta,indent=2))
 args=SimpleNamespace(person='first',image=str(a.batch/row['image']),camera=str(a.out/'camera.npz'),prompt=row['caption'],prompt_file='',caption='',refs='',num_blocks=a.blocks,steps=0,cfg_scale=0.,seed=a.seed,output=str(a.out/'native_output'),name=a.out.name+'_'+a.scene,camera_convention='c2w',keep_workspace=True,ckpt='',wan_dir='',tokenizer_dir='')
 env=dict(os.environ,MG35_REPO=str(REPO),MG_LOOP_META=str(a.out/'input_meta.json'),DA3_MODEL_PATH=MG.resolve_weight('da3',''),DIFFSYNTH_SKIP_DOWNLOAD='true');env['PYTHONPATH']=str(REPO)
 os.chdir(REPO);out_root,tmp,ws=MG.build_workspace(args);index=MG.build_index(ws,sys.executable,env);wan=MG.resolve_weight('wan','')
 cmd=[sys.executable,str(R/'mg35_stream_launch.py'),'--config',str(REPO/'configs/infer_first_person.yaml'),'--model_id_with_origin_paths',f'{wan}:diffusion_pytorch_model*.safetensors,{wan}:models_t5_umt5-xxl-enc-bf16.pth,{wan}:Wan2.2_VAE.pth','--tokenizer_path',MG.resolve_weight('tokenizer',''),'--trained_dit',MG.resolve_weight('ckpt_first',''),'--dataset_index_path',index,'--val_dataset_index_path',index,'--num_val_batches','1','--num_validation_blocks',str(a.blocks),'--validation_sample_offset','0','--output_path',str(a.out/'raw'),'--log_dir_name','run','--validation_seed',str(a.seed),'--vae_decode_tiled']
 t=time.time();subprocess.run(cmd,cwd=REPO,env=env,check=True)
 with np.load(a.out/'frames.npz') as z:assert z['frames'].shape==(1+84*a.blocks,704,1280,3),z['frames'].shape
 meta.update(model='matrixgame',seconds=time.time()-t,deviations=['Whole-module CPU offload; original dtypes and generation settings retained.','Native84-frame generation; export already-decoded block RGB, no final joint re-decode. Baseline/sham/correction use identical export.']);(a.out/'run.json').write_text(json.dumps(meta,indent=2))
if __name__=='__main__':main()
