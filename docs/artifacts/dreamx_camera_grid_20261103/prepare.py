"""Input-only native camera-grid preparation; no calibration from motion."""
import ast,hashlib,json,sys,shutil
from pathlib import Path
import numpy as np
import torch
from PIL import Image
W=Path('/home/hj/Alican');R=Path(__file__).resolve().parent
sys.path.insert(0,str(W/'camera-closed-loop/src'))
from camera_loop.camera_grid import CameraGrid
repo=W/'analysis/plan_base_restart_20261008/model_readiness_20261027/repos/dreamx'
entry=repo/'inference_ar_forcing.py';camera=repo/'utils/trajectory_processor.py'
source=W/'analysis/plan_base_restart_20261008/continuous_eval_20261029/batch_continuous1009'
B=R/'batch';B.mkdir(exist_ok=False)
for sub in ['images','trajectories','captions','references']:(B/sub).mkdir()
ns=dict(np=np,torch=torch)
def extract(path,names):
 nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
 assert {n.name for n in nodes}==set(names)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
extract(camera,['Camera']);extract(entry,['_invert_SE3','get_relative_pose','cam_params_to_prope_dict'])
tree=ast.parse(entry.read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
resizes=[n for n in ast.walk(main) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='Resize']
assert len(resizes)==1
h,w=ast.literal_eval(resizes[0].args[0]);row=np.zeros(19);row[7:]=np.eye(4)[:3].reshape(-1)
encoded=ns['cam_params_to_prope_dict']([ns['Camera'](row)],device='cpu',dtype=torch.bfloat16,chunk_relative=True)
kn=encoded['K'][0,0].float().numpy().astype(float);K=kn.copy();K[0]*=w;K[1]*=h
index=json.loads((source/'index.json').read_text());selected=['02267acf','c7418b7d','0238de14','924228bf'];reports=[];newrows=[]
for scene in selected:
 original=next(x for x in index['scenes'] if x['name']==scene);p=source/'trajectories'/f'{scene}.npz'
 with np.load(p) as z:old={k:z[k].copy() for k in z.files}
 image=np.asarray(Image.open(source/original['image']).convert('RGB'))
 g=CameraGrid((image.shape[1],image.shape[0]),(w,h),old['intrinsics'][0],K)
 rgb,mask=g.rgb(image);depth,valid=g.depth(old['anchor_depth'])
 assert mask.all(),'Native FOV extends outside the supplied anchor'
 assert valid.mean()>=.95,'Insufficient transformed anchor depth'
 Image.fromarray(rgb).save(B/'images'/f'{scene}.png')
 new={k:v.copy() for k,v in old.items()}
 for key in ['w2c','w2c_overrun','source_frame_ids']:new[key]=new[key][:69]
 new['intrinsics']=np.repeat(K[None],69,axis=0);new['anchor_depth']=depth.astype(np.float32)
 new['image_width']=np.int32(w);new['image_height']=np.int32(h)
 np.savez_compressed(B/'trajectories'/f'{scene}.npz',**new)
 for key in ['w2c','w2c_overrun','source_frame_ids']:np.testing.assert_array_equal(new[key],old[key][:69])
 np.testing.assert_array_equal(new['metres_per_anchor_unit'],old['metres_per_anchor_unit'])
 shutil.copyfile(source/'captions'/f'{scene}.json',B/'captions'/f'{scene}.json')
 # Reference RGB is transformed only for offline scoring/review. None enters
 # generation, scale fitting, online observation or the policy.
 refs=B/'references'/scene;refs.mkdir();reference_hashes=[]
 for frame in range(69):
  original_frame=Path(original['frames_dir'])/f'frame_{frame+1:05d}.jpg'
  im=np.asarray(Image.open(original_frame).convert('RGB'))
  fg=CameraGrid((im.shape[1],im.shape[0]),(w,h),old['intrinsics'][frame],K)
  transformed,coverage=fg.rgb(im);assert coverage.all()
  dst=refs/f'frame_{frame+1:05d}.png';Image.fromarray(transformed).save(dst)
  reference_hashes.append(dict(frame=frame,source=str(original_frame),source_sha256=hashlib.sha256(original_frame.read_bytes()).hexdigest(),sha256=hashlib.sha256(dst.read_bytes()).hexdigest()))
 newrow=dict(original,image=f'images/{scene}.png',frames=69,frames_overrun=69,frames_dir=str(refs),raw_reference_frames_dir=original['frames_dir'],reference_extension='.png',pixel_contract='camera_grid_initial_intrinsics_v1')
 newrows.append(newrow)
 xy,_=g.coordinates();sample=np.array([[0,w-1,w//2],[0,h-1,h//2],[1,1,1]],float)
 source_pixels=old['intrinsics'][0]@np.linalg.inv(K)@sample
 ray_error=float(np.max(np.abs(np.linalg.inv(old['intrinsics'][0])@source_pixels-np.linalg.inv(K)@sample)))
 assert ray_error<1e-12
 reports.append(dict(scene=scene,source_trajectory=str(p),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),source_anchor=str(source/original['image']),source_anchor_sha256=hashlib.sha256((source/original['image']).read_bytes()).hexdigest(),rgb_coverage=float(mask.mean()),valid_depth_fraction=float(valid.mean()),ray_error=ray_error,source_K=old['intrinsics'][0].tolist(),native_K=K.tolist(),sampled_source_pixel_bounds=[xy[0].min(),xy[0].max(),xy[1].min(),xy[1].max()],pose_arrays_exact=True,depth_unit_unchanged=True,references=reference_hashes))
 print(scene,'coverage',valid.mean(),flush=True)
index.update(resolution_hw=[h,w],frames=69,selected_count=len(newrows),scenes=newrows,selection='Same four prespecified scenes; initial-only reprojection to released BF16 DreamX K. Separate preprocessing contract, not a replacement of old experiments.',pixel_contract=dict(native_normalized_K=kn.tolist(),size_hw=[h,w],interpolation='bilinear RGB and z depth; missing depth remains NaN',calibration_inputs='initial RGB dimensions and intrinsics; fixed native encoder K',reference_policy='Reference frames reprojected for offline evaluation only; never enter generation or online calibration'))
(B/'index.json').write_text(json.dumps(index,indent=2)+'\n')
report=dict(passed=True,scope='Input preparation and geometry qualification only; no model generations or efficacy result',native_dtype='torch.bfloat16',native_normalized_K=kn.tolist(),sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [entry,camera,Path(__file__),W/'camera-closed-loop/src/camera_loop/camera_grid.py']},batch=str(B),rows=reports,outputs={str(p.relative_to(B)):hashlib.sha256(p.read_bytes()).hexdigest() for p in B.rglob('*') if p.is_file()})
(R/'qualification.json').write_text(json.dumps(report,indent=2)+'\n')
print('Prepared',len(reports),'scenes and',69*len(reports),'offline reference frames',flush=True)
