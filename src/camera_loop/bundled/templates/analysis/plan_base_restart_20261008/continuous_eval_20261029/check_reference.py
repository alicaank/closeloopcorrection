"""CPU preflight: alignment split catches perturbed trajectories and accepts pure gauge changes."""
from pathlib import Path
import ast,json
import numpy as np
from scipy.spatial.transform import Rotation
R=Path(__file__).resolve().parent
# Import numerical validator without importing CUDA/ViPE modules.
p=ast.parse((R/'dense_pose.py').read_text());nodes=[n for n in p.body if isinstance(n,ast.FunctionDef) and n.name in ['fit_similarity','validate']];scope={'np':np};exec(compile(ast.Module(body=nodes,type_ignores=[]),'validator','exec'),scope)
t=np.linspace(0,3,201);P=np.repeat(np.eye(4)[None],201,0);P[:,:3,3]=np.stack([t,np.sin(t),.2*np.cos(2*t)],axis=-1);P[:,:3,:3]=Rotation.from_euler('y',t[:,None]*.1).as_matrix()
rot=Rotation.from_euler('xyz',[.2,.1,-.3]).as_matrix();Q=P.copy();Q[:,:3,3]=2.3*(P[:,:3,3]@rot.T)+[1,2,3];Q[:,:3,:3]=rot@P[:,:3,:3]
ids=np.arange(0,201,5);truth={'c2w':Q[ids],'dense_ids':ids};good=scope['validate'](P,truth);assert good['passed']
bad=P.copy();bad[ids[1::2],:3,3]+=[1,0,0];failed=scope['validate'](bad,truth);assert not failed['passed']
for p in R.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
# Assert source rewrite targets exactly one final whole-history VAE export block.
source=Path('@WORKSPACE@/sota_repos/matrix_game/Matrix-Game-3.5/examples/wanvideo/pipeline/mosaic/validation.py').read_text();assert source.count('        history_frames = _decode_latents_to_numpy_frames(')==1
(R/'cpu_checks.json').write_text(json.dumps(dict(gauge_invariance_passed=True,heldout_error_detection_passed=True,matrix_export_patch_target_unique=True),indent=2));print('CPU checks passed')
