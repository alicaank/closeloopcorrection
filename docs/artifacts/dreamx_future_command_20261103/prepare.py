"""Freeze the matched future-only DreamX ablation after CPU/native preflight."""
import hashlib,json,os,subprocess,time
from pathlib import Path
W=Path('/home/hj/Alican');R=Path(__file__).resolve().parent;P=W/'analysis/memory_generalization_20261103/cohort_resume'
PY=W/'baselines/venv_vmem/bin/python';NATIVE=W/'analysis/plan_base_restart_20261008/model_readiness_20261027/envs/dreamx/bin/python'
B=W/'analysis/dreamx_camera_grid_20261103/batch'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 return h.hexdigest()
assert not (R/'manifest.json').exists(),'Preserve frozen pilot'
parent=json.loads((P/'freeze.json').read_text())
for path,h in parent['sources'].items():assert sha(path)==h,path
prepared=json.loads((B.parent/'qualification.json').read_text());verified=json.loads((B.parent/'verification.json').read_text());assert prepared['passed'] and verified['passed']
for path,h in prepared['outputs'].items():assert sha(B/path)==h,path
cases={}
for scene in ['0238de14','02267acf']:
 for seed in [7,107]:
  key=f'dreamx_{scene}_s{seed}';cases[key]=dict(model='dreamx',scene=scene,seed=seed,batch=str(B),frames=69,latents=18,prefix_latents=6,first_affected_frame=33,regime='moving development' if scene=='0238de14' else 'near-stationary stress')
protocol=dict(method='dreamx_future_fractional_v1_camera_grid',scope='Separate command-only ablation; does not replace the frozen measured-memory method or its original forty-pair cohort.',cases=cases,
 arms={'A':'Observer-free unchanged native commands in the same host','S':'Observer runs; commands unchanged','F':'Same observer, commanded memory, nominal first feedback issuance then accepted fractional recovery'},
 controls='Same native BF16 K, prepared anchor/depth grid, weights, seed, complete horizon and original targets in all arms. No memory rewrite, probe generations or fitted inverse gains.',
 policy=dict(recovery_fraction=.5,speed_multiplier=2.,native_stride=4,nominal_boundaries=1,memory_anchor='Declared previous native camera sample, distinct from latest emitted RGB',authority='Passive execution receipts only; no differential-gain claim'),
 gates=['Complete 69-frame A/S raw RGB identity before F','Exact S/F frames 0 through 32 and latent/cache/RNG receipts through latent 9','All initial native camera/noise inputs exact; first feedback command nominal'],
 scoring='Independent offline VIPE, original timestamped poses and preserved anchor-depth units; primary mean position frames 33–68; endpoint, rotation, actuation and visual review separately.',
 qualification='CPU tests do not establish native GPU parity. GPU gates precede interpretation; losses/abstentions retained. Separate input FOV contract cannot be pooled with historical DreamX runs.')
(R/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=os.pathsep.join([str(R/'source'),str(R)]),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
with (R/'logs/cpu_native_tests.log').open('x') as f:subprocess.run([str(PY),str(R/'test_future_hook.py')],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
with (R/'logs/source_compile.log').open('x') as f:subprocess.run([str(PY),str(R/'qualify_source.py')],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
with (R/'logs/native_cli.log').open('x') as f:subprocess.run([str(NATIVE),str(R/'run.py'),'--help'],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
jobs=[]
for key,c in cases.items():
 for arm in ['A','S','F']:
  if arm=='F':jobs.append(dict(name=key+'_identity',case=key,argv=[str(PY),str(R/'check.py'),'--case',key,'--kind','identity'],required=True))
  argv=[str(NATIVE),str(R/'run.py'),'--batch',str(B),'--scene',c['scene'],'--seed',str(c['seed']),'--out',str(R/'runs'/key/arm),'--arm',arm,'--latents','18','--prefix-latents','6','--memory-policy','commanded']
  jobs.append(dict(name=key+'_'+arm,case=key,argv=argv,required=key==next(iter(cases))))
  jobs.append(dict(name=key+'_'+arm+'_retain',case=key,argv=[str(PY),str(R/'retain.py'),'--case',key,'--arm',arm],required=True))
 jobs.append(dict(name=key+'_attribution',case=key,argv=[str(PY),str(R/'check.py'),'--case',key,'--kind','attribution'],required=True))
 jobs.append(dict(name=key+'_score',case=key,argv=[str(PY),str(R/'score_future.py'),'--case',key],required=False))
sources={p:h for p,h in parent['sources'].items() if not Path(p).is_relative_to(P)}
files=[*R.glob('*.py'),R/'protocol.json',R/'source_validation.json',P/'freeze.json',B.parent/'qualification.json',B.parent/'verification.json',B.parent/'prepare.py',*[p for p in (R/'source').rglob('*') if p.is_file() and '__pycache__' not in p.parts],*[p for p in B.rglob('*') if p.is_file()]]
sources.update({str(p):sha(p) for p in files})
m=dict(protocol=protocol,jobs=jobs,sources=sources,created=time.time(),parent_scientific_freeze_sha256=sha(P/'freeze.json'))
(R/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');proof=dict(passed=True,manifest_sha256=sha(R/'manifest.json'),checks=['Four actual native-encoder CPU tests','Exact runtime source overlay compiles after unchanged native context-cache pass','Native interpreter CLI help','Prepared input/reference hashes verified','Original parent source freeze unchanged'],gpu_qualified=False)
(R/'cpu_validation.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(dict(cases=len(cases),generations=12,jobs=len(jobs),manifest_sha256=proof['manifest_sha256'])))
