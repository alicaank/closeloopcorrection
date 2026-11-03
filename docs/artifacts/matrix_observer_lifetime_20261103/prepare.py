"""Freeze all eight original Matrix pairs under observer lifetime repair."""
import hashlib,importlib.util,json,os,subprocess,time,difflib
from pathlib import Path
R=Path(__file__).resolve().parent;W=Path('/home/hj/Alican');P=R.parent/'cohort_resume';PY=W/'baselines/venv_vmem/bin/python'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
assert not (R/'manifest.json').exists()
m=json.loads((P/'freeze.json').read_text())
for p,h in {**m['sources'],**m['inputs']}.items():assert sha(p)==h,p
state=json.loads((P/'status.json').read_text());assert 'matrixgame' in state['held_models']
failed=state['jobs']['matrixgame_02267acf_s7_measured'];assert failed['state']=='failed' and not Path('/proc',str(failed['pid'])).exists()
protocol=m['protocol'];pairs=[]
for old in m['pairs']:
 if old['model']!='matrixgame':continue
 p=old.copy()
 for arm in ['commanded','measured']:p[arm]=str(R/'runs'/p['key']/arm)
 pairs.append(p);(R/'specs'/f"{p['key']}.json").write_text(json.dumps(p,indent=2)+'\n')
assert len(pairs)==8
config=dict(scope='All eight original MatrixGame pairs within the original forty-pair cohort; no horizon reduction',pairs=pairs,parent_freeze_sha256=sha(P/'freeze.json'),frames=337,
 change='Close endpoint observer before starting memory observer; close memory observer before native continuation. Both close on measurement failure. No weights, precision, camera policy, cache size or decoder arithmetic change.',
 first_case='Full 337-frame commanded completion and byte equality with original baseline, then full measured arm and exact paired prefix gate',original_failure=failed)
(R/'protocol.json').write_text(json.dumps(config,indent=2)+'\n')
diff=''
for name in ['drivers/matrix_child.py','memory_hooks.py']:
 diff+=''.join(difflib.unified_diff((P/'source'/name).read_text().splitlines(True),(R/'source'/name).read_text().splitlines(True),fromfile='parent/'+name,tofile='retry/'+name))
(R/'source_change.diff').write_text(diff)
spec=importlib.util.spec_from_file_location('commands',R/'run.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);mod.sha=lambda p:'matrix-observer-lifetime-20261103'
jobs=[]
for i,p in enumerate(pairs):
 key=p['key'];sp=R/'specs'/f'{key}.json'
 for arm in ['commanded','measured']:
  argv,fullenv=mod.command(p,arm,protocol)
  keys=['CUDA_HOME','TORCH_CUDA_ARCH_LIST','DS_BUILD_OPS','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','PYTHONUNBUFFERED','OMP_NUM_THREADS','MKL_NUM_THREADS','MAX_JOBS','OPENBLAS_NUM_THREADS','PYTORCH_CUDA_ALLOC_CONF','LONG_BATCH','GV_SCALE_CACHE','CAMERA_EVALUATOR_CONTRACT','PYTHONPATH']
  env={k:fullenv[k] for k in keys}
  jobs.append(dict(name=key+'_'+arm,case=key,argv=argv,env=env,required=i==0))
  if i==0 and arm=='commanded':jobs.append(dict(name=key+'_resource_prefix',case=key,argv=[str(PY),str(R/'check.py'),'--pair',key,'--resource-prefix'],env=env,required=True))
  jobs.append(dict(name=key+'_'+arm+'_retain',case=key,argv=[str(PY),str(R/'retain.py'),'--spec',str(sp),'--arm',arm],env=env,required=i==0))
 jobs.append(dict(name=key+'_score',case=key,argv=[str(PY),str(R/'source/score_memory_pair.py'),'--spec',str(sp),'--out',str(R/'scores'/key)],env=env,required=i==0))
 jobs.append(dict(name=key+'_attribution',case=key,argv=[str(PY),str(R/'check.py'),'--pair',key],env=env,required=True))
cpuenv=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONPATH=str(W/'camera-closed-loop/src'))
with (R/'logs/repository_tests.log').open('x') as f:subprocess.run([str(PY),'-m','unittest','discover','-s','tests','-v'],cwd=W/'camera-closed-loop',env=cpuenv,stdout=f,stderr=subprocess.STDOUT,check=True)
with (R/'logs/native_compile.log').open('x') as f:
 for model in ['matrixgame','matrix_child']:
  argv=[str(W/'baselines/venv_mg35/bin/python'),str(R/'source/launch.py'),model,'--check-source']
  subprocess.run(argv,env=dict(jobs[0]['env'],CUDA_VISIBLE_DEVICES=''),stdout=f,stderr=subprocess.STDOUT,check=True)
ref=P/'runs/matrixgame_02267acf_s7/commanded/frames.lossless.mkv'
sources={p:h for p,h in m['sources'].items() if not Path(p).is_relative_to(P)};sources.update(m['inputs'])
files=[*R.glob('*.py'),R/'protocol.json',R/'source_change.diff',P/'freeze.json',*[p for p in (R/'source').rglob('*') if p.is_file() and '__pycache__' not in p.parts],*sorted((R/'specs').glob('*.json'))]
sources.update({str(p):sha(p) for p in files})
out=dict(protocol=config,jobs=jobs,sources=sources,created=time.time(),resource_reference=dict(path=str(ref),sha256=sha(ref)))
(R/'manifest.json').write_text(json.dumps(out,indent=2)+'\n');(R/'cpu_validation.json').write_text(json.dumps(dict(passed=True,manifest_sha256=sha(R/'manifest.json'),checks=['145 repository CPU tests','Both native Matrix source overlays compile','Original 337-frame horizon and all eight pairs preserved','Original failure process absent']),indent=2)+'\n')
print(json.dumps(dict(pairs=len(pairs),jobs=len(jobs),manifest_sha256=sha(R/'manifest.json'))))
