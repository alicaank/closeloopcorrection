"""Frozen resource-only retry of all eight SANA pairs in the original cohort."""
import hashlib,importlib.util,json,os,subprocess,time,difflib
from pathlib import Path
R=Path(__file__).resolve().parent;W=Path('/home/hj/Alican');PARENT=R.parent/'cohort_resume';PY=W/'baselines/venv_vmem/bin/python'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not (R/'manifest.json').exists()
parent=json.loads((PARENT/'freeze.json').read_text())
for path,h in {**parent['sources'],**parent['inputs']}.items():assert sha(path)==h,path
protocol=parent['protocol'];pairs=[]
for p in parent['pairs']:
    if p['model']!='sana':continue
    p=p.copy()
    for arm in ('commanded','measured'):p[arm]=str(R/'runs'/p['key']/arm)
    pairs.append(p);(R/'specs'/f"{p['key']}.json").write_text(json.dumps(p,indent=2)+'\n')
assert len(pairs)==8 and {(p['scene'],p['seed']) for p in pairs}=={(s,k) for s in protocol['scenes'] for k in protocol['seeds']}
config=dict(scope='Resource-only SANA supplement to the original 40-pair cohort, not a replacement or reduced generalization cohort.',
    parent_freeze_sha256=sha(PARENT/'freeze.json'),pairs=pairs,original_protocol=protocol,
    change='Move idle stage-1 weights to GPU BEFORE decoder CPU offload at the endpoint callback; retain observer-time placement for both observers and close them before restoration. No dtype, camera, cache-length or model arithmetic change. Dump the main Python stack before a guarded generation stop.',
    first_case_required='Original failed sana_02267acf_s7: 121-frame commanded completion and exact preserved-prefix parity, then measured completion and paired prefix attribution.',
    remaining_cases='All seven remaining original SANA pairs, retaining losses and abstentions. Resource/prefix failures hold this supplement.',
    option='CAMERA_SANA_OBSERVER_GPU_RESIDENCY=1; default remains disabled outside this attempt.')
previous=R.parent/'sana_observer_residency'
failed=json.loads((previous/'status.json').read_text())
assert failed['phase']=='held' and not Path('/proc',str(failed['pid'])).exists()
partial=previous/'runs/sana_02267acf_s7/commanded/frames.npz'
assert sha(partial)==sha(PARENT/'runs/sana_02267acf_s7/commanded/frames.npz')
config['previous_attempt']=dict(status=failed,partial_frames=97,partial_sha256=sha(partial),observation='97 RGB frames saved exactly, but only three observer-residency entries. Guard interval precedes fourth observer entry; decoder CPU transfer is a candidate peak, not stack-proven causality.')
(R/'protocol.json').write_text(json.dumps(config,indent=2)+'\n')
diff=''
for name in ['drivers/sana.py','memory_hooks.py']:
    diff+=''.join(difflib.unified_diff((PARENT/'source'/name).read_text().splitlines(True),(R/'source'/name).read_text().splitlines(True),fromfile='parent/'+name,tofile='residency/'+name))
(R/'source_change.diff').write_text(diff)
spec=importlib.util.spec_from_file_location('commands',R/'run.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
# command() binds the manifest hash; supply its final contract at execution.
module.sha=lambda path:'sana-observer-transition-20261103'
jobs=[]
for i,pair in enumerate(pairs):
    key=pair['key'];sp=R/'specs'/f'{key}.json'
    for arm in ('commanded','measured'):
        argv,env=module.command(pair,arm,protocol)
        keys=['CUDA_HOME','TORCH_CUDA_ARCH_LIST','DS_BUILD_OPS','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','PYTHONUNBUFFERED','OMP_NUM_THREADS','MKL_NUM_THREADS','MAX_JOBS','OPENBLAS_NUM_THREADS','PYTORCH_CUDA_ALLOC_CONF','LONG_BATCH','GV_SCALE_CACHE','CAMERA_EVALUATOR_CONTRACT','PYTHONPATH']
        env={k:env[k] for k in keys};env.update(CAMERA_SANA_OBSERVER_GPU_RESIDENCY='1',CAMERA_RESIDENCY_LOG=str(R/'runs'/key/arm/'resource_residency.jsonl'))
        jobs.append(dict(name=key+'_'+arm,case=key,argv=argv,env=env,required=i==0))
        if i==0 and arm=='commanded':jobs.append(dict(name=key+'_resource_prefix',case=key,argv=[str(PY),str(R/'check.py'),'--pair',key,'--resource-prefix'],env=env,required=True))
        jobs.append(dict(name=key+'_'+arm+'_retain',case=key,argv=[str(PY),str(R/'retain.py'),'--spec',str(sp),'--arm',arm],env=env,required=i==0))
    jobs.append(dict(name=key+'_score',case=key,argv=[str(PY),str(R/'source/score_memory_pair.py'),'--spec',str(sp),'--out',str(R/'scores'/key)],env=env,required=i==0))
    jobs.append(dict(name=key+'_attribution',case=key,argv=[str(PY),str(R/'check.py'),'--pair',key],env=env,required=True))
cpuenv=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONPATH=str(W/'camera-closed-loop/src'))
with (R/'logs/repository_tests.log').open('x') as log:
    subprocess.run([str(PY),'-m','unittest','discover','-s','tests','-v'],cwd=W/'camera-closed-loop',env=cpuenv,stdout=log,stderr=subprocess.STDOUT,check=True)
argv=jobs[0]['argv'];split=argv.index('--')
with (R/'logs/native_compile.log').open('x') as log:
    subprocess.run(argv[:split]+['--check-source'],env=dict(jobs[0]['env'],CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT,check=True)
ref=PARENT/'runs/sana_02267acf_s7/commanded/frames.npz'
sources={p:h for p,h in parent['sources'].items() if not Path(p).is_relative_to(PARENT)}
sources.update(parent['inputs'])
files=[*R.glob('*.py'),R/'protocol.json',R/'source_change.diff',PARENT/'freeze.json',
       *[p for p in (R/'source').rglob('*') if p.is_file() and '__pycache__' not in p.parts],*sorted((R/'specs').glob('*.json'))]
sources.update({str(p):sha(p) for p in files})
m=dict(protocol=config,jobs=jobs,sources=sources,created=time.time(),resource_reference=dict(path=str(ref),sha256=sha(ref)))
(R/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
(R/'cpu_validation.json').write_text(json.dumps(dict(passed=True,manifest_sha256=sha(R/'manifest.json'),checks=['153 CPU tests','Native SANA overlay source compiles','Parent freeze verified','Original eight SANA cases retained']),indent=2)+'\n')
print(json.dumps(dict(pairs=len(pairs),generation_jobs=16,total_jobs=len(jobs),sources=len(sources))))
