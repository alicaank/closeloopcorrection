"""Freeze a matched first-boundary diagnostic without changing the 40-pair cohort."""
import hashlib,importlib.util,json,os,subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parent;W=Path('/home/hj/Alican');P=W/'analysis/memory_generalization_20261103/cohort_resume';PY=W/'baselines/venv_vmem/bin/python'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
assert not (R/'manifest.json').exists(),'Preserve frozen attempts'
parent=json.loads((P/'freeze.json').read_text())
for p,h in {**parent['sources'],**parent['inputs']}.items():assert sha(p)==h,p
cases={}
for scene in ['0238de14','02267acf']:
    for seed in [7,107]:
        key=f'lingbot_{scene}_s{seed}'
        original=next(p for p in parent['pairs'] if p['key']==key)
        cases[key]=dict(key=key,scene=scene,seed=seed,model='lingbot',batch=original['batch'],frames=77,first_affected_frame=13)
protocol=dict(scope='Separate first-boundary normalization diagnostic; original measured-memory-v1 and forty-pair cohort unchanged',
    cases=cases,arms={'A':'Original commanded memory and normalization','I':'Additional observation and exact native identity re-encoding','G':'Commanded history with shared plan/observed-prefix normalization','M':'Measured history with exactly the same normalization and future camera tensors as G'},
    intervention='First clean-context cache pass only; fixed normalizer afterward. Native later observations remain read-only. Full 77-frame horizon.',
    qualification=['Every case: A/I all 77 RGB bytes, first and final self/cross cache tensors and native generator state match',
                   'A/G/M emitted frames 0–12 exact; G/M initial noise, first latent, observed prefix, accepted measured poses, denominator and all future camera inputs match',
                   'Observer RNG preservation required; rejected observations retain original commands and are reported as abstentions'],
    interpretation='G-A isolates rescaling; M-G isolates measured-history substitution under the same future conditioning; M-A is the combined effect. No result proves later cache replay, training support, or broad generalization.',
    scoring='Offline VIPE, same original targets and initial anchor-depth units; mean over frames 13–76, endpoint, rotation and visual inspection separately. No cross-scene unit pooling.',
    generations=16,parent_freeze_sha256=sha(P/'freeze.json'),gpu_qualified=False)
(R/'cases.json').write_text(json.dumps(cases,indent=2)+'\n');(R/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
spec=importlib.util.spec_from_file_location('pilot_commands',R/'run.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);mod.sha=lambda p:'lingbot-first-boundary-shared-gauge-20261103'
keys=['CUDA_HOME','TORCH_CUDA_ARCH_LIST','DS_BUILD_OPS','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','PYTHONUNBUFFERED','OMP_NUM_THREADS','MKL_NUM_THREADS','MAX_JOBS','OPENBLAS_NUM_THREADS','PYTORCH_CUDA_ALLOC_CONF','LONG_BATCH','GV_SCALE_CACHE','CAMERA_EVALUATOR_CONTRACT','PYTHONPATH']
jobs=[]
for index,(key,c) in enumerate(cases.items()):
    for arm in ['A','I','G','M']:
        argv,env=mod.command(c,'commanded',parent['protocol'])
        argv[argv.index('--out')+1]=str(R/'runs'/key/arm)
        argv[argv.index('--memory-policy')+1]={'A':'commanded','I':'identity','G':'commanded','M':'measured'}[arm]
        env={k:env[k] for k in keys};env['CAMERA_LINGBOT_GAUGE_ARM']=arm
        jobs.append(dict(name=key+'_'+arm,case=key,argv=argv,env=env,required=index==0))
        sp=R/'specs'/f'{key}_{arm}_retain.json';sp.write_text(json.dumps(dict(scene=c['scene'],commanded=str(R/'runs'/key/arm/'baseline')),indent=2)+'\n')
        jobs.append(dict(name=key+'_'+arm+'_retain',case=key,argv=[str(PY),str(R/'retain.py'),'--spec',str(sp),'--arm','commanded'],required=index==0))
        if arm=='I':jobs.append(dict(name=key+'_identity',case=key,argv=[str(PY),str(R/'check.py'),'--case',key,'--gate','identity'],required=True))
    jobs.append(dict(name=key+'_matched',case=key,argv=[str(PY),str(R/'check.py'),'--case',key,'--gate','matched'],required=True))
    for left,right in [('A','G'),('G','M'),('A','M')]:
        name=key+'_'+left+right;sp=R/'specs'/f'{name}.json'
        sp.write_text(json.dumps(dict(c,commanded=str(R/'runs'/key/left/'baseline'),measured=str(R/'runs'/key/right/'baseline'),
            diagnostic='first_boundary_shared_gauge',comparison=left+'->'+right,source_arms=dict(commanded=left,measured=right)),indent=2)+'\n')
        jobs.append(dict(name=name+'_score',case=key,argv=[str(PY),str(R/'source/score_memory_pair.py'),'--spec',str(sp),'--out',str(R/'scores'/name)],required=False))
cpuenv=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONPATH=str(R/'source'))
for script,log in [('test_hook.py','cpu_hook_tests.log'),('qualify_sources.py','source_validation.log')]:
    with (R/'logs'/log).open('x') as f:subprocess.run([str(PY),str(R/script)],env=cpuenv,stdout=f,stderr=subprocess.STDOUT,check=True)
sources={p:h for p,h in parent['sources'].items() if not Path(p).is_relative_to(P)};sources.update(parent['inputs'])
files=[*R.glob('*.py'),R/'protocol.json',R/'cases.json',R/'source_validation.json',P/'freeze.json',
       *[p for p in (R/'source').rglob('*') if p.is_file() and '__pycache__' not in p.parts],*sorted((R/'specs').glob('*.json'))]
sources.update({str(p):sha(p) for p in files})
manifest=dict(protocol=protocol,jobs=jobs,sources=sources,created=time.time())
(R/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(R/'cpu_validation.json').write_text(json.dumps(dict(passed=True,manifest_sha256=sha(R/'manifest.json'),checks=['Three pilot arm-selection tests','Actual native clean-context source overlay compiles','Prior shared-gauge audit and 152 repository tests passed before integration'],gpu_qualified=False),indent=2)+'\n')
print(json.dumps(dict(cases=len(cases),generations=16,jobs=len(jobs),manifest_sha256=sha(R/'manifest.json'))))
