"""Freeze all remaining 32 original cases into four independently guarded queues."""
import importlib.util
import json
import os
import subprocess
import time
from pathlib import Path
from evidence import eligibility,sha
W = Path('/home/hj/Alican')
R = Path(__file__).resolve().parent
ROOT = R.parent
P = W/'analysis/memory_generalization_20261103/cohort_resume'
PY = W/'baselines/venv_vmem/bin/python'
base = json.loads((P/'freeze.json').read_text())
hosts = json.loads((ROOT/'host_preparation.json').read_text())['hosts']
state = json.loads((P/'status.json').read_text())
spec = importlib.util.spec_from_file_location('parent_commands',P/'run.py')
command_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(command_module)
command_module.sha = lambda p:'legacy-matched-native-host-v1'
cpuenv = dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
with (R/'cpu_tests.log').open('x') as f:
    subprocess.run([str(PY),str(R/'test_evidence.py')],env=cpuenv,stdout=f,stderr=subprocess.STDOUT,check=True)
receipts = {}
for model,host in hosts.items():
    q = ROOT/'hosts'/model
    if (q/'manifest.json').exists(): raise FileExistsError('Preserve frozen queue '+model)
    parent_path = Path(host['parent_manifest'])
    assert sha(parent_path)==host['parent_manifest_sha256']
    parent = json.loads(parent_path.read_text())
    for path,h in {**parent['sources'],**parent.get('inputs',{})}.items(): assert sha(path)==h,path
    for name,h in host['source_files'].items(): assert sha(q/'source'/name)==h,name
    parent_pairs = parent['pairs'] if model=='lingbot' else parent['protocol']['pairs']
    pairs = [p.copy() for p in parent_pairs if p['model']==model]
    assert len(pairs)==8
    if model=='lingbot':
        for pair in pairs:
            old = state['pairs'].get(pair['key'],{})
            if old.get('state')=='reused':
                reference = json.loads(Path(old['score']).read_text())['spec']
                for arm in ['commanded','measured']: pair[arm]=reference[arm]
    cases = {p['key']:dict(key=p['key'],model=model,scene=p['scene'],seed=p['seed'],batch=p['batch'],
        **base['protocol']['schedules'][model]) for p in pairs}
    parent_hashes = {x['driver']:x['original']['sha256'] for x in host['source_checks']}
    copied_hashes = {x['driver']:x['copied']['sha256'] for x in host['source_checks']}
    protocol = dict(model=model,cases=cases,first_case=pairs[0]['key'],
        scope='All eight original cases for this model within the full five-model, 40-case comparison.',
        arms={'B':'Native future commands, commanded history, original causal endpoint observer',
              'S':'Full-horizon native identity/sham command encoding, commanded history',
              'L':'Existing correction arm and commanded history; explicit legacy actuator as documented',
              'M':'Baseline future policy and measured memory'},
        reuse='Only complete baseline-arm outputs with the correct model/scene/seed, memory policy and exact original generated-source hashes (including children); copied source equivalence already qualified. Decode lossless RGB, verify retained raw hash and horizon, bind artifact hashes. Otherwise generate a fresh matched arm.',
        execution='First full-horizon B/S identity; then all eight B/L comparisons; only then M and complete triplets. Scientific/registration failures retained. A memory failure cannot suppress already completed legacy comparisons.',
        gates='Exact raw B/S full-horizon identity on the first prespecified case; exact B/L and B/L/M pre-intervention RGB and matching intrinsics for every scored case.',
        primary_metric='Mean fixed-time position from the original first-affected frame to the original native endpoint. Rotation, endpoint, whole-rollout, quality, unavailable registration and failures remain visible.',
        legacy_caveat='Preserve the actual legacy host/controller and its observer acceptance behavior. No newly fitted gain, bounded non-reversal rule or measured-memory rewrite is added to L. DreamX uses its separately documented corrected common host.',
        parent_manifest=str(parent_path),parent_manifest_sha256=sha(parent_path),
        parent_overlay_hashes=parent_hashes,copied_overlay_hashes=copied_hashes,
        guards='4 GiB launch / 2 GiB runtime disk reserve; 2 GiB host-memory reserve; shared GPU locks')
    (q/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    command_module.R = q
    arm_jobs = {}
    eligibility_snapshot = {}
    def env_for(key,arm):
        pair = next(p for p in pairs if p['key']==key)
        _,env = command_module.command(pair,'commanded',base['protocol'])
        keys=['CUDA_HOME','TORCH_CUDA_ARCH_LIST','DS_BUILD_OPS','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE',
            'PYTHONUNBUFFERED','OMP_NUM_THREADS','MKL_NUM_THREADS','MAX_JOBS','OPENBLAS_NUM_THREADS',
            'PYTORCH_CUDA_ALLOC_CONF','LONG_BATCH','GV_SCALE_CACHE','CAMERA_EVALUATOR_CONTRACT','PYTHONPATH']
        env = {k:env[k] for k in keys}
        if model=='sana':
            env.update(CAMERA_SANA_OBSERVER_GPU_RESIDENCY='1',
                CAMERA_RESIDENCY_LOG=str(q/'runs'/key/arm/'resource_residency.jsonl'))
        return env
    for pair in pairs:
        key=pair['key'];arm_jobs[key]={};eligibility_snapshot[key]={}
        for arm in ['B','S','L','M']:
            policy='measured' if arm=='M' else 'commanded'
            argv,_ = command_module.command(pair,policy,base['protocol'])
            argv[argv.index('--out')+1]=str(q/'runs'/key/arm)
            native_arm={'B':'baseline','S':'sham','L':'correction','M':'baseline'}[arm]
            flag='--arms' if model=='lingbot' else '--arm'
            argv[argv.index(flag)+1]=native_arm
            directory=q/'runs'/key/arm
            if model=='lingbot':directory/=native_arm
            job=dict(argv=argv,output=str(directory))
            if arm in ['B','M']:
                job['reference']=pair['measured' if arm=='M' else 'commanded']
                eligible,reason=eligibility(cases[key],arm,job['reference'],parent_hashes)
                eligibility_snapshot[key][arm]=dict(metadata_eligible=eligible,reason=reason,
                    note='Snapshot only; runtime verifies metadata/source and raw RGB again')
            arm_jobs[key][arm]=job
    jobs=[]
    def add(key,name,script,args,required=False,arm='B'):
        jobs.append(dict(name=key+'_'+name,case=key,argv=[str(PY),str(R/script),'--root',str(q),'--case',key,*args],env=env_for(key,arm),required=required))
    for i,pair in enumerate(pairs):
        key=pair['key']
        add(key,'B','ensure_arm.py',['--arm','B'],required=i==0)
        if i==0:
            add(key,'S','ensure_arm.py',['--arm','S'],required=True,arm='S')
            add(key,'identity','gate.py',['--kind','identity'],required=True)
        add(key,'L','ensure_arm.py',['--arm','L'],arm='L')
        add(key,'legacy_gate','gate.py',['--kind','legacy'])
        add(key,'legacy_score','score.py',['--kind','legacy'])
    for pair in pairs:
        key=pair['key']
        add(key,'M','ensure_arm.py',['--arm','M'],arm='M')
        add(key,'triplet_gate','gate.py',['--kind','triplet'])
        add(key,'triplet_score','score.py',['--kind','triplet'])
    native = Path(arm_jobs[pairs[0]['key']]['B']['argv'][0])
    with (q/'logs/source_compile.log').open('x') as f:
        for driver,h in copied_hashes.items():
            result=subprocess.run([str(native),str(q/'source/launch.py'),driver,'--check-source'],
                env=dict(cpuenv,PYTHONPATH=str(q/'source')),capture_output=True,text=True,check=True)
            f.write(result.stdout);proof=json.loads(result.stdout)
            assert proof['compiled'] and proof['sha256']==h,driver
    # Inherited external code/input hashes plus the complete cloned host and shared helpers.
    sources={p:h for p,h in parent['sources'].items() if not Path(p).is_relative_to(Path(host['parent'])/'source')}
    sources.update(parent.get('inputs',{}))
    files=[*R.glob('*.py'),R/'cpu_tests.log',q/'run_queue.py',q/'protocol.json',ROOT/'host_preparation.json',parent_path,
        *[p for p in (q/'source').rglob('*') if p.is_file() and '__pycache__' not in p.parts]]
    sources.update({str(p):sha(p) for p in files})
    manifest=dict(protocol=protocol,arm_jobs=arm_jobs,jobs=jobs,sources=sources,
        created=time.time(),eligibility_at_preparation=eligibility_snapshot)
    (q/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    proof=dict(passed=True,manifest_sha256=sha(q/'manifest.json'),gpu_qualified=False,
        checks=['Eight reuse/attribution CPU tests','Native-interpreter generated source compilation',
                'Copied host byte equality and normalized overlay equality to its parent',
                'All inherited source/input hashes verified','Original eight cases and horizons retained'])
    (q/'cpu_validation.json').write_text(json.dumps(proof,indent=2)+'\n')
    receipts[model]=dict(cases=8,jobs=len(jobs),mandatory_new_generations=9,
        maximum_generations_with_fallback=25,
        metadata_eligible=sum(x['metadata_eligible'] for v in eligibility_snapshot.values() for x in v.values()),
        manifest_sha256=proof['manifest_sha256'])
    print(json.dumps(dict(model=model,**receipts[model])),flush=True)
(R/'preparation.json').write_text(json.dumps(receipts,indent=2)+'\n')
