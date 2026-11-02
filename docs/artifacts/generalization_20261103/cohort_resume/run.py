"""Freeze and execute the predeclared measured-memory v1 development cohort."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import fcntl
import psutil

ROOT=Path('/home/hj/Alican');R=Path(__file__).resolve().parent
A=ROOT/'analysis/plan_base_restart_20261008';OLD=ROOT/'analysis/measured_memory_20261101'
PKG=ROOT/'camera-closed-loop';PY=ROOT/'baselines/venv_vmem/bin/python'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_pair_inputs(batch,scene,frames):
    import numpy as np
    from PIL import Image
    rows=json.loads((batch/'index.json').read_text())['scenes'];row=next(x for x in rows if x['name']==scene)
    with np.load(batch/'trajectories'/f'{scene}.npz') as z:
        poses=z['w2c_overrun' if 'w2c_overrun' in z else 'w2c'];depth=z['anchor_depth'];K=z['intrinsics'][0]
        if len(poses)<frames or poses.shape[1:]!=(4,4) or not np.isfinite(poses[:frames]).all():raise ValueError('Invalid/short trajectory')
        if depth.shape!=(480,864) or not np.isfinite(K).all() or K.shape!=(3,3):raise ValueError('Input geometry contract mismatch')
        if np.mean(np.isfinite(depth)&(depth>0))<.2:raise ValueError('Insufficient anchor depth')
    if Image.open(batch/row['image']).size!=(864,480):raise ValueError('Invalid source dimensions')
    files=[batch/'index.json',batch/'trajectories'/f'{scene}.npz',batch/row['image'],batch/'captions'/f'{scene}.json']
    if not all(p.is_file() for p in files):raise ValueError('Missing cohort input')
    return files


def prepare():
    if (R/'freeze.json').exists():raise FileExistsError('Existing freeze preserved')
    protocol=json.loads((R/'protocol.json').read_text());batch=ROOT/protocol['batch']
    rows=json.loads((batch/'index.json').read_text())['scenes']
    names=protocol['scenes']
    available={r['name'] for r in rows}
    if not set(names)<=available:raise ValueError('Missing prospectively selected scene')
    inputs=[]
    for scene in names:inputs+=validate_pair_inputs(batch,scene,max(v['frames'] for v in protocol['schedules'].values()))
    # Validate the already installed exact-caption cache, without loading CUDA.
    import torch
    cache_path=A/'continuous_eval_20261029/lyra_t5.pt'
    cache=torch.load(cache_path,map_location='cpu',weights_only=False)
    for scene in names:
        captions=json.loads((batch/'captions'/f'{scene}.json').read_text())
        if any(c not in cache.get('prompts',{}) for c in captions.values()):raise ValueError('Missing exact Lyra caption')
    inputs.append(cache_path)
    source=R/'source';source.mkdir(exist_ok=False)
    bundle=PKG/'src/camera_loop/bundled/templates/analysis/measured_memory_20261101'
    for f in bundle.glob('*.py'):
        text=f.read_text().replace('@WORKSPACE@',str(ROOT)).replace('@CONDA_ROOT@','/home/hj/miniconda3').replace('@CACHE_ROOT@','/home/hj/.cache').replace('@HOME_ROOT@','/home/hj')
        (source/f.name).write_text(text)
    # Preserve the native driver file identities/import roots; only expose its seed.
    path=source/'overlays.py';text=path.read_text();needle="    elif model in ('matrixgame','lyra'):\n"
    insert='''    elif model in ('matrixgame','lyra'):
        if model=='lyra':
            source=replace_once(source,"p.add_argument('--frames',type=int,default=481);a=p.parse_args()", "p.add_argument('--frames',type=int,default=481);p.add_argument('--seed',type=int,default=7);a=p.parse_args()")
            source=replace_once(source,"'--seed','7'", "'--seed',str(a.seed)")
            source=replace_once(source,"frames=a.frames,seed=7", "frames=a.frames,seed=a.seed")
'''
    if text.count(needle)!=1:raise ValueError('Lyra seed-overlay insertion point changed')
    path.write_text(text.replace(needle,insert))
    frozen_package=source/'camera_loop';frozen_package.mkdir()
    for f in (PKG/'src/camera_loop').glob('*.py'):shutil.copyfile(f,frozen_package/f.name)
    for f in (PKG/'src/camera_loop').glob('*.json'):shutil.copyfile(f,frozen_package/f.name)
    sys.path.insert(0,str(source));from overlays import DRIVERS,driver_source
    driver_dir=source/'drivers';driver_dir.mkdir()
    for model,path in DRIVERS.items():
        relative=path.relative_to(ROOT) if path.is_relative_to(ROOT) else None
        template=PKG/'src/camera_loop/bundled/templates'/relative if relative else None
        if model!='dreamx' and template is not None and template.exists():
            text=template.read_text().replace('@WORKSPACE@',str(ROOT)).replace('@HOME_ROOT@','/home/hj').replace('@CONDA_ROOT@','/home/hj/miniconda3').replace('@CACHE_ROOT@','/home/hj/.cache')
            (driver_dir/(model+'.py')).write_text(text)
    overlay=source/'overlays.py';text=overlay.read_text().replace('source=DRIVERS[model].read_text()',"source=(R/'drivers'/(model+'.py')).read_text() if (R/'drivers'/(model+'.py')).exists() else DRIVERS[model].read_text()");overlay.write_text(text)
    import importlib,overlays;importlib.reload(overlays);driver_source=overlays.driver_source
    for model in ['dreamx','lingbot','sana','matrixgame','lyra','astronex','worldwarp','matrix_child','lyra_child']:
        driver_source(model)
    deps=list(DRIVERS.values())
    V=A/'continuous_eval_20261029/scale_20261030/correction_v3_20261030'
    deps += [V/n for n in ['observer_worker.py','guarded_observer.py','geometry_quality.py','source_manifest.json','snapshots/spatia_da3_worker.py','snapshots/realized_pose.py']]
    deps += [A/'vipe_eval_20261030'/n for n in ['camera.py','worker.py']]
    deps += [A/'new_models_loop_20261028/common.py',A/'continuous_eval_20261029/anchor_scale.py',A/'fair_common_20261025/fair_common.py']
    # Hash Python/config code of each selected native repository (not checkpoints).
    metadata=json.loads((PKG/'src/camera_loop/bundled/repos.json').read_text());revisions={}
    for model in protocol['models']:
        repo=ROOT/metadata[model]['path']
        p=subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],capture_output=True,text=True,check=True)
        revisions[model]=dict(path=str(repo),commit=p.stdout.strip())
        for f in repo.rglob('*'):
            if f.is_file() and f.suffix in ['.py','.yaml','.yml'] and not any(x in f.parts for x in ['.git','.venv','__pycache__','checkpoints','weights']):deps.append(f)
    local=[R/'run.py',R/'report.py',R/'retain.py',R/'protocol.json',*[p for p in source.rglob('*') if p.is_file() and '__pycache__' not in p.parts]]
    for sub in ['runs','logs','scores','specs','calibration']:(R/sub).mkdir(exist_ok=True)
    pairs=[]
    for seed in protocol['seeds']:
        for scene in protocol['scenes']:
            for model in protocol['models']:
                key=f'{model}_{scene}_s{seed}';d=R/'runs'/key
                pairs.append(dict(key=key,model=model,scene=scene,seed=seed,batch=str(batch),
                    commanded=str(d/'commanded'/('baseline' if model=='lingbot' else '')),
                    measured=str(d/'measured'/('baseline' if model=='lingbot' else '')),
                    first_affected_frame=protocol['schedules'][model]['first_affected_frame']))
    frozen=dict(protocol=protocol,pairs=pairs,created=time.time(),native_revisions=revisions,
        sources={str(f):sha(f) for f in sorted(set(local+deps))},inputs={str(f):sha(f) for f in sorted(set(inputs))},
        source_export='Current portable bundle, explicit Lyra seed forwarding, content-addressed evaluator and verified lossless retention; virtual native driver import roots retained.')
    (R/'freeze.json').write_text(json.dumps(frozen,indent=2))
    for pair in pairs:(R/'specs'/f"{pair['key']}.json").write_text(json.dumps(pair,indent=2))
    print(json.dumps(dict(prepared=True,pairs=len(pairs),sources=len(frozen['sources']),inputs=len(frozen['inputs']))))


def command(pair,arm,protocol):
    model=pair['model'];scene=pair['scene'];seed=str(pair['seed']);schedule=protocol['schedules'][model]
    prep=A/'model_readiness_20261027';py=prep/'envs'/model/'bin/python'
    if model=='matrixgame':py=ROOT/'baselines/venv_mg35/bin/python'
    if model=='lyra':py=PY
    out=R/'runs'/pair['key']/arm
    args=['--batch',pair['batch'],'--scene',scene,'--seed',seed,'--out',str(out)]
    if model=='dreamx':args+=['--latents',str(schedule['latents']),'--prefix-latents',str(schedule['prefix_latents'])]
    elif model=='matrixgame':args+=['--arm','baseline','--blocks',str(schedule['blocks'])]
    elif model=='lingbot':args+=['--arms','baseline','--frames',str(schedule['frames'])]
    else:args+=['--arm','baseline','--frames',str(schedule['frames'])]
    env=dict(os.environ,CUDA_HOME='/home/hj/miniconda3/envs/cuda128',TORCH_CUDA_ARCH_LIST='12.0',DS_BUILD_OPS='0',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MAX_JOBS='2',
        OPENBLAS_NUM_THREADS='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',LONG_BATCH=pair['batch'],
        GV_SCALE_CACHE=str(R/'calibration'/pair['key']))
    for key in ['LD_LIBRARY_PATH','GV_DA3_FAKE','CAMERA_MEMORY_VERIFY_REPLAY','CUDA_VISIBLE_DEVICES']:env.pop(key,None)
    env['CAMERA_EVALUATOR_CONTRACT']='memory-multiboundary:'+sha(R/'freeze.json')
    env['PYTHONPATH']=os.pathsep.join(map(str,[R/'source',ROOT/'analysis/zigzag_followup_v2/frozen',A/'fresh_teacher_20261021',ROOT,
        ROOT/'training_runs/joint_samplerfix_20261012',A/'memorizon_plugins',ROOT/'training_runs/memory1500_resplat_20260905/python_deps']))
    return [str(py),str(R/'source/launch.py'),model,'--memory-policy',arm,'--',*args],env


def verify(frozen,inputs=False):
    for p,h in frozen['sources'].items():
        if sha(p)!=h:raise RuntimeError('Frozen source changed: '+p)
    if inputs:
        for p,h in frozen['inputs'].items():
            if sha(p)!=h:raise RuntimeError('Frozen input changed: '+p)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--check',action='store_true');args=parser.parse_args()
    if args.prepare:prepare();return
    frozen=json.loads((R/'freeze.json').read_text());verify(frozen,True)
    if args.check:
        for pair in frozen['pairs']:
            for arm in ['commanded','measured']:
                argv,_=command(pair,arm,frozen['protocol'])
                if not Path(argv[0]).is_file():raise FileNotFoundError(argv[0])
        print('Frozen sources, inputs, and generation commands verified');return
    if (R/'status.json').exists():raise FileExistsError('Existing queue preserved; use a new version for reruns')
    proof=json.loads((R/'cpu_validation.json').read_text())
    if not proof['passed'] or proof['freeze_sha256']!=sha(R/'freeze.json'):raise RuntimeError('CPU qualification does not match freeze')
    state=dict(method_id=frozen['protocol']['method_id'],freeze_sha256=sha(R/'freeze.json'),pid=os.getpid(),started=time.time(),phase='waiting_for_gpu',jobs={},pairs={},held_models={},failures=[])
    def save():
        tmp=R/'status.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(R/'status.json')
    save();handles=[];active=None
    def stop():
        if active is None:return
        try:children=psutil.Process(active.pid).children(recursive=True)
        except psutil.Error:children=[]
        for child in reversed(children):
            try:child.terminate()
            except psutil.Error:pass
        try:os.killpg(active.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:active.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:os.killpg(active.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            active.wait()
        _,alive=psutil.wait_procs(children,timeout=5)
        for child in alive:
            try:child.kill()
            except psutil.Error:pass
    def interrupt(signum,frame):stop();raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,interrupt)
    signal.signal(signal.SIGINT,interrupt)
    def job(key,argv,env):
        nonlocal active
        verify(frozen)
        if sha(R/'freeze.json')!=state['freeze_sha256']:raise RuntimeError('Freeze manifest changed')
        if shutil.disk_usage(R).free<4*2**30:raise RuntimeError('Disk reserve reached; preserve artifacts and hold queue')
        deadline=time.time()+60
        while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            if time.time()>deadline:raise RuntimeError('GPU occupied outside this queue')
            time.sleep(2)
        state['phase']=key;record=dict(state='running',argv=argv,started=time.time());state['jobs'][key]=record;save()
        with (R/'logs'/f'{key}.log').open('x') as log:
            active=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);record['pid']=active.pid;save()
            reason=None
            while active.poll() is None:
                available=psutil.virtual_memory().available
                record['min_host_available_bytes']=min(record.get('min_host_available_bytes',available),available)
                save()
                if time.time()-record['started']>7200:reason='job_timeout'
                if shutil.disk_usage(R).free<2*2**30:reason='disk_guard'
                if psutil.virtual_memory().available<2*2**30:reason='host_memory_guard'
                if reason:stop();break
                time.sleep(5)
            rc=active.wait()
            if reason:rc=rc or 124
            record['stop_reason']=reason
            if rc:stop()
            active=None
        record.update(state='completed' if rc==0 else 'failed',returncode=rc,finished=time.time());save()
        if reason in ('disk_guard','host_memory_guard'):raise RuntimeError(f'Reserve guard reached: {reason}')
        if rc:raise RuntimeError(f'{key} failed ({rc})')
    try:
        C=A/'continuous_eval_20261029';S=C/'scale_20261030'
        for d in [C,C/'repair_20261029',S,S/'measured_gate_cohort',A/'vipe_eval_20261030']:
            f=(d/'queue.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX);handles.append(f)
        for pair in frozen['pairs']:
            key=pair['key'];model=pair['model']
            if model in state['held_models']:
                state['pairs'][key]=dict(state='held',reason=state['held_models'][model]);save();continue
            if key in frozen.get('reuse_pairs',{}):
                reuse=frozen['reuse_pairs'][key]
                for path,value in reuse['hashes'].items():
                    if sha(path)!=value:raise RuntimeError('Frozen reused evidence changed: '+path)
                score=json.loads(Path(reuse['score']).read_text())
                state['pairs'][key]=dict(state='reused',score=reuse['score'],prefix_exact=score['prefix_rgb_exact'],
                    position_valid={a:v['position_valid'] for a,v in score['arms'].items()},note=reuse['note'])
                save();continue
            try:
                for arm in ['commanded','measured']:
                    argv,env=command(pair,arm,frozen['protocol']);job(key+'_'+arm,argv,env)
                    if key=='sana_02267acf_s7' and arm=='commanded':
                        job(key+'_resource_parity',[str(PY),str(R/'resource_parity.py')],env)
                    job(key+'_'+arm+'_retain',[str(PY),str(R/'retain.py'),'--spec',str(R/'specs'/f'{key}.json'),'--arm',arm],env)
                job(key+'_score',[str(PY),str(R/'source/score_memory_pair.py'),'--spec',str(R/'specs'/f'{key}.json'),'--out',str(R/'scores'/key)],env)
                score=json.loads((R/'scores'/key/'score.json').read_text())
                valid=score['comparison_valid'] is True
                state['pairs'][key]=dict(state='completed' if valid else 'confounded',prefix_exact=score['prefix_rgb_exact'],position_valid={a:v['position_valid'] for a,v in score['arms'].items()},score=str(R/'scores'/key/'score.json'))
                if not valid:state['held_models'][model]='Pre-intervention prefix mismatch'
                save()
            except Exception as exc:
                state['failures'].append(dict(pair=key,error=repr(exc)));state['pairs'][key]=dict(state='failed',reason=repr(exc));save()  # Runtime/geometry failures remain pair-level; other prespecified scenes continue.
                if 'resource_parity' in str(exc):
                    state['held_models'][model]='Resource overlay prefix mismatch; qualification failed';save();continue
                if 'host_memory_guard' in str(exc):
                    state['held_models'][model]='Host-memory guard: '+str(exc);save();continue
                if any(term in str(exc).lower() for term in ['frozen','freeze','reserve']):raise
        state['phase']='finished_with_holds' if state['held_models'] or state['failures'] else 'finished'
    except KeyboardInterrupt:state['phase']='interrupted'
    except Exception as exc:state.update(phase='held',error=repr(exc))
    finally:
        stop()
        state['finished']=time.time();save()
        for f in handles:f.close()
    if state['phase']!='finished':raise SystemExit(1)


if __name__=='__main__':main()
