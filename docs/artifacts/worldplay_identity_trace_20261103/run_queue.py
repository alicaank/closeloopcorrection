"""Frozen WorldPlay identity trace; unchanged full-horizon A/S only."""
import fcntl,hashlib,json,os,shutil,signal,subprocess,time
from pathlib import Path
import psutil
R=Path(__file__).resolve().parent;ROOT=R.parents[1];A=ROOT/'analysis/plan_base_restart_20261008'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify(m):
    for p,h in m['hashes'].items():
        if sha(p)!=h:raise RuntimeError('Frozen source/input changed: '+p)
    for p,record in m['weights'].items():
        s=Path(p).stat()
        if [s.st_size,s.st_mtime_ns]!=record:raise RuntimeError('Checkpoint changed: '+p)
    proof=json.loads((R/'cpu_validation.json').read_text())
    if not proof['passed'] or proof['manifest_sha256']!=sha(R/'manifest.json'):raise RuntimeError('CPU proof does not match frozen manifest')

def environment():
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1',
        OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2',MAX_JOBS='2',
        MASTER_ADDR='127.0.0.1',MASTER_PORT='29647',
        CAMERA_BATCH=str(R/'batch'),LONG_BATCH=str(R/'batch'),CAMERA_LOOP_WORKSPACE=str(ROOT),
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',CUDA_HOME='/home/hj/miniconda3/envs/cuda128',TORCH_CUDA_ARCH_LIST='12.0')
    for k in ['LD_LIBRARY_PATH','CUDA_VISIBLE_DEVICES','GV_DA3_FAKE','CAMERA_LOOP_UNIT_PROVIDER','RANK','LOCAL_RANK','WORLD_SIZE']:env.pop(k,None)
    env['PYTHONPATH']=os.pathsep.join(map(str,[R/'frozen',R,ROOT,ROOT/'training_runs/memory1500_resplat_20260905/python_deps']))
    return env

def compare(m,model,seed,arms,whole):
    records=[]
    for arm in arms:
        p=R/'runs'/model/arm/f"{m['scene']}_s{seed}.json"
        data=json.loads(p.read_text())['online_response'];records.append(data)
    hashes=[d['prefix_hashes'] if whole else d['prefix_hashes'][:1] for d in records]
    exact=all(x==hashes[0] for x in hashes[1:])
    if not hashes[0]:raise RuntimeError('Missing generated-prefix receipts')
    if not exact:raise RuntimeError('Decoded RGB equality failed: '+str((model,seed,arms,whole)))
    return dict(exact=exact,scope='all observed decoded prefixes' if whole else 'first decoded chunk',
        boundaries=len(hashes[0]),arms=arms,frame_id=hashes[0][-1]['frame_id'])

def main():
    path=R/'status.json'
    if path.exists():raise FileExistsError('Existing queue state preserved')
    m=json.loads((R/'manifest.json').read_text());verify(m);env=environment()
    state=dict(phase='waiting_for_gpu_lock',pid=os.getpid(),started=time.time(),manifest_sha256=sha(R/'manifest.json'),
        follows='Full-horizon diagnostic of unchanged-command WorldPlay mismatch; no correction arm',jobs={},qualification={},failures=[])
    def save():
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(path)
    save();handles=[];active=None
    def stop():
        if active is None or active.poll() is not None:return
        try:children=psutil.Process(active.pid).children(recursive=True)
        except psutil.Error:children=[]
        for p in reversed(children):
            try:p.terminate()
            except psutil.Error:pass
        try:os.killpg(active.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        _,alive=psutil.wait_procs(children,timeout=10)
        for p in alive:
            try:p.kill()
            except psutil.Error:pass
        try:active.wait(timeout=10)
        except subprocess.TimeoutExpired:os.killpg(active.pid,signal.SIGKILL);active.wait()
    def interrupt(signum,frame):stop();raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    def job(name,argv):
        nonlocal active
        verify(m)
        if shutil.disk_usage(R).free<8*2**30:raise RuntimeError('8 GiB launch reserve reached; no data deleted')
        state['phase']=name;rec=dict(state='running',argv=argv,started=time.time());state['jobs'][name]=rec;save()
        with (R/'logs'/f'{name}.log').open('x') as log:
            job_env=env.copy()
            if '/venv_worldplay/' in argv[0]:job_env.update(RANK='0',LOCAL_RANK='0',WORLD_SIZE='1')
            active=subprocess.Popen(argv,env=job_env,cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            rec['pid']=active.pid;save();reason=None
            while active.poll() is None:
                if time.time()-rec['started']>28800:reason='8_hour_job_timeout'
                if shutil.disk_usage(R).free<4*2**30:reason='disk_guard'
                if psutil.virtual_memory().available<2*2**30:reason='host_memory_guard'
                if reason:stop();break
                time.sleep(5)
            rc=active.wait();active=None
        rec.update(state='completed' if rc==0 and reason is None else 'failed',returncode=rc,stop_reason=reason,finished=time.time());save()
        if rec['state']!='completed':raise RuntimeError(f'{name}: {rc}, {reason}')
    try:
        while shutil.disk_usage(R).free < 8*2**30:
            state['phase']='waiting_for_disk';state['free_bytes']=shutil.disk_usage(R).free;save()
            time.sleep(30)
        state['phase']='waiting_for_gpu_lock';save()
        C=A/'continuous_eval_20261029';S=C/'scale_20261030'
        for root in [C,C/'repair_20261029',S,S/'measured_gate_cohort',A/'vipe_eval_20261030']:
            f=(root/'queue.lock').open('a');handles.append(f);fcntl.flock(f,fcntl.LOCK_EX)
        state['phase']='waiting_for_gpu_idle';save()
        while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():time.sleep(15)
        for entry in m['jobs']:
            job(entry['key'], entry['argv'])
        job('compare_trace', [str(ROOT/'baselines/venv_vmem/bin/python'), str(R/'compare.py')])
        result=json.loads((R/'comparison.json').read_text())
        state['qualification']['worldplay']=result['gate']
        if not result['gate']['exact_commands'] or not result['gate']['exact_prefixes']:
            raise RuntimeError('WorldPlay traced A/S identity failed; comparison retained; correction remains held')
        state['phase']='finished'
    except KeyboardInterrupt:state['phase']='interrupted'
    except Exception as exc:state['phase']='held';state['failures'].append(dict(error=repr(exc)))
    finally:
        stop();state['finished']=time.time();save()
        for f in handles:f.close()

if __name__=='__main__':main()
