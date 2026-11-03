"""Small serial diagnostics with fixed sources and explicit resource bounds."""
import fcntl,hashlib,json,os,signal,subprocess,time,shutil
from pathlib import Path
import psutil
W=Path('/home/hj/Alican');R=Path('/home/hj/Alican/analysis/lingbot_shared_gauge_20261103/pilot');A=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
RESUME_SHA='01d4c96cf234e45bf815ed25ab906a947a243c21e2e0c8fef9a666b5428ecd03'
def main():
    if (A/'status.json').exists():raise FileExistsError('Preserve queue state')
    m=json.loads((R/'manifest.json').read_text());proof=json.loads((R/'cpu_validation.json').read_text())
    assert proof['passed'] and proof['manifest_sha256']==sha(R/'manifest.json')
    state=dict(pid=os.getpid(),phase='waiting_for_gpu_lock',started=time.time(),manifest_sha256=sha(R/'manifest.json'),jobs={},failures=[],held_cases={})
    state['resume_provenance']=json.loads((A/'resume.json').read_text())
    def save():
        p=A/'status.tmp';p.write_text(json.dumps(state,indent=2)+'\n');p.replace(A/'status.json')
    def verify():
        assert sha(R/'status.json') == '20519d8c4bf464168bda47126adef44c03b1abe32ce6aca4a96225fc5f641cb9'
        assert sha(R/'run_queue.py') == '5b0083e1df30413b42be9c4a0738f9c8439ccabff93ead21d04f355b08838369'
        assert sha(A/'resume.json') == RESUME_SHA
        assert sha(R/'manifest.json')==state['manifest_sha256']
        for p,h in m['sources'].items():
            if sha(p)!=h:raise RuntimeError('Frozen source/input changed: '+p)
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
        try:active.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:os.killpg(active.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            active.wait()
        _,alive=psutil.wait_procs(children,timeout=5)
        for child in alive:
            try:child.kill()
            except psutil.Error:pass
    def interrupt(signum,frame):stop();raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2',
        CUDA_HOME='/home/hj/miniconda3/envs/cuda128',DS_BUILD_OPS='0',MAX_JOBS='2',TORCH_CUDA_ARCH_LIST='12.0',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    for key in ['LD_LIBRARY_PATH','CUDA_VISIBLE_DEVICES','GV_DA3_FAKE','LOCAL_RANK','RANK','WORLD_SIZE']:env.pop(key,None)
    env['GV_SCALE_CACHE']=str(R/'scale_calibration')
    env['PYTHONPATH']=os.pathsep.join(map(str,[R/'source',R,W,W/'training_runs/memory1500_resplat_20260905/python_deps']))
    try:
        C=W/'analysis/plan_base_restart_20261008/continuous_eval_20261029';S=C/'scale_20261030'
        for d in [C,C/'repair_20261029',S,S/'measured_gate_cohort',W/'analysis/plan_base_restart_20261008/vipe_eval_20261030']:
            f=(d/'queue.lock').open('a');handles.append(f);fcntl.flock(f,fcntl.LOCK_EX)
        verify()
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU occupied outside queue')
        for entry in m['jobs']:
            if entry['name'] in set():continue
            if entry['case'] in state['held_cases']:
                state['jobs'][entry['name']]=dict(state='skipped',reason='case qualification failed');save();continue
            verify()
            if shutil.disk_usage(R).free<4*2**30:raise RuntimeError('4 GiB launch reserve; originals retained')
            key=entry['name'];state['phase']=key;rec=dict(state='running',started=time.time(),argv=entry['argv']);state['jobs'][key]=rec;save()
            with (A/'logs'/f'{key}.log').open('x') as log:
                active=subprocess.Popen(entry['argv'],env=dict(env,**entry.get('env',{})),cwd=W,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);rec['pid']=active.pid;save();reason=None
                while active.poll() is None:
                    available=psutil.virtual_memory().available
                    rec['min_host_available_bytes']=min(rec.get('min_host_available_bytes',available),available)
                    try: processes=[psutil.Process(active.pid),*psutil.Process(active.pid).children(recursive=True)];rss=sum(p.memory_info().rss for p in processes if p.is_running())
                    except psutil.Error:rss=0
                    rec['peak_process_tree_rss_bytes']=max(rec.get('peak_process_tree_rss_bytes',0),rss)
                    sample=dict(time=time.time(),job=key,host_available_bytes=available,processes=[])
                    try:
                        for proc in [psutil.Process(active.pid),*psutil.Process(active.pid).children(recursive=True)]:
                            sample['processes'].append(dict(pid=proc.pid,ppid=proc.ppid(),rss_bytes=proc.memory_info().rss))
                    except psutil.Error:sample['partial']=True
                    with (R/'logs/resource_samples.jsonl').open('a') as trace:trace.write(json.dumps(sample)+'\n')
                    save()
                    if shutil.disk_usage(R).free<2*2**30:reason='disk_guard'
                    if psutil.virtual_memory().available<2*2**30:reason='host_memory_guard'
                    if time.time()-rec['started']>5400:reason='timeout'
                    if reason:stop();break
                    time.sleep(5)
                rc=active.wait()
                if rc:stop()
                active=None
            rec.update(state='completed' if rc==0 and reason is None else 'failed',returncode=rc,stop_reason=reason,finished=time.time());save()
            if rec['state']=='failed':
                message=f'{key}: {rc} / {reason}';state['failures'].append(message);state['held_cases'][entry['case']]=message;save()
                if reason in ['disk_guard','host_memory_guard'] or entry.get('required'):raise RuntimeError(message)
            deadline=time.time()+45
            while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                if time.time()>deadline:raise RuntimeError('GPU still occupied after job exit')
                time.sleep(2)
        state['phase']='finished' if not state['failures'] else 'finished_with_failures'
    except KeyboardInterrupt:state['phase']='interrupted'
    except Exception as exc:state['phase']='held';state['failures'].append(repr(exc))
    finally:
        stop();state['finished']=time.time();save()
        for f in handles:f.close()
if __name__=='__main__':main()
