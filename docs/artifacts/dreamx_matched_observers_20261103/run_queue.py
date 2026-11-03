"""One diagnostic under the existing GPU locks and resource reserves."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import psutil

ROOT = Path('/home/hj/Alican')
HERE = Path(__file__).resolve().parent


def main():
    if (HERE / 'status.json').exists(): raise FileExistsError('Preserve existing attempt')
    state = dict(pid=os.getpid(), phase='waiting_for_gpu_lock', started=time.time())
    handles = []; child = None
    def save():
        path = HERE / 'status.tmp'; path.write_text(json.dumps(state, indent=2) + '\n'); path.replace(HERE / 'status.json')
    def stop():
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try: child.wait(timeout=15)
            except subprocess.TimeoutExpired: os.killpg(child.pid, signal.SIGKILL); child.wait()
    def interrupt(signum, frame): stop(); raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, interrupt); signal.signal(signal.SIGINT, interrupt)
    save()
    try:
        c = ROOT / 'analysis/plan_base_restart_20261008/continuous_eval_20261029'; s = c / 'scale_20261030'
        for folder in [c, c/'repair_20261029', s, s/'measured_gate_cohort', ROOT/'analysis/plan_base_restart_20261008/vipe_eval_20261030']:
            f = (folder / 'queue.lock').open('a'); handles.append(f); fcntl.flock(f, fcntl.LOCK_EX)
        manifest = json.loads((HERE / 'manifest.json').read_text())
        for path, digest in manifest['sources'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
        assert json.loads((HERE/'cpu_validation.json').read_text())['passed']
        if shutil.disk_usage(HERE).free < 4*2**30: raise RuntimeError('4 GiB launch reserve')
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(): raise RuntimeError('GPU occupied outside queue')
        env = dict(os.environ, HF_HUB_OFFLINE='1', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
        for name in ('LD_LIBRARY_PATH','CUDA_VISIBLE_DEVICES','GV_DA3_FAKE'): env.pop(name, None)
        with (HERE/'run.log').open('x') as log:
            child = subprocess.Popen([str(ROOT/'baselines/venv_vmem/bin/python'),str(HERE/'compare.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state.update(phase='running', child_pid=child.pid, job_started=time.time()); save()
            while child.poll() is None:
                if shutil.disk_usage(HERE).free < 2*2**30: raise RuntimeError('2 GiB runtime disk reserve')
                if psutil.virtual_memory().available < 2*2**30: raise RuntimeError('2 GiB host reserve')
                if time.time()-state['job_started'] > 5400: raise RuntimeError('90 minute job limit')
                time.sleep(5)
            state.update(returncode=child.returncode, phase='finished' if child.returncode == 0 else 'failed')
    except KeyboardInterrupt: state['phase']='interrupted'
    except Exception as exc: state.update(phase='held', reason=repr(exc))
    finally:
        stop(); state['finished']=time.time(); save()
        for handle in handles: handle.close()


if __name__ == '__main__': main()
