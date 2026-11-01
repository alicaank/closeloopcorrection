"""Shared measured-memory IPC and receipts; no native sources are modified."""
import atexit
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT=Path('@WORKSPACE@');R=Path(__file__).resolve().parent
A=ROOT/'analysis/plan_base_restart_20261008'
from memory_contract import stage_memory_write


def policy():
    p=os.environ.get('CAMERA_MEMORY_POLICY','commanded')
    if p not in ['commanded','identity','measured']:raise ValueError('Unknown memory policy')
    return p


def append(out,record):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    with (out/'memory_events.jsonl').open('a') as f:f.write(json.dumps(record,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n')


class Observer:
    def __init__(self,out):
        self.out=Path(out);self.out.mkdir(parents=True,exist_ok=True);self.index=0
        self.log=(self.out/'worker.log').open('a')
        env=dict(os.environ,HF_HUB_OFFLINE='1',GV_OBSERVER_QUALITY='enforce')
        env.pop('LD_LIBRARY_PATH',None);env.pop('GV_DA3_FAKE',None)
        worker=A/'continuous_eval_20261029/scale_20261030/correction_v3_20261030/observer_worker.py'
        self.p=subprocess.Popen([str(ROOT/'baselines/venv_vmem/bin/python'),str(worker)],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,bufsize=1,env=env)
        line=self.p.stdout.readline()
        if not line.startswith('READY '):raise RuntimeError('Memory observer failed to initialize: '+line)
        self.ready=json.loads(line[6:]);assert not self.ready.get('fake')
        atexit.register(self.close)

    def measure(self,frames,K,depth,ids):
        ids=np.asarray(ids,dtype=int);frames=np.asarray(frames)
        if np.any(ids<0) or np.any(ids>=len(frames)) or np.any(np.diff(ids)<=0):raise ValueError('Noncausal query')
        result=np.repeat(np.eye(4)[None],len(ids),axis=0);positive=ids>0
        if not positive.any():return result,dict(passed=True,anchor_only=True)
        req=self.out/'request.npz';dst=self.out/'response.npz'
        np.savez(req,anchor=frames[0],frames=frames[1:],K=np.asarray(K,np.float32),
            query=ids[positive],anchor_depth=np.asarray(depth,np.float32),max_views=np.int64(16),tail=np.int64(min(8,int(positive.sum()))))
        self.p.stdin.write(json.dumps(dict(cmd='measure',req=str(req),out=str(dst)))+'\n');self.p.stdin.flush()
        line=self.p.stdout.readline()
        if not line:raise RuntimeError('Memory observer stopped')
        info=json.loads(line)
        if info.get('ok'):
            with np.load(dst) as z:result[positive]=z['poses']
        else:result[positive]=np.nan
        passed=bool(info.get('ok') and np.isfinite(result).all()
            and all(c.get('quality',{}).get('passed',False) for c in info.get('calls',[]))
            and info.get('calls'))
        self.index+=1
        np.savez_compressed(self.out/f'measurement_{self.index:03d}.npz',poses=result,frame_ids=ids)
        (self.out/f'measurement_{self.index:03d}.json').write_text(json.dumps(info,indent=2))
        req.unlink(missing_ok=True)
        return result,dict(passed=passed,observer=info)

    def close(self):
        if getattr(self,'p',None) is not None and self.p.poll() is None:
            try:self.p.stdin.write('{"cmd":"quit"}\n');self.p.stdin.flush();self.p.wait(timeout=30)
            except (BrokenPipeError,subprocess.TimeoutExpired):self.p.terminate();self.p.wait(timeout=15)
        if getattr(self,'log',None) is not None and not self.log.closed:self.log.close()


def choose(old,measured,ids,available,quality):
    if policy()=='commanded':return stage_memory_write(ids,old,old,available_last_frame=available,quality={'passed':False})
    if policy()=='identity':measured=old;quality={'passed':True}
    return stage_memory_write(ids,old,measured,available_last_frame=available,quality=quality)


def digest(tensor):
    import torch
    return hashlib.sha256(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def replace_once(source,old,new):
    if source.count(old)!=1:raise RuntimeError('Native hook changed: '+old[:100])
    return source.replace(old,new)
