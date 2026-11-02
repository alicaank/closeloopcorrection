"""Opt-in runtime bridge for command-only Astronex/WorldPlay adaptation."""
import ast,atexit,hashlib,json,os,random,subprocess,sys
from pathlib import Path
from contextlib import contextmanager
import numpy as np
from .online_response import OnlineCommandController
from .evidence import select_window

@contextmanager
def preserve_rng():
    import torch
    py=random.getstate();ns=np.random.get_state()
    devices=list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    try:
        with torch.random.fork_rng(devices=devices):yield
    finally:random.setstate(py);np.random.set_state(ns)


def native_navigation_step(workspace,model):
    root=Path(workspace)
    if model=='astronex':
        path=root/'sota_repos/astronex_world/utils/camera_trajectory.py';tree=ast.parse(path.read_text())
        nodes=[n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_STEP' for t in n.targets)]
        if len(nodes)!=1:raise ValueError('Astronex native navigation contract changed')
        step=float(ast.literal_eval(nodes[0].value))
    elif model=='worldplay':
        path=root/'sota_repos/worldplay/hyvideo/generate.py';tree=ast.parse(path.read_text())
        nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='parse_pose_string']
        if len(nodes)!=1:raise ValueError('WorldPlay native navigation contract changed')
        scope=dict(np=np);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)
        step=float(scope['parse_pose_string']('w-1')[0]['forward'])
    else:raise ValueError('Unsupported response host')
    if not np.isfinite(step) or step<=0:raise ValueError('Invalid native navigation step')
    return step,dict(source=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),native_step=step,
        semantic='released command convention, not physical calibration')


class BoundaryRuntime:
    def __init__(self,plan,frame_ids,*,model,arm,out,anchor,K,depth,workspace):
        self.root=Path(workspace);self.out=Path(out);self.out.mkdir(parents=True,exist_ok=False)
        self.anchor=np.asarray(anchor);self.K=np.asarray(K);self.depth=np.asarray(depth)
        step,self.contract=native_navigation_step(workspace,model)
        self.control=OnlineCommandController(plan,frame_ids,native_step=step,arm=arm)
        self.model=model;self.worker=None;self.log=None;self.previous=[];self.count=0;self.hashes=[]
        atexit.register(self.close)
        self.save()

    def start(self):
        self.log=(self.out/'observer.log').open('x')
        env=dict(os.environ,CAMERA_LOOP_WORKSPACE=str(self.root),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2')
        env.pop('GV_DA3_FAKE',None);env.pop('LD_LIBRARY_PATH',None)
        python=str(self.root/'baselines/venv_vmem/bin/python')
        self.worker=subprocess.Popen([python,str(Path(__file__).with_name('response_observer.py'))],
            env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,bufsize=1)
        ready=self.worker.stdout.readline()
        if not ready.startswith('READY '):raise RuntimeError('Response observer failed to start; see '+str(self.out/'observer.log'))
        if json.loads(ready[6:]).get('fake'):raise RuntimeError('Fake observer forbidden')

    def observe(self,frames,index,next_end):
        images=np.asarray(frames,np.uint8)
        if len(images)-1!=self.control.ids[index]:raise ValueError('Decoded output does not match observed latent endpoint')
        images=images.copy();images[0]=self.anchor
        ids=select_window(len(images)-1,self.previous)
        request=self.out/'request.npz';result=self.out/f'poses_{self.count:03d}.npz'
        np.savez(request,views=images[ids],frame_ids=ids,K=self.K,anchor_depth=self.depth)
        with preserve_rng():
            if self.worker is None:self.start()
            self.worker.stdin.write(json.dumps(dict(cmd='measure',req=str(request),out=str(result)))+'\n');self.worker.stdin.flush()
            line=self.worker.stdout.readline()
            if not line:raise RuntimeError('Response observer exited')
            info=json.loads(line)
            if not info.get('ok'):raise RuntimeError(str(info))
        with np.load(result) as z:measured=z['poses'][-1].copy()
        evidence=info['quality'];evidence['reference_depth_unit']=float(np.median(self.depth[np.isfinite(self.depth)&(self.depth>0)]))
        self.previous=info['accepted_ids'];self.count+=1
        self.hashes.append(dict(frame_id=len(images)-1,sha256=hashlib.sha256(images.tobytes()).hexdigest()))
        future,event=self.control.boundary(index,measured,evidence,next_end=next_end,available_frame=len(images)-1)
        event['evidence']=evidence
        self.save();request.unlink()
        return future,event,measured

    def metadata(self):return dict(model=self.model,contract=self.contract,prefix_hashes=self.hashes,**self.control.metadata())
    def save(self):
        path=self.out/'response.json';tmp=path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.metadata(),indent=2,allow_nan=False)+'\n');tmp.replace(path)
    def close(self):
        if self.worker is not None and self.worker.poll() is None:
            try:
                self.worker.stdin.write('{"cmd":"quit"}\n');self.worker.stdin.flush();self.worker.wait(timeout=30)
            except (BrokenPipeError,subprocess.TimeoutExpired):self.worker.terminate();self.worker.wait(timeout=20)
        if self.log is not None:self.log.close()
