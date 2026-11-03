"""Exercise native callback control flow while replacing model/IPC operations."""
import ast,contextlib,hashlib,json,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import camera_loop

ROOT=Path(camera_loop.__file__).parent/'bundled/templates/analysis'

def function(path,name,scope):
    n=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[n],type_ignores=[]),str(path),'exec'),scope)
    return scope[name]


class MatrixObserverLifetime(unittest.TestCase):
    def test_endpoint_worker_closes_before_next_phase_and_on_failure(self):
        for fail in (False,True):
            with self.subTest(fail=fail),tempfile.TemporaryDirectory() as tmp:
                calls=[];ns=types.SimpleNamespace
                class Tensor:
                    def __init__(self,a):self.a=a
                    def cpu(self):return self
                    def numpy(self):return self.a
                class Worker:
                    def __init__(self,out):calls.append('start')
                    def measure(self,*args):
                        calls.append('measure')
                        if fail:raise RuntimeError('measure failed')
                        return np.eye(4),{'ok':True}
                    def close(self):calls.append('close')
                plan=np.repeat(np.eye(4)[None],253,axis=0)
                scope=dict(np=np,torch=ns(random=ns(fork_rng=lambda **kw:contextlib.nullcontext())),
                    worker=None,PoseWorker=Worker,arm='baseline',plan=plan,prefix=np.zeros((85,2,2,3),np.uint8),
                    out=Path(tmp),K=np.eye(3),depth=np.ones((2,2)),events=[],hashlib=hashlib,json=json,native_w2c=lambda a:a)
                cb=function(ROOT/'plan_base_restart_20261008/continuous_eval_20261029/mg35_stream_launch.py','observe',scope)
                camera={'noisy_latent_indices_prope_extrinsic':Tensor(plan[85:169]),'clean_latent_indices_prope_extrinsic':Tensor(plan[81:85])}
                pipe=ns(load_models_to_device=lambda names:calls.append('offload'))
                if fail:
                    with self.assertRaisesRegex(RuntimeError,'measure failed'):cb(1,pipe,camera)
                else:
                    cb(1,pipe,camera);cb(1,pipe,camera)
                    self.assertEqual(len(scope['events']),2)
                self.assertIsNone(scope['worker'])
                self.assertEqual(calls,(['offload','start','measure','close'] if fail else ['offload','start','measure','close']*2))

    def test_memory_worker_closes_before_geometry_rebuild_and_on_failure(self):
        for fail in (False,True):
            with self.subTest(fail=fail):
                calls=[];ns=types.SimpleNamespace
                def measure(*args):
                    calls.append('measure')
                    if fail:raise RuntimeError('memory failed')
                    return ns(applied=False,poses=np.repeat(np.eye(4)[None],84,axis=0),reason='rejected'),{'passed':False}
                scope=dict(np=np,MATRIX={},policy=lambda:'measured',measured_write=measure,
                           close_observer=lambda out:calls.append('close'),append=lambda *a:calls.append('receipt'))
                cb=function(ROOT/'measured_memory_20261101/memory_hooks.py','matrix_context',scope)
                fake=ns(random=ns(fork_rng=lambda **kw:contextlib.nullcontext()))
                local=dict(out='unused',lo=84,camera={},all_commands=np.repeat(np.eye(4)[None],85,axis=0),
                           prefix=[],K=np.eye(3),depth=np.ones((2,2)),pipe=ns(load_models_to_device=lambda names:None))
                with patch.dict('sys.modules',{'torch':fake}):
                    if fail:
                        with self.assertRaisesRegex(RuntimeError,'memory failed'):cb(local)
                    else:cb(local)
                self.assertEqual(calls,['measure','close']+([] if fail else ['receipt']))


if __name__=='__main__':unittest.main()
