"""Exercise the actual bundled callback without loading native model weights."""
import ast
import contextlib
import tempfile
import types
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import camera_loop


class SanaObserverLifetime(unittest.TestCase):
    def callback(self, directory, fail=False, residency=None):
        path=Path(camera_loop.__file__).parent/'bundled/templates/analysis/plan_base_restart_20261008/new_models_loop_20261028/sana_run.py'
        main=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='main')
        cb=next(n for n in main.body if isinstance(n,ast.FunctionDef) and n.name=='callback')
        outer=ast.parse('def build():\n worker=None\n return None').body[0]
        outer.body[1:]=[cb,ast.Return(value=ast.Name(id='callback',ctx=ast.Load()))]
        module=ast.fix_missing_locations(ast.Module(body=[outer],type_ignores=[]))
        workers=[]
        class Worker:
            def __init__(self,out):
                if residency is not None:assert residency['active'], 'Move weights before worker startup'
                self.closed=False;workers.append(self)
            def measure(self,frames,K,depth):
                if fail:raise RuntimeError('measurement failed')
                return np.eye(4),{'ok':True}
            def close(self):self.closed=True
        ns=types.SimpleNamespace
        def move_decoder(device):
            if residency is not None:assert residency['active'], 'Move stage-1 weights before decoder CPU offload'
        pipe=ns(model=object(),_move_vae_decoder_for_streaming=move_decoder,
                config=ns(vae=ns(vae_stride=(8,32,32))),refiner_settings=ns(block_size=3))
        scope=dict(np=np,PoseWorker=Worker,
                   torch=ns(cuda=ns(empty_cache=lambda:None),random=ns(fork_rng=lambda **kw:contextlib.nullcontext())),
                   a=ns(out=directory,arm='baseline',frames=121),prefix=[np.zeros((2,2,3),np.uint8)],
                   state={'pipe':pipe,'latent_T':16},K=np.eye(3),depth=np.ones((2,2)),events=[])
        import hashlib,json
        scope.update(hashlib=hashlib,json=json)
        exec(compile(module,str(path),'exec'),scope)
        return scope['build'](),workers,scope

    def test_release_before_next_chunk_and_recreate(self):
        with tempfile.TemporaryDirectory() as directory:
            callback,workers,scope=self.callback(Path(directory))
            frames=np.zeros((24,2,2,3),np.uint8)
            callback(frames,1,0)
            self.assertEqual(len(workers),1);self.assertTrue(workers[0].closed)
            callback(frames,25,1)
            self.assertEqual(len(workers),2);self.assertTrue(all(w.closed for w in workers))
            self.assertEqual([e['observed_frame'] for e in scope['events']],[24,48])
            self.assertTrue(all(not e['correction_applied'] for e in scope['events']))

    def test_release_when_measurement_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            callback,workers,_=self.callback(Path(directory),fail=True)
            with self.assertRaisesRegex(RuntimeError,'measurement failed'):
                callback(np.zeros((24,2,2,3),np.uint8),1,0)
            self.assertEqual(len(workers),1);self.assertTrue(workers[0].closed)

    def test_worker_lifetime_inside_resource_scope(self):
        for fail in (False,True):
            with self.subTest(fail=fail),tempfile.TemporaryDirectory() as directory:
                residency={'active':False}
                callback,workers,_=self.callback(Path(directory),fail=fail,residency=residency)
                @contextlib.contextmanager
                def placement(model):
                    residency['active']=True
                    try:yield
                    finally:
                        self.assertTrue(workers and all(w.closed for w in workers))
                        residency['active']=False
                with patch('camera_loop.resource_residency.observer_residency',placement):
                    if fail:
                        with self.assertRaisesRegex(RuntimeError,'measurement failed'):
                            callback(np.zeros((24,2,2,3),np.uint8),1,0)
                    else:callback(np.zeros((24,2,2,3),np.uint8),1,0)
                self.assertFalse(residency['active'])

    def test_restore_resource_scope_when_decoder_offload_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            residency={'active':False}
            callback,workers,scope=self.callback(Path(directory),residency=residency)
            def fail_offload(device):
                self.assertTrue(residency['active'])
                raise RuntimeError('decoder offload failed')
            scope['state']['pipe']._move_vae_decoder_for_streaming=fail_offload
            @contextlib.contextmanager
            def placement(model):
                residency['active']=True
                try:yield
                finally:residency['active']=False
            with patch('camera_loop.resource_residency.observer_residency',placement):
                with self.assertRaisesRegex(RuntimeError,'decoder offload failed'):
                    callback(np.zeros((24,2,2,3),np.uint8),1,0)
            self.assertFalse(residency['active'])
            self.assertEqual(workers,[])


if __name__=='__main__':unittest.main()
