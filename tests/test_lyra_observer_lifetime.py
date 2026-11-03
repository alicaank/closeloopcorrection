"""Execute the bundled callback with fake IPC and native continuation."""
import ast
import contextlib
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
import torch
import camera_loop


class LyraObserverLifetime(unittest.TestCase):
    def test_recreated_worker_closes_before_restore_and_native_continuation(self):
        path = Path(camera_loop.__file__).parent / 'bundled/templates/analysis/plan_base_restart_20261008/continuous_eval_20261029/lyra_plugin.py'
        node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'gv_step')
        for fail in (None, 'start', 'measure'):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as tmp:
                calls = []
                class Worker:
                    def __init__(self, out):
                        calls.append('start')
                        if fail == 'start': raise RuntimeError('start failed')
                    def measure(self, *args):
                        calls.append('measure')
                        if fail == 'measure': raise RuntimeError('measure failed')
                        return np.eye(4), {}
                    def close(self): calls.append('close')
                class Model:
                    def parameters(self): return iter([torch.zeros(1)])
                    def to(self, device): calls.append('local_to')
                trajectory = Path(tmp) / 'trajectory.npz'
                np.savez(trajectory, w2c=np.repeat(np.eye(4)[None],3,axis=0), intrinsics=np.eye(3)[None], anchor_depth=np.ones((2,2)))
                meta = Path(tmp) / 'meta.json'
                meta.write_text(json.dumps(dict(out=tmp, trajectory=str(trajectory), pose_scale=1)))
                pipe = types.SimpleNamespace(history_frames=torch.zeros(1,3,2,2,2), start_index=0,
                    cam_w2c=torch.eye(4).repeat(1,2,1,1), local_da3_model=Model(), model=object())
                scope = dict(__file__=str(path), np=np, torch=torch, json=json, os=os, sys=__import__('sys'),
                    ARM={'name':'baseline'}, _event=lambda **kw:calls.append('receipt'),
                    _orig_step=lambda *a,**kw:calls.append('native'),
                    ARI=types.SimpleNamespace(_offload_diffusion_to_cpu=lambda *a:calls.append('offload'),
                                             _restore_diffusion_to_gpu=lambda *a:calls.append('restore')))
                exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
                fake = types.SimpleNamespace(PoseWorker=Worker, rejoin=lambda *a:None)
                with patch.dict('sys.modules',{'common':fake}), patch.dict(os.environ,{'LONG_LYRA_META':str(meta)}), \
                     patch.object(torch.random,'fork_rng',return_value=contextlib.nullcontext()), patch.object(torch.cuda,'empty_cache'):
                    call=lambda:scope['gv_step'](pipe, cam_w2c_chunk=torch.eye(4)[None,None], intrinsics_chunk=None)
                    if fail:
                        with self.assertRaisesRegex(RuntimeError,fail+' failed'):call()
                    else:call();call()
                expected=['local_to','offload','start']
                if fail != 'start':expected+=['measure','close']
                expected+=['local_to','restore']
                if fail is None:expected+=['receipt','native'];expected*=2
                self.assertEqual(calls,expected)
                self.assertFalse(hasattr(pipe,'_long_observer'))
