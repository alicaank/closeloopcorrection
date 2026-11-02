import ast,random,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from camera_loop.native_response_runtime import native_navigation_step,preserve_rng
from camera_loop.integrations import BUNDLE

BASE=BUNDLE/'templates/analysis/plan_base_restart_20261008'
def extract(path,name,scope):
    node=next(n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.FunctionDef) and n.name==name)
    node.decorator_list=[];exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope);return scope[name]

class ResponseRuntimeTests(unittest.TestCase):
    def test_native_steps_read_from_current_adapter_contract(self):
        root=Path(__file__).resolve().parents[2]
        for model in ('astronex','worldplay'):
            if not (root/'sota_repos').exists():self.skipTest('Native checkout unavailable')
            step,receipt=native_navigation_step(root,model)
            self.assertGreater(step,0);self.assertEqual(len(receipt['sha256']),64)
            self.assertIn('not physical',receipt['semantic'])
    def test_observer_rng_isolation_even_on_rejection(self):
        import torch
        random.seed(3);np.random.seed(4);torch.manual_seed(5)
        before=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
        with self.assertRaises(RuntimeError):
            with preserve_rng():random.random();np.random.rand();torch.rand(2);raise RuntimeError('rejected')
        self.assertEqual(before[0],random.getstate());np.testing.assert_array_equal(before[1][1],np.random.get_state()[1]);self.assertTrue(torch.equal(before[2],torch.get_rng_state()))
    def test_worldplay_native_command_encoding_does_not_require_physical_scale(self):
        import torch
        mem=np.tile(np.eye(4),(8,1,1));mem[:,0,3]=np.arange(8)*.12
        host=SimpleNamespace(mem=mem,response=object(),unit=None,Kn=np.eye(3),wp=SimpleNamespace(device='cpu'))
        fn=extract(BASE/'fair_common_20261025/wp_fair.py','wp_rows',dict(np=np,torch=torch,action_label=lambda a,b:0))
        vm,ks,act=fn(host,4,8)
        np.testing.assert_allclose(np.linalg.inv(vm.numpy()[0]),mem[4:8],atol=1e-6)
    def test_astronex_updates_only_future_native_rows(self):
        import torch
        n=24;cmd=np.tile(np.eye(4),(n,1,1));future=cmd[8:16].copy();future[:,0,3]=np.arange(8)+2
        class Runtime:
            def observe(self,gen,index,end):
                self.args=(index,end);return future,dict(commands_differ_from_bootstrap=True),np.eye(4)
        runtime=Runtime();v=dict(response=runtime,measured={},events=[],cmd=list(cmd.copy()))
        vm=torch.eye(4).repeat(1,n,1,1);old=vm[:,:8].clone()
        fn=extract(BASE/'long_domain_20261028/astronex_run.py','_measure_and_act',dict(np=np,torch=torch,BLOCK=8,NLAT=n))
        fn({},v,{},np.zeros((29,2,2,3)),{},vm,8,0,False)
        self.assertEqual(runtime.args,(7,16));self.assertTrue(torch.equal(vm[:,:8],old))
        np.testing.assert_allclose(vm[0,8:16],np.linalg.inv(future))
    def test_host_sources_compile(self):
        for rel in ['long_domain_20261028/astronex_run.py','fair_common_20261025/wp_fair.py']:
            p=BASE/rel;compile(p.read_text(),str(p),'exec')
            self.assertIn('--online-response',p.read_text())

if __name__=='__main__':unittest.main()
