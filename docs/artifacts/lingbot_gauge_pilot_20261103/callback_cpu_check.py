"""Exercise the frozen first-boundary callback with real native CPU encoding."""
import contextlib,importlib.util,json,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
R=Path(__file__).resolve().parent/'pilot';sys.path.insert(0,str(R/'source'))
import lingbot_gauge_hook as hook
import memory_hooks as mm
P=Path('/home/hj/Alican/analysis/plan_base_restart_20261008/model_readiness_20261027/repos/lingbot/wan/utils/cam_utils.py')
spec=importlib.util.spec_from_file_location('cpu_native_camera',P);cam=importlib.util.module_from_spec(spec);spec.loader.exec_module(cam)


class CallbackTests(unittest.TestCase):
    def run_arm(self,arm,accepted):
        with tempfile.TemporaryDirectory() as tmp:
            calls=[];out=Path(tmp)
            class Model:
                def cpu(self):calls.append('offload')
                def to(self,device):calls.append('restore')
            plan=np.repeat(np.eye(4,dtype=np.float32)[None],20,axis=0);plan[:,0,3]=np.arange(20)*.01
            raw=cam.compute_relative_poses(torch.tensor(plan),framewise=True,normalize_trans=False)
            norm=float(raw[:,:3,3].norm(dim=-1).max())
            measured=plan[:4].copy();measured[:,0,3]*=50
            pipe=types.SimpleNamespace(device='cpu',param_dtype=torch.bfloat16,model=Model(),
                vae=types.SimpleNamespace(decode=lambda x:[torch.zeros(3,13,8,8)]))
            obs=types.SimpleNamespace(norm=norm,commands=plan.copy(),memory_context=dict(out=out,K=np.eye(3),depth=np.ones((8,8),np.float32)))
            pipe._loop_observer=obs
            L=dict(self=pipe,chunk_id=0,chunk_size=4,num_inference_chunk=5,h=8,w=8,lat_h=1,lat_w=1,
                Ks=torch.tensor([[4.,4.,4.,4.]]),pred_latent_chunks=[torch.zeros(3,4,1,1)],
                x0=torch.zeros(3,4,1,1),noise=torch.zeros(3,20,1,1),seed_g=torch.Generator().manual_seed(7))
            fake=types.SimpleNamespace(measure=lambda *a:(measured,{'passed':accepted}))
            with patch.dict('sys.modules',{'wan.utils.cam_utils':cam}):
                native=raw.clone();native[:,:3,3]/=norm
                encoded=hook.encode(native,L['Ks'],L);L['c2ws_plucker_emb']=encoded.clone()
                L['c2ws_plucker_emb_chunk']=L['c2ws_plucker_emb'].split(4,dim=2)
                L['kwargs']=dict(dit_cond_dict={'c2ws_plucker_emb':L['c2ws_plucker_emb_chunk'][0].chunk(1,dim=0)})
                with patch.dict('os.environ',{'CAMERA_LINGBOT_GAUGE_ARM':arm}), \
                    patch.object(torch.random,'fork_rng',return_value=contextlib.nullcontext()), \
                    patch.object(torch.cuda,'get_rng_state_all',return_value=[]), \
                    patch.object(mm,'observer',return_value=fake),patch.object(mm,'close_observer',side_effect=lambda out:calls.append('close')):
                    hook.before_clean(L)
            receipt=json.loads((out/'gauge_receipt.json').read_text())
            self.assertEqual(calls,[] if arm=='A' else ['offload','close','restore'])
            np.testing.assert_array_equal(obs.commands,plan)
            if arm in ('A','I') or not accepted:self.assertTrue(torch.equal(encoded,L['c2ws_plucker_emb']))
            self.assertEqual(receipt['rng_before'],receipt['rng_after'])
            return L['c2ws_plucker_emb'],receipt

    def test_accepted_arms_have_identical_future_and_distinct_clean_history(self):
        a,ar=self.run_arm('A',True);i,ir=self.run_arm('I',True)
        g,gr=self.run_arm('G',True);m,mr=self.run_arm('M',True)
        self.assertTrue(torch.equal(a,i));self.assertTrue(torch.equal(g[:,:,4:],m[:,:,4:]))
        self.assertFalse(torch.equal(g[:,:,:4],m[:,:,:4]));self.assertFalse(torch.equal(a[:,:,4:],g[:,:,4:]))
        self.assertEqual(gr['denominator'],mr['denominator'])

    def test_rejected_observation_is_exact_native_conditioning_for_every_arm(self):
        outputs=[self.run_arm(arm,False) for arm in ('A','I','G','M')]
        for x,r in outputs:
            self.assertTrue(torch.equal(outputs[0][0],x));self.assertFalse(r['observation_accepted'])
            self.assertFalse(r['gauge_applied']);self.assertFalse(r['memory_write_applied'])


if __name__=='__main__':unittest.main()
