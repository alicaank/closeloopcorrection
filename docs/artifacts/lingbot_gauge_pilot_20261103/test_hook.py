import importlib.util,sys,types,unittest
from pathlib import Path
import numpy as np
import torch
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'source'))
from lingbot_gauge_hook import select_relative


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.plan=np.repeat(np.eye(4,dtype=np.float32)[None],20,axis=0)
        self.plan[1:,0,3]=np.arange(1,20)*.001
        self.history=self.plan[:4].copy();self.history[1:,0,3]*=50
        self.norm=float(torch.tensor(self.plan[:,:3,3]).norm(dim=-1).max())
    def test_identity_and_rejection_are_native_float32_exact(self):
        expected=torch.tensor(self.plan);expected[:,:3,3]/=self.norm
        for arm in ['A','I','G','M']:
            x,r=select_relative(self.plan,self.history,arm,False,self.norm)
            self.assertTrue(torch.equal(x,expected));self.assertFalse(r['gauge_applied'])
        x,_=select_relative(self.plan,self.history,'I',True,self.norm)
        self.assertTrue(torch.equal(x,expected))
    def test_rescaling_and_memory_share_every_future_input(self):
        g,a=select_relative(self.plan,self.history,'G',True,self.norm)
        m,b=select_relative(self.plan,self.history,'M',True,self.norm)
        self.assertTrue(torch.equal(g[4:],m[4:]));self.assertEqual(a['denominator'],b['denominator'])
        self.assertFalse(a['memory_write_applied']);self.assertTrue(b['memory_write_applied'])
        self.assertFalse(torch.equal(g[:4],m[:4]));self.assertLess(float(m[4:,:3,3].norm(dim=-1).max()),1)
    def test_unknown_arm_fails(self):
        with self.assertRaises(ValueError):select_relative(self.plan,self.history,'other',True,self.norm)


if __name__=='__main__':unittest.main()
