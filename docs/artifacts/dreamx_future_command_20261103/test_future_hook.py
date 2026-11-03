"""Native-encoder CPU checks for the exact pilot command-commit path."""
import ast,sys,types,unittest
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
R=Path(__file__).resolve().parent;W=Path('/home/hj/Alican');sys.path[:0]=[str(R),str(R/'source')]
from future_hook import encode,next_camera_chunk
from camera_loop.fractional_recovery import fractional_memory_continuation
repo=W/'analysis/plan_base_restart_20261008/model_readiness_20261027/repos/dreamx'
ns=dict(np=np,torch=torch)
for p,names in [(repo/'utils/trajectory_processor.py',['Camera']),(repo/'inference_ar_forcing.py',['_invert_SE3','get_relative_pose','cam_params_to_prope_dict'])]:
 nodes=[n for n in ast.parse(p.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
 assert {n.name for n in nodes}==set(names);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns)
entry=types.SimpleNamespace(**ns)

class NativeFutureTests(unittest.TestCase):
 def setUp(self):
  self.plan=np.repeat(np.eye(4)[None],69,axis=0);self.plan[:,0,3]=np.arange(69)*.002
  self.ids=np.array([0,17,28,29,32]);self.m=self.plan[self.ids].copy();self.m[1:,0,3]-=.02
  self.m[-2,:3,:3]=Rotation.from_euler('y',7,degrees=True).as_matrix()
  self.camera=encode(entry,self.plan,'cpu',torch.bfloat16)
 def call(self,arm='F',**kw):
  args=dict(available_frame=32,anchor_frame=29,end=9,total=18,feedback_index=1,arm=arm,quality={'passed':True},camera=self.camera,frame_seq_length=880);args.update(kw)
  return next_camera_chunk(entry,self.plan,self.plan,self.m,self.ids,**args)
 def test_observer_only_and_first_feedback_are_native_exact(self):
  for arm,index in [('A',1),('S',1),('F',0)]:
   commands,camera,r=self.call(arm,feedback_index=index)
   np.testing.assert_array_equal(commands,self.plan)
   self.assertTrue(torch.equal(camera['viewmats'],self.camera['viewmats']));self.assertFalse(r['native_input_changed'])
 def test_only_next_slice_changes_with_correct_native_anchor(self):
  commands,camera,r=self.call();lo,hi=9*880,12*880
  self.assertTrue(r['native_input_changed']);self.assertTrue(torch.equal(camera['viewmats'][:,:lo],self.camera['viewmats'][:,:lo]))
  self.assertTrue(torch.equal(camera['viewmats'][:,hi:],self.camera['viewmats'][:,hi:]));self.assertTrue(torch.equal(camera['K'],self.camera['K']))
  desired=fractional_memory_continuation(self.plan[:42],self.m,self.ids,native_stride=4,recovery_fraction=.5,speed_multiplier=2.)
  expected=np.linalg.inv(self.m[-2])@desired.commands[[33,37,41]]
  actual=np.linalg.inv(camera['viewmats'][0,lo:hi:880].double().numpy())
  np.testing.assert_allclose(actual,expected,atol=.002,rtol=.002)
  np.testing.assert_array_equal(commands[:33],self.plan[:33]);np.testing.assert_array_equal(commands[42:],self.plan[42:])
 def test_abstention_rebuilds_relative_reference_after_prior_command(self):
  issued,camera,_=self.call();ids=np.array([0,29,40,41,44]);m=self.plan[ids]
  new,after,r=next_camera_chunk(entry,self.plan,issued,m,ids,available_frame=44,anchor_frame=41,end=12,total=18,feedback_index=2,arm='F',quality={'passed':False},camera=camera,frame_seq_length=880)
  np.testing.assert_array_equal(new,issued);self.assertFalse(r['applied'])
  self.assertEqual(r['reason'],'observation_rejected');self.assertTrue(r['native_input_changed'])
  expected=encode(entry,issued,'cpu',torch.bfloat16)
  self.assertTrue(torch.equal(after['viewmats'][:,12*880:15*880],expected['viewmats'][:,12*880:15*880]))
  self.assertTrue(torch.equal(after['viewmats'][:,:12*880],camera['viewmats'][:,:12*880]))
 def test_causal_timestamp_and_intrinsics_mismatches_fail(self):
  for kw in [dict(anchor_frame=32),dict(available_frame=29),dict(end=8)]:
   with self.assertRaises(ValueError):self.call(**kw)
  bad={k:v.clone() for k,v in self.camera.items()};bad['K']*=2
  with self.assertRaisesRegex(ValueError,'intrinsics'):self.call(camera=bad)

if __name__=='__main__':unittest.main()
