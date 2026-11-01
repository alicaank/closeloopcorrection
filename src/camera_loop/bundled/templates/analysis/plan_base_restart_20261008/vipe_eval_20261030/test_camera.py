import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from camera import register,errors,path_metrics
from score import output_K
class Camera(unittest.TestCase):
 def test_saturated_disparity_cannot_establish_a_scale(self):
  raw=np.eye(4)[None];reference=np.ones((32,32))
  for saturated in [.1,1000.]:
   with self.subTest(depth=saturated),self.assertRaises(ValueError):
    register(raw,np.full((32,32),saturated),reference)
 def test_saturation_is_excluded_before_scale_fitting(self):
  raw=np.eye(4)[None];native=np.ones((32,32))*2;native[:8]=1000
  _,quality=register(raw,native,np.ones((32,32))*6)
  self.assertEqual(quality['scale'],3);self.assertEqual(quality['saturated_pixels'],256)
  self.assertEqual(quality['valid_pixels'],768)
 def test_rotation_only_target_has_no_translation_ratio(self):
  target=np.repeat(np.eye(4)[None],8,axis=0)
  target[:,:3,:3]=Rotation.from_euler('y',np.linspace(0,.4,8)[:,None]).as_matrix()
  measured=target.copy();measured[:,0,3]=np.linspace(0,.3,8)
  metrics=path_metrics(measured,target)
  self.assertIsNone(metrics['path_length_ratio']);self.assertFalse(metrics['path_length_ratio_valid'])
  self.assertEqual(metrics['path_length_ratio_reason'],'insufficient_target_translation')
  self.assertAlmostEqual(metrics['measured_path_length'],.3)
 def test_translation_ratio_and_tiny_target(self):
  target=np.repeat(np.eye(4)[None],8,axis=0);target[:,0,3]=np.linspace(0,.3,8)
  measured=target.copy();measured[:,:3,3]*=2
  self.assertAlmostEqual(path_metrics(measured,target)['path_length_ratio'],2)
  target[:,:3,3]*=1e-10
  self.assertIsNone(path_metrics(measured,target)['path_length_ratio'])
 def test_path_metrics_reject_incomplete_or_nonfinite_trajectories(self):
  target=np.repeat(np.eye(4)[None],8,axis=0)
  with self.assertRaises(ValueError):path_metrics(target[:-1],target)
  measured=target.copy();measured[1,0,3]=np.nan
  with self.assertRaises(ValueError):path_metrics(measured,target)
 def test_anchor_registration_preserves_known_position_and_rotation(self):
  p=np.repeat(np.eye(4)[None],8,axis=0);p[:,0,3]=np.linspace(0,1,8);p[:,:3,:3]=Rotation.from_euler('y',np.linspace(0,.4,8)[:,None]).as_matrix()
  native=p.copy();native[:,:3,3]/=3;g=np.eye(4);g[:3,:3]=Rotation.from_euler('xyz',[.2,.3,.1]).as_matrix();g[:3,3]=[2,3,4];native=g@native
  out,q=register(native,np.ones((32,32))*2,np.ones((32,32))*6)
  np.testing.assert_allclose(out,p,atol=1e-12);self.assertEqual(q['scale'],3)
  pe,re=errors(out,p);self.assertLess(pe.max(),1e-12);self.assertLess(re.max(),1e-10)
 def test_motion_error_not_fitted_away(self):
  p=np.repeat(np.eye(4)[None],8,axis=0);p[:,0,3]=np.linspace(0,.3,8)
  out,_=register(p,np.ones((32,32)),np.ones((32,32)))
  self.assertAlmostEqual(out[-1,0,3],.3)
 def test_rejects_poor_geometry_and_invalid_rotations(self):
  p=np.eye(4)[None];ref=np.ones((32,32));bad=ref.copy();bad[:30]=np.nan
  with self.assertRaises(ValueError):register(p,ref,bad)
  with self.assertRaises(ValueError):register(p,ref,ref[:16])
  p[0,0,0]=2
  with self.assertRaises(ValueError):register(p,ref,ref)
 def test_native_crop_intrinsics(self):
  k=np.array([[400.,0,432],[0,400,240],[0,0,1]])
  np.testing.assert_allclose(output_K('worldwarp',k,(864,480),(720,480))[:2,2],[360,240])
  np.testing.assert_allclose(output_K('lingbot',k,(864,480),(832,464))[:2,2],[416,232])
  np.testing.assert_allclose(output_K('spatia',k,(864,480),(1248,704))[:2,2],[624,352])
if __name__=='__main__':unittest.main()
