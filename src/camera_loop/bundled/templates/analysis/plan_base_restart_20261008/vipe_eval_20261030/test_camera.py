import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from camera import register,errors
from score import output_K
class Camera(unittest.TestCase):
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
