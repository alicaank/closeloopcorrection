import unittest
import numpy as np
from camera_loop.camera_grid import CameraGrid


class CameraGridTests(unittest.TestCase):
    def setUp(self):
        self.K=np.array([[8.,0,4],[0,8,3],[0,0,1]])
        yy,xx=np.mgrid[:7,:9]
        self.rgb=np.stack([xx*10,yy*10,xx+yy],axis=-1).astype(np.uint8)

    def test_identity_is_exact_and_does_not_mutate_inputs(self):
        original=self.K.copy();g=CameraGrid((9,7),(9,7),self.K,self.K)
        rgb,mask=g.rgb(self.rgb);d,valid=g.depth(np.full((7,9),2.5))
        np.testing.assert_array_equal(rgb,self.rgb)
        self.assertTrue(mask.all() and valid.all());np.testing.assert_array_equal(d,2.5)
        np.testing.assert_array_equal(self.K,original)
        self.K[0,0]=100
        self.assertEqual(g.source_K[0,0],8)
        k=np.array([[388.894115215,.137,4.3],[0,384.421415886,3.2],[0,0,1]])
        same=CameraGrid((9,7),(9,7),k,k)
        rgb,mask=same.rgb(self.rgb)
        self.assertTrue(mask.all());np.testing.assert_array_equal(rgb,self.rgb)

    def test_narrower_fov_samples_same_rays_for_rgb_and_depth(self):
        target=self.K.copy();target[0,0]*=2;target[1,1]*=2
        g=CameraGrid((9,7),(9,7),self.K,target);xy,valid=g.coordinates()
        rgb,_=g.rgb(self.rgb);depth,_=g.depth(2+self.rgb[:,:,0]/10)
        np.testing.assert_allclose(depth,2+xy[0])
        np.testing.assert_allclose(rgb[:,:,0],np.rint(10*xy[0]))
        self.assertTrue(valid.all())
        p=np.array([2.,1.,1.]);source=self.K@np.linalg.inv(target)@p
        np.testing.assert_allclose(np.linalg.inv(self.K)@source,np.linalg.inv(target)@p)

    def test_no_metric_depth_fallback_for_missing_support(self):
        target=self.K.copy();target[0,0]/=2;target[1,1]/=2
        g=CameraGrid((9,7),(9,7),self.K,target)
        d=np.full((7,9),3.);d[3,4]=np.nan
        depth,mask=g.depth(d)
        self.assertFalse(mask[3,4]);self.assertTrue(np.isnan(depth[3,4]))
        self.assertFalse(mask[0,0]);self.assertTrue(np.isnan(depth[0,0]))
        self.assertTrue(np.all(depth[mask]==3))

    def test_invalid_depth_neighbours_are_not_blended_into_valid_geometry(self):
        target=self.K.copy();target[0,2]+=.5
        g=CameraGrid((9,7),(9,7),self.K,target)
        d=np.full((7,9),4.);d[3,4]=0
        output,mask=g.depth(d)
        self.assertFalse(mask[3,4]);self.assertFalse(mask[3,5])
        self.assertTrue(np.isnan(output[3,4]))

    def test_depth_units_are_preserved_under_common_unit_change(self):
        target=self.K.copy();target[:2,:2]*=1.5
        g=CameraGrid((9,7),(9,7),self.K,target);d=np.arange(63).reshape(7,9)+1.
        a,ma=g.depth(d);b,mb=g.depth(d*123.4)
        np.testing.assert_array_equal(ma,mb);np.testing.assert_allclose(b/123.4,a)

    def test_invalid_contracts_fail(self):
        for shape in [(0,7),(9.,7),(True,7)]:
            with self.assertRaises(ValueError):CameraGrid(shape,(9,7),self.K,self.K)
        for k in [np.zeros((3,3)),self.K*np.nan,np.eye(4)]:
            with self.assertRaises(ValueError):CameraGrid((9,7),(9,7),k,self.K)
        g=CameraGrid((9,7),(9,7),self.K,self.K)
        with self.assertRaises(ValueError):g.depth(np.ones((8,9)))
        with self.assertRaises(ValueError):g.rgb(self.rgb.astype(float))


if __name__=='__main__':unittest.main()
