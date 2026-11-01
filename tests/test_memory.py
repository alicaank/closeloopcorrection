import unittest
import numpy as np
from camera_loop.memory import stage_memory_write, bridge_future


class MemoryContractTests(unittest.TestCase):
    def setUp(self):
        self.old=np.repeat(np.eye(4,dtype=np.float32)[None],3,axis=0)
        self.old[:,0,3]=[0,1,2]

    def stage(self,new,**kw):
        return stage_memory_write([0,4,8],self.old,new,available_last_frame=8,quality=kw.get('quality',{'passed':True}))

    def test_identity_preserves_exact_dtype_bytes_and_ownership(self):
        write=self.stage(self.old.astype(float))
        self.assertFalse(write.applied)
        self.assertEqual(write.reason,'identity')
        self.assertEqual(write.poses.dtype,self.old.dtype)
        self.assertEqual(write.poses.tobytes(),self.old.tobytes())
        self.assertFalse(np.shares_memory(write.poses,self.old))

    def test_rejection_keeps_original_memory(self):
        new=self.old.copy();new[1:,0,3]*=2
        for q in ({},{'passed':False}):
            write=self.stage(new,quality=q)
            self.assertFalse(write.applied)
            np.testing.assert_array_equal(write.poses,self.old)

    def test_only_causal_unique_sorted_integer_frame_ids(self):
        for ids in ([0,4,9],[0,4,4],[0,8,4],[-1,4,8],[0.,4.,8.]):
            with self.assertRaises(ValueError):
                stage_memory_write(ids,self.old,self.old,available_last_frame=8,quality={'passed':True})

    def test_bad_geometry_and_changed_anchor_are_rejected(self):
        for entry,value in (((1,0,0),2),((1,3,0),1),((1,0,3),np.nan),((0,0,3),1)):
            new=self.old.copy();new[entry]=value
            self.assertFalse(self.stage(new).applied)
        new=self.old.copy();new[1,:3,:3]*=-1
        self.assertFalse(self.stage(new).applied)

    def test_valid_writes_do_not_mutate_input(self):
        new=self.old.copy();new[1:,0,3]=[.8,1.5]
        before=self.old.copy();write=self.stage(new)
        self.assertTrue(write.applied)
        np.testing.assert_array_equal(self.old,before)
        np.testing.assert_array_equal(write.poses,new)

    def test_bridge_preserves_relative_command_and_identity(self):
        measured=np.eye(4);measured[1,3]=3
        stored=np.eye(4);stored[:2,:2]=[[0,-1],[1,0]];stored[0,3]=2
        command=bridge_future(self.old,measured,stored)
        np.testing.assert_allclose(np.linalg.inv(stored)@command,np.linalg.inv(measured)@self.old)
        same=bridge_future(self.old,stored,stored)
        self.assertEqual(same.tobytes(),self.old.tobytes())


if __name__=='__main__':unittest.main()
