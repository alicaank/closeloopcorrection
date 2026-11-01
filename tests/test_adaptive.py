import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from camera_loop import CameraAlignment
from camera_loop.adaptive import AdaptiveController, DirectionalResponse, RecoveryConfig, vector, transform
from camera_loop.evidence import select_window, assess_window


def plan(n=40):
    p=np.repeat(np.eye(4)[None],n,axis=0)
    p[:,0,3]=np.arange(n)*.02
    p[:,:3,:3]=Rotation.from_euler('z',(np.arange(n)*.01)[:,None]).as_matrix()
    return p


class Adaptive(unittest.TestCase):
    def make(self, arm='B', direction='c2w', axes='opencv', scale=1):
        p=plan()
        return AdaptiveController(p,np.arange(len(p))*4,CameraAlignment(scale,direction,axes,evidence='test'),arm=arm)

    def evidence(self, frame, passed=True):
        return dict(frame_id=frame,passed=passed,confidence=.8 if passed else 0.,previous_chunk_size=4)

    def test_native_speed_and_transported_acceleration_with_large_error(self):
        c=self.make();original=c.commands.copy();m=c.plan[3].copy();m[:3,3]+=[3,-2,1]
        native,event=c.next_chunk(4,8,m,self.evidence(12))
        previous=vector(c.issued[3])/4;previous_R=c.issued[3,:3,:3]
        for command in native:
            transported=np.r_[previous_R.T@previous[:3],previous_R.T@previous[3:]]
            velocity=vector(command)/4
            for k,sl in enumerate([slice(0,3),slice(3,6)]):
                self.assertLessEqual(np.linalg.norm(velocity[sl]),c.limits[k]+1e-10)
                self.assertLessEqual(np.linalg.norm(velocity[sl]-transported[sl]),c.limits[k]*4/c.config.acceleration_frames+1e-10)
            previous,previous_R=velocity,command[:3,:3]
        np.testing.assert_array_equal(c.commands[:4],original[:4])
        self.assertGreater(event['endpoint_tracking_error'],1)
        self.assertGreater(min(event['return_horizon_frames']),16)
        self.assertLess(event['bridge_max_error'],1e-12)

    def test_quality_rejection_does_not_read_pose_or_learn_response(self):
        c=self.make('C');old=c.response.gains().copy()
        output,event=c.next_chunk(4,8,None,self.evidence(12,False))
        self.assertFalse(event['accepted']);self.assertFalse(any(event['response_update_axes']))
        self.assertTrue(np.isfinite(output).all());np.testing.assert_array_equal(c.response.gains(),old)
        with self.assertRaises(ValueError):c.next_chunk(4,8,None,self.evidence(12,False))

    def test_rejects_stale_and_future_observations(self):
        for frame in [8,16]:
            with self.subTest(frame=frame),self.assertRaises(ValueError):
                self.make().next_chunk(4,8,np.eye(4),self.evidence(frame))

    def test_encoding_decoding_and_history_bridge_all_conventions(self):
        reference=None
        for direction in ['c2w','w2c']:
            for axes in ['opencv','opengl']:
                for scale in [.1,1,10]:
                    c=self.make(direction=direction,axes=axes,scale=scale)
                    m=c.plan[3].copy();m[1,3]+=.1
                    native,event=c.next_chunk(4,8,m,self.evidence(12))
                    self.assertLess(event['bridge_max_error'],1e-11)
                    absolute=c._native(c.commands[3])
                    for j,command in enumerate(native,4):
                        absolute=absolute@(command if direction=='c2w' else np.linalg.inv(command))
                        np.testing.assert_allclose(c._reference(absolute),c.commands[j],atol=1e-11)
                    if reference is None:reference=c.commands[4:8].copy()
                    np.testing.assert_allclose(c.commands[4:8],reference,atol=1e-10)

    def test_B_C_identical_before_response_evidence(self):
        b,c=self.make('B'),self.make('C')
        for lo in [4,8]:
            m=b.plan[lo-1].copy();m[0,3]*=.7
            ob,_=b.next_chunk(lo,lo+4,m,self.evidence((lo-1)*4))
            oc,_=c.next_chunk(lo,lo+4,m,self.evidence((lo-1)*4))
            np.testing.assert_array_equal(ob,oc)

    def test_response_requires_excitation_and_directional_agreement(self):
        response=DirectionalResponse(RecoveryConfig())
        x=transform(np.array([1.,0,0,0,0,0]))
        wrong=transform(np.array([0.,1,0,0,0,0]))
        for _ in range(8):self.assertFalse(any(response.update(x,wrong,np.ones(6),1.)))
        np.testing.assert_array_equal(response.gains(),np.ones(6))
        y=transform(np.array([.8,0,0,0,0,0]))
        for _ in range(8):response.update(x,y,np.ones(6),1.)
        self.assertLess(response.gains()[0],1.)
        np.testing.assert_array_equal(response.gains()[1:],np.ones(5))


class Evidence(unittest.TestCase):
    def fixture(self):
        ids=np.array([0,3,7,11,15]);cameras=np.repeat(np.eye(4)[None],len(ids),axis=0)
        depth=np.ones((len(ids),96,96))*2
        yy,xx=np.mgrid[:96,:96]
        intensity=(128+60*np.sin(xx*.2)+50*np.cos(yy*.3)).astype(np.uint8)
        images=np.repeat(np.repeat(intensity[None,:,:,None],3,axis=3),len(ids),axis=0)
        K=np.repeat(np.array([[80.,0,48],[0,80,48],[0,0,1]])[None],len(ids),axis=0)
        return ids,cameras,depth,depth[0],K,images

    def test_window_is_causal_and_retains_shared_frames(self):
        ids=select_window(20,[0,3,7,11,15])
        self.assertTrue(set([3,7,11,15]).issubset(ids));self.assertLessEqual(len(ids),16)
        self.assertEqual(ids[0],0);self.assertEqual(ids[-1],20)

    def test_support_and_repeated_pose_consistency(self):
        args=self.fixture();registered,first=assess_window(*args)
        self.assertTrue(first['passed'],first)
        previous=dict(zip(args[0],registered))
        _,second=assess_window(*args,previous=previous)
        self.assertTrue(second['passed'],second);self.assertGreater(second['confidence'],first['confidence'])

    def test_shared_pose_disagreement_rejects_valid_SE3(self):
        args=self.fixture();previous=dict(zip(args[0],args[1].copy()))
        for i in list(previous):
            if i:previous[i][0,3]=1
        _,q=assess_window(*args,previous=previous)
        self.assertFalse(q['passed']);self.assertIn('shared_position_disagreement',q['reasons'])

    def test_untextured_images_do_not_certify_geometry(self):
        args=list(self.fixture());args[-1]=np.full_like(args[-1],128)
        _,q=assess_window(*args)
        self.assertFalse(q['passed']);self.assertIn('insufficient_projected_image_support',q['reasons'])

    def test_missing_overlap_cannot_be_substituted_by_target_agreement(self):
        _,q=assess_window(*self.fixture(),previous={0:np.eye(4)})
        self.assertFalse(q['passed']);self.assertIn('insufficient_shared_frames',q['reasons'])


class HostBoundary(unittest.TestCase):
    def test_measured_memory_bridge_does_not_rewrite_issued_history(self):
        for direction in ['c2w','w2c']:
            for axes in ['opencv','opengl']:
                p=plan();c=AdaptiveController(p,np.arange(len(p)),CameraAlignment(7,direction,axes,evidence='host test'))
                old=c.commands.copy();old_issued=c.issued.copy()
                m=p[3].copy();m[:3,3]+=[.3,-.2,.1]
                stored=m.copy();stored[1,3]+=.04  # actual memory writer may use a different observer
                rel,event=c.next_chunk(4,8,m,dict(frame_id=3,passed=True,confidence=.8),stored_boundary=stored)
                running=c._native(stored)
                for j,x in enumerate(rel,4):
                    running=running@(x if direction=='c2w' else np.linalg.inv(x))
                    np.testing.assert_allclose(c._reference(running),c.commands[j],atol=1e-12)
                np.testing.assert_array_equal(c.commands[:4],old[:4])
                np.testing.assert_array_equal(c.issued[:4],old_issued[:4])
                self.assertEqual(event['boundary_source'],'host_rebuilt_memory')
                self.assertLess(event['bridge_max_error'],1e-12)

    def test_rejected_feedback_still_uses_actual_memory_labels(self):
        p=plan();c=AdaptiveController(p,np.arange(len(p)),CameraAlignment(1,evidence='host test'))
        stored=p[3].copy();stored[1,3]+=2
        rel,event=c.next_chunk(4,8,None,dict(frame_id=3,passed=False),stored_boundary=stored)
        np.testing.assert_allclose(np.linalg.inv(stored)@c.commands[4],rel[0],atol=1e-12)
        self.assertFalse(event['accepted']);self.assertIsNone(c.last_observation_index)

    def test_dreamx_delayed_anchor_only_updates_ungenerated_camera_tokens(self):
        ids=np.r_[0,1+4*np.arange(11)];p=plan(len(ids))
        c=AdaptiveController(p,ids,CameraAlignment(1,evidence='native camera sampling'),initial_count=3)
        before=c.commands.copy()
        with self.assertRaises(ValueError):c.next_chunk(3,6,p[2],dict(frame_id=5,passed=True,confidence=.5),available_last_frame=9)
        np.testing.assert_array_equal(c.commands,before)
        _,e=c.next_chunk(3,6,p[2],dict(frame_id=5,passed=True,confidence=.5),available_last_frame=8)
        self.assertEqual(e['observation_delay_frames'],3)
        self.assertEqual(e['future_frame_ids'],[9,13,17])


if __name__=='__main__':unittest.main()
