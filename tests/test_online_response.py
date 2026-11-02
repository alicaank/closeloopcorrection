import unittest
import numpy as np
from camera_loop.online_response import OnlineAffineResponse,OnlineCommandController

class OnlineResponseTests(unittest.TestCase):
    def test_history_separation_and_forward_validation(self):
        m=OnlineAffineResponse();rng=np.random.default_rng(7)
        for i in range(12):
            x=rng.uniform(-1,1,3);h=rng.uniform(-1,1,3)
            e=m.update(x,.6*x+.2*h+.1,h,frame_id=i+1,noise=.001)
            if i<5:self.assertFalse(any(e['active']))
        self.assertTrue(all(e['active']))
        for fit in m.candidate:np.testing.assert_allclose(fit,[.6,.2,.1],atol=1e-10)
    def test_constant_commands_never_identified(self):
        m=OnlineAffineResponse()
        for i in range(12):e=m.update(np.ones(3),np.ones(3)*2,np.ones(3)*.5,frame_id=i+1,noise=.01)
        self.assertFalse(any(e['active']))
        self.assertTrue(all(x['reason']=='insufficient_command_variation' for x in e['axes']))
    def test_history_only_does_not_create_gain(self):
        m=OnlineAffineResponse();rng=np.random.default_rng(3)
        for i in range(14):
            x=rng.normal(size=3);h=rng.normal(size=3)
            e=m.update(x,.5*h+.2,h,frame_id=i+1,noise=.01)
        self.assertFalse(any(e['active']))
    def test_failed_prediction_and_rejection_disable_inverse(self):
        m=OnlineAffineResponse();rng=np.random.default_rng(3)
        for i in range(10):
            x=rng.normal(size=3);h=rng.normal(size=3)
            m.update(x,.8*x+.1*h,h,frame_id=i+1,noise=.001)
        self.assertTrue(all(m.active))
        m.update(np.ones(3),np.ones(3)*10,np.ones(3),frame_id=11,noise=.001)
        self.assertFalse(any(m.active))
        m.update(None,None,None,frame_id=12,noise=0,accepted=False)
        self.assertFalse(any(m.active))
        with self.assertRaises(ValueError):m.update(None,None,None,frame_id=12,noise=0,accepted=False)
    def fixture(self,arm='C'):
        p=np.tile(np.eye(4),(41,1,1));p[:,0,3]=np.arange(41)*.03
        return OnlineCommandController(p,np.arange(41)*4,native_step=.08,arm=arm)
    def evidence(self,j):return dict(frame_id=j*4,passed=True,confidence=.9,bootstrap=False,reference_depth_unit=2.,shared_position_p90_depth_fraction=.001)
    def test_bootstrap_not_physical_scale(self):
        c=self.fixture();self.assertFalse(c.metadata()['physical_units_identified'])
        np.testing.assert_allclose(np.diff(c.commands[:,0,3]),.08)
    def test_rejection_does_not_read_pose_or_rewrite_history(self):
        c=self.fixture();before=c.commands.copy()
        _,e=c.boundary(3,None,dict(frame_id=12,passed=False),next_end=8,available_frame=12)
        np.testing.assert_array_equal(before[:4],c.commands[:4]);self.assertFalse(e['applied'])
        with self.assertRaises(ValueError):c.boundary(3,None,{},next_end=8,available_frame=12)
    def test_future_and_skipped_boundaries_rejected(self):
        c=self.fixture()
        with self.assertRaises(ValueError):c.boundary(3,np.eye(4),self.evidence(3),next_end=8,available_frame=16)
        c.boundary(3,np.eye(4),self.evidence(3),next_end=8,available_frame=12)
        with self.assertRaises(ValueError):c.boundary(8,np.eye(4),self.evidence(8),next_end=12,available_frame=32)
    def test_baseline_sham_exact_with_every_boundary_update(self):
        a,s=self.fixture('A'),self.fixture('S')
        for i,end in [(3,8),(7,12),(11,16)]:
            p=a.plan[i].copy();p[0,3]*=.6
            for c in (a,s):c.boundary(i,p,self.evidence(i),next_end=end,available_frame=i*4)
        np.testing.assert_array_equal(a.commands,s.commands);np.testing.assert_array_equal(a.commands,a.base)
        self.assertEqual(len(a.response.events),3)
    def test_pure_rotation_no_translation_calibration(self):
        p=np.tile(np.eye(4),(8,1,1));c=OnlineCommandController(p,np.arange(8)*4,native_step=.08)
        _,e=c.boundary(3,np.eye(4),self.evidence(3),next_end=8,available_frame=12)
        self.assertEqual(c.bootstrap,0);self.assertFalse(e['applied'])

    def test_changing_response_is_relearned_with_forgetting(self):
        m=OnlineAffineResponse();rng=np.random.default_rng(11)
        for i in range(70):
            x=rng.uniform(-1,1,3);h=rng.uniform(-1,1,3);gain=.8 if i<20 else .4
            m.update(x,gain*x+.1*h,h,frame_id=i+1,noise=.001)
        self.assertTrue(all(m.active))
        for fit in m.candidate:self.assertAlmostEqual(fit[0],.4,places=5)

    def test_closed_loop_synthetic_plant_updates_future_with_speed_bounds(self):
        rng=np.random.default_rng(19);p=np.tile(np.eye(4),(129,1,1))
        p[1:,:3,3]=np.cumsum(rng.uniform(-.04,.04,(128,3)),axis=0)
        c=OnlineCommandController(p,np.arange(129)*4,native_step=.08)
        position=np.zeros(3);history=np.zeros(3);last=0;applied=0
        for index in range(3,128,4):
            duration=(index-last)*4
            x=(c.commands[index,:3,3]-c.commands[last,:3,3])/duration/c.native_speed
            velocity=(.6*x+.1*history+.02)*c.reference_speed
            position+=velocity*duration;history=velocity/c.reference_speed
            m=np.eye(4);m[:3,3]=position
            old=c.commands[:index+1].copy()
            _,e=c.boundary(index,m,dict(frame_id=index*4,passed=True,confidence=1.,shared_position_p90_depth_fraction=.00001),next_end=min(index+5,129),available_frame=index*4)
            np.testing.assert_array_equal(old,c.commands[:index+1]);applied+=e['applied']
            v=np.diff(c.commands[index:index+5,:3,3],axis=0)/4
            self.assertLessEqual(np.linalg.norm(v,axis=1).max(),e['native_speed_ceiling']+1e-10)
            last=index
        self.assertGreater(applied,0)
        self.assertFalse(c.metadata()['physical_units_identified'])

class ResponseRegressionTests(unittest.TestCase):
    def test_stationary_start_cannot_amplify_later_commands(self):
        for epsilon in [0.,1e-12,1e-8,.01]:
            p=np.tile(np.eye(4),(5,1,1));p[:,0,3]=[0,epsilon,1,2,3]
            c=OnlineCommandController(p,np.arange(5)*4,native_step=.08)
            self.assertLessEqual(np.linalg.norm(np.diff(c.base[:,:3,3],axis=0),axis=1).max(),.08+1e-12)
            self.assertAlmostEqual(c.bootstrap,.08,places=10)

    def test_rejection_after_correction_preserves_slew_and_history(self):
        p=np.tile(np.eye(4),(12,1,1));p[:,0,3]=np.arange(12)*.03
        c=OnlineCommandController(p,np.arange(12)*4,native_step=.08)
        ceiling=c.config.speed_multiplier*c.native_speed
        c.commands[1,0,3]=-ceiling*4
        before=c.commands[:2].copy();previous=(before[1,:3,3]-before[0,:3,3])/4
        _,event=c.boundary(1,None,dict(frame_id=4,passed=False),next_end=4,available_frame=4)
        velocities=np.diff(c.commands[1:4,:3,3],axis=0)/4
        changes=np.diff(np.vstack([previous,velocities]),axis=0)
        self.assertLessEqual(np.linalg.norm(changes,axis=1).max(),ceiling*4/12+1e-12)
        self.assertTrue(any(event['clipped']));self.assertFalse(event['applied'])
        np.testing.assert_array_equal(before,c.commands[:2])

    def test_chunk_model_inversion_matches_chunk_prediction_at_multiple_cadences(self):
        class Identified:
            active=np.array([True,False,False])
            candidate=[np.array([1.,.2,0.]),None,None]
            def update(self,*args,**kwargs):return {'active':self.active.tolist(),'axes':[]}
        for steps in [4,8,16]:
            p=np.tile(np.eye(4),(65,1,1));p[:,0,3]=np.arange(65)*.03
            c=OnlineCommandController(p,np.arange(65)*4,native_step=.08)
            c.response=Identified();c.response_duration=steps*4;c.previous_duration=steps*4
            c.previous_velocity=np.array([.5*c.reference_speed,0,0])
            m=p[steps].copy();m[0,3]*=.5
            _,e=c.boundary(steps,m,dict(frame_id=steps*4,passed=True,confidence=1.),next_end=steps*2+1,available_frame=steps*4)
            duration=steps*4
            expected=(p[steps*2,0,3]-p[steps,0,3])/duration+(p[steps,0,3]-m[0,3])/max(64,duration)
            self.assertFalse(any(e['clipped']))
            self.assertAlmostEqual(e['predicted_chunk_reference_velocity'][0],expected,places=12)
            issued=np.array(e['issued_chunk_native_velocity'])[0]/c.native_speed
            self.assertAlmostEqual(issued+.2*.5,expected/c.reference_speed,places=12)

    def test_different_future_duration_abstains_from_inverse(self):
        class Identified:
            active=np.array([True,False,False])
            candidate=[np.array([1.,.2,0.]),None,None]
            def update(self,*args,**kwargs):return {'active':self.active.tolist(),'axes':[]}
        p=np.tile(np.eye(4),(21,1,1));p[:,0,3]=np.arange(21)*.03
        c=OnlineCommandController(p,np.arange(21)*4,native_step=.08)
        c.response=Identified();c.response_duration=16;c.previous_duration=16
        c.previous_velocity=np.array([c.reference_speed,0,0])
        _,e=c.boundary(4,p[4],dict(frame_id=16,passed=True,confidence=1.),next_end=7,available_frame=16)
        self.assertFalse(e['applied']);self.assertEqual(e['inverse_abstention_reason'],'chunk_duration_mismatch')
        np.testing.assert_array_equal(c.commands,c.base)

if __name__=='__main__':unittest.main()
