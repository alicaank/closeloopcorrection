"""First-boundary arithmetic audit; no generation, fitting or native cache edits."""
from pathlib import Path
import hashlib,importlib.util,json,sys
import numpy as np
import torch
from scipy.spatial.transform import Rotation
W=Path('/home/hj/Alican');R=Path(__file__).resolve().parent
sys.path.insert(0,str(W/'camera-closed-loop/src'))
from camera_loop.conditioning_gauge import shared_translation_gauge
native=W/'analysis/plan_base_restart_20261008/model_readiness_20261027/repos/lingbot/wan/utils/cam_utils.py'
spec=importlib.util.spec_from_file_location('native_camera_gauge_audit',native)
cam=importlib.util.module_from_spec(spec);spec.loader.exec_module(cam)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
state_path=W/'analysis/memory_generalization_20261103/cohort_resume/status.json'
state=json.loads(state_path.read_text());rows=[]
for scene in ['02267acf','c7418b7d','0238de14','924228bf']:
    key=f'lingbot_{scene}_s7';score_path=Path(state['pairs'][key]['score'])
    score=json.loads(score_path.read_text());out=Path(score['spec']['measured'])
    rp=out/'run.json';ep=out/'memory_events.jsonl';run=json.loads(rp.read_text());event=json.loads(ep.read_text().splitlines()[0])
    assert event['frame_ids']==[0,4,8,12] and score['prefix_rgb_exact'] is True
    row=dict(key=key,source_hashes={str(p):sha(p) for p in [score_path,rp,ep]},
        first_boundary_frames=event['frame_ids'],write_applied=event['applied'],reason=event['reason'],
        original_plan_normalizer=run['native_translation_normalizer'])
    if not event['applied']:
        row['interpretation']='Observer rejected: stored poses are fallback commands, not accepted measured geometry. No measured-envelope claim.'
        rows.append(row);continue
    plan=torch.tensor(run['latent_commands'],dtype=torch.float32)
    history=torch.tensor(event['stored'],dtype=torch.float32)
    raw_plan=cam.compute_relative_poses(plan,framewise=True,normalize_trans=False)
    raw_history=cam.compute_relative_poses(history,framewise=True,normalize_trans=False)
    norm=run['native_translation_normalizer']
    assert float(raw_plan[:,:3,3].norm(dim=-1).max())==norm
    current=raw_history.clone();current[:,:3,3]/=norm
    normalized_plan=raw_plan.clone();normalized_plan[:,:3,3]/=norm
    torch.testing.assert_close(normalized_plan,cam.compute_relative_poses(plan,framewise=True),rtol=0,atol=0)
    candidate=shared_translation_gauge(raw_plan.numpy(),raw_history.numpy())
    # Reduced ray grid only; origin channels are spatially constant. These
    # samples qualify encoder arithmetic, not image quality or training support.
    K=np.array(run['K']);K[0]*=8/(2*K[0,2]);K[1]*=8/(2*K[1,2])
    ks=torch.tensor([[K[0,0],K[1,1],K[0,2],K[1,2]]],dtype=torch.float32)
    old_features=cam.get_plucker_embeddings(current,ks,8,8)
    control=cam.get_plucker_embeddings(torch.tensor(candidate.commanded_control.copy(),dtype=torch.float32),ks,8,8)
    measured=cam.get_plucker_embeddings(torch.tensor(candidate.measured_history.copy(),dtype=torch.float32),ks,8,8)
    assert torch.equal(control[4:],measured[4:])
    # The released function stores ray origin + direction, despite its name.
    torch.testing.assert_close(old_features[...,0:3],current[:,:3,3,None,None].permute(0,2,3,1).expand(-1,8,8,-1),rtol=0,atol=0)
    maximum=float(measured[...,:3].norm(dim=-1).max());assert maximum<=1+2e-7
    row.update(accepted_native_steps_over_plan_max=current[:,:3,3].norm(dim=-1).tolist(),
        first_history_max_rotation_deg=float(Rotation.from_matrix(raw_history[:,:3,:3].numpy()).magnitude().max()*180/np.pi),
        current_bf16_all_finite=bool(torch.isfinite(old_features.to(torch.bfloat16)).all()),
        candidate=candidate.receipt,candidate_max_origin_norm=maximum,
        candidate_future_max_origin_norm=float(measured[4:,...,:3].norm(dim=-1).max()),
        native_future_features_exact_between_matched_controls=bool(torch.equal(control[4:],measured[4:])),
        direction_channels_unchanged_by_translation_rescale=bool(torch.equal(old_features[...,3:],measured[:4,...,3:])),
        interpretation='Conditioning magnitude evidence before first intervention. Finite BF16 values do not establish operational support; shared rescaling is not an efficacy result.')
    rows.append(row)
result=dict(scope='Four seed-7 first-boundary cases, 77 RGB frames; three accepted observations and one abstention. No future observed RGB/pose or output score selects the normalizer.',
    native_source=str(native),native_sha256=sha(native),script_sha256=sha(__file__),
    helper_sha256=sha(W/'camera-closed-loop/src/camera_loop/conditioning_gauge.py'),
    native_feature_semantics='Concatenated ray origin and direction, not cross-product Plucker moment.',
    rows=rows,gpu_tested=False,cache_replay_qualified=False,training_support_qualified=False,
    deployed=False,normalizer_is_physical_scale=False)
(R/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps([dict(key=r['key'],applied=r['write_applied'],max_step_ratio=max(r.get('accepted_native_steps_over_plan_max',[0])),future_max=r.get('candidate_future_max_origin_norm')) for r in rows],indent=2))
