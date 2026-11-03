"""CPU-only intended-command and repeated-observer audit; no causal gain fit."""
import hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
W=Path('/home/hj/Alican');O=Path(__file__).resolve().parent;hashes={}
def track(p):
 p=Path(p);hashes[str(p.relative_to(W))]=hashlib.sha256(p.read_bytes()).hexdigest();return p
def read(p):return json.loads(track(p).read_text())
def flips(v,nominal,frames):
 norm=np.linalg.norm(v,axis=1);floor=max(1e-8,.01*float(np.median(np.linalg.norm(nominal,axis=1))))
 dot=np.sum(v[1:]*v[:-1],axis=1)
 hit=(dot<0)&(norm[1:]>floor)&(norm[:-1]>floor)
 return dict(count=int(hit.sum()),frames=frames[1:][hit].tolist(),minimum_step=floor,
             angles_deg=np.degrees(np.arccos(np.clip(dot[hit]/(norm[1:][hit]*norm[:-1][hit]),-1,1))).tolist())
def main():
 for source in ['lingbot_pair.py','common.py']:
  track(W/'analysis/plan_base_restart_20261008/new_models_loop_20261028'/source)
 track(W/'analysis/plan_base_restart_20261008/spatia_plugins/spatia_da3_worker.py')
 original=read(W/'analysis/legacy_rejoin_failure_audit_20261103/audit.json')
 visual=read(W/'analysis/legacy_rejoin_failure_audit_20261103/visual_review.json')
 rows=[];traces={}
 for pair in original['lingbot_native_replay']:
  out=dict(id=pair['id'],arms={})
  for arm,old in pair['arms'].items():
   d=read(old['source']);ids=np.asarray(d['camera_frame_ids']);plan=np.asarray(d['plan'])[ids]
   cmd=np.asarray(d['latent_commands']);nominal=np.diff(plan[:,:3,3],axis=0);steps=nominal.copy()
   events=[]
   for e in d['events']:
    lo=(e['chunk']+1)*4;hi=lo+4;m=np.asarray(e['measured'])
    start=m if arm=='correction' else cmd[lo-1]
    chain=np.concatenate([start[None],cmd[lo:hi]])
    steps[lo-1:hi-1]=np.diff(chain[:,:3,3],axis=0)
    events.append(dict(frame=e['endpoint_frame'],finite_pose=bool(np.isfinite(m).all()),
                       worker_ok=e['measurement']['ok'],valid_poses=e['measurement']['n_valid'],
                       correction_applied=e['correction_applied']))
   keep=ids[1:]<=160;v=steps[keep];n=nominal[keep];frames=ids[1:][keep]
   result=flips(v,n,frames);target=flips(n,n,frames)
   out['arms'][arm]=dict(reversals=result,target_reversals=target,events=events,
                         native_translation_normalizer=d['native_translation_normalizer'],
                         max_step_in_plan_normalizer=float(np.linalg.norm(v,axis=1).max()/d['native_translation_normalizer']))
   tangent=n/np.linalg.norm(n,axis=1)[:,None]
   traces[pair['id'],arm]=(frames,np.sum(v*tangent,axis=1)/d['native_translation_normalizer'])
   assert result['count']==old['direction_reversals']
  assert out['arms']['baseline']['reversals']['count']==0
  assert out['arms']['correction']['target_reversals']['count']==0
  rows.append(out)
 assert len(rows)==16
 onsets={'lingbot_re10k160_d3f042a2_s107':48,'lingbot_re10k160_e43058cb_s7':24,'lingbot_re10k160_74310829_s7':16}
 continuation=[]
 for row in rows:
  if row['id'] not in onsets:continue
  onset=onsets[row['id']];events=[e for e in row['arms']['correction']['events'] if e['frame']>=onset]
  assert all(e['finite_pose'] and e['worker_ok'] and e['valid_poses']==1 and e['correction_applied'] for e in events)
  continuation.append(dict(id=row['id'],first_visible_degradation_sample=onset,
                           subsequent_accepted_feedback_frames=[e['frame'] for e in events],
                           limitation='Sampled visual onset, not a frame-exact automatic detection. Finite poses do not validate geometry.'))
 score=read(W/'analysis/spatia_command_gain_20261102/scores/spatia_0238de14_s7.json')
 with np.load(track(Path(score['batch'])/'trajectories/0238de14.npz')) as z:
  target=np.linalg.inv(z['w2c_overrun' if 'w2c_overrun' in z else 'w2c'].astype(float))[:481]
 axis=np.linalg.svd(target[:,:3,3]-target[:,:3,3].mean(axis=0),full_matrices=False)[2][0]
 if np.dot(axis,target[240,:3,3]-target[0,:3,3])<0:axis=-axis
 oldsteps=read(W/'analysis/control_mechanism_audit_20261103/repeated_boundary_steps.json');repeats=[];progress=[]
 for r in oldsteps:
  with np.load(track(W/r['source'])) as z:poses=z['c2w'];ids=z['frame_ids']
  assert np.array_equal(ids,np.arange(481))
  projected=poses[:,:3,3]@axis
  delta={str(f):float(projected[f+8]-projected[f]) for f in [264,336,408]}
  assert all(abs(delta[k]-r['steps'][k])<1e-10 for k in delta)
  repeats.append(dict(source=r['source'],projected_steps=delta));progress.append(projected)
 command_checks=[];saved_commands={}
 for arm in ['A','B','C']:
  run=read(W/f'analysis/spatia_command_gain_20261102/runs/spatia_0238de14_s7/{arm}/run.json')
  commands=np.asarray(run.get('commands',run.get('commanded')));plan=np.asarray(run.get('plan',run.get('planned')))
  ids=np.asarray(run.get('native_camera_frame_ids',run.get('pose_frame_ids',range(len(commands)))))
  if len(commands)!=len(ids):commands=commands[ids];plan=plan[ids]
  keep=ids<=480;commands=commands[keep];plan=plan[keep];ids=ids[keep]
  check=flips(np.diff(commands[:,:3,3],axis=0),np.diff(plan[:,:3,3],axis=0),ids[1:])
  saved_commands[arm]=commands
  assert check['count']==0;command_checks.append(dict(arm=arm,check=check))
 assert np.array_equal(saved_commands['B'],saved_commands['C'])
 result=dict(scope='Historical LingBot 16 pairs, frames 0–160; Spatia 481-frame pilot and four observer evaluations. CPU audit only.',
  definition='Adjacent intended translation vectors have negative dot product; each exceeds 1% median planned step, with minimum 1e-8. Common anchor/world basis, measured start at each corrected LingBot chunk. Threshold is a diagnostic convention, not observer uncertainty.',
  lingbot=rows,lingbot_feedback_after_visual_degradation=continuation,
  spatia=dict(command_checks=command_checks,B_C_commands_exact=True,repeated_observer_steps=repeats,projection_axis=axis.tolist(),
              limitation='Backward projected progress and RGB discontinuities do not distinguish physical camera reversal from inconsistent generated geometry.'),
  dreamx_correction=dict(runs=len(original['dreamx_native_transition_replay']),
    corrected_reversal_counts=[r['measured_anchor_step_reversals'] for r in original['dreamx_native_transition_replay']],
    limitation='Existing verified replay, not fresh generation or physical-motion measurement.'),
  input_sha256=hashes,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 (O/'audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 fig,axs=plt.subplots(3,1,figsize=(11,10),layout='constrained')
 for ax,seed in zip(axs[:2],[7,107]):
  key=f'lingbot_re10k160_d3f042a2_s{seed}'
  for arm,color in [('baseline','#6b7280'),('correction','#bd332f')]:
   x,y=traces[key,arm];ax.plot(x,y,'o-',ms=3,label=arm,color=color)
  ax.axhline(0,color='black',lw=.8);ax.set_title(f'LingBot d3f042a2 / seed {seed}: intended commands, not observed camera motion')
  ax.set_ylabel('Step along planned direction\n/ maximum planned step');ax.set_xlabel('RGB frame');ax.grid(alpha=.2);ax.legend()
  if seed==107:ax.axvspan(48,160,color='grey',alpha=.14,label='After first degraded sample');ax.text(52,ax.get_ylim()[1]*.75,'Degraded RGB from sample 48',fontsize=9)
 ax=axs[2]
 for i,trace in enumerate(progress):ax.plot(np.arange(481),trace,label=['B evaluation','C evaluation (same RGB)','Repeat 2','Repeat 3'][i],alpha=.85)
 ax.plot(np.arange(481),target[:,:3,3]@axis,'k--',label='Reference');ax.set_xlim(320,424)
 for frame in [336,408]:ax.axvspan(frame,frame+8,color='grey',alpha=.16)
 ax.set_title('Spatia: repeated observer estimates of the same RGB, around continuation seams')
 ax.set_xlabel('RGB frame');ax.set_ylabel('Projected position\n(anchor-depth units)');ax.grid(alpha=.2);ax.legend(ncol=3,fontsize=8)
 fig.savefig(O/'commands_and_observer.png',dpi=140);plt.close(fig)
 print(json.dumps(dict(lingbot_pairs=16,reversal_range=[min(r['arms']['correction']['reversals']['count'] for r in rows),max(r['arms']['correction']['reversals']['count'] for r in rows)],post_degradation_feedback=continuation,spatia=command_checks,source_hashes=len(hashes))))
if __name__=='__main__':main()
