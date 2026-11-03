"""Hardlink exact historical output videos; no re-encoding or path removal."""
import importlib.util,json,os,shutil,stat,time
from pathlib import Path
R=Path(__file__).resolve().parent;W=R.parents[1]
spec=importlib.util.spec_from_file_location('archive_dedup',R/'deduplicate.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
sha=helper.sha;identity=helper.identity
ALLOW=[W/'analysis'/p for p in [
 'command_gain_abc_20261101/runs','online_response_v2_resume_20261103/runs',
 'lingbot_rotation_guard_20261031/runs','runtime_repairs_20261031/lingbot/runs',
 'memory_generalization_20261103/cohort_resume/runs','memory_generalization_20261103/matrix_observer_lifetime/runs',
 'shared_controller_pilot_v2/runs','runtime_repairs_20261030/resume/runs','spatia_command_gain_20261102/runs',
 'plan_base_restart_20261008/continuous_eval_20261029/scale_20261030/measured_gate_cohort/raw',
 'plan_base_restart_20261008/continuous_eval_20261029/scale_20261030/measured_gate_cohort/validation',
 'plan_base_restart_20261008/continuous_eval_20261029/repair_20261029/raw',
 'plan_base_restart_20261008/continuous_eval_20261029/repair_20261029/validation',
 'plan_base_restart_20261008/continuous_eval_20261029/validation']]
def main():
 assert not (R/'video_deduplication.json').exists()
 source=R/'video_duplicates_verified.json';groups=[];protected=helper.protected_state()
 for g in json.loads(source.read_text())['groups']:
  paths=[Path(x['path']) for x in g['files']]
  if not all(any(p.is_relative_to(root) for root in ALLOW) for p in paths):continue
  assert all(p.suffix=='.mp4' and p.resolve()==p and stat.S_ISREG(p.lstat().st_mode) for p in paths)
  assert all(sha(p)==g['sha256'] for p in paths)
  assert len({(p.stat().st_dev,p.stat().st_mode,p.stat().st_uid,p.stat().st_gid) for p in paths})==1
  paths.sort(key=lambda p:(-p.stat().st_nlink,str(p)));keep=paths[0]
  changes=[p for p in paths[1:] if p.stat().st_ino!=keep.stat().st_ino]
  assert all(p.stat().st_nlink==1 for p in changes)
  if changes:groups.append(dict(keep=str(keep),replace=list(map(str,changes)),sha256=g['sha256'],before={str(p):identity(p) for p in paths}))
 helper.not_open([Path(p) for g in groups for p in g['before']])
 plan=dict(source_sha256=sha(source),groups=groups,excluded='Model-repository asset copies and non-identical lossless container bytes are not changed.')
 (R/'video_deduplication_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
 receipt=dict(started=time.time(),free_before=shutil.disk_usage(W).free,status='running',linked=[])
 def save():(R/'video_deduplication.json').write_text(json.dumps(receipt,indent=2)+'\n')
 save()
 for g in groups:
  keep=Path(g['keep']);assert sha(keep)==g['sha256']
  for name in g['replace']:
   p=Path(name);assert identity(p)==g['before'][name] and sha(p)==g['sha256']
   helper.not_open([p,keep]);tmp=p.with_name('.'+p.name+'.dedup-video');assert not tmp.exists()
   os.link(keep,tmp);os.replace(tmp,p)
   assert p.stat().st_ino==keep.stat().st_ino and sha(p)==g['sha256']
   receipt['linked'].append(dict(path=name,keep=str(keep),bytes=g['before'][name][2],sha256=g['sha256']));save()
 assert helper.protected_state()==protected
 assert all(sha(Path(p))==g['sha256'] for g in groups for p in g['before'])
 receipt.update(status='complete',finished=time.time(),free_after=shutil.disk_usage(W).free,
                logical_bytes_deduplicated=sum(x['bytes'] for x in receipt['linked']),
                all_paths_and_hashes_preserved=True,protected_tree_stats_unchanged=True)
 save();print(json.dumps({k:v for k,v in receipt.items() if k!='linked'}))
if __name__=='__main__':main()
