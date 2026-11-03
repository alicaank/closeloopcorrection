"""Deduplicate only exact archived RGB NPZs; keep every path and byte."""
import hashlib,json,os,shutil,stat,time
from pathlib import Path
W=Path('/home/hj/Alican');R=Path(__file__).resolve().parent
SOURCE=W/'analysis/legacy_rejoin_matched_20261103/disk_resume_duplicates.json'
ALLOWED=[W/'analysis'/p for p in ['boundary_causal_probe_20260921/results',
 'boundary_checkpoint_cross_20260922/results','zigzag_followup_v2/runs/qualification_vmem',
 'zigzag_followup_v1/runs/qualification_vmem','aligned_history_long_20260923/parent_replay',
 'online_memory_pose_20260919/rollouts','pose_servo_se3_20260918','persistent_gaussians_20260918/rollouts']]
PROTECTED=[W/p for p in ['training_runs/gaussian_recovery500_prep_20260907',
 'training_runs/gaussian_base5000_20260907','gaussverse/outputs/gate_omega/ahr_corrective_step8000_v2_depth4',
 'gaussverse/outputs/gate_omega/init_wan22_walking_v3']]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def identity(p):
 s=p.lstat();return [s.st_dev,s.st_ino,s.st_size,s.st_mode,s.st_uid,s.st_gid,s.st_mtime_ns,s.st_nlink]
def protected_state():
 return {str(p):identity(p) for root in PROTECTED for p in sorted(root.rglob('*'))}
def not_open(paths):
 names=set(map(str,paths))
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit():continue
  for fd in (proc/'fd').glob('*'):
   try:target=os.readlink(fd)
   except OSError:continue
   assert target not in names, ('open archive',target)
def main():
 assert not (R/'deduplication.json').exists()
 protected=protected_state();groups=[]
 for g in json.loads(SOURCE.read_text())['groups']:
  paths=[Path(x['path']) for x in g['files']]
  if not all(any(p.is_relative_to(root) for root in ALLOWED) for p in paths):continue
  assert all(p.name=='frames.npz' and p.resolve()==p and stat.S_ISREG(p.lstat().st_mode) for p in paths)
  if len({tuple(identity(p)[:2]) for p in paths})<2:continue
  assert all(sha(p)==g['sha256'] for p in paths)
  assert len({(p.stat().st_dev,p.stat().st_mode,p.stat().st_uid,p.stat().st_gid) for p in paths})==1
  # Reuse the inode with most existing links; replace only private allocations.
  paths.sort(key=lambda p:(-p.stat().st_nlink,str(p)))
  keep=paths[0];changes=[p for p in paths[1:] if p.stat().st_ino!=keep.stat().st_ino]
  assert all(p.stat().st_nlink==1 for p in changes)
  groups.append(dict(keep=str(keep),replace=[str(p) for p in changes],sha256=g['sha256'],
                     before={str(p):identity(p) for p in paths}))
 not_open([Path(p) for g in groups for p in g['before']])
 plan=dict(source=str(SOURCE),source_sha256=sha(SOURCE),groups=groups,protected_stat_entries=len(protected))
 (R/'deduplication_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
 receipt=dict(started=time.time(),free_before=shutil.disk_usage(W).free,linked=[],status='running')
 def save():(R/'deduplication.json').write_text(json.dumps(receipt,indent=2)+'\n')
 save()
 for g in groups:
  keep=Path(g['keep']);assert sha(keep)==g['sha256']
  for name in g['replace']:
   p=Path(name);assert identity(p)==g['before'][name] and sha(p)==g['sha256']
   tmp=p.with_name('.frames.npz.dedup-v2');assert not tmp.exists()
   os.link(keep,tmp);os.replace(tmp,p)
   assert p.stat().st_ino==keep.stat().st_ino and sha(p)==g['sha256']
   receipt['linked'].append(dict(path=name,keep=str(keep),bytes=g['before'][name][2],sha256=g['sha256']));save()
 assert protected_state()==protected
 assert all(sha(Path(p))==g['sha256'] for g in groups for p in g['before'])
 receipt.update(status='complete',finished=time.time(),free_after=shutil.disk_usage(W).free,
                logical_bytes_deduplicated=sum(r['bytes'] for r in receipt['linked']),
                all_paths_and_hashes_preserved=True,protected_tree_stats_unchanged=True)
 save();print(json.dumps({k:v for k,v in receipt.items() if k!='linked'}))
if __name__=='__main__':main()
