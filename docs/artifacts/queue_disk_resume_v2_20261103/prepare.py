"""Second disk-hold resume: preserve both prior attempts and frozen protocols."""
import difflib,hashlib,importlib.util,json,time
from pathlib import Path
import psutil
W=Path('/home/hj/Alican');A=Path(__file__).resolve().parent
PRIOR=W/'analysis/queue_disk_resume_20261103'
spec=importlib.util.spec_from_file_location('first_resume',PRIOR/'prepare.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
sha=old.sha;replace_once=old.replace_once
def main():
 prepared=[]
 for label,root in old.ROOTS.items():
  dst=A/label;assert not dst.exists()
  manifest=json.loads((root/'manifest.json').read_text())
  proof=json.loads((root/'cpu_validation.json').read_text())
  assert proof['passed'] and proof['manifest_sha256']==sha(root/'manifest.json')
  for path,digest in manifest['sources'].items():assert sha(path)==digest,path
  states=[];completed=[];logs={};previous=[]
  for attempt in [root,PRIOR/label]:
   state=json.loads((attempt/'status.json').read_text());states.append(state)
   assert state['phase']=='held' and not psutil.pid_exists(state['pid'])
   assert state['failures']==["RuntimeError('4 GiB launch reserve; originals retained')"] and not state['held_cases']
   assert state['manifest_sha256']==sha(root/'manifest.json')
   if attempt!=root:
    p=json.loads((attempt/'resume.json').read_text())
    assert state['resume_provenance']==p and p['original_status_sha256']==sha(root/'status.json')
    assert set(p['completed_jobs_skipped'])==set(completed)
    assert p['original_runner_sha256']==sha(root/'run_queue.py')
    assert all(sha(name)==digest for name,digest in p['completed_log_sha256'].items())
    previous.append(dict(root=str(attempt),status_sha256=sha(attempt/'status.json'),
                         runner_sha256=sha(attempt/'run_queue.py'),resume_sha256=sha(attempt/'resume.json')))
   for job in manifest['jobs']:
    rec=state['jobs'].get(job['name'])
    if rec is None:continue
    assert rec['state']=='completed' and rec['returncode']==0 and rec['stop_reason'] is None
    assert rec['argv']==job['argv'] and job['name'] not in completed
    completed.append(job['name']);path=attempt/'logs'/f"{job['name']}.log";logs[str(path)]=sha(path)
   assert set(state['jobs'])<=set(completed)
  source=(root/'run_queue.py').read_text()
  modified=replace_once(source,'R=Path(__file__).resolve().parent',f'R=Path({str(root)!r});A=Path(__file__).resolve().parent')
  for name in ['status.json','status.tmp']:modified=modified.replace(f"R/'{name}'",f"A/'{name}'")
  modified=modified.replace("R/'logs'","A/'logs'")
  check=(f"assert sha(R/'status.json') == {sha(root/'status.json')!r}\n"
         f"        assert sha(R/'run_queue.py') == {sha(root/'run_queue.py')!r}\n"
         "        assert sha(A/'resume.json') == RESUME_SHA\n        ")
  for p in previous:
   for filename,key in [('status.json','status_sha256'),('run_queue.py','runner_sha256'),('resume.json','resume_sha256')]:
    check+=f"assert sha(Path({str(Path(p['root'])/filename)!r})) == {p[key]!r}\n        "
  modified=replace_once(modified,'def verify():\n        ','def verify():\n        '+check)
  modified=replace_once(modified,"for entry in m['jobs']:\n",f"for entry in m['jobs']:\n            if entry['name'] in {completed!r}:continue\n")
  modified=replace_once(modified,'def save():\n',"state['resume_provenance']=json.loads((A/'resume.json').read_text())\n    def save():\n")
  dst.mkdir();(dst/'logs').mkdir()
  record=dict(label=label,original_root=str(root),original_pid=states[0]['pid'],
              original_status_sha256=sha(root/'status.json'),original_manifest_sha256=sha(root/'manifest.json'),
              original_runner_sha256=sha(root/'run_queue.py'),previous_attempts=previous,
              completed_jobs_skipped=completed,completed_log_sha256=logs,
              remaining_jobs=[j['name'] for j in manifest['jobs'] if j['name'] not in completed],
              unchanged=['job argv','runtime/source paths','environment','4 GiB launch reserve','2 GiB disk/host runtime reserve',
                         'five GPU locks','model protocol','horizons','seeds','output paths'],
              changed=['separate attempt status/log paths','skip successful jobs from both prior attempts'],prepared=time.time())
  (dst/'resume.json').write_text(json.dumps(record,indent=2)+'\n')
  modified=replace_once(modified,'def main():\n',f"RESUME_SHA={sha(dst/'resume.json')!r}\ndef main():\n")
  compile(modified,str(dst/'run_queue.py'),'exec');(dst/'run_queue.py').write_text(modified)
  (dst/'source_change.diff').write_text(''.join(difflib.unified_diff(source.splitlines(True),modified.splitlines(True),fromfile=str(root/'run_queue.py'),tofile=str(dst/'run_queue.py'))))
  prepared.append(dict(label=label,root=str(dst),runner_sha256=sha(dst/'run_queue.py'),resume_sha256=sha(dst/'resume.json'),
                       skipped=len(completed),remaining=len(record['remaining_jobs']),frozen_source_files_verified=len(manifest['sources'])))
 (A/'preparation.json').write_text(json.dumps(dict(passed=True,attempts=prepared),indent=2)+'\n')
 print(json.dumps(prepared,indent=2))
if __name__=='__main__':main()
