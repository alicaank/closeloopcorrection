"""Check real resume accounting and reject corrupted provenance in memory."""
import ast,hashlib,json
from pathlib import Path
W=Path('/home/hj/Alican');R=W/'analysis/memory_generalization_20261103'
script=R/'coverage_snapshot.py';tree=ast.parse(script.read_text())
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='effective_state')
root=R/'matrix_observer_lifetime';attempt=W/'analysis/queue_disk_resume_v2_20261103/matrix_memory'
snapshots={}
for base in [root,W/'analysis/queue_disk_resume_20261103/matrix_memory',attempt]:
 for name in ['status.json','resume.json']:
  p=base/name
  if p.exists():snapshots[str(p)]=p.read_bytes()
def run(mutate=None):
 captured={}
 def read(p):
  raw=snapshots.get(str(p),None)
  if raw is None:raw=p.read_bytes()
  captured[str(p)]=hashlib.sha256(raw).hexdigest();value=json.loads(raw)
  if mutate:mutate(p,value)
  return value
 def sha(p):return hashlib.sha256(snapshots[str(p)] if str(p) in snapshots else p.read_bytes()).hexdigest()
 scope=dict(R=R,Path=Path,read=read,sha=sha,captured=captured)
 exec(compile(ast.Module(body=[fn],type_ignores=[]),str(script),'exec'),scope)
 return scope['effective_state'](root,'matrix_memory')
baseline=run()
assert len(baseline['previous_queue_attempts'])==2
assert baseline['resume_status']==str(attempt/'status.json')
assert len(baseline['jobs'])==sum(len(json.loads(raw)['jobs']) for path,raw in snapshots.items() if path.endswith('/status.json'))
oldjob=next(iter(json.loads(snapshots[str(root/'status.json')])['jobs']))
def duplicate(p,d):
 if p==attempt/'status.json':d['jobs'][oldjob]={'state':'completed','returncode':0}
def altered_skip(p,d):
 if p==attempt/'resume.json':d['completed_jobs_skipped']=[]
 if p==attempt/'status.json':d['resume_provenance']['completed_jobs_skipped']=[]
def altered_prior_hash(p,d):
 if p==attempt/'resume.json':d['previous_attempts'][0]['status_sha256']='0'*64
 if p==attempt/'status.json':d['resume_provenance']['previous_attempts'][0]['status_sha256']='0'*64
tests=[]
for case in [duplicate,altered_skip,altered_prior_hash]:
 try:run(case)
 except AssertionError:tests.append(case.__name__)
 else:raise AssertionError('Corrupt provenance accepted: '+case.__name__)
result=dict(passed=True,job_records_joined=len(baseline['jobs']),retained_previous_attempts=2,
            corruptions_rejected=tests,script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
            input_sha256={p:hashlib.sha256(raw).hexdigest() for p,raw in snapshots.items()})
(Path(__file__).parent/'coverage_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='input_sha256'}))
