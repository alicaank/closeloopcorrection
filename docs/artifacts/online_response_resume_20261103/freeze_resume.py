"""Rebind the unstarted environment-fixed pilot, preserving its algorithm."""
from pathlib import Path
import ast,hashlib,importlib.util,json,os,subprocess,time,types
R=Path(__file__).resolve().parent;W=R.parents[1];OLD=W/'analysis/online_response_v2_retry_20261102'
PY=W/'baselines/venv_vmem/bin/python'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not (R/'manifest.json').exists()
parent=json.loads((OLD/'manifest.json').read_text())
for p,h in parent['hashes'].items():assert sha(p)==h,p
m=json.loads(json.dumps(parent).replace(str(OLD),str(R)))
# Path relocation only. Do not silently export newer controller code.
for p in (OLD/'frozen').rglob('*.py'):
    q=R/p.relative_to(OLD)
    assert q.read_text()==p.read_text().replace(str(OLD),str(R)),p
files=[*R.glob('*.py'),*list((R/'tests').glob('*.py'))]
m.update(created=time.time(),resumes_manifest_sha256=sha(OLD/'manifest.json'),
         resume_scope='Same original 10-generation, 4-score online-response pilot; new output paths and corrected per-model rank environment. No controller/scoring/scene/horizon change.')
m['hashes']={p:sha(p) for p in m['hashes']}
m['hashes'].update({str(p):sha(p) for p in files})
m['hashes'][str(OLD/'manifest.json')]=sha(OLD/'manifest.json')
(R/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
spec=importlib.util.spec_from_file_location('pilot_queue',R/'run_queue.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
env=module.environment();env['CUDA_VISIBLE_DEVICES']=''
assert not any(k in env for k in ['RANK','LOCAL_RANK','WORLD_SIZE'])
checks=[]
def check(name,argv):
    start=time.time();log=R/'logs'/f'{name}.log'
    with log.open('x') as f:subprocess.run(argv,env=env,cwd=W,stdout=f,stderr=subprocess.STDOUT,check=True)
    checks.append(dict(name=name,argv=argv,passed=True,seconds=time.time()-start,log_sha256=sha(log)))
for model in ['astronex','worldplay']:
    group=next(g for g in m['groups'] if g['model']==model);argv=group['jobs'][0]['argv'][:2]+['--help'];check(model+'_help',argv)
check('score_help',[str(PY),str(R/'score.py'),'--help'])
check('frozen_regressions',[str(PY),'-m','unittest','discover','-s',str(R/'tests'),'-v'])
check('import_binding',[str(PY),'-c',"import camera_loop; from pathlib import Path; assert Path(camera_loop.__file__).is_relative_to(Path("+repr(str(R/'frozen'))+")); print(camera_loop.__file__)"])
# Execute the native distributed decision using the actual stripped environment.
native=W/'sota_repos/astronex_world/inference/sample.py';tree=ast.parse(native.read_text())
branch=next(n for n in tree.body if isinstance(n,ast.If) and ast.unparse(n.test)=='args.sp_size > 1')
scope=dict(args=types.SimpleNamespace(sp_size=1),os=types.SimpleNamespace(environ=env),torch=types.SimpleNamespace(device=lambda value:value))
exec(compile(ast.fix_missing_locations(ast.Module(body=[branch],type_ignores=[])),str(native),'exec'),scope)
assert scope['world_size']==1 and scope['local_rank']==0
proof=dict(passed=True,manifest_sha256=sha(R/'manifest.json'),checks=checks,
    native_distributed_branch_check=dict(passed=True,source_sha256=sha(native),inherited_rank_variables_stripped=True,world_size=1,local_rank=0),
    scope='Frozen-module CPU regression and native CLI/branch checks; no generation outcome or efficacy claim')
(R/'cpu_validation.json').write_text(json.dumps(proof,indent=2)+'\n')
module.verify(m)
print(json.dumps(dict(generation_jobs=m['generation_jobs'],score_groups=len(m['groups']),cpu_checks=len(checks),manifest_sha256=sha(R/'manifest.json'))))
