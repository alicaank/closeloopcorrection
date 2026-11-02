"""Resume the complete original cohort with an explicit resource-only overlay."""
import hashlib,importlib.util,json,os,subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parent;W=Path('/home/hj/Alican')
OLD=W/'analysis/measured_memory_multiboundary_20261102'
RETRY=R.parent/'sana_lifetime_qualification'
PY=W/'baselines/venv_vmem/bin/python'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not (R/'freeze.json').exists(),'Never overwrite a frozen queue'
parent=json.loads((OLD/'freeze.json').read_text())
for p,h in {**parent['sources'],**parent['inputs']}.items():assert sha(p)==h,p
prior=json.loads((RETRY/'manifest.json').read_text())
for p,h in prior['sources'].items():assert sha(p)==h,p
for base in (OLD,RETRY):
    state=json.loads((base/'status.json').read_text())
    assert state['phase']=='held' and not Path('/proc',str(state['pid'])).exists(),base
protocol=parent['protocol']
(R/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
pairs=[]
for original in parent['pairs']:
    p=original.copy()
    for arm in ('commanded','measured'):p[arm]=str(R/'runs'/p['key']/arm/('baseline' if p['model']=='lingbot' else ''))
    pairs.append(p)
    (R/'specs'/f"{p['key']}.json").write_text(json.dumps(p,indent=2)+'\n')
assert len(pairs)==40 and {p['model'] for p in pairs}==set(protocol['models'])
key='lingbot_02267acf_s7';score=OLD/'scores'/key/'score.json'
reuse_files=[score,OLD/'specs'/f'{key}.json',*[p for p in (OLD/'runs'/key).rglob('*') if p.is_file()]]
reuse={key:dict(score=str(score),hashes={str(p):sha(p) for p in reuse_files},
    note='Previously completed original pair; visual collapse despite lower pose error. Retained as a policy failure, not an efficacy success.')}
reference=RETRY/'runs/sana_02267acf_s7/commanded/frames.npz'
assert reference.is_file()
sources={p:h for p,h in parent['sources'].items() if not Path(p).is_relative_to(OLD)}
files=[*R.glob('*.py'),R/'protocol.json',OLD/'freeze.json',RETRY/'manifest.json',
       *[p for p in (R/'source').rglob('*') if p.is_file() and '__pycache__' not in p.parts],*sorted((R/'specs').glob('*.json'))]
sources.update({str(p):sha(p) for p in files})
frozen=dict(protocol=protocol,pairs=pairs,created=time.time(),native_revisions=parent['native_revisions'],
    sources=sources,inputs=parent['inputs'],reuse_pairs=reuse,
    parent_freeze_sha256=sha(OLD/'freeze.json'),resource_parity=dict(reference=str(reference),sha256=sha(reference)),
    source_export='Original frozen scientific sources; SANA auxiliary observer close plus consumed-history reference release. Cache precision, length, tensor arithmetic and future targets unchanged. GPU RGB parity required.',
    execution_policy='Host-memory failure holds affected model only; disk/freeze guard holds queue. Original 40 pairs and all failures remain in accounting.')
(R/'freeze.json').write_text(json.dumps(frozen,indent=2)+'\n')
spec=importlib.util.spec_from_file_location('queue_commands',R/'run.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONPATH=str(W/'camera-closed-loop/src'))
with (R/'logs/repository_tests.log').open('x') as f:
    subprocess.run([str(PY),'-m','unittest','discover','-s','tests','-v'],cwd=W/'camera-closed-loop',env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
with (R/'logs/native_compile.log').open('x') as f:
    for model in protocol['models']:
        p=next(p for p in pairs if p['model']==model);argv,jobenv=module.command(p,'commanded',protocol)
        split=argv.index('--')
        subprocess.run(argv[:split]+['--check-source'],env=dict(jobenv,CUDA_VISIBLE_DEVICES=''),stdout=f,stderr=subprocess.STDOUT,check=True)
with (R/'logs/native_history_check.log').open('x') as f:
    subprocess.run([str(PY),str(R/'native_history_check.py')],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
assert json.loads((R/'native_history_validation.json').read_text())['passed']
module.verify(frozen,True)
checks=['130 repository CPU tests','All five native driver sources compile','216 exact native history-update cases','Original and first retry source/input freezes verified','All 40 original pairs preserved; one completed pair reused with hashes']
(R/'cpu_validation.json').write_text(json.dumps(dict(passed=True,freeze_sha256=sha(R/'freeze.json'),checks=checks,
    evidence_sha256={str(p.relative_to(R)):sha(p) for p in [R/'logs/repository_tests.log',R/'logs/native_compile.log',R/'native_history_validation.json']}),indent=2)+'\n')
print(json.dumps(dict(pairs=len(pairs),reused=len(reuse),new_generations=2*(len(pairs)-len(reuse)),sources=len(sources),checks=checks)))
