"""New immutable attempt: same eight cases, explicit allocator-only option."""
import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
W=Path('/home/hj/Alican');R=Path(__file__).resolve().parent
OLD=R.parent/'hosts/sana';HELP=R.parent/'other_models';PY=W/'baselines/venv_vmem/bin/python'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
if (R/'manifest.json').exists():raise FileExistsError('Preserve frozen retry')
old=json.loads((OLD/'manifest.json').read_text())
for p,h in old['sources'].items():assert sha(p)==h,p
failed=W/'analysis/queue_disk_resume_v2_20261103/legacy_sana/status.json'
status=json.loads(failed.read_text());assert status['phase']=='held' and not Path(f"/proc/{status['pid']}/cmdline").exists()
assert status['jobs']['sana_02267acf_s7_B']['stop_reason']=='host_memory_guard'
shutil.copytree(OLD/'source',R/'source',ignore=shutil.ignore_patterns('__pycache__','*.pyc'),dirs_exist_ok=True)
shutil.copy2(W/'camera-closed-loop/src/camera_loop/resource_residency.py',R/'source/camera_loop/resource_residency.py')
(R/'helpers').mkdir(exist_ok=True)
for p in HELP.glob('*.py'):
    if p.name!='prepare.py':shutil.copy2(p,R/'helpers'/p.name)
for name in ('logs','runs','bindings','gates','scores'):(R/name).mkdir(exist_ok=True)
ensure=R/'helpers/ensure_arm.py';s=ensure.read_text();needle="directory = Path(spec['reference']) if reused else Path(spec['output'])"
assert s.count(needle)==1
s=s.replace(needle,"if a.case==m['protocol']['first_case'] and a.arm=='B':\n    reused=False;reason='mandatory_fresh_allocator_qualification'\n"+needle);ensure.write_text(s)
runner=(OLD/'run_queue.py').read_text()
needle='''                    if reason:
                        if entry['argv'][1].endswith('/launch.py') and active.poll() is None:
                            try:os.kill(active.pid,signal.SIGUSR1);time.sleep(0.2)
                            except ProcessLookupError:pass
                        stop();break'''
assert runner.count(needle)==1
runner=runner.replace(needle,'''                    if reason:
                        diagnostics=[]
                        for process in processes:
                            try:
                                command=process.cmdline();memory=process.memory_full_info()
                                diagnostics.append(dict(pid=process.pid,command=command,rss=memory.rss,pss=getattr(memory,'pss',None),uss=getattr(memory,'uss',None)))
                                if str(R/'source/launch.py') in command:process.send_signal(signal.SIGUSR1)
                            except psutil.Error:pass
                        (R/'logs'/f'{key}.resource_stop.json').write_text(json.dumps(diagnostics,indent=2)+'\\n')
                        time.sleep(1);stop();break''')
# Guard collection must still work if the process vanished during sampling.
runner=runner.replace('                    try: processes=', '                    processes=[]\n                    try: processes=')
(R/'run_queue.py').write_text(runner)
protocol=json.loads(json.dumps(old['protocol']))
refs=[OLD/'runs/sana_02267acf_s7/B/frames.npz',W/'analysis/memory_generalization_20261103/cohort_resume/runs/sana_02267acf_s7/commanded/frames.npz']
protocol['resource_retry']=dict(parent=str(OLD),parent_manifest_sha256=sha(OLD/'manifest.json'),failed_status=str(failed),failed_status_sha256=sha(failed),change='Opt-in gc.collect plus glibc malloc_trim after stage-1 CPU-to-GPU placement and around CPU restoration; no live tensors evicted, no policy/dtype/context/horizon change.',qualification='Fresh 121-frame B, exact preserved 73- and 97-frame prefixes, then original full B/S identity and all original pair gates.',prefix_references=[dict(path=str(p),sha256=sha(p)) for p in refs])
cpuenv=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(R/'source'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
compile_result=subprocess.run([str(PY),str(R/'source/launch.py'),'sana','--check-source'],env=cpuenv,capture_output=True,text=True,check=True)
(R/'logs/source_compile.log').write_text(compile_result.stdout+compile_result.stderr)
compiled=json.loads(compile_result.stdout);assert compiled['compiled']
protocol['copied_overlay_hashes']['sana']=compiled['sha256']
arm_jobs=json.loads(json.dumps(old['arm_jobs']))
for case,arms in arm_jobs.items():
    for arm,job in arms.items():
        job['argv']=[x.replace(str(OLD),str(R)) for x in job['argv']]
        job['output']=job['output'].replace(str(OLD),str(R))
jobs=[]
for original in old['jobs']:
    job=json.loads(json.dumps(original));job['argv']=[x.replace(str(OLD),str(R)).replace(str(HELP),str(R/'helpers')) for x in job['argv']]
    job['env']={k:v.replace(str(OLD),str(R)) if isinstance(v,str) else v for k,v in job['env'].items()}
    job['env']['CAMERA_HOST_ALLOCATOR_TRIM']='1';jobs.append(job)
    if job['name']=='sana_02267acf_s7_B':
        jobs.append(dict(name='sana_02267acf_s7_resource_prefix',case=job['case'],required=True,env=job['env'],argv=[str(PY),str(R/'prefix_gate.py'),'--root',str(R),'--case',job['case']]))
(R/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
sources={p:h for p,h in old['sources'].items() if not Path(p).is_relative_to(OLD)}
sources.update({str(p):sha(p) for p in [OLD/'manifest.json',failed,*refs]})
for p in R.rglob('*'):
    if p.is_file() and '__pycache__' not in p.parts:sources[str(p)]=sha(p)
m=dict(protocol=protocol,arm_jobs=arm_jobs,jobs=jobs,sources=sources,created=time.time())
(R/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
assert len(protocol['cases'])==8 and all(c['frames']==121 for c in protocol['cases'].values())
log=(R/'repository_tests.log').read_text();assert '\nOK\n' in log and 'FAILED' not in log
proof=dict(passed=True,manifest_sha256=sha(R/'manifest.json'),gpu_qualified=False,checks=['155 repository CPU tests','Real allocator call preserves live CPU tensor storage/value and Torch RNG','Exception-safe device restoration and trim lifecycle','Native overlay compilation','Original eight 121-frame cases retained; mandatory original-prefix and B/S gates'])
(R/'cpu_validation.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(dict(cases=8,jobs=len(jobs),manifest_sha256=proof['manifest_sha256'])))
