"""All original cases, with retries visible rather than replacing failures."""
from pathlib import Path
import collections,hashlib,json,time
R=Path(__file__).resolve().parent;Q=R/'cohort_resume';S=R/'sana_observer_residency';M=R/'matrix_observer_lifetime';L=R/'lyra_observer_resources'
captured={}
T=R/'sana_observer_transition'
def read(p):
    raw=p.read_bytes();captured[str(p)]=hashlib.sha256(raw).hexdigest();return json.loads(raw)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def effective_state(root,label):
    original=read(root/'status.json')
    combined=original;previous=[];previous_root=root
    for name in ['queue_disk_resume_20261103','queue_disk_resume_v2_20261103']:
        attempt=R.parent/name/label
        if not (attempt/'status.json').exists():break
        provenance=read(attempt/'resume.json');resumed=read(attempt/'status.json')
        assert provenance['original_root']==str(root)
        assert provenance['original_status_sha256']==captured[str(root/'status.json')]
        assert provenance['original_manifest_sha256']==sha(root/'manifest.json')
        assert resumed['resume_provenance']==provenance
        assert set(provenance['completed_jobs_skipped'])==set(combined['jobs'])
        assert all(v['state']=='completed' and v['returncode']==0 for v in combined['jobs'].values())
        for prior in provenance.get('previous_attempts',[]):
            for filename,key in [('status.json','status_sha256'),('run_queue.py','runner_sha256'),('resume.json','resume_sha256')]:
                path=Path(prior['root'])/filename
                assert sha(path)==prior[key]
                captured[str(path)]=prior[key]
        assert not set(combined['jobs']).intersection(resumed['jobs']), 'Resume reran a completed job'
        item={k:combined[k] for k in ['pid','phase','failures','finished']}
        item.update(status=str(previous_root/'status.json'),sha256=sha(previous_root/'status.json'))
        previous.append(item)
        # Combine in memory for coverage only. Never rewrite an attempt record.
        combined=dict(resumed,jobs={**combined['jobs'],**resumed['jobs']})
        combined.update(previous_queue_attempt=item,previous_queue_attempts=previous.copy(),resume_status=str(attempt/'status.json'))
        previous_root=attempt
    return combined
freeze=read(Q/'freeze.json');state=read(Q/'status.json');supp=read(S/'status.json');matrix_supp=effective_state(M,'matrix_memory');lyra_supp=effective_state(L,'lyra_memory')
visual=read(R/'visual_review/manifest.json');visual_by_key={x['key']:x for x in visual['pairs']}
transition=effective_state(T,'sana_transition')
matrix_manifest=read(M/'manifest.json')
assert matrix_manifest['protocol']['parent_freeze_sha256']==sha(Q/'freeze.json')
resource_gate=read(M/'resource_prefix.json') if (M/'resource_prefix.json').exists() else {}
matrix_visual_path=M/'visual_review/manifest.json'
matrix_visual=read(matrix_visual_path) if matrix_visual_path.exists() else {'pairs':[]}
matrix_reviews={r['key']:r for r in matrix_visual['pairs']}
rows=[]
for pair in freeze['pairs']:
    key=pair['key'];model=pair['model'];entry=state['pairs'].get(key,{})
    phase=entry.get('state','pending')
    jobs={k:v for k,v in state['jobs'].items() if k.startswith(key+'_')}
    if not entry:
        if any(j['state']=='running' for j in jobs.values()):phase='running'
        elif model in state['held_models']:phase='held_by_model'
    row={k:pair[k] for k in ['key','model','scene','seed']}
    row.update(execution_state=phase,reason=entry.get('reason',state['held_models'].get(model)),jobs=jobs)
    if 'score' in entry:
        p=Path(entry['score']);score=read(p)
        row.update(score=str(p),score_sha256=sha(p),prefix_exact=score['prefix_rgb_exact'],
            comparison_valid=score['comparison_valid'],accepted_writes=score['accepted_memory_writes'],
            delta=score['delta'],position_valid={a:x['position_valid'] for a,x in score['arms'].items()})
        review=visual_by_key.get(key)
        if review and review['score_sha256']==row['score_sha256']:
            row['visual_review']={k:review[k] for k in ['review_status','classification'] if k in review}
    if model in ('sana','matrixgame','lyra'):
        supplement={'sana':supp,'matrixgame':matrix_supp,'lyra':lyra_supp}[model]
        sj={k:v for k,v in supplement['jobs'].items() if k.startswith(key+'_')}
        row['resource_supplement']=dict(queue_phase=supplement['phase'],jobs=sj,
            note='Separate original-horizon retry; the parent execution failure/hold remains in this record.')
        if 'previous_queue_attempt' in supplement:
            row['resource_supplement'].update(previous_queue_attempt=supplement['previous_queue_attempt'],previous_queue_attempts=supplement['previous_queue_attempts'],resume_status=supplement['resume_status'])
    if model=='sana':
        row['additional_resource_attempts']=[dict(name='transition_order',queue_phase=transition['phase'],
            jobs={k:v for k,v in transition['jobs'].items() if k.startswith(key+'_')})]
        if 'previous_queue_attempt' in transition:
            row['additional_resource_attempts'][0].update(previous_queue_attempt=transition['previous_queue_attempt'],previous_queue_attempts=transition['previous_queue_attempts'],resume_status=transition['resume_status'])
    if model=='matrixgame':
        scorepath=M/'scores'/key/'score.json';gatepath=scorepath.with_name('attribution_gate.json')
        if gatepath.exists():
            gate=read(gatepath);score=read(scorepath)
            assert resource_gate.get('passed') is True and resource_gate['new_frames']==337
            assert gate['passed'] is True and score['comparison_valid'] is True and score['prefix_rgb_exact'] is True
            assert score['horizon']==freeze['protocol']['schedules'][model]['frames']-1
            assert score['spec']['key']==key and score['spec']['first_affected_frame']==85
            retry=dict(attempt='matrix_observer_lifetime',score=str(scorepath),score_sha256=sha(scorepath),
                qualification_gate=str(gatepath),qualification_gate_sha256=sha(gatepath),
                resource_parity_gate=str(M/'resource_prefix.json'),resource_parity_gate_sha256=sha(M/'resource_prefix.json'),
                horizon=score['horizon'],comparison_valid=True,prefix_exact=True,
                accepted_writes=score['accepted_memory_writes'],delta=score['delta'],
                position_valid={a:v['position_valid'] for a,v in score['arms'].items()})
            review=matrix_reviews.get(key)
            if review and review['score_sha256']==retry['score_sha256']:
                retry['visual_review']={k:review[k] for k in ['review_status','classification'] if k in review}
            row['qualified_retry']=retry
    row['has_qualified_result']=bool(row.get('comparison_valid') or row.get('qualified_retry'))
    rows.append(row)
assert len(rows)==40 and len({r['key'] for r in rows})==40
models={m:dict(planned=8,execution_counts=dict(collections.Counter(r['execution_state'] for r in rows if r['model']==m)),
               scored=sum('score' in r for r in rows if r['model']==m)) for m in freeze['protocol']['models']}
for model,counts in models.items():
    assert sum(counts['execution_counts'].values())==8,model
    counts['qualified_unique_pairs']=sum(r['has_qualified_result'] for r in rows if r['model']==model)
    counts['qualified_retries']=sum('qualified_retry' in r for r in rows if r['model']==model)

out=dict(snapshot_time=time.time(),original_planned_pairs=40,models=models,rows=rows,
    interpretation='Coverage accounting only. Execution success, active writes, numeric pose changes and visual quality are distinct. Pending/held cases are not efficacy results; retries do not erase failures.',
    queue_processes={name:dict(pid=s['pid'],phase=s['phase'],process_exists=Path('/proc',str(s['pid'])).exists()) for name,s in [('cohort',state),('sana_supplement',supp),('matrix_supplement',matrix_supp),('lyra_supplement',lyra_supp),('sana_transition',transition)]},
    input_hashes={**captured,**{str(p):captured.get(str(p),None) or sha(p) for p in [Q/'freeze.json',Q/'status.json',S/'manifest.json',S/'status.json',M/'manifest.json',M/'status.json',L/'manifest.json',L/'status.json',R/'visual_review/manifest.json',T/'manifest.json',T/'status.json']}})
(R/'coverage_snapshot.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(models,indent=2))
