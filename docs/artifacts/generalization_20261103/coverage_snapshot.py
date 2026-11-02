"""All original cases, with retries visible rather than replacing failures."""
from pathlib import Path
import collections,hashlib,json,time
R=Path(__file__).resolve().parent;Q=R/'cohort_resume';S=R/'sana_observer_residency'
captured={}
def read(p):
    raw=p.read_bytes();captured[str(p)]=hashlib.sha256(raw).hexdigest();return json.loads(raw)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
freeze=read(Q/'freeze.json');state=read(Q/'status.json');supp=read(S/'status.json')
visual=read(R/'visual_review/manifest.json');visual_by_key={x['key']:x for x in visual['pairs']}
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
    if model=='sana':
        sj={k:v for k,v in supp['jobs'].items() if k.startswith(key+'_')}
        row['resource_supplement']=dict(queue_phase=supp['phase'],jobs=sj,
            note='Separate original-horizon retry; the parent execution failure/hold remains in this record.')
    rows.append(row)
assert len(rows)==40 and len({r['key'] for r in rows})==40
models={m:dict(planned=8,execution_counts=dict(collections.Counter(r['execution_state'] for r in rows if r['model']==m)),
               scored=sum('score' in r for r in rows if r['model']==m)) for m in freeze['protocol']['models']}
for model,counts in models.items():assert sum(counts['execution_counts'].values())==8,model
out=dict(snapshot_time=time.time(),original_planned_pairs=40,models=models,rows=rows,
    interpretation='Coverage accounting only. Execution success, active writes, numeric pose changes and visual quality are distinct. Pending/held cases are not efficacy results; retries do not erase failures.',
    queue_processes={name:dict(pid=s['pid'],phase=s['phase'],process_exists=Path('/proc',str(s['pid'])).exists()) for name,s in [('cohort',state),('sana_supplement',supp)]},
    input_hashes={str(p):captured.get(str(p),None) or sha(p) for p in [Q/'freeze.json',Q/'status.json',S/'manifest.json',S/'status.json',R/'visual_review/manifest.json']})
(R/'coverage_snapshot.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(models,indent=2))
