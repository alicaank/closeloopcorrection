"""Reconcile the prespecified cohort, retaining failures, abstentions and tradeoffs."""
from pathlib import Path
import hashlib,json,statistics
R=Path(__file__).resolve().parent;W=R.parents[1];D=W/'camera-closed-loop/docs';E=W/'analysis/measured_memory_v1_20261101'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
freeze=read(E/'freeze.json');state=read(E/'status.json')
visual_source=D/'memory_component_evidence_20261102.json'
review=read(visual_source)['historical_and_expanded']['expanded_review']
visual=review['visual_review']['pairs'];audits={x['pair']:x for x in review['audit']}
rows=[];sources={str(p):sha(p) for p in [E/'freeze.json',E/'status.json',visual_source]}
for spec in freeze['pairs']:
    key=spec['key'];row={k:spec[k] for k in ['key','model','scene','seed']}
    row['state']=state['pairs'].get(key,{}).get('state','not_started')
    p=E/'scores'/key/'score.json'
    if not p.exists():
        row['reason']=state['pairs'].get(key,{}).get('reason');rows.append(row);continue
    s=read(p);sources[str(p)]=sha(p)
    assert audits[key]['score_sha256']==sha(p),'Visual review no longer bound to this score'
    a=s['arms']['commanded'];b=s['arms']['measured']
    identical=bool(a.get('frame_sha256') and a['frame_sha256']==b.get('frame_sha256') and a['K']==b['K'])
    row.update(comparison_valid=s['comparison_valid'],prefix_exact=s['prefix_rgb_exact'],
               accepted_writes=s['accepted_memory_writes'],identical_rgb_and_intrinsics=identical,
               visual_review=visual[key],metrics={})
    for metric in ['position_mean','position_endpoint','rotation_mean_deg','rotation_endpoint_deg']:
        x=a['post_intervention'][metric];raw=b['post_intervention'][metric]
        y=x if identical else raw
        valid=bool(s['comparison_valid'] and x is not None and y is not None)
        if metric.startswith('position'):valid=valid and a['position_valid'] and (a['position_valid'] if identical else b['position_valid'])
        delta=(y-x) if valid else None
        row['metrics'][metric]=dict(commanded=x,measured=y,original_measured=raw,valid=valid,
            outcome=('unavailable' if not valid else 'win' if delta<0 else 'loss' if delta>0 else 'tie'),
            delta=delta,reduction_percent=(100*(x-y)/x if valid and x>0 else None))
    rows.append(row)
models={}
for model in freeze['protocol']['models']:
    rs=[r for r in rows if r['model']==model];complete=[r for r in rs if 'metrics' in r]
    item=dict(prespecified=len(rs),completed=len(complete),active=sum(r['accepted_writes']>0 for r in complete),
              abstained=sum(r['accepted_writes']==0 for r in complete),
              active_scenes=sorted({r['scene'] for r in complete if r['accepted_writes']>0}),metrics={})
    for metric in ['position_mean','rotation_mean_deg']:
        counts={v:sum(r['metrics'][metric]['outcome']==v for r in complete) for v in ['win','loss','tie','unavailable']}
        gains=[r['metrics'][metric]['reduction_percent'] for r in complete if r['accepted_writes']>0 and r['metrics'][metric]['reduction_percent'] is not None]
        counts['active_reduction_percent_range']=[min(gains),max(gains)] if gains else None
        item['metrics'][metric]=counts
    models[model]=item
out=dict(scope='Retrospective accounting of a prespecified two-scene development cohort. No failed scene is removed and no efficacy confidence interval is inferred from two scenes.',
         exact_input_rule='For identical RGB, K and fixed scene reference, reuse the commanded estimate; keep original measured score for provenance. Invalid reference registration remains unavailable.',
         model_aggregation='Within-model paired outcomes only; no pooled raw errors across native horizons or units.',
         rows=rows,models=models,sources=sources,script_sha256=sha(Path(__file__)))
(D/'generalization_evidence_20261103.json').write_text(json.dumps(out,indent=2)+'\n')
lines=['| Model | Complete / planned | Active / abstained | Mean-position W/L/T/unavailable | Mean-rotation W/L/T/unavailable | Active mean-position reduction |',
       '| --- | --- | --- | --- | --- | --- |']
for model,m in models.items():
    def counts(metric):return '/'.join(str(m['metrics'][metric][v]) for v in ['win','loss','tie','unavailable'])
    interval=m['metrics']['position_mean']['active_reduction_percent_range']
    gain='—' if interval is None else f'{interval[0]:+.2f}% to {interval[1]:+.2f}%'
    lines.append(f"| {model} | {m['completed']}/{m['prespecified']} | {m['active']}/{m['abstained']} | {counts('position_mean')} | {counts('rotation_mean_deg')} | {gain} |")
(R/'cohort_table.md').write_text('\n'.join(lines)+'\n')
assert len(rows)==20 and sum('metrics' in r for r in rows)==18
print('\n'.join(lines))
