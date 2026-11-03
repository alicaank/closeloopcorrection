"""Descriptive accounting of every frozen primary pair; never selects successes."""
from pathlib import Path
import collections
import hashlib
import json
import time

ROOT = Path(__file__).resolve().parent
captured = {}


def read(path):
    raw = path.read_bytes()
    captured[str(path)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


freeze = read(ROOT / 'cohort_resume/freeze.json')
state = read(ROOT / 'cohort_resume/status.json')
visual = read(ROOT / 'visual_review/manifest.json')
reviews = {r['key']: r for r in visual['pairs']}
rows = []
for pair in freeze['pairs']:
    key = pair['key']
    p = state['pairs'].get(key, {})
    jobs = [v for k, v in state['jobs'].items() if k.startswith(key + '_')]
    status = p.get('state', 'running' if any(j['state'] == 'running' for j in jobs) else
                   'held_by_model' if pair['model'] in state['held_models'] else 'pending')
    row = dict(key=key, model=pair['model'], scene=pair['scene'], seed=pair['seed'],
               state=status, reason=p.get('reason', state['held_models'].get(pair['model'])))
    if 'score' in p:
        path = Path(p['score'])
        s = read(path)
        assert s['spec']['key'] == key
        valid = all(v['position_valid'] for v in s['arms'].values())
        metrics = {}
        for metric in ['position_mean', 'position_endpoint', 'rotation_mean_deg', 'rotation_endpoint_deg']:
            a = s['arms']['commanded']['post_intervention'][metric]
            b = s['arms']['measured']['post_intervention'][metric]
            if metric.startswith('position') and not valid:
                assert a is None or b is None
                metrics[metric] = dict(commanded=a, measured=b, outcome='unavailable')
                continue
            assert a is not None and b is not None
            metrics[metric] = dict(commanded=a, measured=b,
                outcome='lower' if b < a else 'higher' if b > a else 'equal',
                measured_minus_commanded=b-a,
                reduction_percent=100*(a-b)/a if a > 0 else None)
        row.update(score=str(path), score_sha256=captured[str(path)], frames=s['horizon']+1,
                   prefix_exact=s['prefix_rgb_exact'], comparison_valid=s['comparison_valid'],
                   accepted_writes=s['accepted_memory_writes'],
                   rejected_writes=len(s['rejected_memory_writes']), position_valid=valid,
                   metrics=metrics)
        r = reviews.get(key)
        if r and r['score_sha256'] == row['score_sha256'] and r['review_status'] == 'sampled_visual_review_complete':
            inspection = read(Path(r['inspection_record']))
            assert captured[r['inspection_record']] == r['inspection_sha256']
            row['visual_assessment'] = inspection['classification']
            row['visual_scope'] = inspection.get('method', 'Six-frame sampled visual review')
        if row['accepted_writes'] == 0:
            audit_path = ROOT / 'visual_review' / (key + '.abstention_rgb.json')
            if audit_path.exists():
                audit = read(audit_path)
                assert audit['score_sha256'] == row['score_sha256']
                row['full_rgb_equality_verified'] = audit['full_rgb_exact']
    rows.append(row)
assert len(rows) == 40 and len({r['key'] for r in rows}) == 40
models = {}
for model in freeze['protocol']['models']:
    subset = [r for r in rows if r['model'] == model]
    scored = [r for r in subset if 'metrics' in r]
    eligible = [r for r in scored if r['comparison_valid'] is True and r['prefix_exact'] is True]
    models[model] = dict(planned=len(subset), scored=len(scored),
        execution_counts=dict(collections.Counter(r['state'] for r in subset)),
        active_pairs=sum(r['accepted_writes'] > 0 for r in eligible),
        zero_write_pairs=sum(r['accepted_writes'] == 0 for r in eligible),
        full_rgb_equality_verified=sum(r.get('full_rgb_equality_verified') is True for r in eligible),
        visually_reviewed=sum('visual_assessment' in r for r in scored),
        qualified_pairs=len(eligible),
        metric_outcomes={m:dict(collections.Counter(r['metrics'][m]['outcome'] for r in eligible))
                         for m in ['position_mean', 'position_endpoint', 'rotation_mean_deg', 'rotation_endpoint_deg']})
    assert len(subset) == 8

result = dict(snapshot_time=time.time(), original_pairs=40, primary_method='measured-memory-v1',
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    main_queue_phase=state['phase'], models=models, rows=rows, input_sha256=captured,
    interpretation='Descriptive development-cohort accounting. Strict numeric lower/higher/equal is not significance or overall success. Invalid position stays unavailable; missing executions stay missing. Visual assessments are unblinded qualitative reviews, not a calibrated quality metric. No raw-error mean is pooled across scenes or models, and no primary metric or inclusion rule is changed. Resource retry outcomes must be reconciled separately without overwriting these original attempts.')
(ROOT / 'primary_cohort_summary.json').write_text(json.dumps(result, indent=2)+'\n')
lines = ['# Original repeated-memory cohort: descriptive accounting', '',
         'All 40 frozen cases remain below. Lower/higher counts are strict numerical comparisons, not significant effects or overall visual success. Missing position is unavailable. Resource retry results remain separate from original failed/held attempts.', '',
         '| Model | Scored / planned | Active / zero-write | Mean-position lower / higher / equal / unavailable | Reviewed |',
         '| --- | ---: | ---: | --- | ---: |']
for name, m in models.items():
    c=m['metric_outcomes']['position_mean']
    counts=' / '.join(str(c.get(k,0)) for k in ['lower','higher','equal','unavailable'])
    lines.append(f"| {name} | {m['scored']} / {m['planned']} | {m['active_pairs']} / {m['zero_write_pairs']} | {counts} | {m['visually_reviewed']} |")
lines += ['', '## Every frozen case', '', '| Case | Execution | Position: commanded → measured | Writes | Visual assessment |', '| --- | --- | --- | ---: | --- |']
for row in rows:
    if 'metrics' in row:
        m=row['metrics']['position_mean']
        value='Unavailable' if m['outcome']=='unavailable' else f"{m['commanded']:.6f} → {m['measured']:.6f}"
        writes=str(row['accepted_writes'])
    else:
        value='Missing';writes='—'
    lines.append(f"| {row['key']} | {row['state']} | {value} | {writes} | {row.get('visual_assessment','Not yet reviewed / not generated')} |")
lines += ['', 'Position values use each scene’s anchor-depth units. Models have different native horizons. The JSON contains endpoint and rotation outcomes, source hashes, exact-prefix qualification and abstention RGB checks. These tables introduce no aggregate efficacy claim.', '']
(ROOT / 'primary_cohort_summary.md').write_text('\n'.join(lines))
print(json.dumps(models, indent=2))
