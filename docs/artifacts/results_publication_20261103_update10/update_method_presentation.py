"""Publish Method averages and a complete, evidence-qualified failure ledger."""
import copy
import datetime
import gzip
import hashlib
import json
import math
import re
import shutil
from pathlib import Path
from package_data import write_snapshot

ROOT = Path(__file__).resolve().parent
D = ROOT / 'dist'
W = ROOT.parents[1]
before = json.loads(gzip.decompress((D / 'snapshot.json.gz').read_bytes()))
data = copy.deepcopy(before)
n = data['research']['nov3']
audit = json.loads((D / n['legacy_failures']['source']).read_text())
ids = {c['id'] for c in audit['cases']}
assert len(ids) == 80
assert sum(c['position_regression'] for c in audit['cases']) == 21
assert sum(c['mean_improves_endpoint_worsens'] for c in audit['cases']) == 8

def evidence(path):
    b = path.read_bytes()
    dest = D / 'evidence' / ('method_update_' + path.name)
    dest.write_bytes(b)
    return 'evidence/' + dest.name

report = n['links']['legacy_rejoin_failures_20261103.md']
zigzag = n['zigzag']['report']
details = {}

def detail(pair, when, problem, cause, certainty, frame=None, source=report):
    details[pair] = dict(when=when, problem=problem, cause=cause,
                         certainty=certainty, frame=frame, evidence=source)

detail('lingbot_re10k160_d3f042a2_s107',
       'After boundary 44: degradation visible at 48; noise collapse by 64. Sampled onset, not exact first bad frame.',
       'Large commands precede scene collapse; mean position worsens 91.7%. Finite observer poses continue afterward.',
       'Normalized steps reach 3.95 after frame 44 and 17.07 after 60. Large inputs are a supported contributor. The finite-only observation gate demonstrably fails to reject damaged imagery; amplitude alone is not isolated.',
       'Supported contributor', 48, zigzag)
for scene, frame, maximum in [('74310829',16,'10.42'),('e43058cb',24,'15.26')]:
    detail(f'lingbot_re10k160_{scene}_s7', f'First degraded inspected sample: frame {frame}; accepted feedback continues at 28–156.',
           'Visible scene degradation after correction, with finite poses still accepted.',
           f'Large native commands (whole-run maximum {maximum}) and missing scene-quality rejection are observed. The maximum alone does not locate the trigger or prove amplitude is the sole cause.',
           'Supported contributor',frame,zigzag)
detail('dreamx_re10k160_e43058cb_s7','Across the 160-frame rollout; exact visual onset not established.',
       'Both arms miss the reference transition into the kitchen. Position worsens while rotation improves.',
       'Recovery steps reach 23.43× the corresponding planned step, with a three-frame observation delay. Aggressive recovery is a candidate contributor; observer versus generator/history effects are not isolated. Native-anchor replay finds zero intended command reversals.',
       'Cause unresolved')
detail('sana_rejoin_v3_e43058cb_s7','Observer disagreement measured at frame 144; exact visual onset not established.',
       'Position and rotation worsen; sampled rooms depart from the reference layout.',
       'DA3 and VIPE differ by 1.196 anchor units and 79.27° at frame 144. This establishes observer disagreement, not which observer is correct or that gain alone causes the failure.',
       'Cause unresolved',144)
detail('sana_rejoin_v3_0f12b97e_s7','Disagreement grows between frames 24 and 144.',
       'Position worsens, while rotation and LPIPS improve.',
       'Observer position disagreement grows from 0.00238 to 0.14909 anchor units. A tracking/appearance tradeoff is observed; its causal source is not isolated.',
       'Cause unresolved',144)
detail('matrixgame_dl3dv480_0238de14_s7','Contrast seam at frame 337 (inspected 334–339); regression scored over 480 frames.',
       'Position 0.01574 → 0.02622 and rotation 0.840° → 1.214°; image scores improve slightly.',
       'A separate same-latent experiment establishes a decoder/export contribution to seams. It does not prove decoding causes this historical tracking regression; both arms show the seam.',
       'Cause unresolved',337)
detail('lyra_rejoin_v3_61b27b61_s7','Across the 480-frame garden rollout; exact onset not established.',
       'Garden layout changes; position 0.01892 → 0.04653 and mean rotation 0.147° → 0.677°.',
       'Scene/geometry drift is visible. Observer versus generation effects remain unresolved; no universal angular noise floor applies.',
       'Cause unresolved')
detail('spatia_rejoin_v3_a126efe8_s7','Across the 480-frame hall rollout, including continuation seams; exact onset not established.',
       'Mean and endpoint position worsen despite slightly better image scores; hall layout drifts.',
       'Image discontinuities are observed, but physical camera reversal versus geometry inconsistency is unresolved. Memory-label/handoff causality has not been isolated.',
       'Cause unresolved')
detail('astronex_minwm76_minwm_001231_s7','Throughout the 76-frame stationary-reference case.',
       'Mean position 0.04193 → 0.07538. Huge path ratios are misleading on zero target translation.',
       'The ratio failure is a confirmed zero-denominator artifact. The pose regression itself remains unresolved; moving people and reference-motion calibration complicate interpretation.',
       'Confirmed metric artifact')
for scene in ['74310829','d3f042a2']:
    detail(f'vmem_re10k40_{scene}_s7','Across the 40-view outbound-and-return cycle; endpoint alone hides intermediate error.',
           'Combined correction regresses; command-only is worse still. On 74310829 the sampled view stretches into a patio-wall close-up instead of the kitchen.',
           'Four-arm evidence shows memory materially changes the result. Commands replay correctly; per-window normalization and observer disagreement remain unresolved. This is not proof of a single scale or bridge bug.',
           'Cause unresolved')

ledger = []
for c in audit['cases']:
    if not (c['position_regression'] or c['mean_improves_endpoint_worsens']):
        continue
    outcome = 'Position regression' if c['position_regression'] else 'Endpoint tradeoff'
    row = dict(pair=c['id'], model=c['model'], scene=c['scene'], seed=c['seed'],
               cohort=c['cohort'], scope='Method', policy='Method + measured memory' if c['model']=='vmem' else 'Method',
               outcome=outcome, horizon=c['horizon'], metrics=c['metrics'])
    row.update(details.get(c['id'], dict(
        when=f"Scored horizon: frame 0–{c['horizon']}; exact onset not reviewed.",
        problem='Mean position is worse than baseline.' if c['position_regression'] else 'Mean position improves, but final position worsens.',
        cause='No case-specific mechanism has been isolated. This row records a numerical outcome; it is not a diagnosis of visible collapse.',
        certainty='Not yet diagnosed', frame=None, evidence=n['legacy_failures']['source'])))
    ledger.append(row)

# Include the missing position case, without turning unavailable into zero.
c = next(c for c in audit['cases'] if c['model']=='sana' and c['scene']=='c1496c1b')
ledger.append(dict(pair=c['id'], model=c['model'], scene=c['scene'], seed=c['seed'], cohort=c['cohort'],
    scope='Method',policy='Method',outcome='Position unavailable',horizon=c['horizon'],metrics=c['metrics'],
    when='Full scored horizon; position registration unavailable.',problem='No valid positional comparison.',
    cause='Position is excluded from the mean; this does not establish a generator failure.',
    certainty='Cause unresolved',frame=None,evidence=report))

for p in data['pairs']:
    if p.get('experiment')=='dreamx_future' and p['scene']=='02267acf' and p['seed']==7:
        ledger.append(dict(pair=p['id'],model=p['model'],scene=p['scene'],seed=7,cohort=p['cohort'],scope='Ablation / control',
          policy='F · bounded future commands',outcome='Position regression',horizon=p['horizon'],metrics=None,
          when='Post-intervention frames 33–68; speed clipping at boundaries 44 and 56.',
          problem='Mean position worsens 5.99%; endpoint worsens 5.25%.',
          cause='The requested catch-up is mostly unserved after clipping; online/offline observer RGB preprocessing also differs. These facts do not isolate generator authority.',
          certainty='Cause unresolved',frame=44,evidence=n['dreamx_future_results']['audit']))
    if p.get('cohort')=='memory_repeated' and p['model']=='lingbot' and p['scene']=='02267acf':
        ledger.append(dict(pair=p['id'],model=p['model'],scene=p['scene'],seed=p['seed'],cohort=p['cohort'],scope='Ablation / control',
          policy='M · measured memory',outcome='Visual failure',horizon=p['horizon'],metrics=None,
          when='After measured-history intervention on a near-stationary plan; exact onset is not established here.',
          problem='Lower pose error can coexist with visible loss of scene content.',
          cause='Measured history steps reach about 70–106× the plan-normalized scale. The whole memory rewrite causes the change; amplitude alone remains unisolated. This is separate from the Method RE10K failures.',
          certainty='Supported contributor',frame=None,evidence=zigzag))

new_reports={name:evidence(W/'camera-closed-loop/docs'/name) for name in ['online_response_review_20261103.md','worldplay_identity_trace_20261103.md','sana_host_allocator_retry_20261103.md']}
for model,scene,policy,outcome,when,problem,cause,certainty,report_name in [
 ('worldplay','0238de14','A/S · identity control','Identity failure','Prefixes match through 44; first mismatch at boundary 60.',
  'Identical commands produce unequal RGB; C is held.', 'Mismatch precedes camera-based memory selection. Remaining numerical/state/decoder cause awaits tracing.', 'Cause unresolved','worldplay_identity_trace_20261103.md'),
 ('astronex','0238de14','C · online response','No correction activated','20 boundaries / 637 RGB frames.',
  'All A/S/C frames and commands match; this is abstention, not improvement.', 'No axis passes response-identification gates. Command variation and supported positive slope are insufficient; physical units remain unidentified.', 'Confirmed abstention','online_response_review_20261103.md'),
 ('sana','02267acf','Baseline · 121-frame qualification','Runtime failure','Stopped after a retained 73-frame prefix, during the third observation scope.',
  'Full horizon did not complete because available host memory crossed the guard.', 'Host-memory guard termination is confirmed; retained allocator pages are a hypothesis. The allocator-reclamation retry is queued.', 'Confirmed resource stop','sana_host_allocator_retry_20261103.md')]:
    ledger.append(dict(pair=None,model=model,scene=scene,seed=7,cohort='qualification',scope='Ablation / control',policy=policy,
       outcome=outcome,horizon=None,metrics=None,when=when,problem=problem,cause=cause,certainty=certainty,frame=None,evidence=new_reports[report_name]))

# Recompute every displayed mean from the audit's case-level measurements.
for row in n['legacy']['rows']:
    cases=[c for c in audit['cases'] if (c['model'],c['cohort'],c['seed'])==(row['model'],row['cohort'],row['seed'])]
    for key, m in row['metrics'].items():
        valid=[c['metrics'][key] for c in cases if all(isinstance(c['metrics'][key][side],(int,float)) and math.isfinite(c['metrics'][key][side]) for side in ['baseline','correction'])]
        assert len(valid)==m['valid_pairs']
        for side in ['baseline','correction']:
            assert math.isclose(sum(x[side] for x in valid)/len(valid),m[side+'_mean'],rel_tol=1e-12,abs_tol=1e-12)

n['method_presentation']=dict(updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    pair_ids=sorted(ids),failure_ledger=ledger,reviewed_cases=details,
    metric_policy='Equal weight per valid scene within each model/cohort/seed. Percent change is the ratio of cohort means; no pooled cross-model mean.',
    source=n['legacy_failures']['source'],reports=new_reports)
for p in data['pairs']:
    if p['id'] in ids:
        p['right_label']='Method + measured memory' if p['model']=='vmem' else 'Method'

def rename(value):
    if isinstance(value,str):
        return re.sub(r'legacy rejoin','Method',value,flags=re.I).replace('matched legacy study','matched Method study').replace('Baseline / legacy / memory-only','Baseline / Method / memory-only')
    if isinstance(value,list): return [rename(x) for x in value]
    if isinstance(value,dict): return {k:rename(v) for k,v in value.items()}
    return value
data=rename(data)
for p,old in zip(data['pairs'],before['pairs']):
    assert p['id']==old['id'] and p['arms']==old['arms'], 'Scientific data changed'
assert len(data['pairs'])==203
write_snapshot(data)
catalog=rename(json.loads((D/'evidence/catalog.json').read_text()))
(D/'evidence/catalog.json').write_text(json.dumps(catalog,indent=2)+'\n')
(D/'catalog.js').write_text('window.ABLATION_CATALOG='+json.dumps(catalog,separators=(',',':'))+';')

for name in ['index.html','nov3.js','research.js']:
    p=D/name;p.write_text(rename(p.read_text()))
p=D/'index.html';s=p.read_text()
s=s.replace('CURRENT EVIDENCE · METHOD SELECTION OPEN','METHOD · RESULTS AND LIMITS')
s=s.replace('HISTORICAL LEAD','BASELINE → METHOD')
s=s.replace('Method deserves the direct comparison.','Method: average results')
s=s.replace('Earlier cohorts show substantial tracking gains. These percentages compare rejoin with each cohort’s own baseline; they are not a matched ranking against memory-only.','Each row compares the Method with its own baseline. Averages, scene counts and regressions are shown together. Different datasets, horizons and observers remain separate.')
s=s.replace('<details><summary>All 10 historical cohorts · improvements, regressions and limits</summary>', '<div id="methodAverages"><label class="metricPicker">Metric <select id="methodAverageMetric"><option value="camera_mean">Mean position error ↓</option><option value="camera_endpoint">Endpoint position error ↓</option><option value="rotation_mean">Mean rotation error ↓</option><option value="rotation_endpoint">Endpoint rotation error ↓</option><option value="lpips">LPIPS ↓</option><option value="dino">DINO similarity ↑</option></select></label>')
s=s.replace('<th>Mean position change</th><th>Lower / valid</th><th>Mean rotation change</th>','<th>Baseline mean</th><th>Method mean</th><th>Change</th><th>Improved / valid</th>')
s=s.replace('</tbody></table></div></details><p class="fine" id="legacySource">','</tbody></table></div><p class="fine" id="methodAverageNote"></p></div><p class="fine" id="legacySource">',1)
s=s.replace('<details id="legacyFailures"><summary>Failure investigation · commands, observers and memory</summary>', '<details id="legacyFailures"><summary>Case investigations · observations and causal limits</summary>')
s=s.replace('<h3 class="memoryEvidenceTitle">', '''<section id="failureScenarios" aria-labelledby="failureTitle"><div class="sectionhead"><div><span class="eyebrow">FAILURES, TRADEOFFS &amp; ABSTENTIONS</span><h2 id="failureTitle">Where it fails, when, and why</h2></div></div><p class="sectionLead">Every Method position regression and endpoint tradeoff is listed. Confirmed observations are separated from possible causes. Frame numbers refer to each run’s own timeline; “not reviewed” means the onset is unknown.</p><div class="recentFilters"><label>Model<select id="failureModel"><option value="all">All models</option></select></label><label>Intervention<select id="failureScope"><option value="all">Method + ablations / controls</option><option value="Method">Method only</option><option value="Ablation / control">Ablations / controls only</option></select></label><label>Diagnosis<select id="failureCertainty"><option value="all">All evidence levels</option></select></label><label>Scene / problem<input id="failureSearch" type="search" placeholder="Scene ID or symptom"/></label></div><p id="failureLedgerCount" class="fine" aria-live="polite"></p><div id="failureLedger" class="failureLedger"></div></section><h3 class="memoryEvidenceTitle">''',1)
s=s.replace('<a href="#workspace">Compare results</a>','<a href="#methodAverages">Averages</a><a href="#failureScenarios">Failures</a><a href="#workspace">Compare results</a>')
s=s.replace('<button data-mode="L">L · Method</button>','<button data-mode="L">Method</button>')
s=s.replace('<strong>Method selection is open.</strong> Method is a leading candidate based on historical results.','<strong>Method and ablations.</strong> Method denotes the rejoin policy shown in the averages above. Its historical results are retained; matched comparison against alternatives is still underway.')
s=s.replace('We are comparing these policies on matched runs before selecting the paper method.','Memory-only, bounded commands and online gain learning remain separately labelled ablations.')
s=s.replace('<script src="clarity.js"></script>','<script src="method-results.js"></script><script src="clarity.js"></script>')
s=s.replace('A separate future-command experiment is running; its first observer/sham control passed.','The separate future-command pilot is complete: moving-scene gains are modest and near-stationary outcomes are mixed.')
p.write_text(s)
p=D/'research.js';s=p.read_text().replace("tag:'Leading candidate · Method'","tag:'Method · command rejoin'").replace("title:'L rejoins the planned path through future commands.'","title:'The Method rejoins the planned path through future commands.'");p.write_text(s)
proof=dict(preserved_comparisons=len(data['pairs']),unchanged_arm_data=True,method_pairs=len(ids),cohorts=len(n['legacy']['rows']),
    failure_entries=len(ledger),position_regressions=21,endpoint_tradeoffs=8,mean_recomputation_passed=True,
    snapshot_sha256=hashlib.sha256((D/'snapshot.json.gz').read_bytes()).hexdigest())
(ROOT/'method_presentation_integrity.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(proof))
