"""Recompute legacy evidence from published paired records and verify source hashes."""
from pathlib import Path
import json,gzip,hashlib,time
W=Path('/home/hj/Alican');R=Path(__file__).resolve().parent
snapshot=W/'sites/camera-correction-results/dist/snapshot.json.gz';d=json.loads(gzip.decompress(snapshot.read_bytes()));summary=json.loads((W/'analysis/correction_summary_20261101.json').read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
specs=[(r['model'],r['cohort'],r['seed']) for r in summary['historical_vggt']]+[(r['model'],'rejoin_v3',7) for r in summary['current_vipe']]
rows=[]
for model,cohort,seed in specs:
 pairs=[p for p in d['pairs'] if (p['model'],p['cohort'],p['seed'])==(model,cohort,seed)];assert pairs
 sources=[]
 for p in pairs:
  source=W/p['source'];assert sha(source)==p['sha256'],source
  sources.append({'pair':p['id'],'path':str(source),'sha256':p['sha256'],'scene':p['scene'],'horizon':p['horizon'],'position_valid':p.get('position_valid')})
 metrics={}
 for key in ['camera_mean','camera_endpoint','rotation_mean','rotation_endpoint','lpips','dino']:
  v=[(p['arms']['baseline']['metrics'].get(key),p['arms']['correction']['metrics'].get(key)) for p in pairs]
  v=[(a,b) for a,b in v if a is not None and b is not None]
  if not v:continue
  a=sum(a for a,b in v)/len(v);b=sum(b for a,b in v)/len(v)
  metrics[key]={'valid_pairs':len(v),'baseline_mean':a,'correction_mean':b,'change_percent':100*(b/a-1) if a else None,'lower':sum(b<a for a,b in v),'higher':sum(b>a for a,b in v),'equal':sum(b==a for a,b in v)}
 rows.append({'model':model,'cohort':cohort,'seed':seed,'pairs':len(pairs),'estimator':sorted({p['estimator'] for p in pairs}),'metrics':metrics,'sources':sources})
out={'created':time.time(),'script_sha256':sha(Path(__file__)),'snapshot_sha256':sha(snapshot),'rows':rows,'interpretation':'Each row compares historical rejoin against its own native baseline. Neither differences across rows nor differences against later memory cohorts establish a matched policy ranking. All listed historical cohorts are retained, including regressions.','method_selection':'Freeze experimental protocols, not the final paper-method choice. Retain legacy rejoin as a central candidate; measured memory and online gain require demonstrated incremental value.'}
(R/'result.json').write_text(json.dumps(out,indent=2)+'\n')
print('Verified',len(rows),'cohort/seed rows and',sum(len(r['sources']) for r in rows),'source-backed comparisons')
