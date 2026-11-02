"""Policy outcomes and repeated-actuation qualification, including zero writes."""
import json
from pathlib import Path
R=Path(__file__).resolve().parent

def report():
 m=json.loads((R/'freeze.json').read_text());rows=[];state=json.loads((R/'status.json').read_text()) if (R/'status.json').exists() else {'phase':'pending','pairs':{}}
 for p in m['pairs']:
  f=Path(m['reuse_pairs'][p['key']]['score']) if p['key'] in m.get('reuse_pairs',{}) else R/'scores'/p['key']/'score.json';r={k:p[k] for k in ['key','model','scene','seed']};r['state']=state['pairs'].get(p['key'],{}).get('state','pending')
  if p['key'] in m.get('reuse_pairs',{}):r['visual_review']=m['reuse_pairs'][p['key']]['note']
  if f.exists():
   d=json.loads(f.read_text());r.update(accepted_writes=d['accepted_memory_writes'],rejected_writes=d['rejected_memory_writes'],identical_inputs=d['identical_observer_inputs'],post_frames=d['horizon']-d['first_affected_frame']+1,comparison_valid=d['comparison_valid'],delta=d['delta'])
  rows.append(r)
 qualifications={}
 for model in m['protocol']['models']:
  rs=[r for r in rows if r['model']==model];active=sorted({r['scene'] for r in rs if r.get('comparison_valid') is True and r.get('accepted_writes',0)>=2})
  qualifications[model]=dict(completed=sum('accepted_writes' in r for r in rs),multiwrite_scenes=active,qualified=len(active)>=2,zero_write_pairs=sum(r.get('accepted_writes')==0 for r in rs))
 out=dict(phase=state['phase'],rows=rows,actuation_qualification=qualifications,planned_pairs=len(rows),held_models=state.get('held_models',{}),failures=state.get('failures',[]),scope='All original 40 development pairs retained, including pending, holds, abstentions and visual collapse. Actuation qualification is coverage, not proof of efficacy; visual reviews remain required.')
 (R/'results.json').write_text(json.dumps(out,indent=2)+'\n');return out
if __name__=='__main__':print(json.dumps(report()))
