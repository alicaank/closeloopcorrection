"""Read-only audit of failed WorldPlay identity control; MP4 is not raw RGB."""
from pathlib import Path
import hashlib,json,subprocess,time
import numpy as np
R=Path(__file__).resolve().parent;W=R.parents[1];Q=W/'analysis/online_response_v2_resume_20261103';FF=W/'analysis/plan_base_restart_20261008/model_controls_20261024/bin/ffmpeg'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
paths={arm:Q/f'runs/worldplay/{arm}/0238de14_s7.json' for arm in ('A','S')};d={a:json.loads(p.read_text()) for a,p in paths.items()}
fields=['planned','native_commanded','unit','settings','pose_frame_ids','K_out','out_hw','frames_in_mp4','preprocess']
equality={k:d['A'][k]==d['S'][k] for k in fields};assert all(equality.values())
commands={a:np.array(x['native_commanded']) for a,x in d.items()};np.testing.assert_array_equal(commands['A'],commands['S'])
hashes={a:x['online_response']['prefix_hashes'] for a,x in d.items()};assert len(hashes['A'])==len(hashes['S'])==20
checks=[{'boundary':i,'frame_id':x['frame_id'],'raw_prefix_equal':x==y,'A':x['sha256'],'S':y['sha256']} for i,(x,y) in enumerate(zip(hashes['A'],hashes['S']))]
first=next(x for x in checks if not x['raw_prefix_equal']);assert first['frame_id']==60
for a,x in d.items():assert not any(e['applied'] or e['commands_differ_from_bootstrap'] for e in x['online_response']['events'])
# Stream two lossy exports; never retain full native-resolution clips in RAM.
H,B=d['A']['out_hw'];frame_bytes=H*B*3;procs={};videos={}
for a,x in d.items():
 p=paths[a].with_suffix('.mp4');videos[a]={'path':str(p),'sha256':sha(p)}
 procs[a]=subprocess.Popen([str(FF),'-v','error','-threads','1','-i',str(p),'-an','-threads','1','-f','rawvideo','-pix_fmt','rgb24','pipe:1'],stdout=subprocess.PIPE)
def read_frame(stream):
 chunks=[];total=0
 while total<frame_bytes:
  b=stream.read(frame_bytes-total)
  if not b:break
  chunks.append(b);total+=len(b)
 assert total in (0,frame_bytes),total
 return b''.join(chunks)
rows=[]
try:
 while True:
  raw={a:read_frame(p.stdout) for a,p in procs.items()}
  if not any(raw.values()):break
  assert all(raw.values())
  x=np.frombuffer(raw['A'],np.uint8);y=np.frombuffer(raw['S'],np.uint8);delta=np.abs(x.astype(np.int16)-y)
  rows.append({'frame_id':len(rows),'mae_255':float(delta.mean()),'max_channel_difference_255':int(delta.max()),'changed_channel_fraction':float(np.mean(delta!=0))})
 for p in procs.values():assert p.wait()==0
finally:
 for p in procs.values():
  if p.poll() is None:p.terminate();p.wait()
assert len(rows)==317,len(rows)
result={'audited_at':time.time(),'scope':'WorldPlay seed 7, scene 0238de14, A versus S, original 20 chunks; failed exact identity control retained.','script_sha256':sha(Path(__file__)),'source_manifest_sha256':sha(Q/'manifest.json'),'source_logs':{a:{'path':str(p),'sha256':sha(p)} for a,p in paths.items()},'metadata_equality':equality,'native_command_shape':list(commands['A'].shape),'command_sha256':sha(paths['A']),'active_command_changes':0,'raw_prefix_checks':checks,'last_equal_raw_checkpoint':44,'first_unequal_raw_checkpoint':60,'first_raw_difference_interval_inclusive':[45,60],'retrieval_selection_first_chunk':4,'first_differing_checkpoint_chunk':3,'retrieval_scope':'Chunk 3 uses the complete fixed prefix range; geometric retrieval starts at chunk 4, current_frame_idx >= 16.','videos':videos,'export_differences':rows,'export_summary':{'frames':len(rows),'mean_mae_255':float(np.mean([r['mae_255'] for r in rows])),'max_frame_mae_255':max(r['mae_255'] for r in rows),'first_nonidentical_export_frame':next((r['frame_id'] for r in rows if r['max_channel_difference_255']),None)},'limitations':['Export comparisons use CRF-12 MP4 and cannot locate the first raw RGB difference or quantify a raw regeneration floor.','Raw checkpoint hashes establish divergence but do not identify the cause.','Identical saved commands do not prove all native intermediate tensors or RNG states were identical.','No adaptive WorldPlay C run is qualified; do not relax exact identity based on a small export difference.']}
result['native_command_sha256']=hashlib.sha256(commands['A'].tobytes()).hexdigest();result.pop('command_sha256')
(R/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['export_summary']))
