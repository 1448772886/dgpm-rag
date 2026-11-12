import json,sqlite3,collections,datetime,re
from pathlib import Path
import chromadb,numpy as np
from chromadb.api.client import SharedSystemClient
r=Path('/root/autodl-tmp/V16_run');run='v16_loogle_longdep_q36plus_r2rerun_20261109';base=r/'eval/experiments'/run;reports=json.loads((r/'round1_restore_full_structure_audit.json').read_text());print('STRUCTURE',sum(x['collections']['relation']['changed_vectors']+x['collections']['connection']['changed_vectors'] for x in reports),'changed_edges')
overlapq=overlape=0;changed_total=checked_changed=0;missing=[]
for ti in [i for i in range(20) if i!=13]:
 cp=r/f'eval/checkpoints/{run}/title_{ti}/round1';a=json.loads((cp/'round1_restore_audit.json').read_text())
 for rec in a['source_receipts']:
  ov=set(map(tuple,rec['inner_enhance'])) & set(map(tuple,rec['outer_enhance']))
  if ov:overlapq+=1;overlape+=len(ov)
 c=sqlite3.connect('file:'+str(r/f'eval/database/v16_loogle_longdep_q36plus_149_20261109/{ti}/chroma_data/chroma.sqlite3')+'?mode=ro',uri=True)
 names={str(k):n for k,n in c.execute('select id,name from collections')};hist={}
 for topic,eid,vec in c.execute('select topic,id,vector from embeddings_queue where operation=1 and created_at<=? order by seq_id',(a['cutoff_utc'],)):
  hist[(names[topic.rsplit('/',1)[-1]],eid)]=vec
 c.close()
 cl=chromadb.PersistentClient(path=str(cp/'chroma_data'))
 for name in ['relation','connection']:
  rows=cl.get_collection(name).get(include=['embeddings'])
  for eid,v in zip(rows['ids'],rows['embeddings']):
   if np.linalg.norm(v)>1e-8:
    changed_total+=1
    if (name,eid) in hist:checked_changed+=1
    else:missing.append([ti,name,eid])
 cl._system.stop();SharedSystemClient.clear_system_cache()
print('DOUBLE_ENHANCE_QUESTIONS',overlapq,'overlapping_edge_writes',overlape)
print('NONZERO_MEMORY_VECTORS',changed_total,'WITH_HISTORICAL_VECTOR',checked_changed,'WITHOUT_HISTORY',len(missing))
for ph in ['round1','round2','round3']:
 models=collections.Counter();quick=quickn=0
 for p in base.glob('title_*/'+ph+'/q*.json'):
  x=json.loads(p.read_text());models[x.get('judge_model')]+=1
  t=p.parent/'traces'/p.stem/'c2_trace.json'
  if t.exists():
   d=json.loads(t.read_text());n=len(d.get('quick_chunks',[]));quick+=n;quickn+=n>0
 print('PHASE',ph,'judge_models',dict(models),'questions_with_quick_chunks',quickn,'total_quick_chunks',quick)
(r/'round1_restore_additional_audit.json').write_text(json.dumps({'double_enhance_questions':overlapq,'double_enhance_edge_writes':overlape,'nonzero_memory_vectors':changed_total,'nonzero_with_historical_vector':checked_changed,'missing_historical_vectors':missing},indent=2))
