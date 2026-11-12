import json,collections,sqlite3
from pathlib import Path
import chromadb,numpy as np
from chromadb.api.client import SharedSystemClient
r=Path('/root/autodl-tmp/V16_run');run='v16_loogle_longdep_q36plus_r2rerun_20261109';old=r/'eval/experiments/v16_loogle_longdep_q36plus_149_20261109';reports=[]
for ti in [i for i in range(20) if i!=13]:
 base=chromadb.PersistentClient(path=str(Path('/root/autodl-tmp/shortdep_qwen36_graphs_20261109')/f'title_{ti}/round0/chroma_data'))
 snap=r/f'eval/checkpoints/{run}/title_{ti}/round1';rest=chromadb.PersistentClient(path=str(snap/'chroma_data'))
 a=json.loads((snap/'round1_restore_audit.json').read_text());expected=set()
 for e in a['source_receipts']:
  for key in ['inner_enhance','inner_punish','outer_enhance','outer_punish']:expected.update(map(tuple,e[key]))
 rep={'title':ti,'collections':{},'expected_edge_ids':len(expected),'unexpected_changes':[],'enhanced_but_zero':[]}
 for name in ['entity','chunk','relation','connection']:
  b=base.get_collection(name).get(include=['embeddings','documents','metadatas']);s=rest.get_collection(name).get(include=['embeddings','documents','metadatas'])
  bm={k:(v,d,m) for k,v,d,m in zip(b['ids'],b['embeddings'],b['documents'],b['metadatas'])};sm={k:(v,d,m) for k,v,d,m in zip(s['ids'],s['embeddings'],s['documents'],s['metadatas'])};changed=[]
  for k,(v,d,m) in sm.items():
   if k not in bm:rep['unexpected_changes'].append([name,k,'added']);continue
   bv,bd,bmeta=bm[k]
   if not np.array_equal(v,bv):
    changed.append(k)
    if (name,k) not in expected:rep['unexpected_changes'].append([name,k,'vector'])
   if d!=bd or m!=bmeta:rep['unexpected_changes'].append([name,k,'text_metadata'])
  rep['collections'][name]={'base':len(bm),'restored':len(sm),'changed_vectors':len(changed),'deleted':len(set(bm)-set(sm))}
  if name in ['relation','connection']:
   for receipt in a['source_receipts']:
    for edge in receipt['inner_enhance']+receipt['outer_enhance']:
     if edge[0]==name and (edge[1] not in sm or np.linalg.norm(sm[edge[1]][0])<1e-8):rep['enhanced_but_zero'].append(edge)
 reports.append(rep);print(json.dumps(rep),flush=True)
 for client in [base,rest]:client._system.stop()
 SharedSystemClient.clear_system_cache()
(r/'round1_restore_full_structure_audit.json').write_text(json.dumps(reports,indent=2))
