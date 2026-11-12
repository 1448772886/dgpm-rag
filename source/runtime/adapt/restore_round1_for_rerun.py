"""Reconstruct round1 memory from saved write receipts; verify retained historical vectors."""
import datetime,json,logging,os,shutil,sqlite3,sys
from pathlib import Path
ROOT=Path('/root/autodl-tmp/V16_run')
os.chdir(ROOT);sys.path.insert(0,str(ROOT))
from adapt.v16_config import configure_v16
configure_v16(ROOT)
import numpy as np
import chromadb
from adapt.gpu_embedding import GpuEmbedding
from ReMindRag.database.chromaDB import ChromaDBManager
OLD_RUN='v16_loogle_longdep_q36plus_149_20261109'
NEW_RUN='v16_loogle_longdep_q36plus_r2rerun_20261109'

def restore(ti):
    old=ROOT/'eval/experiments'/OLD_RUN/f'title_{ti}'
    dest=ROOT/'eval/database'/NEW_RUN/str(ti)
    source=Path('/root/autodl-tmp/shortdep_qwen36_graphs_20261109')/f'title_{ti}/round0'
    assert (source/'v15_build_complete.json').is_file()
    assert not dest.exists(),dest
    shutil.copytree(source,dest)
    client=chromadb.PersistentClient(path=str(dest/'chroma_data'))
    db=ChromaDBManager.__new__(ChromaDBManager)
    db.client=client;db.logger=logging.getLogger('restore')
    db.embedding=GpuEmbedding('nomic-ai/nomic-embed-text-v2-moe',str(ROOT/'model_cache'),device='cpu')
    db.relation_collection=client.get_collection('relation');db.connection_collection=client.get_collection('connection')
    manifest=json.loads((ROOT/'round1_replay_manifest_candidate.json').read_text())
    selected=sorted([x for x in manifest if x['title']==ti],key=lambda x:x['qi'])
    expected=len(list((old/'round1').glob('q*.json')))
    if ti==16:
        for qi in [1,2]:
            record=json.loads((old/f'round1/q{qi}.json').read_text())
            trace=json.loads((old/f'round1/traces/q{qi}/c2_trace.json').read_text())
            assert not trace.get('supported_chunks') and not trace.get('supported_entities') and not trace.get('supported_graph_edges')
            assert not (record.get('memory_update') or {}).get('enhanced_edges') and not (record.get('d1') or {}).get('d1_applied')
            selected.append({'title':ti,'qi':qi,'query':record['question'],'inner_enhance':[],'inner_punish':[],'outer_enhance':[],'outer_punish':[],'source_log':'verified_no_memory_write'})
        selected.sort(key=lambda x:x['qi'])
    assert len(selected)==expected,(ti,len(selected),expected)
    for x in selected:
        for key,method in [('inner_enhance',db.enhance_edge_weight),('inner_punish',db.punish_edge_weight),('outer_enhance',db.enhance_edge_weight),('outer_punish',db.punish_edge_weight)]:
            if x[key]:method(x['query'],x[key])
    cutoff=max(p.stat().st_mtime for p in (old/'round1').glob('q*.json'))
    stamp=datetime.datetime.fromtimestamp(cutoff,datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    original=ROOT/'eval/database'/OLD_RUN/str(ti)/'chroma_data/chroma.sqlite3'
    con=sqlite3.connect('file:'+str(original)+'?mode=ro',uri=True)
    names={str(uuid):name for uuid,name in con.execute('select id,name from collections')}
    historic={}
    for seq,topic,eid,vec,encoding in con.execute('select seq_id,topic,id,vector,encoding from embeddings_queue where operation=1 and created_at<=? order by seq_id',(stamp,)):
        name=names[topic.rsplit('/',1)[-1]]
        if name in ['relation','connection'] and vec:historic[(name,eid)]=np.frombuffer(vec,dtype=np.float32)
    con.close()
    mismatches=[]
    for name in ['relation','connection']:
        rows=client.get_collection(name).get(include=['embeddings'])
        current=dict(zip(rows['ids'],rows['embeddings']))
        for (coll,eid),v in historic.items():
            if coll!=name:continue
            error=float(np.max(np.abs(np.array(current[eid])-v)))
            if error>1e-5:mismatches.append({'type':name,'id':eid,'max_abs_error':error})
    audit={'title':ti,'questions_replayed':len(selected),'cutoff_utc':stamp,'historical_vectors_checked':len(historic),'mismatches':mismatches,'source_receipts':selected}
    (dest/'round1_restore_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in audit.items() if k not in ['source_receipts','mismatches']})+' mismatch_count='+str(len(mismatches)),flush=True)
    if mismatches:print(json.dumps(mismatches[:5],ensure_ascii=False),flush=True);return 2
    (dest/'round1_restore_verified.json').write_text(json.dumps({'title':ti,'verified':True,'historical_vectors_checked':len(historic)}))
    return 0
if __name__=='__main__':sys.exit(restore(int(sys.argv[1])))
