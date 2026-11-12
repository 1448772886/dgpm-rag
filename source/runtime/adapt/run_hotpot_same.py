"""Latest pinned V16, Hotpot same/origin 0..96, four independent round states."""
import os
for key in list(os.environ):
    if key.startswith('REMIND_'):del os.environ[key]
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
os.environ.update(TOKENIZERS_PARALLELISM='false',USE_TF='0',USE_FLAX='0',ANONYMIZED_TELEMETRY='False')
import argparse
import collections
import datetime
import fcntl
import gc
import hashlib
import json
import logging
import multiprocessing as mp
from pathlib import Path
import re
import shutil
import sqlite3
import string
import sys
import time
import traceback
from types import SimpleNamespace

RUNTIME=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RUNTIME))
ROOT=Path('/root/autodl-tmp/V16_run')
os.chdir(ROOT)
from adapt import run_neoqa_v16_200 as core
from adapt.v16_config import configure_v16,MODEL
from adapt.v16_memory import commit_question_memory,MEMORY_PROTOCOL
RUN='hotpot_same_v16_q36plus_newprompt_20261111'
OUT=ROOT/'eval/experiments'/RUN
WORK=ROOT/'eval/database'/RUN
SOURCE=ROOT/'eval/database/v16_hotpot_q36_j38flash_97_factv7_origin_20261108'
DATA=ROOT/'eval/dataset_cache/Hotpot/hotpot_dev_distractor_v1.json'
KEYFILE=ROOT/'adapt/.secrets/neoqa_deepseek_bare_20261111.json'
PHASES=('fresh','round1','round2','round3')
core.KEYFILE=KEYFILE
Agent,load,save,sha,emit,totals=core.Agent,core.load,core.save,core.sha,core.emit,core.totals


def normalize(answer):
    answer=answer.lower()
    answer=''.join(c for c in answer if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',answer).split())


def extract(raw):
    blocks=re.findall(r'```cot-ans\s*(.*?)```',raw,re.S|re.I)
    if blocks and blocks[-1].strip():return blocks[-1].strip()
    m=re.search(r'(?:^|\n)(?:Final\s+)?Answer\s*:\s*(.+?)\s*$',raw,re.I)
    if m:return m.group(1).strip()
    if '\n' not in raw.strip() and len(raw)<250:return raw.strip()
    raise RuntimeError('Final answer format could not be extracted')


def prepare():
    manifest=load(OUT/'pinned_runtime.json')
    for name,digest in manifest['sha256'].items():assert sha((RUNTIME/name).read_bytes())==digest,name
    rows=load(DATA)[:97];assert len(rows)==97 and len({r['id'] for r in rows})==97
    groups={};audit=[]
    for i,row in enumerate(rows):
        if i==74:continue
        source=SOURCE/str(i)/'chroma_data/chroma.sqlite3'
        with sqlite3.connect('file:'+str(source)+'?mode=ro',uri=True) as conn:
            assert conn.execute('pragma quick_check').fetchone()[0]=='ok',i
            assert dict(conn.execute('select name,dimension from collections').fetchall())==dict.fromkeys(['entity','chunk','relation','connection'],768)
            docs=[x[0] for x in conn.execute("select m.string_value from embedding_metadata m join embeddings e on e.id=m.id join segments s on s.id=e.segment_id join collections c on c.id=s.collection where c.name='chunk' and m.key='chroma:document'").fetchall()]
        text=' '.join(' '.join(docs).split())
        missing=[]
        for title,sentences in row['context']:
            sentences=[' '.join(s.split()) for s in sentences if len(' '.join(s.split()))>20]
            if sentences and not any(s in text for s in sentences):missing.append(title)
        assert not missing,(i,missing)
        groups[f'q{i:03d}']=[dict(index=i,question_id=row['id'],question=row['question'])]
        audit.append(dict(index=i,question_id=row['id'],chunks=len(docs),contexts=len(row['context']),
                          source_sqlite_sha256=sha(source.read_bytes()),context_coverage_passed=True))
    save(OUT/'graph_audit.json',audit)
    spec=dict(run=RUN,condition='same/origin',model=MODEL,judge_model=MODEL,
        source_dataset_sha256=sha(DATA.read_bytes()),planned_questions=97,eligible_questions=96,
        skipped_indices=[74],skip_reason='Previously confirmed sensitive question',phases=PHASES,
        workers=10,device='cpu',enable_thinking=False,max_jumps_per_question=10,
        request_timeout_seconds=180,retries=8,memory_protocol=MEMORY_PROTOCOL,
        graph_source=str(SOURCE),graph_restore='Clone existing content/topology; reset relation/connection vectors to original zero initialization; preserve entity/chunk vectors',
        fresh='Pristine restored graph, no memory writes',round1='Copy pristine fresh graph; one post-answer commit',
        round2='Inherit completed round1',round3='Inherit completed round2',
        question_input='Unchanged original question only; gold/supporting facts absent',
        scoring='Normalized exact match plus Qwen3.6Plus semantic equality after memory commit; report semantic accuracy and exact match separately',
        tokens='Provider prompt+completion; traversal, answer and scoring separately; no double reasoning count',
        runtime_source=manifest['source'],pinned_runtime_sha256=sha((OUT/'pinned_runtime.json').read_bytes()),
        mapping=[rows[0]['id']]+[x['id'] for x in rows[1:]])
    if (OUT/'manifest.json').exists():assert load(OUT/'manifest.json')==json.loads(json.dumps(spec))
    else:save(OUT/'manifest.json',spec)
    (OUT/'worker_logs').mkdir(exist_ok=True)
    return groups


def question(row,phase,embedding,tokenizer,db):
    from ReMindRag.generator.pathfinder import PathFinder
    from ReMindRag.generator.prompts import generate_rag_ans_prompt
    from adapt.v10_memory import record_result
    folder=OUT/'questions'/f"q{row['index']:03d}"/phase
    folder.mkdir(parents=True,exist_ok=True)
    for key,name in [('REMIND_C2_TRACE_OUT','c2_trace.json'),('REMIND_TEMP_SUMMARY_FILE','evidence_ledger.json'),('REMIND_EVIDENCE_TRACE','evidence.jsonl')]:os.environ[key]=str(folder/name)
    logger=logging.getLogger(f"hotpot.{row['index']}.{phase}");logger.setLevel(logging.DEBUG);logger.propagate=False
    handler=logging.FileHandler(folder/'traversal.log',encoding='utf-8');handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'));logger.addHandler(handler)
    agent=Agent(folder);started=time.time();commit_started=False
    try:
        db.logger=logger
        pf=PathFinder(agent,db,350,logging.WARNING,None);pf.logger=logger
        query=row['question']
        summary,edges=pf.get_query_ans(query,phase!='fresh',2,10)
        assert pf.last_jumps<=10
        save(folder/'retrieval.json',dict(query=query,jumps=pf.last_jumps,entities=pf.entity,chunks=pf.chunk,
            summary=summary,edges=edges,path=pf.path,visited_nodes=pf.c_node_list))
        save(folder/'retrieved_sources.json',db.chunk_collection.get(ids=list(dict.fromkeys(map(str,pf.chunk))),include=['documents','metadatas']) if pf.chunk else dict(ids=[],documents=[],metadatas=[]))
        prompt=generate_rag_ans_prompt.format(chat_history='',query=query,
            rag_summary='\n'.join(f'{k}:{v}' for k,v in summary.items()),edges=str(edges) if edges else '')
        agent.stage='answer'
        raw=agent.generate_response('Answer the question using the retrieved reference information. Put the concise final answer in one cot-ans block.',[dict(role='user',content=prompt)])
        answer=extract(raw)
        trace=getattr(pf,'c2_trace',None)
        if not trace:raise RuntimeError('V16 source trace missing')
        current,_=record_result(folder/'memory_record.json',phase,row['index'],-1,True,trace)
        commit_started=True
        audit=commit_question_memory(SimpleNamespace(database=db,preprocess=SimpleNamespace(path_finder=pf)),current)
        assert not(set(map(tuple,audit['enhanced_edges'])) & set(map(tuple,audit['punished_edges'])))
        save(folder/'memory_update.json',audit)
        # Gold is first accessed only after answer and memory updates.
        gold=load(DATA)[row['index']]['answer']
        exact=normalize(answer)==normalize(gold)
        judge_prompt=("Given one question, there is a groundtruth and a predict answer.\n"
            "Please decide whether they are the same or not in semantic.\n"
            "Please only output True or False.\n"
            f"Question: {query}\ngroundtruth = {gold}\npredicted answer = {answer}\n\n"
            "Only output one word(True or False), without any additional content.")
        agent.stage='scoring'
        judge=agent.generate_response('',[dict(role='user',content=judge_prompt)])
        if judge.strip() not in ('True','False'):raise RuntimeError('Invalid semantic grader output')
        result=dict(status='completed',phase=phase,index=row['index'],question_id=row['question_id'],
            question=query,raw_answer=raw,prediction=answer,gold_answer=gold,correct=judge.strip()=='True',
            exact_match=exact,judge_raw=judge,jumps=pf.last_jumps,retrieved_chunks=len(pf.chunk),memory_update=audit)
    except Exception as exc:
        if commit_started:raise
        error=str(exc).replace(agent.key,'[REDACTED]')[:1600]
        sensitive=bool(re.search('data_inspection|inappropriate|sensitive|content_filter|content.*safety',error,re.I))
        result=dict(status='skipped_sensitive' if sensitive else 'failed',phase=phase,index=row['index'],question_id=row['question_id'],error=error)
        (folder/'exception.txt').write_text(traceback.format_exc().replace(agent.key,'[REDACTED]'))
    finally:
        agent.client.close();logger.removeHandler(handler);handler.close()
    result.update(tokens_traversal=totals([x for x in agent.records if x['stage']=='traversal']),
        tokens_answer=totals([x for x in agent.records if x['stage']=='answer']),
        tokens_scoring=totals([x for x in agent.records if x['stage']=='scoring']),tokens_total=totals(agent.records),elapsed_s=time.time()-started)
    save(folder/'result.json',result)
    emit(dict(event='question_done',phase=phase,index=row['index'],status=result['status'],correct=result.get('correct')))
    if result['status']=='failed':raise RuntimeError(result['error'])
    return result


def child(group,phase,rows,embedding,tokenizer):
    with (OUT/'worker_logs'/f'{group}_{phase}.log').open('ab',buffering=0) as stream:
        os.dup2(stream.fileno(),1);os.dup2(stream.fileno(),2)
        from ReMindRag.database import ChromaDBManager
        from ReMindRag.utils.logger import trace
        import numpy as np
        logging.Logger.trace=trace
        target=WORK/group/f'{phase}.working'
        if target.exists():
            assert target.resolve().is_relative_to(WORK.resolve());shutil.rmtree(target)
        if phase=='fresh':source=SOURCE/str(rows[0]['index'])
        else:
            previous=PHASES[PHASES.index(phase)-1]
            source=WORK/group/previous
            assert (source/'phase_complete.json').exists()
        target.mkdir(parents=True)
        shutil.copytree(source/'chroma_data',target/'chroma_data')
        db=ChromaDBManager(logging.WARNING,None,embedding,None,0.7,0.1,0.5,str(target/'chroma_data'),tokenizer)
        try:
            if phase=='fresh':
                # Graph construction initializes both edge collections at zero.
                # Traversal changes only these vectors, never content/topology.
                restored={}
                for name in ['relation','connection']:
                    c=getattr(db,name+'_collection');ids=c.get(include=[])['ids']
                    for start in range(0,len(ids),200):
                        batch=ids[start:start+200];c.update(ids=batch,embeddings=np.zeros((len(batch),768),dtype=np.float32))
                    e=c.get(include=['embeddings'])['embeddings'];assert np.count_nonzero(e)==0
                    restored[name]=len(ids)
                assert db.entity_collection.count()>0 and db.chunk_collection.count()>0
                save(target/'pristine_restore.json',dict(index=rows[0]['index'],source=str(source),edge_counts=restored,edge_vectors_all_zero=True))
            results=[question(row,phase,embedding,tokenizer,db) for row in rows]
        finally:db.client.close()
        save(target/'phase_complete.json',dict(group=group,phase=phase,statuses=[x['status'] for x in results]))


def summarize():
    phases={}
    for phase in PHASES:
        rows=[load(f) for f in (OUT/'questions').glob(f'*/{phase}/result.json')]
        done=[r for r in rows if r['status']=='completed'];n=len(done)
        phases[phase]=dict(planned=97,eligible=96,completed=n,correct=sum(r['correct'] for r in done),
            accuracy=sum(r['correct'] for r in done)/n if n else None,exact_match=sum(r['exact_match'] for r in done),
            skipped_known_sensitive=1,skipped_sensitive=sum(r['status']=='skipped_sensitive' for r in rows),failed=sum(r['status']=='failed' for r in rows),
            avg_traversal_tokens=sum(r['tokens_traversal']['total_tokens'] for r in done)/n if n else None,
            avg_answer_tokens=sum(r['tokens_answer']['total_tokens'] for r in done)/n if n else None,
            avg_scoring_tokens=sum(r['tokens_scoring']['total_tokens'] for r in done)/n if n else None,
            avg_total_tokens=sum(r['tokens_total']['total_tokens'] for r in done)/n if n else None)
    save(OUT/'summary.json',dict(updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),phases=phases))
    return phases


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight-only',action='store_true');args=parser.parse_args()
    lock=(OUT/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    configure_v16(RUNTIME,api_key_file=KEYFILE)
    groups=prepare();emit(dict(event='preflight_passed',planned=97,eligible=96,workers=10));summarize()
    assert [extract('```cot-ans\nyes\n```'),extract('Answer: Paris')]==['yes','Paris']
    if args.preflight_only:return
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    from adapt.gpu_embedding import GpuEmbedding,resolve_local_model
    from transformers import AutoTokenizer
    embedding=GpuEmbedding('nomic-ai/nomic-embed-text-v2-moe',str(ROOT/'model_cache'),device='cpu',batch_size=8)
    tokenizer=AutoTokenizer.from_pretrained(resolve_local_model(embedding.model_name,str(ROOT/'model_cache')),local_files_only=True,trust_remote_code=True)
    assert embedding.get_hidden_state_size()==768 and embedding.model.max_seq_length==512
    from chromadb.api.client import SharedSystemClient
    assert not SharedSystemClient._identifier_to_system
    gc.collect();gc.freeze();ctx=mp.get_context('fork');pending=collections.deque()
    for group in groups:
        for pi,phase in enumerate(PHASES):
            if not(WORK/group/phase/'phase_complete.json').exists():pending.append((group,pi));break
    active={};events=[]
    while pending or active:
        while pending and len(active)<10:
            group,pi=pending.popleft();proc=ctx.Process(target=child,args=(group,PHASES[pi],groups[group],embedding,tokenizer));proc.start();active[proc.pid]=(proc,group,pi)
            emit(dict(event='worker_started',group=group,phase=PHASES[pi],pid=proc.pid))
        for pid,(proc,group,pi) in list(active.items()):
            if proc.is_alive():continue
            proc.join();phase=PHASES[pi];target=WORK/group/f'{phase}.working';success=proc.exitcode==0 and(target/'phase_complete.json').exists()
            if success:
                os.rename(target,WORK/group/phase)
                if pi+1<len(PHASES):pending.appendleft((group,pi+1))
            events.append(dict(group=group,phase=phase,exitcode=proc.exitcode,complete=success));save(OUT/'worker_events.json',events);emit(dict(event='worker_finished',**events[-1]));del active[pid]
        summarize();time.sleep(3)
    save(OUT/'run_completion.json',dict(finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),complete=all((WORK/g/ph/'phase_complete.json').exists() for g in groups for ph in PHASES),phases=summarize()))


if __name__=='__main__':main()
