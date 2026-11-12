"""NeoQA 200 x four V16 rounds, isolated by timeline and temporal snapshot.

Ten fork children share CPU embedding weights. A child owns one snapshot/round,
then exits before its graph is checkpointed. No gold data enters model requests.
Successful API responses are saved before use, including messages and usage.
"""
from __future__ import annotations

import os
for name in list(os.environ):
    if name.startswith('REMIND_'):
        del os.environ[name]
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
os.environ.update(TOKENIZERS_PARALLELISM='false', USE_TF='0', USE_FLAX='0',
                  ANONYMIZED_TELEMETRY='False')

import argparse
import collections
import datetime
import gc
import hashlib
import inspect
import json
import logging
import multiprocessing as mp
from pathlib import Path
import re
import shutil
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from adapt.v16_config import configure_v16, MODEL, REQUEST_TIMEOUT_SECONDS, REQUEST_RETRIES
from adapt.v16_memory import commit_question_memory, MEMORY_PROTOCOL

RUN = 'neoqa_v16_q36plus_200x4_20261111'
OUT = ROOT / 'eval/experiments' / RUN
WORK = ROOT / 'eval/database' / RUN
PACKAGE = Path('/root/autodl-tmp/neoqa_packages/neoqa_remindrag_five_200_20261110')
KEYFILE = ROOT / 'adapt/.secrets/neoqa_qwen_20261111.json'
PHASES = ('fresh', 'round1', 'round2', 'round3')
SYSTEM = ('Answer the multiple-choice question using the retrieved reference information. '
          'Select one of the provided options. Return only the option letter A, B, C, D, E, F or G.')


def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def save(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, p)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def emit(row):
    print(json.dumps(row, ensure_ascii=False), flush=True)


def parse(raw):
    blocks = re.findall(r'```cot-ans\s*(.*?)```', raw, re.S | re.I)
    text = (blocks[-1] if blocks else raw).strip().replace('**', '')
    # The upstream answer prompt requests analysis. Accept an unambiguous
    # terminal answer line, while never mining letters from the analysis.
    text = text.splitlines()[-1].strip() if text else ''
    match = re.fullmatch(r'(?:(?:Final\s+)?(?:Answer|Choice)\s*:\s*)?\(?([A-G])\)?[.。]?', text, re.I)
    return ord(match.group(1).upper()) - 65 if match else -1


def totals(records):
    keys = ('calls', 'prompt_tokens', 'completion_tokens', 'total_tokens', 'reasoning_tokens')
    return {k: sum(r['usage'].get(k, 0) or 0 for r in records) for k in keys}


class Agent:
    """API attempts follow V16 timeout/retry policy; saved successes are replayable."""
    def __init__(self, folder):
        from openai import OpenAI
        config = load(KEYFILE)
        self.key = config['api_key']
        self.client = OpenAI(**config, timeout=REQUEST_TIMEOUT_SECONDS, max_retries=0)
        self.folder = folder / 'calls'
        self.folder.mkdir(parents=True, exist_ok=True)
        self.ordinal, self.stage, self.records = 0, 'traversal', []

    def generate_response(self, system_prompt, chat_history, max_tokens=None):
        self.ordinal += 1
        messages = ([dict(role='system', content=system_prompt)] if system_prompt else []) + chat_history
        request = dict(model=MODEL, messages=messages, temperature=0, seed=123,
                       extra_body={'enable_thinking': False})
        if max_tokens is not None:
            request['max_tokens'] = max_tokens
        digest = sha(json.dumps(request, sort_keys=True, ensure_ascii=False).encode())
        path = self.folder / f'{self.ordinal:04d}.json'
        if path.exists():
            row = load(path)
            if row['request_sha256'] != digest or row['stage'] != self.stage:
                raise RuntimeError('Saved trajectory differs; refusing to mix API replay with changed traversal')
        else:
            for attempt in range(REQUEST_RETRIES + 1):
                try:
                    start = time.time()
                    response = self.client.chat.completions.create(**request)
                    choice = response.choices[0]
                    u = response.usage
                    if u is None:
                        raise RuntimeError('Provider response omitted usage; cannot report valid token metrics')
                    reasoning = getattr(getattr(u, 'completion_tokens_details', None), 'reasoning_tokens', 0) or 0
                    row = dict(ordinal=self.ordinal, stage=self.stage, request_sha256=digest,
                               messages=messages, raw=choice.message.content or '', model=response.model,
                               finish_reason=choice.finish_reason, elapsed_s=time.time()-start,
                               enable_thinking=False, reasoning_content_present=bool(getattr(choice.message, 'reasoning_content', None)),
                               usage=dict(calls=1, prompt_tokens=u.prompt_tokens, completion_tokens=u.completion_tokens,
                                          total_tokens=u.total_tokens, reasoning_tokens=reasoning))
                    save(path, row)
                    break
                except Exception as exc:
                    # Reject invalid credentials/billing/content blocks without retry storms.
                    msg = str(exc).replace(self.key, '[REDACTED]')[:1200]
                    with (self.folder.parent / 'api_errors.jsonl').open('a') as f:
                        f.write(json.dumps(dict(at=time.time(), ordinal=self.ordinal, attempt=attempt,
                                                error=msg, status_code=getattr(exc, 'status_code', None))) + '\n')
                    if getattr(exc, 'status_code', None) in (400, 401, 402, 403, 404) or attempt == REQUEST_RETRIES:
                        raise RuntimeError(msg) from None
                    time.sleep(min(2 * 2**attempt, 60))
        self.records.append(row)
        if row['reasoning_content_present'] or row['usage']['reasoning_tokens']:
            raise RuntimeError('Unexpected thinking output in non-thinking experiment')
        if row['finish_reason'] == 'length' or not row['raw'].strip():
            raise RuntimeError('Empty or truncated model response')
        emit(dict(question=self.folder.parent.name, call=self.ordinal, stage=self.stage,
                  tokens=row['usage']['total_tokens']))
        return row['raw']


def question(row, phase, embedding, tokenizer, db):
    from ReMindRag.generator.pathfinder import PathFinder
    from ReMindRag.generator.prompts import generate_rag_ans_prompt
    from adapt.v10_memory import record_result
    folder = OUT / 'questions' / f"q{row['index']:03d}" / phase
    folder.mkdir(parents=True, exist_ok=True)
    for key, filename in [('REMIND_C2_TRACE_OUT', 'c2_trace.json'),
                          ('REMIND_TEMP_SUMMARY_FILE', 'evidence_ledger.json'),
                          ('REMIND_EVIDENCE_TRACE', 'evidence.jsonl')]:
        os.environ[key] = str(folder / filename)
    logger = logging.getLogger(f"neoqa.{row['index']}.{phase}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    handler = logging.FileHandler(folder / 'traversal.log', encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(handler)
    agent = Agent(folder)
    started, commit_started = time.time(), False
    try:
        db.logger = logger
        pf = PathFinder(agent, db, 350, logging.WARNING, None)
        pf.logger = logger
        options = '\n'.join(f'{chr(65+i)}. {x}' for i, x in enumerate(row['options']))
        query = f"Date of question: {row['date']}\nQuestion: {row['question']}\n\nOptions:\n{options}"
        summary, edges = pf.get_query_ans(query, phase != 'fresh', 2, 10)
        assert pf.last_jumps <= 10, 'Question exceeded shared 10-hop budget'
        save(folder / 'retrieval.json', dict(query=query, jumps=pf.last_jumps,
             entities=pf.entity, chunks=pf.chunk, summary=summary, edges=edges,
             path=pf.path, visited_nodes=pf.c_node_list))
        retrieved = db.chunk_collection.get(ids=list(dict.fromkeys(map(str, pf.chunk))),
                        include=['documents', 'metadatas']) if pf.chunk else dict(ids=[], documents=[], metadatas=[])
        save(folder / 'retrieved_sources.json', retrieved)
        prompt = generate_rag_ans_prompt.format(chat_history='', query=query,
                 rag_summary='\n'.join(f'{k}:{v}' for k, v in summary.items()), edges=str(edges) if edges else '')
        agent.stage = 'answer'
        raw = agent.generate_response(SYSTEM, [dict(role='user', content=prompt)])
        choice = parse(raw)
        trace = getattr(pf, 'c2_trace', None)
        if not trace:
            raise RuntimeError('Required V16 source trace missing')
        current, _ = record_result(folder / 'memory_record.json', phase, row['index'], choice, choice >= 0, trace)
        # Gold is excluded from memory. Commit only after a valid model answer.
        if choice >= 0:
            commit_started = True
            audit = commit_question_memory(SimpleNamespace(database=db,
                    preprocess=SimpleNamespace(path_finder=pf)), current)
            assert not (set(map(tuple, audit['enhanced_edges'])) & set(map(tuple, audit['punished_edges'])))
        else:
            audit = dict(protocol=MEMORY_PROTOCOL, skip_reason='invalid_option', enhanced_edges=[], punished_edges=[])
        save(folder / 'memory_update.json', audit)
        gold = next(x for x in load(PACKAGE / 'timelines' / row['graph_id'] / 'gold.json')
                    if x['question_id'] == row['question_id'])
        result = dict(status='completed', phase=phase, index=row['index'], question_id=row['question_id'],
                      graph_id=row['graph_id'], snapshot=row['snapshot'], date=row['date'],
                      raw_answer=raw, predicted_option_1based=choice+1 if choice>=0 else None,
                      gold_option_1based=gold['answer_option_1based'], correct=choice+1 == gold['answer_option_1based'],
                      invalid_choice=choice<0, jumps=pf.last_jumps, retrieved_chunks=len(pf.chunk), memory_update=audit)
    except Exception as exc:
        error = str(exc).replace(agent.key, '[REDACTED]')[:1600]
        if commit_started:
            # Never continue with a possibly half-written memory transaction.
            raise
        sensitive = bool(re.search(r'data_inspection|inappropriate|sensitive|content_filter|content.*safety', error, re.I))
        result = dict(status='skipped_sensitive' if sensitive else 'failed', phase=phase,
                      index=row['index'], question_id=row['question_id'], graph_id=row['graph_id'], error=error)
        (folder / 'exception.txt').write_text(traceback.format_exc().replace(agent.key, '[REDACTED]'))
    finally:
        agent.client.close()
        logger.removeHandler(handler)
        handler.close()
    result.update(tokens_traversal=totals([x for x in agent.records if x['stage']=='traversal']),
                  tokens_answer=totals([x for x in agent.records if x['stage']=='answer']),
                  tokens_total=totals(agent.records), elapsed_s=time.time()-started)
    save(folder / 'result.json', result)
    emit(dict(event='question_done', phase=phase, index=row['index'], status=result['status'],
              correct=result.get('correct'), tokens=result['tokens_total']['total_tokens']))
    # Credential/billing errors stop this job, rather than failing all 200 silently.
    if result['status']=='failed' and re.search(r'invalid.*key|insufficient|quota|arrearage|401|402', result['error'], re.I):
        raise RuntimeError(result['error'])
    return result


def child(group, phase, rows, embedding, tokenizer):
    log = OUT / 'worker_logs' / f'{group}_{phase}.log'
    with log.open('ab', buffering=0) as stream:
        os.dup2(stream.fileno(), 1)
        os.dup2(stream.fileno(), 2)
        from adapt.restore_neoqa_graph import restore
        from ReMindRag.database import ChromaDBManager
        from ReMindRag.utils.logger import trace
        logging.Logger.trace = trace
        target = WORK / group / f'{phase}.working'
        # Rebuild the round transaction on restart; replay API successes, then
        # reapply memory in fixed question order. Never resume a half commit.
        if target.exists():
            assert target.resolve().is_relative_to(WORK.resolve())
            shutil.rmtree(target)
        if phase in ('fresh', 'round1'):
            restore(PACKAGE / rows[0]['snapshot'], target)
        else:
            previous = PHASES[PHASES.index(phase)-1]
            source = WORK / group / previous
            assert (source / 'phase_complete.json').exists()
            shutil.copytree(source, target)
            (target / 'phase_complete.json').unlink()
        # Independently reopen and verify every restored collection's content/vector
        # equality for pristine rounds before the first model request.
        if phase in ('fresh', 'round1'):
            import chromadb
            import numpy as np
            client = chromadb.PersistentClient(path=str(target))
            for name in ('entity', 'chunk', 'relation', 'connection'):
                expected = load(PACKAGE / rows[0]['snapshot'] / f'{name}.json')
                vectors = np.load(PACKAGE / rows[0]['snapshot'] / f'{name}_vectors.npz')['vectors']
                coll = client.get_collection(name)
                assert coll.count() == len(expected['ids'])
                for start in range(0, len(expected['ids']), 250):
                    ids = expected['ids'][start:start+250]
                    got = coll.get(ids=ids, include=['documents', 'metadatas', 'embeddings'])
                    byid = {v:i for i,v in enumerate(got['ids'])}
                    for off, key in enumerate(ids):
                        j = byid[key]
                        assert got['documents'][j] == expected['documents'][start+off]
                        assert got['metadatas'][j] == expected['metadatas'][start+off]
                        assert np.allclose(got['embeddings'][j], vectors[start+off], rtol=0, atol=1e-7)
            client.close()
            emit(dict(event='snapshot_verified', group=group, phase=phase))
        db = ChromaDBManager(logging.WARNING, None, embedding, None, 0.7, 0.1, 0.5, str(target), tokenizer)
        try:
            results = [question(q, phase, embedding, tokenizer, db) for q in rows]
        finally:
            db.client.close()
        save(target / 'phase_complete.json', dict(group=group, phase=phase,
             question_indices=[q['index'] for q in rows], statuses=[r['status'] for r in results]))


def summarize():
    report = {}
    for phase in PHASES:
        rows = [load(p) for p in (OUT / 'questions').glob(f'*/{phase}/result.json')]
        done = [x for x in rows if x['status']=='completed']
        n = len(done)
        report[phase] = dict(planned=200, completed=n, correct=sum(x['correct'] for x in done),
            accuracy=sum(x['correct'] for x in done)/n if n else None,
            failed=sum(x['status']=='failed' for x in rows),
            skipped_sensitive=sum(x['status']=='skipped_sensitive' for x in rows),
            invalid_choice=sum(x['invalid_choice'] for x in done),
            avg_traversal_tokens=sum(x['tokens_traversal']['total_tokens'] for x in done)/n if n else None,
            avg_answer_tokens=sum(x['tokens_answer']['total_tokens'] for x in done)/n if n else None,
            avg_total_tokens=sum(x['tokens_total']['total_tokens'] for x in done)/n if n else None,
            total_tokens_all_attempted_questions=sum(x['tokens_total']['total_tokens'] for x in rows))
    save(OUT / 'summary.json', dict(updated_at=datetime.datetime.now().isoformat(), phases=report))
    return report


def prepare():
    rows = load(PACKAGE / 'questions200.json')
    # Package indices are 1..40 separately within each timeline.
    # Preserve those labels while assigning globally unique output identities.
    for index, row in enumerate(rows, 1):
        row['source_index'], row['index'] = row['index'], index
    assert len(rows)==len({q['question_id'] for q in rows})==len({q['index'] for q in rows})==200
    assert set(collections.Counter(q['graph_id'] for q in rows).values()) == {40}
    groups = collections.defaultdict(list)
    for q in sorted(rows, key=lambda x:x['index']):
        assert len(q['options'])==7 and isinstance(q['date'], str)
        snapshot = (PACKAGE / q['snapshot']).resolve()
        assert snapshot.is_relative_to(PACKAGE.resolve()) and snapshot.is_dir()
        assert snapshot.name == f"event_{q['cutoff_event']:02d}"
        news = load(PACKAGE / 'timelines' / q['graph_id'] / 'news.json')
        allowed = {a['article_id'] for a in news if a['created_at'] <= q['cutoff_event']}
        chunks = load(snapshot / 'chunk.json')
        assert all(m['article_id'] in allowed for m in chunks['metadatas'])
        gold = next(g for g in load(PACKAGE / 'timelines' / q['graph_id'] / 'gold.json') if g['question_id']==q['question_id'])
        assert 1 <= gold['answer_option_1based'] <= 7
        groups[f"{q['graph_id']}_{snapshot.name}"].append(q)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'worker_logs').mkdir(exist_ok=True)
    manifest = dict(run=RUN, model=MODEL, enable_thinking=False, workers=10,
        phases=PHASES, questions=200, answers=800, max_jumps_per_question=10,
        timeout_seconds=REQUEST_TIMEOUT_SECONDS, retries=REQUEST_RETRIES,
        memory_protocol=MEMORY_PROTOCOL, snapshot_groups=len(groups),
        memory_scope='One independent working graph per timeline/cutoff snapshot. Fixed index order within group. No cross-date memory.',
        fresh='Pristine graph, no writes', round1='Pristine graph, one post-answer memory commit',
        round2='Inherit completed round1 graph', round3='Inherit completed round2 graph',
        retrieval_input='Question date + unchanged question + A-G options; no gold or evidence annotations',
        scoring='Local exact 1-based option equality, no LLM answer judge',
        token_metrics='Provider prompt + completion tokens. Traversal and final generation separated. Reasoning tokens not added twice.',
        groups={k:[q['index'] for q in v] for k,v in groups.items()},
        question_mapping=[dict(index=q['index'], source_index=q['source_index'],
                              graph_id=q['graph_id'], question_id=q['question_id']) for q in rows],
        question_sha256=sha((PACKAGE / 'questions200.json').read_bytes()),
        code_sha256={str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in
          [Path(__file__), ROOT/'adapt/v16_config.py', ROOT/'adapt/v16_memory.py',
           ROOT/'ReMindRag/generator/pathfinder.py', ROOT/'ReMindRag/generator/fact_v7.py',
           ROOT/'ReMindRag/generator/e1_seed_selection.py', ROOT/'ReMindRag/database/chromaDB.py']})
    path = OUT / 'manifest.json'
    if path.exists():
        assert load(path)==json.loads(json.dumps(manifest)), 'Run inputs or code changed'
    else:
        save(path, manifest)
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary-only', action='store_true')
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    if args.summary_only:
        emit(summarize()); return
    configure_v16(ROOT, api_key_file=KEYFILE)
    from ReMindRag.generator.e1_seed_selection import score_second_seed
    assert score_second_seed(list('abcde'), [0,1,2,3,4], {'a':set('bcde')})[:2] == ('a','b')
    assert [parse(x) for x in ['A','G','Answer: F','The answer could be A or B','H']] == [0,6,5,-1,-1]
    groups = prepare()
    emit(dict(event='preflight_passed', questions=200, groups=len(groups)))
    if args.preflight_only:
        return
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from adapt.gpu_embedding import GpuEmbedding, resolve_local_model
    from transformers import AutoTokenizer
    emit(dict(event='loading_shared_cpu_embedding'))
    embedding = GpuEmbedding('nomic-ai/nomic-embed-text-v2-moe', str(ROOT/'model_cache'), device='cpu', batch_size=8)
    tokenizer = AutoTokenizer.from_pretrained(resolve_local_model(embedding.model_name, str(ROOT/'model_cache')),
                                              local_files_only=True, trust_remote_code=True)
    assert embedding.get_hidden_state_size()==768 and embedding.model.max_seq_length==512
    from chromadb.api.client import SharedSystemClient
    assert not SharedSystemClient._identifier_to_system
    gc.collect(); gc.freeze()
    ctx = mp.get_context('fork')
    pending = collections.deque()
    for group in groups:
        for pi, phase in enumerate(PHASES):
            if not (WORK / group / phase / 'phase_complete.json').is_file():
                pending.append((group, pi)); break
    active, events = {}, []
    while pending or active:
        while pending and len(active)<10:
            group, pi = pending.popleft()
            process = ctx.Process(target=child, args=(group, PHASES[pi], groups[group], embedding, tokenizer))
            process.start()
            active[process.pid] = process, group, pi
            emit(dict(event='worker_started', group=group, phase=PHASES[pi], pid=process.pid))
        for pid, (process, group, pi) in list(active.items()):
            if process.is_alive():
                continue
            process.join()
            phase = PHASES[pi]
            target = WORK / group / f'{phase}.working'
            success = process.exitcode==0 and (target/'phase_complete.json').is_file()
            if success:
                os.rename(target, WORK / group / phase)
                if pi+1<len(PHASES):
                    pending.appendleft((group, pi+1))
            events.append(dict(group=group, phase=phase, exitcode=process.exitcode, complete=success))
            save(OUT/'worker_events.json', events)
            emit(dict(event='worker_finished', **events[-1]))
            del active[pid]
        summarize()
        time.sleep(3)
    save(OUT/'run_completion.json', dict(finished_at=datetime.datetime.now().isoformat(),
          complete=all((WORK/g/p/'phase_complete.json').exists() for g in groups for p in PHASES), phases=summarize()))
    emit(dict(event='run_finished'))


if __name__=='__main__':
    main()
