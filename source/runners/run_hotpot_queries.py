"""Hotpot similar/different: frozen same-run V16, one global ten-worker queue."""
from pathlib import Path
import argparse
import collections
import datetime
import fcntl
import gc
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import sqlite3
import sys
import time

ROOT = Path('/root/autodl-tmp/V16_run')
RUN = 'hotpot_queries_v16_q36plus_newprompt_20261112'
OUT = ROOT / 'eval/experiments' / RUN
WORK = ROOT / 'eval/database' / RUN
SAME = ROOT / 'eval/experiments/hotpot_same_v16_q36plus_newprompt_20261111'
SAME_WORK = ROOT / 'eval/database/hotpot_same_v16_q36plus_newprompt_20261111'
RUNTIME = OUT / 'runtime'
SOURCE_VIEW = WORK / 'pristine_sources'
KEYFILE = ROOT / 'adapt/.secrets/loogle_qwen_next_20261111.json'
CONDITIONS = ('similar', 'different')
PHASES = ('fresh', 'round1', 'round2', 'round3')
DATA_ROOT = ROOT / 'eval/dataset_cache/Hotpot'
WORKERS = 10


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def emit(**data):
    print(json.dumps(data, ensure_ascii=False), flush=True)


def prepare():
    if not RUNTIME.exists():
        shutil.copytree(SAME / 'runtime', RUNTIME, ignore=shutil.ignore_patterns('__pycache__'))
    pinned = read(SAME / 'pinned_runtime.json')
    for name, expected in pinned['sha256'].items():
        assert digest(RUNTIME / name) == expected, ('runtime differs from same', name)
    assert digest(RUNTIME / 'adapt/run_hotpot_same.py') == digest(SAME / 'runtime/adapt/run_hotpot_same.py')
    original_path = DATA_ROOT / 'hotpot_dev_distractor_v1.json'
    original = read(original_path)[:97]
    variants = {c: read(DATA_ROOT / f'hotpot_dev_distractor_{c}.json')[:97] for c in CONDITIONS}
    assert len(original) == 97
    for c, rows in variants.items():
        assert len(rows) == 97
        assert all(isinstance(x['question'], str) and x['question'].strip() and isinstance(x['answer'], str) for x in rows)
    audits = []
    SOURCE_VIEW.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(original):
        if i == 74:
            continue
        source = SAME_WORK / f'q{i:03d}' / 'fresh'
        assert (source / 'phase_complete.json').exists()
        dbfile = source / 'chroma_data/chroma.sqlite3'
        with sqlite3.connect('file:' + str(dbfile) + '?mode=ro', uri=True) as con:
            assert con.execute('pragma quick_check').fetchone()[0] == 'ok'
            assert dict(con.execute('select name,dimension from collections')) == dict.fromkeys(['entity', 'chunk', 'relation', 'connection'], 768)
            docs = [x[0] for x in con.execute("select m.string_value from embedding_metadata m join embeddings e on e.id=m.id join segments s on s.id=e.segment_id join collections c on c.id=s.collection where c.name='chunk' and m.key='chroma:document'")]
        norm = lambda x: ' '.join(x.split())
        text = norm(' '.join(docs))
        context = dict(row['context'])
        missing = [(title, si) for title, si in row['supporting_facts'] if norm(context[title][si]) not in text]
        assert not missing, (i, 'support sentences missing', missing)
        link = SOURCE_VIEW / str(i)
        if link.is_symlink():
            assert link.resolve() == source.resolve()
        else:
            assert not link.exists()
            link.symlink_to(source, target_is_directory=True)
        audits.append(dict(index=i, question_id=row['id'], graph=str(source), sqlite_sha256=digest(dbfile),
                           chunks=len(docs), support_sentences=len(row['supporting_facts']), support_coverage_passed=True))
    manifest = dict(run=RUN, conditions=CONDITIONS, phases=PHASES, workers_global=WORKERS,
        question_indices=list(range(97)), skipped_indices=[74], eligible_per_condition=96,
        runtime_source=str(SAME / 'runtime'), runtime_sha256=pinned['sha256'],
        same_runner_sha256=digest(RUNTIME / 'adapt/run_hotpot_same.py'),
        graph_source=str(SAME_WORK), graph_rebuilt=False,
        graph_protocol='Identical pristine same/fresh graph content, topology and embeddings; independent condition/round copies. Fresh edge vectors reset to original zero initialization.',
        memory_protocol='fresh: no memory writes; round1: copy fresh and commit once after answer; round2/3: inherit preceding phase of the same query condition.',
        variant_alignment='First 97 author-provided variant rows aligned by original row index; no gold rewriting or question generation.',
        original_dataset_sha256=digest(original_path),
        datasets={c: dict(path=str(DATA_ROOT / f'hotpot_dev_distractor_{c}.json'), sha256=digest(DATA_ROOT / f'hotpot_dev_distractor_{c}.json')) for c in CONDITIONS},
        model='qwen3.6-plus', judge_model='qwen3.6-plus', enable_thinking=False,
        max_jumps_per_question=10, timeout_seconds=180, retries=8,
        token_protocol='Provider prompt+completion, traversal/answer/scoring separately; no graph-building calls.',
        scoring='Condition-specific gold accessed only after answer and memory commit, using unchanged same-run semantic grader and normalized EM.')
    if (OUT / 'manifest.json').exists():
        assert read(OUT / 'manifest.json') == json.loads(json.dumps(manifest))
    else:
        save(OUT / 'manifest.json', manifest)
    save(OUT / 'graph_audit.json', audits)
    save(OUT / 'question_mapping.json', [dict(index=i, original_id=row['id'], original_question=row['question'],
        variants={c: variants[c][i] for c in CONDITIONS}, skipped=i == 74) for i, row in enumerate(original)])
    return {f'q{i:03d}': [dict(index=i, question_id=row['id'], question='')] for i, row in enumerate(original) if i != 74}, variants


def bind(engine, condition):
    engine.RUN = RUN + '_' + condition
    engine.OUT = OUT / condition
    engine.WORK = WORK / condition
    engine.SOURCE = SOURCE_VIEW
    engine.DATA = DATA_ROOT / f'hotpot_dev_distractor_{condition}.json'
    engine.KEYFILE = KEYFILE
    engine.core.KEYFILE = KEYFILE
    (engine.OUT / 'worker_logs').mkdir(parents=True, exist_ok=True)


def summarize(engine):
    result = {}
    for condition in CONDITIONS:
        bind(engine, condition)
        result[condition] = engine.summarize()
    save(OUT / 'summary.json', dict(updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), conditions=result))
    return result


def worker(engine, condition, group, phase, rows, embedding, tokenizer):
    bind(engine, condition)
    engine.child(group, phase, rows, embedding, tokenizer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / 'run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    groups, variants = prepare()
    sys.path.insert(0, str(RUNTIME))
    from adapt import run_hotpot_same as engine
    engine.configure_v16(RUNTIME, api_key_file=KEYFILE)
    engine.core.KEYFILE = KEYFILE
    save(OUT / 'supervisor.pid.json', dict(pid=os.getpid(), host='region-42.seetacloud.com', port=19065))
    summarize(engine)
    emit(event='preflight_passed', graph_count=len(groups), conditions=CONDITIONS, workers_global=WORKERS,
         runtime_identical_to_same=True, shared_question_budget=10)
    if args.preflight_only:
        return
    agent = engine.Agent(OUT / 'preflight_api')
    try:
        reply = agent.generate_response('', [dict(role='user', content='Reply only OK.')], max_tokens=16)
        emit(event='api_preflight_passed', model='qwen3.6-plus', reply=reply, tokens=engine.totals(agent.records))
    finally:
        agent.client.close()
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from adapt.gpu_embedding import GpuEmbedding, resolve_local_model
    from transformers import AutoTokenizer
    embedding = GpuEmbedding('nomic-ai/nomic-embed-text-v2-moe', str(ROOT / 'model_cache'), device='cpu', batch_size=8)
    tokenizer = AutoTokenizer.from_pretrained(resolve_local_model(embedding.model_name, str(ROOT / 'model_cache')), local_files_only=True, trust_remote_code=True)
    assert embedding.get_hidden_state_size() == 768 and embedding.model.max_seq_length == 512
    from chromadb.api.client import SharedSystemClient
    assert not SharedSystemClient._identifier_to_system
    gc.collect()
    gc.freeze()
    ctx = mp.get_context('fork')
    pending = collections.deque()
    for group in groups:
        for condition in CONDITIONS:
            for pi, phase in enumerate(PHASES):
                if not (WORK / condition / group / phase / 'phase_complete.json').exists():
                    pending.append((condition, group, pi))
                    break
    active, events = {}, []
    while pending or active:
        while pending and len(active) < WORKERS:
            condition, group, pi = pending.popleft()
            rows = [dict(row, question=variants[condition][row['index']]['question']) for row in groups[group]]
            proc = ctx.Process(target=worker, args=(engine, condition, group, PHASES[pi], rows, embedding, tokenizer))
            proc.start()
            active[proc.pid] = (proc, condition, group, pi)
            emit(event='worker_started', condition=condition, group=group, phase=PHASES[pi], pid=proc.pid, active_global=len(active))
        for pid, (proc, condition, group, pi) in list(active.items()):
            if proc.is_alive():
                continue
            proc.join()
            phase = PHASES[pi]
            target = WORK / condition / group / f'{phase}.working'
            success = proc.exitcode == 0 and (target / 'phase_complete.json').exists()
            if success:
                os.rename(target, WORK / condition / group / phase)
                if pi + 1 < len(PHASES):
                    pending.appendleft((condition, group, pi + 1))
            event = dict(condition=condition, group=group, phase=phase, exitcode=proc.exitcode, complete=success)
            events.append(event)
            save(OUT / 'worker_events.json', events)
            emit(event='worker_finished', **event)
            del active[pid]
        summarize(engine)
        time.sleep(3)
    summary = summarize(engine)
    complete = all((WORK / c / g / ph / 'phase_complete.json').exists() for c in CONDITIONS for g in groups for ph in PHASES)
    save(OUT / 'run_completion.json', dict(finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), complete=complete, conditions=summary))


if __name__ == '__main__':
    main()
