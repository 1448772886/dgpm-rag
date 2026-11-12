"""Run three clean V16 memory rounds with ten CPU workers sharing model pages.

Only the parent loads/warmups the embedding model. Linux fork workers share
its read-only parameters; each opens its own graph and owns all mutable query
state. Workers exit after one article/round, before their graph is snapshotted.
"""
from __future__ import annotations

import collections
import datetime
import gc
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from adapt.v16_config import BUILD_MODEL, JUDGE_MODEL, MODEL, WORKERS, configure_v16
from adapt.v16_memory import MEMORY_PROTOCOL

RUN = "v16_loogle_longdep_q36plus_singlecommit_20261110"
FRESH_SOURCE = "v16_loogle_longdep_q36plus_149_20261109"
GRAPH_MASTER = Path("/root/autodl-tmp/shortdep_qwen36_graphs_20261109")
STAGE = ROOT / "loogle_longdep_v16_20261109_data"
OUT = ROOT / "eval/experiments" / RUN
DB = ROOT / "eval/database" / RUN
CHECKPOINTS = ROOT / "eval/checkpoints" / RUN
LOGS = ROOT / "loogle_singlecommit_worker_logs"
PHASES = ("round1", "round2", "round3")


def emit(value):
    print(json.dumps(value, ensure_ascii=False), flush=True)


def child_main(ti, phase, components):
    log = LOGS / f"title_{ti}_{phase}.log"
    with log.open("ab", buffering=0) as handle:
        os.dup2(handle.fileno(), 1)
        os.dup2(handle.fileno(), 2)
        from adapt import run_loogle_longdep_v16 as runner
        runner.RUN = RUN
        runner.OUT = OUT
        try:
            result = runner.run_title(ti, phases=(phase,), components=components)
            emit(result)
            if result.get("status") != "complete":
                raise RuntimeError("Article/round did not complete")
        except BaseException:
            import traceback
            traceback.print_exc()
            raise


def verify_phase(ti, phase, expected):
    folder = OUT / f"title_{ti}"
    skipped = {int(p.stem.split("q")[-1]) for p in folder.glob("SKIPPED_q*.json")}
    missing = []
    for qi in range(expected):
        if qi in skipped:
            continue
        path = folder / phase / f"q{qi}.json"
        if not path.is_file():
            missing.append(qi)
            continue
        row = json.loads(path.read_text())
        audit = row.get("memory_update") or {}
        assert audit.get("protocol") == MEMORY_PROTOCOL, path
        assert not (set(map(tuple, audit["enhanced_edges"])) &
                    set(map(tuple, audit["punished_edges"]))), path
    if missing:
        raise RuntimeError(f"title_{ti} {phase}: missing questions {missing}")
    return len(skipped)


def snapshot(ti, phase):
    target = CHECKPOINTS / f"title_{ti}" / phase
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(DB / str(ti), target)
    (target / "snapshot_complete.json").write_text(json.dumps({
        "title": ti, "phase": phase, "memory_protocol": MEMORY_PROTOCOL,
        "at": datetime.datetime.now().isoformat(),
    }))


def prepare():
    if OUT.exists() or DB.exists() or CHECKPOINTS.exists():
        raise RuntimeError("Run directory already exists; refusing to mix or overwrite results")
    titles, expected = [], {}
    OUT.mkdir(parents=True)
    DB.mkdir(parents=True)
    LOGS.mkdir(exist_ok=True)
    for ti in range(20):
        source = GRAPH_MASTER / f"title_{ti}/round0"
        target = OUT / f"title_{ti}"
        target.mkdir()
        if not (source / "v15_build_complete.json").is_file():
            (target / "SKIPPED_GRAPH_INCOMPLETE.json").write_text(json.dumps({
                "title_index": ti, "reason": "Qwen3.6Plus round0 graph incomplete"}))
            continue
        rows = json.loads((STAGE / f"choices/title_{ti}.json").read_text())
        expected[ti] = len(rows)
        titles.append(ti)
        shutil.copytree(source, DB / str(ti))
        previous = ROOT / "eval/experiments" / FRESH_SOURCE / f"title_{ti}"
        assert len(list((previous / "fresh").glob("q*.json"))) == len(rows), ti
        shutil.copytree(previous / "fresh", target / "fresh")
        summary = previous / "fresh_summary.json"
        if summary.is_file():
            shutil.copy2(summary, target / summary.name)
    (OUT / "run_manifest.json").write_text(json.dumps({
        "run": RUN, "fresh_source": FRESH_SOURCE,
        "memory_protocol": MEMORY_PROTOCOL, "workers": WORKERS,
        "model": MODEL, "judge_model": JUDGE_MODEL,
        "rerun_phases": list(PHASES), "valid_questions": sum(expected.values()),
        "graph_source": str(GRAPH_MASTER),
        "embedding": "shared read-only CPU model pages via Linux fork",
        "started": datetime.datetime.now().isoformat(),
    }, indent=2))
    return titles, expected


def main():
    configure_v16(ROOT)
    assert WORKERS == 10
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from adapt.eval_hotpot_adapted import build_components
    emit({"status": "warming_shared_embedding", "workers": WORKERS})
    components = build_components(
        backbone=MODEL, judge_model=JUDGE_MODEL, build_backbone=BUILD_MODEL,
        api_key_file=str(ROOT / "adapt/.secrets/qwen36plus_bailian.json"),
        build_disable_thinking=True, traverse_disable_thinking=True)
    components["embedding"].sentence_embedding("example")
    # No database or network connection has been opened before forking.
    from chromadb.api.client import SharedSystemClient
    assert not SharedSystemClient._identifier_to_system
    gc.collect()
    gc.freeze()
    titles, expected = prepare()
    ctx = multiprocessing.get_context("fork")
    pending = collections.deque((ti, 0) for ti in titles)
    active, results = {}, []
    emit({"status": "started", "workers": WORKERS, "titles": titles,
          "valid_questions": sum(expected.values()), "memory_protocol": MEMORY_PROTOCOL})
    while pending or active:
        while pending and len(active) < WORKERS:
            ti, phase_index = pending.popleft()
            phase = PHASES[phase_index]
            process = ctx.Process(target=child_main, args=(ti, phase, components))
            process.start()
            active[process.pid] = (process, ti, phase_index)
            emit({"status": "started", "title": ti, "phase": phase, "pid": process.pid})
        finished = False
        for pid, (process, ti, phase_index) in list(active.items()):
            if process.is_alive():
                continue
            process.join()
            phase = PHASES[phase_index]
            result = {"title": ti, "phase": phase, "exit": process.exitcode}
            try:
                if process.exitcode:
                    raise RuntimeError(f"Worker exited with code {process.exitcode}")
                result["skipped_sensitive"] = verify_phase(ti, phase, expected[ti])
                snapshot(ti, phase)
                result["status"] = "complete"
                if phase_index + 1 < len(PHASES):
                    # Continue this article with a fresh child, releasing all
                    # database and per-question state from the previous round.
                    pending.appendleft((ti, phase_index + 1))
            except Exception as exc:
                result.update(status="failed", error=repr(exc))
            results.append(result)
            emit(result)
            del active[pid]
            finished = True
            (OUT / "phase_completion.json").write_text(json.dumps(results, indent=2))
        if not finished:
            time.sleep(1)
    (OUT / "run_completion.json").write_text(json.dumps(results, indent=2))
    emit({"status": "finished", "completed_article_rounds":
          sum(x["status"] == "complete" for x in results),
          "failed_article_rounds": sum(x["status"] != "complete" for x in results)})


if __name__ == "__main__":
    main()
