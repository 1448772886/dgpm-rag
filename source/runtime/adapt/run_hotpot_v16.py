"""Run origin, similar and different HotpotQA through the same V16 pipeline.

One 20-worker pool is shared by all 291 questions. Each question's graph is
built once from the original context and cloned before memory rounds so each
query condition has an independent graph-weight state.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import gc
import json
import multiprocessing
import os
import shutil
import time
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

from adapt.v16_config import (JUDGE_MODEL, MAX_JUMPS, MODEL, WORKERS,
                              REQUEST_RETRIES, REQUEST_TIMEOUT_SECONDS,
                              configure_v16, validate_v16_environment)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "eval" / "dataset_cache" / "Hotpot"
CONDITIONS = ("origin", "similar", "different")
PHASES = ("fresh", "round1", "round2", "round3")
RUNS = {
    "origin": "v16_hotpot_q36_j38flash_97_factv7_origin_20261108",
    "similar": "v16_hotpot_q36_j38flash_97_factv7_similar_20261108",
    "different": "v16_hotpot_q36_j38flash_97_factv7_different_20261108",
}
DATASETS = {
    "origin": "hotpot_dev_distractor_v1.json",
    "similar": "hotpot_dev_distractor_similar.json",
    "different": "hotpot_dev_distractor_different.json",
}


def _is_sensitive(message: str) -> bool:
    text = message.lower()
    return any(term in text for term in (
        "sensitive", "content filter", "content policy", "risk control",
        "inappropriate content", "safety violation",
    ))


def _clone_graph(source: Path, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)


def _run_condition(comp, qi: int, condition: str, phases=PHASES) -> None:
    from adapt.eval_hotpot_adapted import run_single
    from adapt.metered_llm import MeteredOpenaiAgent
    from adapt.v10_memory import record_result
    from adapt.v16_memory import MEMORY_PROTOCOL, commit_question_memory

    run_name = RUNS[condition]
    out = ROOT / "eval" / "experiments" / run_name / f"q{qi}"
    trace_root = out / "traces"
    out.mkdir(parents=True, exist_ok=True)
    trace_root.mkdir(parents=True, exist_ok=True)
    for previous in out.glob("round[123].json"):
        saved = json.loads(previous.read_text(encoding="utf-8"))
        if (saved.get("memory_update") or {}).get("protocol") != MEMORY_PROTOCOL:
            raise RuntimeError("Legacy memory updates detected; start a new run from round0 graphs")
    if (out / "SKIPPED.json").exists() or (out / "FAILED.json").exists():
        return

    for phase in phases:
        result_path = out / f"{phase}.json"
        if result_path.is_file():
            continue
        attempt = trace_root / phase
        attempt.mkdir(parents=True, exist_ok=True)
        os.environ["REMIND_C2_TRACE_OUT"] = str(attempt / "c2_trace.json")
        os.environ["REMIND_C2_RECORD_OUT"] = str(attempt / "c2_record.json")
        MeteredOpenaiAgent.reset_usage()
        try:
            result = run_single(comp, qi, run_name, condition,
                                do_update=(phase != "fresh"),
                                max_jumps=MAX_JUMPS)
        except Exception as exc:
            sensitive = _is_sensitive(str(exc))
            (out / ("SKIPPED.json" if sensitive else "FAILED.json")).write_text(
                json.dumps({"question_index": qi, "condition": condition,
                            "phase": phase,
                            "status": "skipped_sensitive" if sensitive else "failed",
                            "error": str(exc)[:2000]}, ensure_ascii=False, indent=2),
                encoding="utf-8")
            return

        trace_file = attempt / "c2_trace.json"
        trace = json.loads(trace_file.read_text(encoding="utf-8")) if trace_file.is_file() else {
            "query": comp["all_data"][qi]["question"], "enough": False,
            "supported_chunks": [], "paths_by_chunk": {}, "quick_chunks": [],
            "quick_chunk_entry_edges": {}, "structured_judge_valid": False,
        }
        current, _ = record_result(attempt / "c2_record.json", phase, qi, -1, True, trace)
        evidence_audit = d1_audit = None
        if phase != "fresh":
            evidence_audit = commit_question_memory(comp["_last_rag"], current)
            d1_audit = evidence_audit["d1"]
        row = {
            "question_index": qi, "condition": condition, "phase": phase,
            "id": comp["all_data"][qi]["id"],
            "question": (comp["similar_data"][qi]["question"] if condition == "similar"
                         else comp["different_data"][qi]["question"] if condition == "different"
                         else comp["all_data"][qi]["question"]),
            "answer": (comp["similar_data"][qi]["answer"] if condition == "similar"
                       else comp["different_data"][qi]["answer"] if condition == "different"
                       else comp["all_data"][qi]["answer"]),
            "correct": bool(result.get("correct")),
            "tokens_retrieval": result.get("tokens_retrieval", {}),
            "tokens_total": result.get("tokens_total", result.get("tokens_retrieval", {})),
            "model": MODEL, "judge_model": JUDGE_MODEL,
            "traversal": "fact_v7", "c2_trace_valid": bool(trace.get("structured_judge_valid")),
            "memory_update": evidence_audit, "d1": d1_audit,
        }
        result_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")


def _worker(qi: int) -> dict:
    os.chdir(ROOT)
    configure_v16(ROOT)
    validate_v16_environment()
    # A previous pool crash may have left a transient FAILED marker. It is
    # safe to remove it here because phase JSON files are never overwritten.
    for condition in CONDITIONS:
        failed = ROOT / "eval" / "experiments" / RUNS[condition] / f"q{qi}" / "FAILED.json"
        if failed.exists():
            failed.unlink()
    from adapt.eval_hotpot_adapted import build_components

    comp = build_components(
        backbone=MODEL, judge_model=JUDGE_MODEL, build_backbone=MODEL,
        api_key_file=str(ROOT / "adapt" / ".secrets" / "qwen36plus_bailian.json"),
        build_disable_thinking=True, traverse_disable_thinking=True,
    )
    # Build the original context graph once, before any memory update.
    _run_condition(comp, qi, "origin", ("fresh",))
    source_db = ROOT / "eval" / "database" / RUNS["origin"] / str(qi)
    # Release the Chroma handles before copying its persisted graph snapshot.
    comp.pop("_last_rag", None)
    gc.collect()
    for condition in ("similar", "different"):
        target_db = ROOT / "eval" / "database" / RUNS[condition] / str(qi)
        if not target_db.exists() and source_db.exists():
            _clone_graph(source_db, target_db)
    _run_condition(comp, qi, "origin", ("round1", "round2", "round3"))
    _run_condition(comp, qi, "similar", PHASES)
    _run_condition(comp, qi, "different", PHASES)
    return {"question_index": qi, "conditions": list(CONDITIONS)}


def _question_complete(qi: int) -> bool:
    """Only resubmit questions whose four phases are still incomplete."""
    for condition in CONDITIONS:
        out = ROOT / "eval" / "experiments" / RUNS[condition] / f"q{qi}"
        if any((out / f"{phase}.json").is_file() for phase in PHASES):
            # A partial question must still be resumed; all four phases are
            # required unless the question was explicitly skipped.
            if not all((out / f"{phase}.json").is_file() for phase in PHASES):
                return False
        elif not (out / "SKIPPED.json").is_file():
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=96)
    args = parser.parse_args()
    if args.workers != WORKERS:
        raise SystemExit(f"V16 requires exactly {WORKERS} global workers; got {args.workers}")
    if args.start < 0 or args.end < args.start:
        raise SystemExit("Invalid question range")
    configure_v16(ROOT)
    validate_v16_environment()
    for condition, filename in DATASETS.items():
        data = json.loads((DATA_DIR / filename).read_text(encoding="utf-8"))
        if len(data) < args.end + 1:
            raise RuntimeError(f"{filename} has only {len(data)} rows")
    indices = list(range(args.start, args.end + 1))
    print(json.dumps({"questions_per_condition": len(indices),
                      "conditions": list(CONDITIONS), "workers_global": args.workers,
                      "phases": list(PHASES), "traversal": "fact_v7",
                      "b1": 2, "a5_gap_fallback": True,
                      "model": MODEL, "judge_model": JUDGE_MODEL,
                      "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
                      "request_retries": REQUEST_RETRIES}, ensure_ascii=False),
          flush=True)
    # Rebuild the pool after every batch.  Each batch has at most one task per
    # worker, so a native model/Chroma crash cannot poison a long-lived pool
    # carrying dozens of questions.  A broken batch is retried from disk; the
    # per-question phase files make the retry idempotent.
    pending = [qi for qi in indices if not _question_complete(qi)]
    ctx = multiprocessing.get_context("spawn")
    batch_no = 0
    while pending:
        batch_no += 1
        batch = pending[:args.workers]
        print(json.dumps({"batch": batch_no, "size": len(batch),
                          "workers": args.workers}, ensure_ascii=False), flush=True)
        futures = {}
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=min(args.workers, len(batch)),
                                                       mp_context=ctx) as pool:
                futures = {pool.submit(_worker, qi): qi for qi in batch}
                for future in concurrent.futures.as_completed(futures):
                    try:
                        print(json.dumps(future.result(), ensure_ascii=False), flush=True)
                    except Exception as exc:
                        print(json.dumps({"question_index": futures[future],
                                          "status": "failed", "error": repr(exc)},
                                         ensure_ascii=False), flush=True)
        except (BrokenProcessPool, OSError) as exc:
            print(json.dumps({"batch": batch_no, "status": "pool_restart",
                              "error": repr(exc)}, ensure_ascii=False), flush=True)
            time.sleep(2)
        pending = [qi for qi in indices if not _question_complete(qi)]
        # A provider outage or permanently crashing question must not spin
        # forever; its own FAILED/SKIPPED marker remains available for audit.
        if len(pending) >= len(batch) and batch_no >= 12:
            raise RuntimeError(f"V16 retry budget exhausted; pending={pending}")


if __name__ == "__main__":
    main()



