from __future__ import annotations

import concurrent.futures
import json
import multiprocessing
import os
import re
import shutil
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from adapt.v16_config import (BUILD_MODEL, JUDGE_MODEL, MAX_JUMPS, MODEL, WORKERS,
                              configure_v16)

RUN = "v16_loogle_longdep_q36plus_149_20261109"
STAGE = ROOT / "loogle_longdep_v16_20261109_data"
OUT = ROOT / "eval" / "experiments" / RUN
GRAPH_MASTER = Path("/root/autodl-tmp/shortdep_qwen36_graphs_20261109")
PHASES = ("fresh", "round1", "round2", "round3")


def sensitive(text: str) -> bool:
    low = text.lower()
    return any(x in low for x in ("sensitive", "content filter", "content policy",
                                  "risk control", "safety violation", "inappropriate content"))


def run_title(ti: int, *, phases=None, components=None) -> dict:
    phases = PHASES if phases is None else tuple(phases)
    qdir = OUT / f"title_{ti}"
    qdir.mkdir(parents=True, exist_ok=True)
    from adapt.v16_memory import MEMORY_PROTOCOL
    for previous in qdir.glob("round[123]/q*.json"):
        saved = json.loads(previous.read_text(encoding="utf-8"))
        if (saved.get("memory_update") or {}).get("protocol") != MEMORY_PROTOCOL:
            raise RuntimeError("Legacy memory updates detected; start a new run from round0 graphs")
    graph_dir = GRAPH_MASTER / f"title_{ti}" / "round0"
    marker_path = graph_dir / "v15_build_complete.json"
    if not marker_path.is_file():
        (qdir / "SKIPPED_GRAPH_INCOMPLETE.json").write_text(
            json.dumps({"title_index": ti,
                        "reason": "Qwen3.6Plus source graph is incomplete; no fallback graph will be built."},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        return {"title_index": ti, "status": "skipped_graph_incomplete"}
    configure_v16(ROOT)
    from adapt.eval_hotpot_adapted import build_components, run_single
    from adapt.metered_llm import MeteredOpenaiAgent
    from adapt.v10_memory import record_result
    from adapt.v16_memory import commit_question_memory

    title = json.loads((STAGE / "corpus_manifest.json").read_text(encoding="utf-8"))["titles"][ti]
    rows = json.loads((STAGE / "choices" / f"title_{ti}.json").read_text(encoding="utf-8"))
    ctx_file = STAGE / "contexts" / f"{ti}.txt"
    context = ctx_file.read_text(encoding="utf-8")
    db_name = RUN
    dbdir = ROOT / "eval" / "database" / db_name / str(ti)
    # Linux workers may inherit the warmed CPU model from the parent before
    # opening any Chroma database or making API requests. Parameters stay
    # read-only; question state and memory writes belong to each child.
    comp = components if components is not None else build_components(
        backbone=MODEL, judge_model=JUDGE_MODEL, build_backbone=BUILD_MODEL,
        api_key_file=str(ROOT / "adapt" / ".secrets" / "qwen36plus_bailian.json"),
        build_disable_thinking=True, traverse_disable_thinking=True)
    comp["all_data"] = [{"id": f"{ti}-{i}", "question": "", "answer": "",
                         "context": context, "supporting_facts": []} for i in range(20)]
    total = {phase: {"done": 0, "correct": 0, "prompt_tokens": 0, "completion_tokens": 0}
             for phase in phases}

    for phase in phases:
        phase_dir = qdir / phase
        phase_dir.mkdir(parents=True, exist_ok=True)
        do_update = phase != "fresh"
        for qi, row in enumerate(rows):
            if (qdir / f"SKIPPED_q{qi}.json").is_file():
                continue
            result_path = phase_dir / f"q{qi}.json"
            if result_path.is_file():
                saved = json.loads(result_path.read_text(encoding="utf-8"))
                s = total[phase]
                s["done"] += 1
                s["correct"] += int(saved.get("correct", False))
                s["prompt_tokens"] += saved.get("tokens_retrieval", {}).get("prompt_tokens", 0)
                s["completion_tokens"] += saved.get("tokens_retrieval", {}).get("completion_tokens", 0)
                continue
            rawq = str(row["question"])
            matches = list(re.finditer(r"(?m)^\s*([A-D])[.)]\s+(.+?)\s*$", rawq))
            if len(matches) != 4:
                raise ValueError(f"{title} q{qi}: expected four choices, found {len(matches)}")
            stem = rawq[:matches[0].start()].strip()
            options = [m.group(2).strip() for m in matches]
            gold = str(row.get("answer", "")).strip().upper()
            if gold not in "ABCD":
                raise ValueError(f"{title} q{qi}: invalid gold choice {gold!r}")
            trace_dir = phase_dir / "traces" / f"q{qi}"
            trace_dir.mkdir(parents=True, exist_ok=True)
            os.environ["REMIND_LOOGLE_CONTEXT_FILE"] = str(ctx_file)
            os.environ["REMIND_LOOGLE_TITLE"] = title
            os.environ["REMIND_C2_TRACE_OUT"] = str(trace_dir / "c2_trace.json")
            os.environ["REMIND_C2_RECORD_OUT"] = str(trace_dir / "c2_record.json")
            MeteredOpenaiAgent.reset_usage()
            try:
                result = run_single(comp, ti, db_name, "origin", do_update=do_update,
                                    max_jumps=MAX_JUMPS, override_query=stem,
                                    override_answer=options["ABCD".index(gold)])
                retrieval_usage = result["tokens_retrieval"]
                answer_data = json.loads((dbdir / "input.json").read_text(encoding="utf-8"))
                evidence = answer_data.get("response", "")
                opt_prompt = ("Answer the multiple-choice question using only the retrieved evidence. "
                              "Return exactly one letter: A, B, C, or D.\n\n"
                              f"Question: {stem}\n" + "\n".join(f"{c}. {o}" for c, o in zip("ABCD", options)) +
                              f"\n\nRetrieved evidence:\n{evidence}")
                before = MeteredOpenaiAgent.get_usage()
                selection = comp["generate_agent"].generate_response("", [{"role": "user", "content": opt_prompt}])
                after = MeteredOpenaiAgent.get_usage()
                choice_match = re.fullmatch(r"\s*([A-Da-d])[.。]?\s*", selection or "")
                chosen = "ABCD".find(choice_match.group(1).upper()) if choice_match else -1
                correct = chosen == "ABCD".index(gold)
                trace_file = trace_dir / "c2_trace.json"
                trace = json.loads(trace_file.read_text(encoding="utf-8")) if trace_file.is_file() else {}
                current, _ = record_result(trace_dir / "c2_record.json", phase, qi,
                                           chosen, chosen >= 0, trace)
                update_audit = d1_audit = None
                if do_update:
                    update_audit = commit_question_memory(comp["_last_rag"], current)
                    d1_audit = update_audit["d1"]
                mc_usage = {k: after.get(k, 0) - before.get(k, 0) for k in after}
                out = {"title_index": ti, "title": title, "question_index": qi,
                       "phase": phase, "question": stem, "options": options,
                       "answer": gold, "prediction": "ABCD"[chosen] if chosen >= 0 else "",
                       "correct": correct, "raw_selection": selection,
                       "tokens_retrieval": retrieval_usage, "tokens_mc_selection": mc_usage,
                       "model": MODEL, "judge_model": JUDGE_MODEL,
                       "build_model": BUILD_MODEL, "traversal": "fact_v7",
                       "memory_update": update_audit, "d1": d1_audit}
                result_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
                s = total[phase]
                s["done"] += 1
                s["correct"] += int(correct)
                s["prompt_tokens"] += retrieval_usage.get("prompt_tokens", 0)
                s["completion_tokens"] += retrieval_usage.get("completion_tokens", 0)
                print(json.dumps({"title_index": ti, "question_index": qi,
                                  "phase": phase, "correct": correct,
                                  "traversal_tokens": retrieval_usage.get("prompt_tokens", 0) +
                                                      retrieval_usage.get("completion_tokens", 0)},
                                 ensure_ascii=False), flush=True)
            except Exception as exc:
                marker = "SKIPPED.json" if sensitive(str(exc)) else "FAILED.json"
                (qdir / marker).write_text(json.dumps({"title": title, "question_index": qi,
                                                       "phase": phase, "error": repr(exc),
                                                       "traceback": traceback.format_exc()},
                                                      ensure_ascii=False, indent=2), encoding="utf-8")
                print(json.dumps({"title_index": ti, "question_index": qi,
                                  "phase": phase, "status": marker,
                                  "error": repr(exc)[:500]}, ensure_ascii=False), flush=True)
                if marker == "SKIPPED.json":
                    (qdir / f"SKIPPED_q{qi}.json").write_text(
                        json.dumps({"title_index": ti, "question_index": qi,
                                    "phase": phase, "reason": "sensitive"}), encoding="utf-8")
                    continue
                return {"title_index": ti, "status": marker, "totals": total}
        summary = total[phase]
        summary["accuracy"] = summary["correct"] / summary["done"] if summary["done"] else 0
        summary["avg_traversal_tokens"] = round((summary["prompt_tokens"] + summary["completion_tokens"]) /
                                                 summary["done"]) if summary["done"] else 0
        (qdir / f"{phase}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"title_index": ti, "status": "complete", "totals": total}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(run_title, ti): ti for ti in range(20)}
        for f in concurrent.futures.as_completed(futures):
            try:
                print(json.dumps(f.result(), ensure_ascii=False), flush=True)
            except Exception as exc:
                print(json.dumps({"title_index": futures[f], "status": "crashed",
                                  "error": repr(exc)}, ensure_ascii=False), flush=True)
    print("LooGLE Longdep V16 run finished", flush=True)


if __name__ == "__main__":
    main()

