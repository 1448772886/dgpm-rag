"""Cross-round C2/D1 commits for the isolated V10 long-document run."""
import json
import os
from pathlib import Path


ROUND_ORDER = {"fresh": 0, "round1": 1, "round2": 2, "round3": 3}


def _edge_key(edge):
    return (str(edge.get("type", "")), str(edge.get("id", "")))


def _atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def record_result(record_path, phase, qi, answer_idx, valid, trace):
    """Persist one MC decision and its question-local source ledger."""
    record = {
        "phase": phase, "qi": int(qi), "answer_idx": int(answer_idx),
        "valid": bool(valid), "query": str(trace.get("query", "")),
        "enough": bool(trace.get("enough", False)),
        "structured_judge_valid": bool(trace.get("structured_judge_valid", False)),
        "supported_chunks": sorted(set(map(str, trace.get("supported_chunks", [])))),
        "supported_entities": sorted(set(map(str, trace.get("supported_entities", [])))),
        "paths_by_chunk": trace.get("paths_by_chunk", {}),
        "supported_graph_edges": trace.get("supported_graph_edges", []),
        "paths_by_graph": trace.get("paths_by_graph", {}),
        "quick_chunks": trace.get("quick_chunks", []),
        "quick_chunk_entry_edges": trace.get("quick_chunk_entry_edges", {}),
        "quick_chunk_routes": trace.get("quick_chunk_routes", {}),
    }
    path = Path(record_path)
    _atomic_json(path, record)
    return record, path


def commit_c2_pairs(rag, record_dir, state_file, current, *, enable_d1=True):
    """Match current round against every earlier eligible round and commit.

    Positive C2: valid equal answer + nonempty common evidence chunk set;
    union both recorded routes to each common chunk and deduplicate edge IDs.
    Legacy V10 D1: only previously C2-enhanced quick-chunk ingress edges are
    candidates; protect quick chunks marked as evidence in either record of
    this pair. V12 passes enable_d1=False and runs its per-round D1 separately.
    """
    phase = current["phase"]
    if phase not in ROUND_ORDER or phase == "fresh" or not current["valid"]:
        return {"phase": phase, "matched_pairs": [], "enhanced_edges": [],
                "d1_candidates": [], "d1_applied": []}

    records = Path(record_dir)
    prior = []
    for old_phase, idx in ROUND_ORDER.items():
        if ROUND_ORDER[old_phase] >= ROUND_ORDER[phase]:
            continue
        p = records / f"{old_phase}_{current['qi']}.json"
        if p.exists():
            row = json.loads(p.read_text(encoding="utf-8"))
            if row.get("valid") and row.get("answer_idx") == current["answer_idx"]:
                common = set(row.get("supported_chunks", [])) & set(current["supported_chunks"])
                common_graph = (set(row.get("paths_by_graph", {})) &
                                set(current.get("paths_by_graph", {})))
                if common or common_graph:
                    prior.append((row, common, common_graph))

    if not prior:
        return {"phase": phase, "matched_pairs": [], "enhanced_edges": [],
                "d1_candidates": [], "d1_applied": []}

    state_path = Path(state_file)
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {
        "enhanced_quick_entries": {}}
    registry = state.setdefault("enhanced_quick_entries", {})
    audit = {"phase": phase, "matched_pairs": [], "enhanced_edges": [],
             "d1_candidates": [], "d1_applied": [], "d1_skipped": []}
    db = rag.database

    for old, common, common_graph in prior:
        pair_id = f"{old['phase']}+{phase}:q{current['qi']}"
        edge_rows = []
        for chunk in sorted(common):
            for rec in (old, current):
                for route in rec.get("paths_by_chunk", {}).get(chunk, []):
                    edge_rows.extend(route)
        for graph_key in sorted(common_graph):
            for rec in (old, current):
                for route in rec.get("paths_by_graph", {}).get(graph_key, []):
                    edge_rows.extend(route)
        unique = []
        seen = set()
        for edge in edge_rows:
            key = _edge_key(edge)
            if key[0] in ("relation", "connection") and key[1] and key not in seen:
                unique.append(key)
                seen.add(key)

        protect_chunks = set(old.get("supported_chunks", [])) | set(current.get("supported_chunks", []))
        protected_graph_edges = {_edge_key(e) for rec in (old, current)
                                 for e in rec.get("supported_graph_edges", [])}
        quick_entries = {}
        for rec in (old, current):
            for chunk, edges in rec.get("quick_chunk_entry_edges", {}).items():
                for edge in edges:
                    quick_entries.setdefault(_edge_key(edge), set()).add(str(chunk))

        if unique:
            db.enhance_edge_weight(current["query"], unique)
        audit["matched_pairs"].append({"pair": pair_id, "common_chunks": sorted(common),
                                        "common_graph_evidence": sorted(common_graph),
                                        "route_edges": len(unique)})
        audit["enhanced_edges"].extend([list(e) for e in unique])

        # Track only edges this C2 mechanism has positively written and that
        # are actual quick-subgraph chunk-entry edges. No virgin edge can be
        # introduced into the D1 candidate set.
        for edge in unique:
            chunks = quick_entries.get(edge, set())
            if not chunks:
                continue
            key = f"{edge[0]}::{edge[1]}"
            entry = registry.setdefault(key, {"type": edge[0], "id": edge[1],
                                               "chunks": [], "pairs": []})
            entry["chunks"] = sorted(set(entry["chunks"]) | chunks)
            if pair_id not in entry["pairs"]:
                entry["pairs"].append(pair_id)

        if not enable_d1:
            continue

        # D1 is pair-local: consider only quick-subgraph chunk ingress edges
        # that occurred in this matched pair and that C2 has already written.
        # Do not let a different question's ledger entry become a candidate.
        pair_quick_chunks = set(old.get("quick_chunks", [])) | set(current.get("quick_chunks", []))
        candidates = []
        for edge, chunks in quick_entries.items():
            key = f"{edge[0]}::{edge[1]}"
            written = registry.get(key)
            if not written:
                continue
            relevant_chunks = chunks & pair_quick_chunks
            # Never apply opposing updates to an edge positively confirmed
            # by this same pair, even if it also introduced another chunk.
            if (not relevant_chunks or relevant_chunks & protect_chunks or
                    edge in protected_graph_edges or
                    edge in set(unique)):
                continue
            candidates.append(edge)
        candidates = list(dict.fromkeys(candidates))
        applied = []
        if candidates:
            db.punish_edge_weight(current["query"], candidates)
            applied = candidates
        audit["d1_candidates"].extend([list(e) for e in candidates])
        audit["d1_applied"].extend([list(e) for e in applied])

    for name in ("enhanced_edges", "d1_candidates", "d1_applied"):
        audit[name] = [list(e) for e in dict.fromkeys(tuple(x) for x in audit[name])]
    _atomic_json(state_path, state)
    return audit

