"""One post-answer memory commit for each V16 question."""
from __future__ import annotations

from .v12_memory import apply_round_d1


MEMORY_PHASES = {"round1", "round2", "round3"}
MEMORY_PROTOCOL = "v16_single_commit"


def _edge_key(edge):
    if isinstance(edge, dict):
        return str(edge.get("type", "")), str(edge.get("id", ""))
    return str(edge[0]), str(edge[1])


def _edges(rows):
    return {key for key in map(_edge_key, rows)
            if key[0] in ("relation", "connection") and key[1]}


def plan_evidence_paths(current):
    """Collect verified chunk, entity and relation routes without writing."""
    if not current.get("structured_judge_valid", False):
        return set()
    edges = set()
    routes = current.get("paths_by_chunk", {})
    for chunk in sorted(set(map(str, current.get("supported_chunks", [])))):
        for route in routes.get(chunk, []):
            edges.update(_edges(route))
    for routes in current.get("paths_by_graph", {}).values():
        for route in routes:
            edges.update(_edges(route))
    edges.update(_edges(current.get("supported_graph_edges", [])))
    return edges


def commit_question_memory(rag, current):
    """Merge all evidence, enhance once, then punish only unprotected edges.

    Retrieval must have deferred its writes. A receipt lives on this question's
    PathFinder plan, so repeated commits cannot apply either update twice.
    Fresh and later rounds each get a new plan when their traversal starts.
    """
    audit = {
        "protocol": MEMORY_PROTOCOL,
        "phase": current.get("phase"), "query": str(current.get("query", "")),
        "supported_chunks": sorted(set(map(str, current.get("supported_chunks", [])))),
        "enhanced_edges": [], "punished_edges": [], "protected_edges": [],
        "skip_reason": None, "d1": None,
    }
    if current.get("phase") not in MEMORY_PHASES:
        audit["skip_reason"] = "not_a_memory_round"
        return audit
    finder = rag.preprocess.path_finder
    plan = getattr(finder, "v16_memory_plan", None)
    if plan is None:
        raise RuntimeError("V16 memory writes were not deferred; refusing a second update")
    if audit["query"] != plan["query"]:
        raise RuntimeError("V16 memory record does not match the traversed question")
    if plan.get("receipt") is not None:
        old = plan["receipt"]
        if (old["phase"], old.get("qi")) != (current.get("phase"), current.get("qi")):
            raise RuntimeError("V16 memory plan belongs to a different question/round")
        return old
    if not plan["update_requested"]:
        audit["skip_reason"] = "retrieval_only"
        return audit

    # Plan both sets before the first write. Positive evidence has priority,
    # including ancestors shared by supported and unsupported chunk paths.
    enhance = _edges(plan["enhance_edges"]) | plan_evidence_paths(current)
    d1 = apply_round_d1(rag, current, enhanced_edges=enhance, write=False)
    candidates = _edges(plan["punish_edges"]) | _edges(d1["d1_applied"])
    protected = (candidates & enhance) | _edges(d1["protected_edges"])
    punish = candidates - enhance
    query = audit["query"]
    if not plan["enhance_applied"]:
        if enhance:
            rag.database.enhance_edge_weight(query, sorted(enhance))
        plan["enhance_applied"] = True
    if not plan["punish_applied"]:
        if punish:
            rag.database.punish_edge_weight(query, sorted(punish))
        plan["punish_applied"] = True

    audit.update({
        "qi": current.get("qi"),
        "enhanced_edges": [list(edge) for edge in sorted(enhance)],
        "punished_edges": [list(edge) for edge in sorted(punish)],
        "protected_edges": [list(edge) for edge in sorted(protected)],
        "skip_reason": None if enhance or punish else "no_eligible_edges",
        "d1": d1,
    })
    plan["receipt"] = audit
    return audit
