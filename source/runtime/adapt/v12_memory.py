"""Question-local D1 pruning for unsupported quick-subgraph chunks."""
from __future__ import annotations


QUICK_ARRIVAL_SOURCES = {"quick_subgraph", "quick_anchor"}


def _key(edge):
    if isinstance(edge, dict):
        return str(edge.get("type", "")), str(edge.get("id", ""))
    return str(edge[0]), str(edge[1])


def _route_keys_to_anchor_ingress(route):
    """Return edges from an unsupported chunk back through its nearest anchor.

    The route is ordered root-to-chunk. If no anchor occurs on it, only the
    chunk's direct ingress edge is selected. An anchor seed has no preceding
    edge, so its anchor-to-chunk connection is the fallback target.
    """
    has_anchor = any(
        str(edge.get(field, "")).startswith("anchor-")
        for edge in route if isinstance(edge, dict)
        for field in ("parent", "child")
    )
    if not has_anchor:
        return [_key(route[-1])] if route else []

    keys = []
    for edge in reversed(route):
        key = _key(edge)
        if key[0] not in ("relation", "connection") or not key[1]:
            continue
        keys.append(key)
        parent = str(edge.get("parent", "")) if isinstance(edge, dict) else ""
        child = str(edge.get("child", "")) if isinstance(edge, dict) else ""
        if child.startswith("anchor-") or parent.startswith("anchor-"):
            if child.startswith("anchor-"):
                break
    return keys


def _routes_for_chunk(current, chunk):
    routes = current.get("quick_chunk_routes", {}).get(chunk)
    if routes is not None:
        return routes
    # Older receipts retain only the direct quick-subgraph ingress edge.
    return [[edge] for edge in current.get("quick_chunk_entry_edges", {}).get(chunk, [])]


def apply_round_d1(rag, current_or_legacy_state, current=None, enhanced_edges=(), *, write=True):
    """Apply D1 once per answered question using its quick-subgraph ledger.

    D1 no longer checks whether C2 previously enhanced an edge. It compares
    this question's quick chunks with its validated evidence, then applies the
    paper-rule penalty to unsupported chunk routes, back through the nearest
    anchor ingress. Shared routes supporting current evidence are protected.
    """
    # Older V12 runners pass a state path and C2's new writes. Accept those
    # arguments for compatibility. Explicit enhancement edges are protected
    # even when they came from another evidence path in the same question.
    if current is None:
        current = current_or_legacy_state
    phase = current.get("phase")
    quick_chunks = set(map(str, current.get("quick_chunks", [])))
    supported = set(map(str, current.get("supported_chunks", [])))
    supported_entities = set(map(str, current.get("supported_entities", [])))
    supported_edges = {
        _key(edge) for edge in current.get("supported_graph_edges", [])
    }
    supported_edges.update(_key(edge) for edge in enhanced_edges)
    judge_valid = bool(current.get("structured_judge_valid", False))
    enough = bool(current.get("enough", False))
    audit = {
        "phase": phase,
        "quick_chunks": len(quick_chunks),
        "supported_chunks": sorted(supported),
        "supported_entities": sorted(supported_entities),
        "unsupported_quick_chunks": [],
        "supported_graph_edges": [list(edge) for edge in sorted(supported_edges)],
        "protected_edges": [],
        "d1_applied": [],
        "structured_judge_valid": judge_valid,
        "enough": enough,
        "skip_reason": None,
    }

    if phase not in ("round1", "round2", "round3"):
        audit["skip_reason"] = "not_a_memory_round"
        return audit
    if not judge_valid:
        audit["skip_reason"] = "structured_judge_invalid"
        return audit
    if not enough:
        audit["skip_reason"] = "answer_not_sufficient"
        return audit
    if not supported and not supported_entities and not supported_edges:
        audit["skip_reason"] = "answerable_without_verified_evidence"
        return audit

    # Protect all graph edges on currently supported chunk/entity routes.
    for field in ("paths_by_chunk", "paths_by_graph"):
        for routes in current.get(field, {}).values():
            for route in routes:
                supported_edges.update(_key(edge) for edge in route)

    unsupported = quick_chunks - supported
    candidates = set()
    for chunk in sorted(unsupported):
        for route in _routes_for_chunk(current, chunk):
            candidates.update(_route_keys_to_anchor_ingress(route))

    protected = candidates & supported_edges
    punish = sorted(candidates - supported_edges)
    if punish and write:
        rag.database.punish_edge_weight(str(current.get("query", "")), punish)

    audit.update({
        "unsupported_quick_chunks": sorted(unsupported),
        "supported_entities": sorted(supported_entities),
        "supported_graph_edges": [list(edge) for edge in sorted(supported_edges)],
        "protected_edges": [list(edge) for edge in sorted(protected)],
        "d1_applied": [list(edge) for edge in punish],
        "skip_reason": None if punish else "no_unprotected_quick_route",
    })
    return audit
