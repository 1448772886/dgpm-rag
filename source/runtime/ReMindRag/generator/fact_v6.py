"""FGAT v6 — evidence-gap-driven retrieval entry correction.

Implements adapt/FGAT_V6.md: original traversal + extended sufficiency judge
that outputs fact bindings and gaps; gaps with proven entity bindings drive
entry correction (switch to bound entity or seed a gap-specific quick subgraph).

Uses ONLY original mechanisms: quick_query, get_entity_edges, edge weights,
LLM-guided expansion, single end-of-question enhance/punish.
"""
import json
import re

V6_JUDGE = """You are evaluating a knowledge-graph search for the question below.

Question: {query}

Current candidate subgraph:
Entities: {entity_list}
Edges: {edge_list}
Chunk summaries: {chunk_summary}

Atomic facts to verify (derived from the question):
{fact_block}

For each fact, determine:
- "status": "solved" if the subgraph evidence answers it, or "unsolved"
- "entity": if solved, the entity id from the subgraph that IS the answer
- "source_chunk": the chunk id that supports it

If ALL facts are solved, set "enough" to true. Otherwise false.
Also list any NEW facts you discover are needed but cannot be resolved from
the current subgraph (these become gaps for the next expansion step).

JSON only:
{{"enough": true/false,
  "facts": [{{"id": "F1", "status": "solved", "entity": "...", "source_chunk": "..."}},
            {{"id": "F2", "status": "unsolved", "entity": "", "source_chunk": ""}}],
  "gaps": ["self-contained question for missing fact 1", "..."]}}
"""

V6_EXPAND = """You are expanding a knowledge-graph search. The question is:
{query}

Current subgraph has these entities: {entity_list}
These facts are still UNSOLVED: {gaps}

Below are candidate edges from the subgraph boundary (unexplored, with weights).
Pick the ONE edge most likely to help resolve an unsolved fact.

Edges (weight sorted, highest first):
{boundary}

Output JSON only: {{"pick": <edge number starting from 1>}}
"""


def parse_v6_judge(raw):
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group())
    except ValueError:
        return None
    if not isinstance(obj, dict) or "enough" not in obj:
        return None
    facts = []
    for item in (obj.get("facts") or []):
        if isinstance(item, dict) and item.get("id"):
            facts.append({"id": str(item["id"]),
                          "status": str(item.get("status", "unsolved")),
                          "entity": str(item.get("entity", "")),
                          "source_chunk": str(item.get("source_chunk", ""))})
    gaps = [str(g) for g in (obj.get("gaps") or []) if isinstance(g, str) and g.strip()]
    return {"enough": obj["enough"] is True, "facts": facts, "gaps": gaps}


def parse_v6_pick(raw):
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group())
        num = obj.get("pick")
        if type(num) is int and num >= 1:
            return num - 1
    except ValueError:
        pass
    body = (raw or "").strip()
    fence = re.fullmatch(r"```(?:cot-ans|json)?\s*(\d+)\s*```", body)
    if fence:
        body = fence.group(1)
    if re.fullmatch(r"\d+", body):
        return int(body) - 1
    return None
