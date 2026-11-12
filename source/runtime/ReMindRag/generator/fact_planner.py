"""Fact-Guided Adaptive Traversal (FGAT) — mechanisms 1-3.

1. Retrieval-oriented atomic fact decomposition with dependencies (inspired by
   KELGoP; the novelty here is that the decomposition DRIVES traversal state).
2. Fact-coverage state controls which candidate nodes the selector may advance.
3. The selector LLM judges candidates against OPEN facts; coverage — not an
   overall sufficiency guess — terminates the traversal.

Gated by env REMIND_TRAVERSAL=fact in PathFinder; "original" (default) leaves
every code path identical to the released baseline. Edge-level memory (the
original enhance/punish) is untouched by this module.
"""

import json
import re

FACT_DECOMPOSE_PROMPT = """
You are a retrieval planner for a knowledge-graph search system. Decompose the question into ATOMIC FACTS that must each be verified against the knowledge graph.

Question: {query}

Rules:
1. Each fact is ONE verifiable atom: a subject entity, a target relation, and a target entity/attribute.
2. Some facts DEPEND on others: a fact that asks about "the successor's birth year" cannot be searched until the fact identifying the successor is verified. Express this with "depends_on" (list of fact ids). Facts with no dependency have "depends_on": [].
3. Do not invent facts the question does not need. At most 4 facts. If the question is already a single verifiable fact, output it as the only fact.
4. Respond with ONLY a JSON object wrapped in the cot-ans block:

Example:
```cot-ans
{{"facts": [
  {{"id": "F1", "text": "Who is the successor of A?", "depends_on": []}},
  {{"id": "F2", "text": "In which year did A's successor receive prize B?", "depends_on": ["F1"]}}
]}}
```
"""

FACT_GUIDED_NEXT_NODE_PROMPT = """
You are a traversal agent searching a knowledge graph to VERIFY ATOMIC FACTS — not merely to find nodes similar to the question.

The knowledge graph: entities connect to entities and to anchor nodes; anchors connect to chunks (the actual text). You are currently at node **{c_node}**.

## FACT STATE
{fact_block}

## RULES
1. Only OPEN facts are actionable. COVERED facts are done — do not pursue them again. BLOCKED facts wait for their dependency — do not guess entities for them.
2. Look at the candidate relations/connections of the current node and choose the ONE node most likely to advance an OPEN fact (its relation or target matches the fact's subject/relation/target).
3. If several candidates advance open facts, pick the one with the higher edge weight. If NONE advances an open fact, fall back to the node most relevant to the question overall.
4. If the current node has no unexplored adjacent nodes, you may return to a previous node on the traversed path (each may have unexplored connections). Never select an already-accessed chunk.

## CONTEXT
Traversed entities: {entity_list}
Accessed chunks: {chunk_list}
Traversed edges: {edge_list}
Anchor chunk titles: {anchor_chunk_titles}
Chunk summaries: {chunk_summary}
Visit history: {c_node_list}

Candidate relations of {c_node}: {relation_cnode}
Candidate connections of {c_node}: {connection_cnode}

First reason about which OPEN fact each candidate could advance, then answer.
Output format — the node (type: id) and the fact it advances, separated by " | ":
```cot-ans
entity:123 | F2
```
"""

FACT_COVERAGE_PROMPT = """
You are a fact-coverage auditor for a knowledge-graph search.

## FACTS
{fact_block}

## CURRENT EVIDENCE
Traversed edges: {edge_list}
Chunk summaries: {chunk_summary}

For EACH fact, judge whether the current evidence (summaries and edges) contains the answer or enough information to answer it:
- "covered": the evidence states or clearly determines the fact's target.
- "not": the evidence is missing, partial, or the fact's dependency is not yet covered (do not guess entities for blocked facts).

Respond with ONLY a JSON object mapping every fact id to "covered" or "not":
```cot-ans
{{"F1": "covered", "F2": "not", "F3": "not"}}
```
"""

SUBQUESTION_JUDGE_PROMPT = """
You are verifying ONE atomic sub-question against a knowledge-graph subgraph built for it.

Sub-question: {sub_question}

Entities in the subgraph: {entity_list}
Chunk summaries: {chunk_summary}
Edges: {edge_list}

1. Judge whether the evidence is sufficient to answer this sub-question ("yes") or not ("no").
2. If yes, give the concise answer.
3. If yes, also give the entity id from the entity list that best represents the answer (this id will seed follow-up sub-questions). If you cannot identify one, leave it empty.

Respond with ONLY a JSON object:
```cot-ans
{{"enough": "yes", "answer": "...", "entity": "..."}}
```
"""

FACT_BOUNDARY_PROMPT = """
You are expanding a candidate subgraph in a knowledge graph to verify ATOMIC FACTS. Multiple sub-questions share this one subgraph; each has its own coverage state.

## FACT STATE
{fact_block}

## CURRENT SUBGRAPH
Entities: {entity_list}
Edges: {edge_list}
Chunk summaries: {chunk_summary}

## BOUNDARY CANDIDATES (unexplored edges from subgraph entities, with weights)
{boundary_block}

## TASK
Choose the ONE boundary candidate whose target node most likely advances an OPEN fact (its relation/target should match the fact's subject-relation-target). Weights indicate importance; fact match decides. If NO candidate advances an open fact, fall back to the candidate most relevant to the overall question: {query}

Output format — the target node and the fact it advances:
```cot-ans
entity:target_id | F2
```
"""

FACT_ANSWER_SEED_PROMPT = """
Sub-question: {sub_question}
Evidence summaries: {chunk_summary}

The answer to this sub-question is an entity. From the candidate entity list below, return the id of the ONE entity that is the answer.

Candidate entities: {entity_list}

Respond with ONLY the entity id.
"""


def _extract_cot_ans(text):
    m = re.search(r"```cot-ans\s*(.*?)```", text or "", re.S)
    return m.group(1).strip() if m else (text or "").strip()


def parse_fact_plan(raw):
    """Parse the decompose output into a fact list; None on failure."""
    body = _extract_cot_ans(raw)
    m = re.search(r"\{.*\}", body, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None
    facts = obj.get("facts")
    if not isinstance(facts, list) or not facts:
        return None
    clean = []
    for f in facts:
        if not isinstance(f, dict):
            continue
        fid = str(f.get("id", "")).strip()
        text = str(f.get("text", "")).strip()
        deps = f.get("depends_on") or []
        if not fid or not text:
            continue
        if not isinstance(deps, list):
            deps = []
        clean.append({"id": fid, "text": text,
                      "depends_on": [str(d).strip() for d in deps if str(d).strip()]})
    ids = {f["id"] for f in clean}
    clean = [f for f in clean if all(d in ids for d in f["depends_on"])]
    return clean or None


def plan_facts(agent, query, logger=None, max_retries=3):
    """Mechanism 1: decompose the query into atomic facts with dependencies.
    Falls back to a single whole-question fact (baseline degradation)."""
    last_err = ""
    for attempt in range(max_retries):
        try:
            raw = agent.generate_response("", [{"role": "user", "content":
                FACT_DECOMPOSE_PROMPT.format(query=query)}])
            facts = parse_fact_plan(raw)
            if facts:
                if logger:
                    logger.info(f"Fact Plan ({len(facts)} facts): {facts}")
                return facts
            last_err = (raw or "")[:120]
        except Exception as e:
            last_err = f"{type(e).__name__}: {str(e)[:100]}"
        if logger:
            logger.warning(f"fact plan retry {attempt + 1}/{max_retries}: {last_err}")
    if logger:
        logger.warning(f"fact plan FAILED ({last_err}) — falling back to whole-question fact")
    return [{"id": "F1", "text": query, "depends_on": []}]


def fact_block(facts, status):
    """Render the FACT STATE block shown to the selector / auditor."""
    lines = []
    for f in facts:
        st = status.get(f["id"], "open")
        deps = f'(waits for {",".join(f["depends_on"])})' if f.get("depends_on") else ""
        lines.append(f'{f["id"]} [{st.upper()}] {f["text"]} {deps}'.strip())
    return "\n".join(lines)


def open_fact_ids(facts, status):
    """Facts that are actionable: open, and all dependencies covered."""
    out = []
    for f in facts:
        st = status.get(f["id"], "open")
        deps_covered = all(status.get(d) == "covered" for d in f.get("depends_on", []))
        if st != "covered" and deps_covered:
            out.append(f["id"])
    return out


def parse_fact_guided_node(raw):
    """Parse '<type>:<id> | <F?>' from the selector output. Returns (node, fact_id_or_None)."""
    body = _extract_cot_ans(raw)
    ms = re.findall(r"(entity|chunk)\s*:\s*([A-Za-z0-9_\-]+)", body)
    if not ms:
        return None, None
    typ, tid = ms[-1]
    node = f"{typ}:{tid}"
    fm = re.search(r"\|\s*(F\d+)", body) or re.search(r"(F\d+)", body)
    return node, (fm.group(1) if fm else None)


def parse_coverage(raw, facts):
    """Parse the auditor JSON into {fact_id: 'covered'|'not'}; None on failure."""
    body = _extract_cot_ans(raw)
    m = re.search(r"\{.*\}", body, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None
    out = {}
    for f in facts:
        v = str(obj.get(f["id"], "")).strip().lower()
        out[f["id"]] = "covered" if v.startswith("covered") or v == "true" else "not"
    return out
