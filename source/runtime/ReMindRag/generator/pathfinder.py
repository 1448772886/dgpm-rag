from ..llms import AgentBase
from ..database import ChromaDBManager
from .prompts import (chunk_summary_prompt, judge_sufficient_information_prompt,
                      find_next_node_prompt, v12_find_next_node_prompt,
                      v12_sufficiency_prompt, v14a_find_next_node_prompt,
                      v14a_sufficiency_prompt,
                      reward_or_punishment_prompt, select_another_node_prompt,
                      generate_rag_ans_prompt)
import os


import os
import json
import re
import numpy as np

from ..utils.decorators import retry_json_parsing ,check_keys, unpack_cot_ans
from ..utils.logger import setup_logger
from typing import List, Dict
import logging



def select_region_diverse_seeds(candidate_ids, candidate_embeddings,
                                query_embedding, chunk_areas, seed_count,
                                overlap_penalty=0.12, similarity_margin=0.08):
    """Select relevant seeds while reducing duplicated chunk regions."""
    if not candidate_ids or seed_count <= 0:
        return []
    q = np.asarray(query_embedding, dtype=float).reshape(-1)
    q_norm = np.linalg.norm(q)
    if q_norm == 0:
        return list(candidate_ids[:seed_count])
    q = q / q_norm
    vectors = np.asarray(candidate_embeddings, dtype=float)
    if vectors.ndim != 2 or len(vectors) != len(candidate_ids):
        return list(candidate_ids[:seed_count])
    norms = np.linalg.norm(vectors, axis=1)
    similarities = np.full(len(candidate_ids), -1.0, dtype=float)
    valid = norms > 0
    similarities[valid] = (vectors[valid] / norms[valid, None]) @ q
    best_similarity = float(np.max(similarities))
    eligible = [i for i, score in enumerate(similarities)
                if score >= best_similarity - similarity_margin]
    for index in np.argsort(-similarities):
        if len(eligible) >= min(seed_count, len(candidate_ids)):
            break
        if int(index) not in eligible:
            eligible.append(int(index))

    selected = []
    covered_chunks = set()
    remaining = set(eligible)
    while remaining and len(selected) < seed_count:
        best_index, best_key = None, None
        for index in remaining:
            area = set(chunk_areas.get(candidate_ids[index], set()))
            overlap = (len(area & covered_chunks) / len(area)) if area else 0.0
            score = float(similarities[index]) - overlap_penalty * overlap
            key = (score, float(similarities[index]), -index)
            if best_key is None or key > best_key:
                best_index, best_key = index, key
        selected.append(candidate_ids[best_index])
        covered_chunks.update(chunk_areas.get(candidate_ids[best_index], set()))
        remaining.remove(best_index)
    return selected


class PathFinder():
    # [reproduction deviation, documented] see the no-progress budget in
    # get_query_ans: counts traversals ended by the budget instead of the
    # LLM judge (reported in the reproduction notes).
    no_progress_triggers = 0

    def __init__(self, agent:AgentBase, database:ChromaDBManager, chunk_summary_threshold, logger_level, log_path):
        self.max_retries = 3

        self.agent = agent
        self.database = database
        self.chunk_summary_threshold = chunk_summary_threshold

        self.entity = []
        self.chunk = []
        self.edge = []
        self.path = []
        self.c_node_list = []
        self.out_dgree = {}
        self.summary = {}
        # Batch evidence mode keeps source text during traversal and performs
        # one sentence-selection request after the question's search ends.
        self._batch_documents = {}
        self._judge_current_raw_chunk_id = None
        self._judge_current_raw_mode = os.environ.get(
            "REMIND_JUDGE_CURRENT_RAW_MODE", "full")

        # FGAT: "fact" = v2 sub-question orchestrator (the original machinery
        # runs ONCE PER SUB-QUESTION: per-fact seed -> original quick subgraph
        # -> per-sub-question sufficiency judge -> original enhance/punish).
        # "original" (default) keeps every code path identical to the release.
        self.traversal_mode = os.environ.get("REMIND_TRAVERSAL", "original")
        self.facts = []
        self.fact_status = {}

        self.logger = setup_logger("PathFinder", logger_level, log_path)

        self.update_dict = {}
        self.confirm_paths = []
        self.viewed_nodes = []
        # A1 keeps only per-question local comparisons.  It is deliberately
        # not a persistent history store: each record is used once at update
        # time and then discarded with the PathFinder instance.
        self.a1_decision_records = []
        # B2 is a question-local source ledger.  It records actual node
        # introduction events, including quick-subgraph construction, and is
        # discarded with this PathFinder instance after the question.
        self.b2_roots = []
        self.b2_arrivals = {}
        self.b2_supported_chunks = set()
        self.b2_supported_entities = set()
        self.b2_supported_edges = set()
        self.b2_quick_chunks = set()
        # V10 defers all persistent memory writes until the MC evaluator has
        # checked the answer and the cross-round C2 gate has selected a pair.
        self.c2_trace = None
        # B4's question-local backtracking trace; unlike the B2 provenance
        # ledger, this records when the traversal resumes an older frontier.
        self.b4_backtrack_ledger = []
        

    def get_query_ans(self, query, do_update, search_keys, max_jumps):
        if self.traversal_mode == "fact_v7":
            from .fact_v7 import initialize
            initialize(self)
        if self.traversal_mode == "fact_v6":
            from .fact_v6 import V6_JUDGE, parse_v6_judge
            self._v6_facts = []
            self._v6_gaps = []
            self._v6_extra_roots = []
            self._v6_metrics = {"switches": 0, "gap_entries": 0, "gap_tried": []}
        if self.traversal_mode == "fact_v5":
            from .fact_v5 import search
            return search(self, query, do_update, search_keys, max_jumps)
        if self.traversal_mode == "fact":
            return self.get_query_ans_fact(query, do_update, search_keys, max_jumps)
        return self._get_query_ans_original(query, do_update, search_keys, max_jumps)

    def _path_trace(self):
        """Stable traversal trace used only for the A2 write gate."""
        return {
            "path": [[p.get("type"), p.get("id"), p.get("from"), p.get("to")]
                     for p in self.path],
            "c_node_list": list(self.c_node_list),
            "entity": list(self.entity),
            "chunk": list(self.chunk),
        }

    def _a2_path_write_gate(self):
        """Save fresh traces and allow A2 writes only on exact replay."""
        out_path = os.environ.get("REMIND_PATH_TRACE_OUT", "")
        expected_path = os.environ.get("REMIND_PATH_TRACE_EXPECTED", "")
        decision_path = os.environ.get("REMIND_PATH_TRACE_DECISION", "")
        trace = self._path_trace()
        if out_path:
            parent = os.path.dirname(out_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(trace, f, ensure_ascii=False, indent=2)
        if not expected_path:
            return True
        try:
            with open(expected_path, "r", encoding="utf-8") as f:
                expected = json.load(f)
        except (OSError, ValueError):
            self.logger.warning("A2 path gate: expected trace unavailable; skip memory write")
            return False
        same = trace == expected
        self.logger.info("A2 path gate: " + ("MATCH, apply memory update" if same
                                             else "MISMATCH, skip memory update"))
        if decision_path:
            parent = os.path.dirname(decision_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(decision_path, "w", encoding="utf-8") as f:
                json.dump({"match": same}, f)
        return same

    def _get_query_ans_original(self, query, do_update, search_keys, max_jumps, seed_override=None):
        self.logger.info("Start PathFinder Process...")
        # Retrieval plans writes; the evaluator commits once after the answer.
        self.v16_memory_plan = None
        if os.environ.get("REMIND_V16_DEFER_MEMORY", "0") == "1":
            self.v16_memory_plan = {
                "query": str(query), "update_requested": bool(do_update),
                "enhance_edges": [], "punish_edges": [],
                "enhance_applied": False, "punish_applied": False,
                "receipt": None,
            }
        self.entity = []
        self.chunk = []
        self.edge = []
        self.path = []
        self.c_node_list = []
        self.out_dgree = {}
        self.summary = {}
        self._batch_documents = {}
        self._judge_current_raw_chunk_id = None
        self._judge_current_raw_mode = os.environ.get(
            "REMIND_JUDGE_CURRENT_RAW_MODE", "full")
        self.a1_decision_records = []
        self.b2_roots = []
        self.b2_arrivals = {}
        self.b2_supported_chunks = set()
        self.b2_supported_entities = set()
        self.b2_supported_edges = set()
        self.b2_quick_chunks = set()
        self.c2_trace = None
        self.b4_backtrack_ledger = []
        # Baseline-only retrieval probe: keep a small programmatic frontier of
        # admissible candidates from previously visited nodes. This is a
        # retrieval-control layer; it does not alter the quick subgraph or
        # persistent memory updates.
        self.frontier_candidates = []
        self.frontier_lookup = {}
        self.frontier_selected = set()

        self.logger.info(f"Get Query: {query}")
        self.query = query
        self.database_query = query
        anchor_seed_ids = set()

        if seed_override is not None:
            # FGAT: dependent sub-questions start from the parent answer entity
            # instead of re-searching seeds from the sub-question text.
            self.entity = [seed_override]
            search_keys = 1
            self.logger.info(f"Seed override: {seed_override}")
        else:
            query_embedding = self.database.embedding.sentence_embedding(query)
            use_diverse_seeds = os.environ.get("REMIND_DIVERSE_SEEDS", "0") == "1"
            use_community_seeds = (
                os.environ.get("REMIND_COMMUNITY_SEEDS", "0") == "1" and
                self.traversal_mode == "original")
            use_top5_farthest = (
                os.environ.get("REMIND_E1_RELEVANCE_DISTANCE", "0") == "1" and
                self.traversal_mode in ("original", "fact_v7") and
                search_keys >= 2)
            candidate_count = 5 if use_top5_farthest else search_keys
            if use_diverse_seeds:
                candidate_count = max(
                    search_keys,
                    int(os.environ.get("REMIND_SEED_CANDIDATES",
                                      max(search_keys * 5, 10))))
            elif use_community_seeds:
                candidate_count = max(
                    search_keys,
                    int(os.environ.get("REMIND_COMMUNITY_SEED_POOL", "10")))
            if use_diverse_seeds:
                results = self.database.entity_collection.query(
                    query_embeddings=[query_embedding],
                    n_results=candidate_count,
                    include=["embeddings"]
                )
            else:
                results = self.database.entity_collection.query(
                    query_embeddings=[query_embedding], n_results=candidate_count,
                    **({"include": ["distances"]}
                       if use_top5_farthest else {}))
            if not results["ids"][0]:
                self.logger.error(f"No Data Found. The data may not be loaded into the database.")
                raise RuntimeError("No Data Found. The data may not be loaded into the database.")

            candidates = results["ids"][0]
            self.entity = candidates[:search_keys]
            if use_top5_farthest and len(candidates) > search_keys:
                try:
                    # Keep rank 1 fixed. Score ranks 2..5 using the
                    # preregistered relevance-distance mix (alpha=0.33).
                    # This replaces only the second initial seed; the later
                    # E1 global fallback remains a separate search mechanism.
                    graph = {}
                    from .e1_seed_selection import score_second_seed
                    relation_data = self.database.relation_collection.get(
                        include=["metadatas"])
                    for metadata in relation_data.get("metadatas", []):
                        left = str(metadata.get("subject_entity_id", ""))
                        right = str(metadata.get("object_entity_id", ""))
                        if left and right:
                            graph.setdefault(left, set()).add(right)
                            graph.setdefault(right, set()).add(left)
                    candidate_distances = results.get("distances", [[]])[0][:5]
                    first, second, audit = score_second_seed(
                        candidates[:5], candidate_distances, graph, alpha=0.33)
                    self.entity = [first, second]
                    self.logger.info(
                        "E1 relevance-distance seeds alpha=0.33: selected=%s candidates=%s",
                        self.entity, audit)
                except Exception as exc:
                    self.entity = candidates[:search_keys]
                    self.logger.warning(
                        "E1 relevance-distance seed selection failed; using original top-k: "
                        f"{type(exc).__name__}: {str(exc)[:160]}")
            if use_community_seeds and len(candidates) > search_keys:
                try:
                    from .community_seeds import (
                        choose_community_seeds,
                        get_community_seed_map,
                    )
                    communities, community_count = get_community_seed_map(
                        self.database,
                        os.environ.get("REMIND_COMMUNITY_CACHE_PATH") or None)
                    self.entity, audit = choose_community_seeds(
                        candidates, communities, search_keys)
                    self.logger.info(
                        "Community seed selection: %s pool=%d communities=%d",
                        audit, len(candidates), community_count)
                except Exception as exc:
                    self.entity = candidates[:search_keys]
                    self.logger.warning(
                        "Community seed selection failed; using original top-k: "
                        f"{type(exc).__name__}: {str(exc)[:160]}")
            if use_diverse_seeds and len(candidates) > search_keys:
                try:
                    chunk_areas = {
                        entity_id: self.database.get_entity_chunk_area(entity_id)
                        for entity_id in candidates
                    }
                    self.entity = select_region_diverse_seeds(
                        candidates,
                        results.get("embeddings", [[]])[0],
                        query_embedding,
                        chunk_areas,
                        search_keys,
                        overlap_penalty=float(os.environ.get(
                            "REMIND_SEED_OVERLAP_PENALTY", "0.12")),
                        similarity_margin=float(os.environ.get(
                            "REMIND_SEED_SIMILARITY_MARGIN", "0.08")),
                    )
                    self.logger.info(
                        "Region-diverse seeds: selected=%s candidates=%d",
                        self.entity, len(candidates))
                except Exception as exc:
                    self.entity = candidates[:search_keys]
                    self.logger.warning(
                        "Region-diverse seed selection failed; using original top-k: "
                        f"{type(exc).__name__}: {str(exc)[:160]}"
                    )
            if (os.environ.get("REMIND_SEED_COVERAGE", "0") == "1" and
                    self.traversal_mode == "original" and
                    search_keys >= 2 and not use_diverse_seeds):
                pool_size = max(search_keys, int(
                    os.environ.get("REMIND_SEED_COVERAGE_POOL", "10")))
                try:
                    pool = self.database.entity_collection.query(
                        query_embeddings=[query_embedding],
                        n_results=pool_size)
                    pool_candidates = list((pool.get("ids") or [[]])[0])
                    first_seed = pool_candidates[0] if pool_candidates else None
                    if first_seed is not None:
                        neighbor_data = self.database.relation_collection.get(
                            where={"$or": [
                                {"subject_entity_id": first_seed},
                                {"object_entity_id": first_seed},
                            ]},
                            include=["metadatas"])
                        neighbors = set()
                        for metadata in neighbor_data.get("metadatas", []):
                            for key in ("subject_entity_id", "object_entity_id"):
                                other = str(metadata.get(key, ""))
                                if other and other != str(first_seed):
                                    neighbors.add(other)
                        first_chunks = set(
                            self.database.get_entity_chunk_area(first_seed))
                        eligible = []
                        for rank, candidate in enumerate(
                                pool_candidates[1:], 1):
                            if candidate in neighbors:
                                continue
                            candidate_chunks = set(
                                self.database.get_entity_chunk_area(
                                    candidate))
                            union = first_chunks | candidate_chunks
                            overlap = (
                                len(first_chunks & candidate_chunks)
                                / len(union) if union else 1.0)
                            new_count = len(candidate_chunks -
                                            first_chunks)
                            if new_count <= 0 or overlap > 0.25:
                                continue
                            eligible.append(
                                (new_count, -rank, candidate, overlap))
                        if eligible:
                            selected = max(eligible)
                            self.entity = [first_seed, selected[2]]
                            self.logger.info(
                                "Seed coverage selection: first=%s "
                                "second=%s new_chunks=%d overlap=%.3f "
                                "pool=%d",
                                first_seed, selected[2], selected[0],
                                selected[3], len(pool_candidates))
                        else:
                            self.entity = pool_candidates[:search_keys]
                            self.logger.info(
                                "Seed coverage fallback: no eligible "
                                "diverse candidate; seeds=%s pool=%d",
                                self.entity, len(pool_candidates))
                except Exception as exc:
                    self.entity = candidates[:search_keys]
                    self.logger.warning(
                        "Seed coverage selection failed; using original top-k: "
                        f"{type(exc).__name__}: {str(exc)[:160]}")
            # Fresh-only anchor seed reranking: use a direct chunk lookup as
            # a silent second entry point, without injecting that chunk as
            # evidence or exposing a candidate pool to the selector.
            if (os.environ.get("REMIND_FRESH_ANCHOR_SEEDS", "0") == "1" and
                    self.traversal_mode == "original" and search_keys >= 2):
                original_seeds = list(self.entity[:search_keys])
                selected_anchor = None
                selected_chunk = None
                selected_distance = None
                try:
                    direct = self.database.chunk_collection.query(
                        query_embeddings=[query_embedding],
                        n_results=min(3, max(1, search_keys)),
                        include=["distances"])
                    direct_ids = (direct.get("ids") or [[]])[0]
                    direct_distances = (direct.get("distances") or [[]])[0]
                    for idx, chunk_id in enumerate(direct_ids):
                        anchor_id = f"anchor-{chunk_id}"
                        if anchor_id in original_seeds:
                            continue
                        try:
                            exists = self.database.entity_collection.get(
                                ids=[anchor_id]).get("ids", [])
                        except Exception:
                            exists = []
                        if exists:
                            selected_anchor = anchor_id
                            selected_chunk = str(chunk_id)
                            selected_distance = (
                                float(direct_distances[idx])
                                if idx < len(direct_distances) else None)
                            break
                    if selected_anchor is not None:
                        self.entity = [original_seeds[0], selected_anchor]
                        anchor_seed_ids.add(selected_anchor)
                        self.logger.info(
                            "Fresh anchor seed selected: original=%s selected=%s "
                            "chunk=%s distance=%s",
                            original_seeds, selected_anchor, selected_chunk,
                            selected_distance)
                    else:
                        self.logger.info(
                            "Fresh anchor seed skipped: no distinct valid anchor "
                            "among direct chunk candidates; original=%s",
                            original_seeds)
                except Exception as exc:
                    self.logger.warning(
                        "Fresh anchor seed lookup failed; using original seeds: "
                        f"{type(exc).__name__}: {str(exc)[:160]}")
            self.logger.debug(f"Initial Entity: {self.entity}")
            if len(self.entity) < search_keys:
                self.logger.warning(f"Can't found enough entities, only found ({len(self.entity)}/{search_keys})")
                search_keys = len(self.entity)

        for entity_iter in range(search_keys):
            if self._b2_enabled():
                self._b2_record_root(self.entity[entity_iter], "query_seed")
            seed_id = self.entity[entity_iter]
            origin_entities, origin_chunks, origin_edges = self.database.quick_query(
                self.database_query, seed_id)
            anchor_seed = seed_id in anchor_seed_ids
            blocked_anchor_chunks = {
                str(edge["to"]) for edge in origin_edges
                if edge["type"] == "connection" and
                str(edge.get("from")) in anchor_seed_ids
            }

            for edge in origin_edges:
                if (edge["type"] == "connection" and
                        (anchor_seed or str(edge.get("from")) in anchor_seed_ids)):
                    continue
                if edge["type"] == "relation":
                    if edge["to"] in self.entity:
                        self.logger.debug(f"Edge {edge['id']} May Form a Loop, Deleted.")
                        continue
                    self.edge.append(f"from entity:{edge['from']} to entity:{edge['to']} with relations: {edge['documents']}")
                    self.path.append({"from":edge['from'], "to":edge['to'], "id":edge['id'], "documents":edge['documents'],"type":"relation"})
                    self._b2_record_arrival(self.path[-1], "quick_subgraph")
                else:
                    if edge["to"] in self.chunk:
                        self.logger.debug(f"Connection {edge['id']} Already Exist, Deleted.")
                        continue
                    self.edge.append(f"from entity:{edge['from']} to chunk:{edge['to']}")
                    self.path.append({"from":edge['from'], "to":edge['to'], "id":edge['id'],"type":"connection"})
                    self._b2_record_arrival(self.path[-1], "quick_subgraph")

            for origin_entity in origin_entities:
                if origin_entity not in self.entity:
                    self.entity.append(origin_entity)

            for origin_chunk in origin_chunks:
                if str(origin_chunk) in blocked_anchor_chunks:
                    continue
                if origin_chunk not in self.chunk:
                    self.chunk.append(origin_chunk)

        # CAHI (Chunk-Anchor Hybrid Initialization): the entity collection
        # mixes named entities and anchor nodes, so an answer-bearing chunk
        # can lose the initial top-k competition to aliases.  If the normal
        # entity seeds produce no chunk at all, retrieve one chunk directly
        # and expose its anchor/chunk pair to the existing traversal.  This
        # is local vector retrieval only; it adds no LLM call and does not
        # modify persistent edge weights.
        if (os.environ.get("REMIND_CAHI", "0") == "1" and not self.chunk):
            try:
                chunk_results = self.database.chunk_collection.query(
                    query_embeddings=[query_embedding],
                    n_results=1,
                    include=["distances"])
                cids = chunk_results.get("ids", [[]])[0]
                distances = chunk_results.get("distances", [[]])[0]
                if cids:
                    chunk_id = str(cids[0])
                    similarity = (1.0 - float(distances[0])) if distances else None
                    anchor_id = f"anchor-{chunk_id}"
                    if anchor_id not in self.entity:
                        self.entity.append(anchor_id)
                    if chunk_id not in self.chunk:
                        self.chunk.append(chunk_id)
                    edge_id = f"{anchor_id}_{chunk_id}"
                    self.edge.append(f"from entity:{anchor_id} to chunk:{chunk_id}")
                    self.path.append({"from": anchor_id, "to": chunk_id,
                                      "id": edge_id, "type": "connection"})
                    self._b2_record_arrival(self.path[-1], "cahi_chunk")
                    self.logger.info(
                        "CAHI: injected direct chunk=%s anchor=%s similarity=%s "
                        "because entity quick subgraph was empty",
                        chunk_id, anchor_id,
                        "%.4f" % similarity if similarity is not None else "unknown")
                else:
                    self.logger.info("CAHI: no direct chunk candidate")
            except Exception as exc:
                self.logger.warning(
                    "CAHI: direct chunk retrieval failed; keeping original "
                    "initialization: %s: %s", type(exc).__name__, str(exc)[:160])
            

        for entity_iter in range(search_keys):
            node_id = self.entity[entity_iter]
            if node_id.startswith("anchor") and node_id not in anchor_seed_ids:
                chunk_id = node_id.split("-")[1]
                if chunk_id not in self.chunk:
                    self.chunk.append(chunk_id)
                    self.edge.append(f"from entity:{node_id} to chunk:{chunk_id}")
                    self.path.append({"from":node_id, "to":chunk_id, "id":f"{node_id}_{chunk_id}", "type":"connection"})
                    self._b2_record_arrival(self.path[-1], "quick_anchor")

        # Freeze the initial fast-subgraph chunk set. D1 compares this set
        # against the evidence routes after the existing sufficiency judge.
        if self._b2_enabled():
            self.b2_quick_chunks = {str(c) for c in self.chunk}

        
        for entity_iter in self.entity:
            self.out_dgree[entity_iter], _ =self.get_out_dgree(entity_iter)
        
        self.logger.info(f"Quick Search Results --- entity:{len(self.entity)}, chunk:{len(self.chunk)}, edge:{len(self.edge)}.")
        self.logger.debug(f"Quick Search Entity List:\n{self.entity}")
        self.logger.debug(f"Quick Search Chunk List:\n{self.chunk}")
        self.logger.debug(f"Quick Search Edge List:\n{self.edge}")
        if self.chunk:
            for chunk_iter in self.chunk:
                self.summary[f"chunk:{chunk_iter}"] = self.get_chunk_summary(chunk_iter)
        self.logger.debug(f"Chunk Summary:\n{self.summary}")

        enough = self._judge_dispatch(query)
        self.logger.info(f"Data Enough? {enough}")
        if not enough:
            self.logger.info(f"Start Search At Entity Node:{self.entity[0]}.")
        c_node = self.entity[0]
        self.c_node_list.append(c_node)

        jumps = 0
        f1_enabled = (self.traversal_mode == "fact_v7" and
                      os.environ.get("REMIND_F1", "0") == "1")
        a5_continue = (self.traversal_mode == "fact_v7" and
                       os.environ.get("REMIND_A5_CONTINUE", "0") == "1")
        if a5_continue:
            self.v7_transition_attempts = set()
        b4_repeat_counts = {}
        # F1 records transitions per unresolved requirement.  A transition
        # already tried under the same requirement is removed from the next
        # selector call, including its reverse (a->b and b->a).  If the
        # requirement changes, the edge becomes eligible again.
        f1_seen = set()

        def f1_topic():
            gap = getattr(self, "v7_gap", None)
            if isinstance(gap, dict):
                return json.dumps({k: gap.get(k, "") for k in
                                   ("id", "question", "entity")},
                                  ensure_ascii=False, sort_keys=True)
            return str(query)

        def f1_edge(source, target):
            return tuple(sorted((str(source), str(target))))

        def f1_allowed(source, target):
            return (not f1_enabled or
                    (f1_edge(source, target), f1_topic()) not in f1_seen)

        def f1_mark(source, target):
            if f1_enabled:
                f1_seen.add((f1_edge(source, target), f1_topic()))

        def f1_frontier(current_node):
            """Return the nearest prior branch with an unvisited F1 edge."""
            if not f1_enabled:
                return None
            topic = f1_topic()
            # Search the whole discovered graph, not only the current DFS
            # stack. A viable branch elsewhere keeps the question alive.
            visit_order = list(dict.fromkeys(
                list(self.c_node_list) + list(self.entity)
            ))
            for node in reversed(visit_order):
                if not node or node == current_node:
                    continue
                relations, connection = self.database.get_entity_edges(query, node)
                candidates = [r["to"] for r in relations
                              if r["to"] not in self.entity and
                              (f1_edge(node, r["to"]), topic) not in f1_seen]
                if connection and connection["to"] not in self.chunk and \
                        (f1_edge(node, connection["to"]), topic) not in f1_seen:
                    candidates.append(connection["to"])
                if candidates:
                    self.logger.info("F1 branch fallback: %s -> %s", current_node, node)
                    return node
            return None

        def a5_frontier(current_node):
            if not a5_continue:
                return None
            graph_state = (tuple(sorted(self.entity)), tuple(sorted(self.chunk)))
            if os.environ.get("REMIND_B4", "0") == "1":
                # B4: prefer the most recently traversed node that still has
                # unexplored graph degree. This is local bookkeeping, not an
                # LLM decision; the graph, prompts, and jump budget stay fixed.
                visit_order = list(self.c_node_list)
                if current_node:
                    visit_order.append(current_node)
                seen = set()
                for node in reversed(visit_order):
                    if not node or node == current_node or node in seen:
                        continue
                    seen.add(node)
                    degree = self.out_dgree.get(node, 0)
                    key = (graph_state, current_node, node)
                    if degree > 0 and key not in self.v7_frontier_attempted:
                        self.v7_frontier_attempted.add(key)
                        event = {"from": current_node, "to": node,
                                 "remaining_degree": degree,
                                 "entity_count": len(self.entity),
                                 "chunk_count": len(self.chunk)}
                        self.b4_backtrack_ledger.append(event)
                        self.logger.info(f"B4 ledger backtrack: {event}")
                        return node
                return None
            options = []
            for node, degree in self.out_dgree.items():
                key = (graph_state, current_node, node)
                if node != current_node and degree > 0 and key not in self.v7_frontier_attempted:
                    options.append((degree, node, key))
            if not options:
                return None
            degree, node, key = max(options)
            self.v7_frontier_attempted.add(key)
            self.logger.info(f"A5 continuation backtrack: {current_node!r} -> {node} "
                             f"(remaining weighted frontier degree={degree})")
            return node
        
        while not enough:
            if self.traversal_mode == "fact_v7":
                self.v7_steps += 1
                if f1_enabled and self.v7_steps > 3 * max_jumps:
                    # F1 bounds selector/backtrack attempts as well as new
                    # graph expansions.  This is the hard no-progress gate.
                    PathFinder.no_progress_triggers += 1
                    self.logger.warning("F1 no-progress budget exhausted: steps=%d max=%d",
                                        self.v7_steps, 3 * max_jumps)
                    do_update = False
                    break
                if jumps >= max_jumps:
                    do_update = False
                    break
            self.logger.debug(f"Out Dgree: {self.out_dgree}")
            processed_relations = []
            processed_path = []
            processed_connection = []
            connection_chunk_id = ""
            if c_node:
                self.logger.info(f"======= Current Entity: {c_node} === Jumps: {jumps} ===========================================================")
                self.logger.debug(f"Entity Now:{self.entity}")
                self.logger.debug(f"Chunks Now:{self.chunk}")
                relations, connection = self.database.get_entity_edges(query, c_node)
                self.logger.trace(f"Relations:\n{relations}")
                self.logger.trace(f"Connections:\n{connection}")
                self.logger.info(f"Current Node Edge --- Relations:{len(relations)}, Connections:{len(connection)==6}.")

                for relation in relations:
                    processed_relation_iter = f"from entity:{relation['from']} to entity:{relation['to']} with relations: {relation['documents']}"

                    if (relation["to"] not in self.entity and
                            f1_allowed(c_node, relation["to"])):
                        self.logger.trace(f"Relation: {processed_relation_iter}")
                        processed_relations.append(processed_relation_iter)
                        processed_path.append({"from":relation['from'], "to":relation['to'], "id":relation['id'], "documents":relation['documents'], "type":"relation"})
                    else:
                        self.logger.debug(f"Entity {relation['to']} Already in Entity Set.")
                
                if connection:
                    if (connection["to"] not in self.chunk and
                            f1_allowed(c_node, connection["to"])):
                        processed_connection = f"from entity:{connection['from']} to chunk:{connection['to']}"
                        processed_path.append({"from":connection['from'], "to":connection['to'], "id":connection['id'], "type":"connection"})
                        connection_chunk_id = connection["to"]
                        self.logger.trace(f"Connection: {processed_connection}")
            else:
                self.logger.info(f"======= Current Node: In Chunk Handle Process ================================================================")

            # A chunk has just been consumed. If it did not make the judge
            # sufficient, resume at a still-unexplored entity frontier instead
            # of repeatedly asking the selector to invent a node from an empty
            # local neighborhood.
            if a5_continue and not c_node and not connection_chunk_id:
                frontier = a5_frontier(c_node)
                if frontier:
                    c_node = frontier
                    continue
            if f1_enabled and not c_node and not connection_chunk_id:
                frontier = f1_frontier(c_node)
                if frontier:
                    c_node = frontier
                    self.c_node_list.append(c_node)
                    continue
                
            anchor_chunk_titles = self.get_anchor_chunk_title(processed_path)
            if self._frontier_enabled():
                self._prepare_global_frontier(query, c_node, processed_path)
            self.logger.debug(f"Find Next Node... With Data Size --- edges:{len(str(self.edge+processed_relations))+len(processed_connection)}, chunk summary:{len(str(self.summary))}, anchor chunk titles:{len(str(anchor_chunk_titles))}")

            # If this node has no fresh adjacent edge, resume from the latest
            # viable frontier before spending another selector call.
            if (a5_continue and os.environ.get("REMIND_B4", "0") == "1"
                    and c_node and not processed_path):
                frontier = a5_frontier(c_node)
                if frontier:
                    c_node = frontier
                    continue
            if f1_enabled and c_node and not processed_path:
                frontier = f1_frontier(c_node)
                if frontier:
                    c_node = frontier
                    self.c_node_list.append(c_node)
                    continue
                # F1 contract: an exhausted node with no reachable
                # unvisited frontier terminates this branch. Do not call the
                # selector with an empty candidate set; that path can produce
                # an empty array and crash while indexing its first element.
                # A dead local branch must still get the V16 gap-entry
                # opportunity.  Global degree can be nonzero because it
                # counts already-visited edges, so waiting for
                # sum(out_dgree)==0 would skip the gap search and can send
                # the selector an empty candidate set.
                if self.traversal_mode == "fact_v7":
                    from .fact_v7 import enter
                    entry = enter(self, max_jumps - jumps)
                    if not entry and a5_continue:
                        from .fact_v7 import enter_global
                        entry = enter_global(self, max_jumps - jumps)
                    if entry:
                        c_node, cost = entry
                        jumps += cost
                        enough = self._judge_dispatch(query)
                        continue
                self.logger.info(
                    "F1: no unvisited frontier or gap entry from %s; terminating branch",
                    c_node,
                )
                do_update = False
                break
            
            if connection_chunk_id:
                new_node = f"chunk:{connection_chunk_id}"
                new_node_type = "chunk"
                new_node_id = connection_chunk_id
                self.edge.append(f"from entity:{c_node} to chunk:{connection_chunk_id}")
                self.path.append({"from":c_node, "to":connection_chunk_id, "id":f"{c_node}_{connection_chunk_id}", "type":"connection"})
                self._b2_record_arrival(self.path[-1], "guided_expansion")

            elif sum(self.out_dgree.values()) == 0:
                if self.traversal_mode == "fact_v7":
                    from .fact_v7 import enter
                    entry = enter(self, max_jumps - jumps)
                    if not entry and a5_continue:
                        from .fact_v7 import enter_global
                        entry = enter_global(self, max_jumps - jumps)
                    if entry:
                        c_node, cost = entry
                        jumps += cost
                        enough = self._judge_dispatch(query)
                        continue
                self.logger.error(f"No More Nodes Can Be Found,")
                # do_update = False
                break
            else:
                find_chunk_already_in = []
                for attempt in range(self.max_retries):
                    try:
                        temp_ans = False
                        new_node = self.find_next_node_ft(c_node, processed_relations, processed_connection, anchor_chunk_titles, find_chunk_already_in)
                        if self.traversal_mode == "fact_v7" and new_node == "gap:entry":
                            temp_ans = True
                            break
                        if not len(new_node.split(":")) == 2:
                            self.logger.error(f"New Node Format Error. Retrying...({attempt+1}/{self.max_retries})")
                            continue
                        new_node_type = new_node.split(":")[0]
                        new_node_id = new_node.split(":")[1]
                        if (f1_enabled and new_node_type in ("entity", "chunk") and
                                not f1_allowed(c_node, new_node_id)):
                            find_chunk_already_in = [
                                {"role": "assistant", "content": new_node},
                                {"role": "user", "content":
                                 "F1 pruned this transition under the unchanged requirement. "
                                 "Choose another unvisited candidate."}]
                            self.logger.info("F1 pruned repeated transition %s -> %s",
                                             c_node, new_node_id)
                            continue
                        # A frontier candidate is a programmatically verified
                        # edge from a previously visited source node.  Accept
                        # it through the same path ledger instead of treating
                        # it as an invalid local-neighbor response.
                        if (self._frontier_enabled() and
                                new_node in self.frontier_lookup):
                            temp_ans = self._accept_frontier_candidate(new_node)
                            if temp_ans:
                                self.logger.info(
                                    "Frontier candidate accepted %s", new_node)
                                break
                        if a5_continue:
                            graph_state = (tuple(sorted(self.entity)), tuple(sorted(self.chunk)))
                            transition_key = (graph_state, c_node, new_node_type, new_node_id)
                            if transition_key in self.v7_transition_attempts:
                                if os.environ.get("REMIND_B4", "0") == "1":
                                    # Allow one repeated selector response to be
                                    # rejected and retried. Backtrack only after
                                    # the same transition is selected repeatedly.
                                    b4_repeat_counts[transition_key] = (
                                        b4_repeat_counts.get(transition_key, 0) + 1)
                                    repeat_count = b4_repeat_counts[transition_key]
                                    self.logger.info(
                                        "B4 repeated transition %s (%d/%d)",
                                        transition_key[1:], repeat_count, 2)
                                    if repeat_count >= 2:
                                        break
                                else:
                                    self.logger.info(
                                        f"A5 continuation rejects repeated transition: {transition_key[1:]}")
                                find_chunk_already_in = [
                                    {"role": "assistant", "content": new_node},
                                    {"role": "user", "content":
                                     "This exact transition was already tried without new evidence. "
                                     "Choose a different unvisited neighbor or backtrack to another frontier."}
                                ]
                                continue
                        if new_node_type == "chunk" or new_node_type == "entity":
                            for path_iter in processed_path:
                                if path_iter["type"] == "connection" and path_iter["to"] == new_node_id and new_node_type == "chunk":
                                    self.edge.append(f"from entity:{path_iter['from']} to chunk:{path_iter['to']}")
                                    self.path.append({"from":path_iter['from'], "to":path_iter['to'], "id":path_iter['id'], "type":"connection"})
                                    self._b2_record_arrival(self.path[-1], "guided_expansion")
                                    temp_ans= True
                                    break
                                if path_iter["type"] == "relation" and path_iter["to"] == new_node_id and new_node_type == "entity":
                                    self.edge.append(f"from entity:{path_iter['from']} to entity:{path_iter['to']} with relations: {path_iter['documents']}")
                                    self.path.append({"from":path_iter['from'], "to":path_iter['to'], "id":path_iter['id'], "type":"relation"})
                                    self._b2_record_arrival(self.path[-1], "guided_expansion")
                                    temp_ans = True
                                    break
                        if temp_ans:
                            break
                        for entity_check_iter in self.entity:
                            if new_node_type == "entity" and new_node_id == entity_check_iter:
                                temp_ans = True
                                break
                        if temp_ans:
                            break
                        if new_node_type=="chunk":
                            anchor_temp_id = f"anchor-{new_node_id}"
                            # if anchor_temp_id in self.entity:
                            if (anchor_temp_id in self.entity) and (new_node_id not in self.chunk):
                                self.edge.append(f"from entity:{anchor_temp_id} to chunk:{new_node_id}")
                                self.path.append({"from":anchor_temp_id, "to":new_node_id, "id":f"{anchor_temp_id}_{new_node_id}", "type":"connection"})
                                self._b2_record_arrival(self.path[-1], "guided_anchor")
                                temp_ans = True
                                break

                            if new_node_id in self.chunk:
                                find_chunk_already_in = [{"role":"assistant","content":new_node},{"role":"user","content":select_another_node_prompt.format(node = new_node, chunks = self.chunk)}]
                        if temp_ans:
                            break
                        self.logger.error(f"No node was identified during the search for a new node. Retrying...({attempt+1}/{self.max_retries})")
                    except Exception as e:
                        raise RuntimeError(f"No node was identified in Knowledge Graph during the search for a new node.")
                    
            
                if not temp_ans:
                    if self._frontier_enabled():
                        alternative = self._pop_frontier_alternative()
                        if alternative is not None:
                            new_node = alternative["label"]
                            new_node_type, new_node_id = new_node.split(":", 1)
                            temp_ans = self._accept_frontier_candidate(new_node)
                            if temp_ans:
                                self.logger.info(
                                    "Frontier beam fallback selected %s", new_node)
                if not temp_ans:
                    if a5_continue:
                        frontier = a5_frontier(c_node)
                        if frontier:
                            c_node = frontier
                            continue
                    if f1_enabled:
                        frontier = f1_frontier(c_node)
                        if frontier:
                            c_node = frontier
                            self.c_node_list.append(c_node)
                            continue
                        from .fact_v7 import enter, enter_global
                        entry = enter(self, max_jumps - jumps) or enter_global(self, max_jumps - jumps)
                        if entry:
                            c_node, cost = entry
                            jumps += cost
                            enough = self._judge_dispatch(query)
                            continue
                    self.logger.error("Failed to find a valid new node after maximum retries.")
                    do_update = False
                    break

                # ``gap:entry`` is a control-flow sentinel, not an LLM-selected
                # graph node. It therefore has no ``new_node_type`` / ``new_node_id``
                # and must not be recorded as a transition or an A1 decision.
                if new_node != "gap:entry":
                    if f1_enabled and new_node_type in ("entity", "chunk"):
                        f1_mark(c_node, new_node_id)
                    if a5_continue and c_node:
                        graph_state = (tuple(sorted(self.entity)), tuple(sorted(self.chunk)))
                        self.v7_transition_attempts.add(
                            (graph_state, c_node, new_node_type, new_node_id))

                    if (self.traversal_mode in ("memory_a1", "memory_a1a2") or
                            (self.traversal_mode == "fact_v7" and
                             os.environ.get("REMIND_B1", "0") in ("1", "2"))):
                        self._record_a1_decision(processed_path, new_node_type, new_node_id)
            
            if self.traversal_mode == "fact_v7" and new_node == "gap:entry":
                from .fact_v7 import enter
                entry = enter(self, max_jumps - jumps)
                if not entry:
                    do_update = False
                    break
                c_node, cost = entry
                jumps += cost
                enough = self._judge_dispatch(query)
                continue

            self.logger.info(f"Select Next Node: {new_node}")
            self.c_node_list.append(c_node)

            if new_node_type == "chunk":
                c_node = ""
                jumps += 1
                chunk_id = new_node_id
                self.chunk.append(chunk_id)
                self.logger.info(f"Step into Chunk: {chunk_id}")
                self.logger.info(f"Update Chunk Summary with Chunk:{chunk_id}.")
                self.summary.update({f"chunk:{chunk_id}":self.get_chunk_summary(chunk_id)})
                self.logger.debug(f"New Chunk Summary:\n{self.summary}")
                if (self.traversal_mode == "fact_v7" and
                        os.environ.get("REMIND_JUDGE_CURRENT_RAW") == "1"):
                    self._judge_current_raw_chunk_id = chunk_id
            else:
                entity_id = new_node_id
                if entity_id in self.entity:
                    self.logger.info(f"Go Back to Entity: {entity_id}")
                else:
                    jumps += 1
                    self.logger.info(f"Step into Entity: {entity_id}")
                    self.entity.append(entity_id)
                    self.out_dgree[entity_id], has_out_entity=self.get_out_dgree(entity_id)
                    for has_out_entity_iter in has_out_entity:
                        self.out_dgree[has_out_entity_iter] -= 1
                c_node = entity_id
            
            enough = self._judge_dispatch(query)
            self.logger.info(f"Data Enough? {enough}")
            if not enough and self.traversal_mode == "fact_v6" and jumps < max_jumps:
                c_node = self._v6_entry_correction(c_node)

            
            if jumps >= max_jumps:
                self.logger.error(f"Reach the Max Jumps: {max_jumps}")
                # do_update = False
                break
            
            self.logger.info("=============================================================================================================")
        
        

        self.last_jumps = jumps
        self.logger.info(f"Stop Search --- Jumps: {jumps}")
        if os.environ.get('REMIND_EVIDENCE_REPRESENTATION') == 'extractive_batch_v1':
            self._finalize_batch_evidence()
        if self.traversal_mode == "residual_v1":
            from .residual import repair_evidence
            roots = list(self.entity[:search_keys])
            extra_roots = repair_evidence(self, query, search_keys)
            roots = list(dict.fromkeys(roots + extra_roots))
            self.entity = roots + [e for e in self.entity if e not in roots]
            search_keys = len(roots)
            if self.residual_metrics["new_chunks"]:
                jumps = max(1, jumps)
        if self.traversal_mode == "fact_v7":
            roots = list(dict.fromkeys(self.entity[:search_keys] + self.v7_roots))
            self.entity = roots + [e for e in self.entity if e not in roots]
            search_keys = len(roots)
            self.logger.info("V7 metrics: " + json.dumps(self.v7_metrics))
        if self.traversal_mode == "fact_v6" and self._v6_extra_roots:
            roots = list(self.entity[:search_keys])
            for v6_root in self._v6_extra_roots:
                if v6_root not in roots:
                    roots.append(v6_root)
            self.entity = roots + [e for e in self.entity if e not in roots]
            search_keys = len(roots)
        if self.traversal_mode == "fact_v6":
            self._v6_metrics["roots_added"] = list(self._v6_extra_roots)
            self.logger.info(f"V6 metrics: {self._v6_metrics}")
        c2_trace_path = os.environ.get("REMIND_C2_TRACE_OUT", "")
        if c2_trace_path and self._b2_enabled():
            self.c2_trace = self._make_c2_trace(enough)
            parent = os.path.dirname(c2_trace_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(c2_trace_path, "w", encoding="utf-8") as f:
                json.dump(self.c2_trace, f, ensure_ascii=False, indent=2)
            self.logger.info("C2 deferred trace written: " + c2_trace_path)
        if os.environ.get("REMIND_C2_DEFER_UPDATE", "0") == "1" and do_update:
            self.logger.info("C2: defer persistent graph update until cross-round pair validation.")
            do_update = False
        b2_has_quick_evidence = (
            self._b2_enabled() and
            bool(getattr(self, "b2_supported_chunks", set()) or
                 getattr(self, "b2_supported_entities", set()) or
                 getattr(self, "b2_supported_edges", set()))
        )
        if jumps == 0 and not b2_has_quick_evidence:
            self.logger.info(f"No LLM-Guided Traversal, Skip Update.")
            do_update = False
        elif jumps == 0 and b2_has_quick_evidence:
            # B2's ledger explicitly covers evidence found in the initial
            # quick subgraph; allow its verified source path to be written
            # even when no later LLM-guided hop was needed. Other modes retain
            # the release behavior above.
            self.logger.info("B2: verified quick-subgraph evidence; allow source-ledger update with 0 guided jumps.")
        if self.traversal_mode in ("memory_a2", "memory_a1a2"):
            path_match = self._a2_path_write_gate()
            if do_update:
                do_update = path_match
        self.logger.debug(f"Start Node: {self.entity[:search_keys]}")
        self.logger.debug(f"Entities : {self.entity}")
        self.logger.debug(f"Chunks : {self.chunk}")
        self.logger.debug(f"Paths:\n{self.path}")
        if do_update:
            self.logger.info("Update Knowledge Graph.")

            final_confirm_paths = []
            punish_paths = []
            b2_enabled = self._b2_enabled()
            if b2_enabled:
                # B2 reuses source-verified facts from the sufficiency judge;
                # no second LLM claim/edge update request is needed.
                self.update_dict = {"edges": [],
                                    "chunks": sorted(getattr(self, "b2_supported_chunks", set()))}
                self.logger.info("B2 direct evidence update: "
                                 f"supported_chunks={self.update_dict['chunks']} "
                                 f"supported_entities={sorted(getattr(self, 'b2_supported_entities', set()))} "
                                 f"supported_graph_edges={len(getattr(self, 'b2_supported_edges', set()))}")
            else:
                self.update_dict = self.get_update_relation_and_chunk()
            str_chunk_list = []
            for update_chunk_iter in self.update_dict["chunks"]:
                str_chunk_list.append(str(update_chunk_iter))
            self.update_dict["chunks"] = str_chunk_list

            self.logger.debug(f"Update Dict: {self.update_dict}")
            
            if not b2_enabled:
                for entity_iter in range(search_keys):
                    self.viewed_nodes = []
                    self.confirm_paths = []
                    if not self.get_update_dfs(self.entity[entity_iter]):
                        self.logger.warning(f"Update: Can't Find Available Path (Start with {self.entity[entity_iter]}).")
                    else:
                        final_confirm_paths = final_confirm_paths + self.confirm_paths
                    
            if self.traversal_mode in ("memory_a1", "memory_a1a2"):
                # A path is positive memory exactly as in the release code.
                # Negative memory is local contrast: only siblings shown to
                # the LLM and rejected at a confirmed decision are punished.
                confirmed = set(final_confirm_paths)
                for decision in self.a1_decision_records:
                    selected = decision["selected"]
                    if selected not in confirmed:
                        # The selected branch did not survive final
                        # attribution, so this decision is uncertain.
                        continue
                    for candidate in decision["candidates"]:
                        if candidate != selected and candidate not in confirmed:
                            punish_paths.append(candidate)
                final_confirm_paths = list(dict.fromkeys(final_confirm_paths))
                punish_paths = list(dict.fromkeys(punish_paths))
                self.logger.info(
                    f"A1 local memory: decisions={len(self.a1_decision_records)}, "
                    f"enhance={len(final_confirm_paths)}, punish={len(punish_paths)}"
                )
            elif b2_enabled:
                final_confirm_paths, b2_report = self._b2_confirmed_paths(
                    getattr(self, "b2_supported_chunks", set()),
                    getattr(self, "b2_supported_entities", set()),
                    getattr(self, "b2_supported_edges", set()))
                memory_label = "B3" if self._b3_enabled() else "B2"
                self.logger.info(f"{memory_label} source-ledger attribution: " +
                                 json.dumps(b2_report, ensure_ascii=False))
                confirmed = set(final_confirm_paths)
                if (os.environ.get("REMIND_D1", "0") == "1" and enough and
                        (getattr(self, "b2_supported_chunks", set()) or
                         getattr(self, "b2_supported_entities", set()) or
                         getattr(self, "b2_supported_edges", set()))):
                    punish_paths, d1_report = self._d1_unrouted_quick_chunk_edges(
                        self.b2_supported_chunks, confirmed,
                        getattr(self, "b2_supported_entities", set()),
                        getattr(self, "b2_supported_edges", set()))
                    self.logger.info("D1 quick-subgraph negative feedback: " +
                                     json.dumps(d1_report, ensure_ascii=False))
                elif os.environ.get("REMIND_D1", "0") == "1":
                    self.logger.info(
                        "D1 skipped: final sufficiency not confirmed or no quote-validated evidence chunks.")
                elif self._b3_enabled():
                    # B3 keeps B2's positive, ledger-attributed enhancement,
                    # but intentionally disables all negative edge updates.
                    punish_paths = []
                    self.logger.info("B3: negative punishment disabled.")
                else:
                    for decision in self.a1_decision_records:
                        selected = decision["selected"]
                        if selected not in confirmed:
                            continue
                        for candidate in decision["candidates"]:
                            if candidate != selected and candidate not in confirmed:
                                punish_paths.append(candidate)
                final_confirm_paths = list(dict.fromkeys(final_confirm_paths))
                punish_paths = list(dict.fromkeys(punish_paths))
                self.logger.info(
                    f"{memory_label} memory: "
                    f"confirmed={len(final_confirm_paths)}, "
                    f"punished={len(punish_paths)}")
            elif (self.traversal_mode == "fact_v7" and
                  os.environ.get("REMIND_B1", "0") == "1"):
                roots = list(dict.fromkeys(
                    list(self.entity[:search_keys]) +
                    list(getattr(self, "v7_roots", []))))
                verified_entities = {str(item.get("source_id"))
                                     for item in getattr(self, "v7_verified_facts", [])
                                     if item.get("source_type") == "entity"}
                verified_edges = {str(item.get("source_id"))
                                  for item in getattr(self, "v7_verified_facts", [])
                                  if item.get("source_type") == "edge"}
                final_confirm_paths, b1_report = self._b1_confirmed_paths(
                    self.update_dict.get("chunks", []), roots,
                    verified_entities, verified_edges)
                self.logger.info("B1 evidence attribution: " +
                                 json.dumps(b1_report, ensure_ascii=False))
                confirmed = set(final_confirm_paths)
                # A1 negative memory: only punish alternatives actually shown
                # to the LLM after the selected edge has confirmed provenance.
                for decision in self.a1_decision_records:
                    selected = decision["selected"]
                    if selected not in confirmed:
                        continue
                    for candidate in decision["candidates"]:
                        if candidate != selected and candidate not in confirmed:
                            punish_paths.append(candidate)
                final_confirm_paths = list(dict.fromkeys(final_confirm_paths))
                punish_paths = list(dict.fromkeys(punish_paths))
                self.logger.info(
                    f"B1/A1 memory: confirmed={len(final_confirm_paths)}, "
                    f"explicitly_rejected={len(punish_paths)}")
            else:
                for path_iter in self.path:
                    path_iter_tuple = (path_iter["type"],path_iter["id"])
                    if path_iter_tuple not in final_confirm_paths:
                        punish_paths.append(path_iter_tuple)

            self.logger.info(f"Enhance {len(final_confirm_paths)} Edges, Punish {len(punish_paths)}.")
            
            if self.traversal_mode == "residual_v1":
                final_confirm_paths = list(dict.fromkeys(final_confirm_paths))
                punish_paths = list(dict.fromkeys(punish_paths))
            if self.v16_memory_plan is not None:
                self.v16_memory_plan.update({
                    "query": str(self.database_query),
                    "enhance_edges": list(dict.fromkeys(final_confirm_paths)),
                    "punish_edges": ([] if self._b3_enabled() else
                                     list(dict.fromkeys(punish_paths))),
                })
                self.logger.info("V16: memory planned; no persistent writes during retrieval.")
            else:
                self.database.enhance_edge_weight(self.database_query, final_confirm_paths)
                self.logger.debug(f"Enhance Edges:\n{final_confirm_paths}")
                if self._b3_enabled():
                    self.logger.info("B3: punishment update call omitted.")
                else:
                    self.database.punish_edge_weight(self.database_query, punish_paths)
                    self.logger.debug(f"Punish Edges:\n{punish_paths}")
        else:
            self.logger.info("Skip Update.")

        if self.traversal_mode == "fact_v7":
            from .fact_v7 import finalize_temp_summary
            finalize_temp_summary(self)
                
        return self.summary, self.edge

    def _finalize_batch_evidence(self):
        """Replace raw traversal documents with one-question evidence extracts."""
        if not self._batch_documents:
            return
        from .extractive_evidence import get_batch_evidence
        from .extractive_evidence import sentence_spans
        documents = [(chunk_id, document, tokens)
                     for chunk_id, (document, tokens) in self._batch_documents.items()]
        evidence = get_batch_evidence(self, documents)
        for chunk_id, document, _tokens in documents:
            self.summary[f"chunk:{chunk_id}"] = evidence[str(chunk_id)]
            self._record_v7_chunk_evidence(chunk_id, evidence[str(chunk_id)])
        extraction_needed = any(
            tokens >= self.chunk_summary_threshold and sentence_spans(document)
            for _cid, (document, tokens) in self._batch_documents.items()
        )
        self.logger.info("Batch evidence finalized: chunks=%s, extraction_calls=%s",
                         len(documents), 1 if extraction_needed else 0)



    def get_chunk_summary(self, chunk_id):
        chunk_data = self.database.chunk_collection.get(ids = [chunk_id], include=['documents','metadatas'])
        chunk_tokens = chunk_data["metadatas"][0]['tokens']
        chunk_document = chunk_data["documents"][0]
        representation = os.environ.get('REMIND_EVIDENCE_REPRESENTATION')
        if representation == 'extractive_batch_v1':
            self._batch_documents[str(chunk_id)] = (chunk_document, chunk_tokens)
            if chunk_tokens < self.chunk_summary_threshold:
                self._record_v7_chunk_evidence(chunk_id, chunk_document)
                return chunk_document
            input_msg = chunk_summary_prompt.format(
                query=self.query,
                entity_list=str(self.entity),
                edge_list=str(self.edge),
                chunk_summary=self.summary,
                chunk_document=chunk_document,
            )
            evidence = self.agent.generate_response('', [{'role': 'user', 'content': input_msg}])
            self._record_v7_chunk_evidence(chunk_id, evidence)
            return evidence
        if representation in ('extractive_v1', 'extractive_gap_v1'):
            from .extractive_evidence import get_evidence
            gap_aware = representation == 'extractive_gap_v1'
            v14a_prompt = os.environ.get('REMIND_V14A', '0') == '1'
            current_gap = (getattr(self, 'v7_gap', None)
                           if (gap_aware or v14a_prompt) and
                           self.traversal_mode == 'fact_v7' else None)
            if (gap_aware and self.traversal_mode == 'fact_v7' and
                    os.environ.get('REMIND_V12B', '0') == '1'):
                current_gap = getattr(self, 'v7_missing_items', [])
            if (gap_aware and not v14a_prompt and self.traversal_mode == 'fact_v7' and
                    os.environ.get('REMIND_V12A_DISABLE_GAP', '0') == '1'):
                current_gap = None
            evidence = get_evidence(self, chunk_id, chunk_document, chunk_tokens,
                                    current_gap=current_gap, gap_aware=gap_aware,
                                    v11_context=self._v11_context_compaction_enabled(),
                                    v12_prompt=os.environ.get('REMIND_V12_PROMPT', '0') == '1',
                                    v14a_prompt=v14a_prompt,
                                    v12b_prompt=os.environ.get('REMIND_V12B', '0') == '1')
            self._record_v7_chunk_evidence(chunk_id, evidence)
            return evidence
        if chunk_tokens < self.chunk_summary_threshold:
            self._record_v7_chunk_evidence(chunk_id, chunk_document)
            return chunk_document
        

        input_msg = chunk_summary_prompt.format(
            query = self.query,
            entity_list = str(self.entity),
            edge_list = str(self.edge),
            chunk_summary = self.summary,
            chunk_document = chunk_document
        )
        response = self.agent.generate_response("", [{"role":"user","content":input_msg}])
        self._record_v7_chunk_evidence(chunk_id, response)
        return response

    def _v11_context_compaction_enabled(self):
        return (self.traversal_mode == 'fact_v7' and
                os.environ.get('REMIND_V11_CONTEXT_COMPACTION', '0') == '1')

    def _v12_context_compaction_enabled(self):
        return (self.traversal_mode == 'fact_v7' and
                os.environ.get('REMIND_V12_PROMPT', '0') == '1')

    def _v11_context_chunks_not_judged(self):
        if not self._v11_context_compaction_enabled():
            return []
        contexts = getattr(self, '_v11_context_evidence', {})
        judged = getattr(self, '_v11_context_judged_chunks', set())
        return sorted(set(contexts) - set(judged))

    def _v12_context_chunks_not_judged(self):
        if not self._v12_context_compaction_enabled():
            return []
        contexts = getattr(self, '_v12_context_evidence', {})
        judged = getattr(self, '_v12_context_judged_chunks', set())
        return sorted(set(contexts) - set(judged))

    def _v12_prompt_summary(self, for_judge=False):
        """Use A5's full first-read window once, then selected compact evidence."""
        summary = dict(self.summary)
        if not self._v12_context_compaction_enabled():
            return summary
        contexts = getattr(self, '_v12_context_evidence', {})
        judged = getattr(self, '_v12_context_judged_chunks', set())
        for key in list(summary):
            chunk_id = str(key).removeprefix('chunk:')
            context = contexts.get(chunk_id)
            if context:
                summary[key] = (context['expanded']
                                if for_judge and chunk_id not in judged
                                else context['compact'])
        return summary

    def _v12_mark_context_judged(self, chunk_ids):
        if not self._v12_context_compaction_enabled():
            return
        judged = getattr(self, '_v12_context_judged_chunks', None)
        if judged is None:
            judged = self._v12_context_judged_chunks = set()
        judged.update(str(chunk_id) for chunk_id in chunk_ids)

    def _v11_prompt_summary(self, for_judge=False):
        """Use full first-read context once, then carry the compact source text."""
        summary = dict(self.summary)
        if not self._v11_context_compaction_enabled():
            return summary
        contexts = getattr(self, '_v11_context_evidence', {})
        judged = getattr(self, '_v11_context_judged_chunks', set())
        for key, value in list(summary.items()):
            chunk_id = str(key).removeprefix('chunk:')
            context = contexts.get(chunk_id)
            if not context:
                continue
            if for_judge and chunk_id not in judged:
                summary[key] = context['expanded']
            else:
                summary[key] = context['compact']
        return summary

    def _v11_mark_context_judged(self, chunk_ids):
        if not self._v11_context_compaction_enabled():
            return
        judged = getattr(self, '_v11_context_judged_chunks', None)
        if judged is None:
            judged = self._v11_context_judged_chunks = set()
        judged.update(str(chunk_id) for chunk_id in chunk_ids)

    def _v11_evidence_state_key(self, query, summary):
        return json.dumps([query, sorted(set(self.entity)), sorted(set(self.edge)),
                           sorted(set(self.chunk)), summary], sort_keys=True)

    def _record_v7_chunk_evidence(self, chunk_id, evidence):
        """Persist evidence entering the current query's temporary ledger."""
        if self.traversal_mode == "fact_v7":
            from .fact_v7 import record_chunk_evidence
            record_chunk_evidence(self, chunk_id, evidence)

    def _frontier_enabled(self):
        """Whether the isolated baseline retrieval probe is active."""
        return (self.traversal_mode == "original" and
                os.environ.get("REMIND_GLOBAL_FRONTIER", "0") == "1")

    def _prepare_global_frontier(self, query, current_node, processed_path):
        """Collect a bounded global candidate frontier without an LLM call.

        The release traversal exposes only the current node's adjacency to the
        selector. This probe also exposes unexplored neighbors of recently
        visited nodes, while leaving the quick subgraph and memory state alone.
        """
        visited_entities = {str(node) for node in self.entity}
        visited_chunks = {str(chunk) for chunk in self.chunk}
        source_nodes = []
        for node in [current_node] + list(reversed(self.c_node_list)):
            if node and node not in source_nodes:
                source_nodes.append(node)
        max_sources = int(os.environ.get("REMIND_FRONTIER_SOURCES", "8"))
        max_candidates = int(os.environ.get("REMIND_FRONTIER_CANDIDATES", "6"))
        candidates = {}
        for source in source_nodes[:max_sources]:
            try:
                relations, connection = self.database.get_entity_edges(query, source)
            except Exception as exc:
                self.logger.debug("Frontier read failed at %s: %s", source, exc)
                continue
            for relation in relations:
                target = str(relation.get("to", ""))
                if not target or target in visited_entities:
                    continue
                label = f"entity:{target}"
                item = {
                    "label": label,
                    "source": source,
                    "type": "relation",
                    "id": str(relation.get("id", f"{source}_{target}")),
                    "to": target,
                    "documents": relation.get("documents", ""),
                    "score": float(relation.get("weight", 0.0) or 0.0),
                }
                previous = candidates.get(label)
                if previous is None or item["score"] > previous["score"]:
                    candidates[label] = item
            if connection:
                target = str(connection.get("to", ""))
                if target and target not in visited_chunks:
                    label = f"chunk:{target}"
                    item = {
                        "label": label,
                        "source": source,
                        "type": "connection",
                        "id": str(connection.get("id", f"{source}_{target}")),
                        "to": target,
                        "documents": connection.get("documents", ""),
                        "score": float(connection.get("weight", 0.0) or 0.0),
                    }
                    previous = candidates.get(label)
                    if previous is None or item["score"] > previous["score"]:
                        candidates[label] = item
        ranked = sorted(candidates.values(),
                        key=lambda item: (-item["score"], item["label"]))
        self.frontier_candidates = ranked[:max_candidates]
        self.frontier_lookup = {item["label"]: item
                                for item in self.frontier_candidates}
        self.logger.info(
            "Global retrieval frontier: sources=%d candidates=%s",
            len(source_nodes[:max_sources]),
            [item["label"] for item in self.frontier_candidates])
        return self.frontier_candidates

    def _accept_frontier_candidate(self, label):
        item = self.frontier_lookup.get(label)
        if item is None or label in self.frontier_selected:
            return False
        self.frontier_selected.add(label)
        edge_id = item["id"]
        if not any(str(path.get("id")) == edge_id for path in self.path):
            if item["type"] == "relation":
                self.edge.append(
                    f"from entity:{item['source']} to entity:{item['to']} "
                    f"with relations: {item['documents']}")
                self.path.append({"from": item["source"], "to": item["to"],
                                  "id": edge_id,
                                  "documents": item["documents"],
                                  "type": "relation"})
            else:
                self.edge.append(
                    f"from entity:{item['source']} to chunk:{item['to']}")
                self.path.append({"from": item["source"], "to": item["to"],
                                  "id": edge_id, "type": "connection"})
        return True

    def _pop_frontier_alternative(self):
        for item in self.frontier_candidates:
            if item["label"] not in self.frontier_selected:
                return item
        return None

    @unpack_cot_ans
    def judge_sufficient_information(self, error_chat_history = None):
        prompt = (v14a_sufficiency_prompt if
                  os.environ.get('REMIND_V14A', '0') == '1' and
                  self.traversal_mode == 'fact_v7' else
                  v12_sufficiency_prompt if
                  os.environ.get('REMIND_V12_PROMPT', '0') == '1' and
                  self.traversal_mode == 'fact_v7' else
                  judge_sufficient_information_prompt)
        input_msg = prompt.format(
            query = self.query,
            entity_list = str(self.entity),
            chunk_list = str(self.chunk),
            edge_list = str(self.edge),
            chunk_summary = (self._v12_prompt_summary(for_judge=True)
                             if self._v12_context_compaction_enabled() else
                             self._v11_prompt_summary(for_judge=True)),
            verified_facts=str(getattr(self, 'v7_verified_facts', [])),
            previous_gap=str(getattr(self, 'v7_previous', None))
        )
        response = self.agent.generate_response("", [{"role":"user","content":input_msg}]+ (error_chat_history or []))
        self.logger.debug(f"Function judge_sufficient_information Output:\n{response}")
        return response

    @unpack_cot_ans
    def find_next_node(self, c_node, relation, connection, anchor_chunk_titles, find_chunk_already_in, error_chat_history = None):
        prompt = (v14a_find_next_node_prompt if
                  os.environ.get('REMIND_V14A', '0') == '1' else
                  v12_find_next_node_prompt if
                  os.environ.get('REMIND_V12_PROMPT', '0') == '1' else
                  find_next_node_prompt)
        input_msg = prompt.format(
            query = self.query,
            entity_list = str(self.entity),
            chunk_list = str(self.chunk),
            edge_list = str(self.edge),
            anchor_chunk_titles = str(anchor_chunk_titles),
            chunk_summary = (self._v12_prompt_summary()
                             if self._v12_context_compaction_enabled() else
                             self._v11_prompt_summary()),
            c_node = c_node,
            c_node_list = str(self.c_node_list),
            relation_cnode = str(relation),
            connection_cnode = str(connection),
            previous_gap = str(getattr(self, 'v7_previous', None))
        )
        if self._frontier_enabled() and self.frontier_candidates:
            lines = [
                "\nGLOBAL RETRIEVAL FRONTIER (programmatically gathered; admissible candidates):"
            ]
            for item in self.frontier_candidates:
                lines.append(
                    f"- {item['label']} | from={item['source']} | "
                    f"weight={item['score']:.4f} | edge={item['id']}")
            lines.append(
                "Choose an exact candidate label from the local adjacency or this "
                "global frontier. Do not invent a node.")
            input_msg += "\n" + "\n".join(lines)
        if self.traversal_mode == "fact_v7":
            from .fact_v7 import selector_suffix
            input_msg += selector_suffix(self)
        response = self.agent.generate_response("", [{"role":"user","content":input_msg}] + find_chunk_already_in + (error_chat_history or []))
        self.logger.debug(f"Function find_next_node Output:\n{response}")
        return response
    
    @check_keys("edges","chunks")
    @unpack_cot_ans
    def get_update_relation_and_chunk(self, error_chat_history = None):
        input_msg = reward_or_punishment_prompt.format(
            query = self.query,
            entity_list = str(self.entity),
            chunk_list = str(self.chunk),
            edge_list = str(self.path),
            chunk_summary = self.summary,
        )
        response = self.agent.generate_response("", [{"role":"user","content":input_msg}] + (error_chat_history or []))
        self.logger.debug(f"Function get_update_relation_and_chunk Output:\n{response}")
        return response
    
    def get_update_dfs(self, node_id):
        if node_id in self.viewed_nodes:
            return False
        ans = False
        for node_path in self.path:
            if node_path["from"] == node_id:
                if node_path["to"] in self.viewed_nodes:
                    continue
                if node_path["type"] == "relation":
                    temp_dfs_ans = self.get_update_dfs(node_path["to"])
                    if (not ans) and (node_path["id"] in self.update_dict["edges"] or temp_dfs_ans):
                        self.confirm_paths.append(("relation",node_path["id"]))
                        ans = True
                else:
                    if (not ans) and node_path["to"] in self.update_dict["chunks"]:
                        self.confirm_paths.append(("connection",node_path["id"]))
                        ans = True

                    
        self.viewed_nodes.append(node_id)
        return ans

    def _b1_confirmed_paths(self, claimed_chunks, roots, claimed_entities=(),
                            claimed_graph_edges=()):
        """Resolve cited chunk/entity/edge evidence through traversed graph edges.

        A target is reinforced only when exactly one simple directed path from
        the run's search roots reaches it. Enumeration stops at two paths;
        ambiguous and unreachable claims remain neutral (no reward/punishment).
        """
        adjacency = {}
        for edge in self.path:
            adjacency.setdefault(str(edge["from"]), []).append(edge)

        report = {"claimed": [], "unique": [], "ambiguous": [], "unreachable": [],
                  "claimed_entities": [], "unique_entities": [],
                  "ambiguous_entities": [], "unreachable_entities": [],
                  "supported_graph_edges": [], "unmatched_graph_edges": []}
        confirmed = []
        for raw_target in claimed_chunks:
            target = str(raw_target).removeprefix("chunk:").strip()
            if not target or target in report["claimed"]:
                continue
            report["claimed"].append(target)
            found = []

            def walk(node, visited_nodes, path):
                if len(found) >= 2:
                    return
                for edge in adjacency.get(str(node), []):
                    next_node = str(edge["to"])
                    if edge["type"] == "connection" and next_node == target:
                        found.append(path + [edge])
                        if len(found) >= 2:
                            return
                    elif edge["type"] == "relation" and next_node not in visited_nodes:
                        walk(next_node, visited_nodes | {next_node}, path + [edge])
                        if len(found) >= 2:
                            return

            for root in roots:
                walk(str(root), {str(root)}, [])
                if len(found) >= 2:
                    break
            if len(found) == 1:
                report["unique"].append(target)
                confirmed.extend((edge["type"], edge["id"]) for edge in found[0])
            elif len(found) > 1:
                report["ambiguous"].append(target)
            else:
                report["unreachable"].append(target)
        # Entity labels are valid evidence only if they are actual visited graph
        # nodes. Reinforce their unique traversed provenance path.
        for target in sorted(set(map(str, claimed_entities))):
            if target not in self.entity:
                report["unreachable_entities"].append(target)
                continue
            report["claimed_entities"].append(target)
            found = []

            def walk_entity(node, visited_nodes, path):
                if len(found) >= 2:
                    return
                for edge in adjacency.get(str(node), []):
                    if edge["type"] != "relation":
                        continue
                    next_node = str(edge["to"])
                    next_path = path + [edge]
                    if next_node == target:
                        found.append(next_path)
                    elif next_node not in visited_nodes:
                        walk_entity(next_node, visited_nodes | {next_node}, next_path)

            if target in roots:
                found.append([])
            else:
                for root in roots:
                    walk_entity(str(root), {str(root)}, [])
                    if len(found) >= 2:
                        break
            if len(found) == 1:
                report["unique_entities"].append(target)
                confirmed.extend((edge["type"], edge["id"]) for edge in found[0])
            elif len(found) > 1:
                report["ambiguous_entities"].append(target)
            else:
                report["unreachable_entities"].append(target)
        # A directly cited graph edge is reinforced only when the exact edge
        # string maps to an edge traversed during this question.
        for cited in sorted(set(map(str, claimed_graph_edges))):
            matched = None
            for edge in self.path:
                if edge["type"] == "relation":
                    rendered = (f"from entity:{edge.get('from')} to entity:{edge.get('to')} "
                                f"with relations: {edge.get('documents', '')}")
                elif edge["type"] == "connection":
                    rendered = f"from entity:{edge.get('from')} to chunk:{edge.get('to')}"
                else:
                    continue
                if rendered == cited and edge.get("id"):
                    matched = edge
                    break
            if matched:
                report["supported_graph_edges"].append(cited)
                confirmed.append((matched["type"], matched["id"]))
            else:
                report["unmatched_graph_edges"].append(cited)
        return list(dict.fromkeys(confirmed)), report



    def get_anchor_chunk_title(self, paths):
        anchor_chunk_title = []
        for entity_iter in self.entity:
            if entity_iter.startswith("anchor"):
                chunk_title = self.database.entity_collection.get(ids=[entity_iter],include=["documents"])["documents"][0]
                anchor_chunk_title.append(f"{entity_iter} connect to chunk which have document:{chunk_title}")

        for path in paths:
            if path["type"] == "relation":
                if path["to"].startswith("anchor"):
                    chunk_title = self.database.entity_collection.get(ids=[path["to"]],include=["documents"])["documents"][0]
                    anchor_chunk_title.append(f"{path['to']} connect to chunk which have document:{chunk_title}")
            else:
                if path["from"].startswith("anchor"):
                    chunk_title = self.database.entity_collection.get(ids=[path["from"]],include=["documents"])["documents"][0]
                    anchor_chunk_title.append(f"{path['from']} connect to chunk:{path['to']} which have document:{chunk_title}")
        
        return anchor_chunk_title
    

    def get_out_dgree(self, entity_id):
        out_dgree = 0
        has_out_entity = []
        relations = self.database.relation_collection.get(
            where={
                "$or": [
                    {"subject_entity_id": entity_id},
                    {"object_entity_id": entity_id}
                ]
            },
            include=['metadatas']
        )

        for relation_iter in range(len(relations["ids"])):
            
            another_node = ""
            if relations["metadatas"][relation_iter]["subject_entity_id"] == entity_id:
                another_node = relations["metadatas"][relation_iter]["object_entity_id"]
            else:
                another_node = relations["metadatas"][relation_iter]["subject_entity_id"]
            
            if another_node not in self.entity:
                out_dgree += 1
            else:
                has_out_entity.append(another_node)
        connection = self.database.connection_collection.get(where={"entity_id":entity_id}, include=['metadatas', 'documents', 'embeddings'])
        if connection["ids"]:
            if str(connection["metadatas"][0]["chunk_id"]) not in self.chunk:
                out_dgree += 1

        return out_dgree, has_out_entity

    # ------------------------------------------------------------------
    # FGAT v2 dispatchers: the sub-question orchestrator (get_query_ans_fact)
    # replaces the whole loop in fact mode, so these per-hop calls are pure
    # pass-throughs kept only for call-site compatibility.
    # ------------------------------------------------------------------

    def judge_sufficient_information_ft(self, error_chat_history=None):
        return self.judge_sufficient_information(error_chat_history=error_chat_history)

    def find_next_node_ft(self, c_node, relation, connection, anchor_chunk_titles, find_chunk_already_in):
        return self.find_next_node(c_node, relation, connection, anchor_chunk_titles, find_chunk_already_in)

    def _record_a1_decision(self, processed_path, new_node_type, new_node_id):
        """Record one actual LLM comparison for A1's local negative memory."""
        candidates = [(p["type"], p["id"]) for p in processed_path]
        if len(candidates) < 2:
            return
        selected = None
        for p in processed_path:
            path_node_type = "entity" if p["type"] == "relation" else "chunk"
            if path_node_type == new_node_type and p["to"] == new_node_id:
                selected = (p["type"], p["id"])
                break
        if selected is None:
            return
        self.a1_decision_records.append({
            "candidates": list(dict.fromkeys(candidates)),
            "selected": selected,
        })

    def _b2_enabled(self):
        return (self.traversal_mode == "fact_v7" and
                os.environ.get("REMIND_B1", "0") == "2")

    def _b3_enabled(self):
        return self._b2_enabled() and os.environ.get("REMIND_B3", "0") == "1"

    def _b2_record_root(self, node_id, source):
        if not self._b2_enabled() or node_id is None:
            return
        node_id = str(node_id)
        if not any(r["node"] == node_id for r in self.b2_roots):
            self.b2_roots.append({"node": node_id, "source": str(source)})

    def _b2_record_arrival(self, edge, source):
        """Record one real parent->child introduction event for B2."""
        if not self._b2_enabled() or not edge:
            return
        target = str(edge.get("to", ""))
        parent = str(edge.get("from", ""))
        if not target or not parent:
            return
        record = {
            "parent": parent, "edge_type": str(edge.get("type", "")),
            "edge_id": str(edge.get("id", "")), "source": str(source)
        }
        bucket = self.b2_arrivals.setdefault(target, [])
        if record not in bucket:
            bucket.append(record)

    def _b2_confirmed_paths(self, supported_chunks, supported_entities=(),
                            supported_edges=()):
        """Attribute positive memory to graph paths/components accepted as evidence."""
        roots = {str(r["node"]) for r in self.b2_roots}
        report = {"supported": [], "reachable": [], "unreachable": [],
                  "supported_entities": [], "reachable_entities": [],
                  "unreachable_entities": [], "supported_graph_edges": [],
                  "unmatched_graph_edges": [], "sources": {}, "enhance_edges": 0}
        confirmed = []

        def walk(node, seen):
            node = str(node)
            if node in roots:
                return [[]]
            if node in seen:
                return []
            paths = []
            for arrival in self.b2_arrivals.get(node, []):
                for prefix in walk(arrival["parent"], seen | {node}):
                    paths.append(prefix + [(arrival["edge_type"], arrival["edge_id"])])
            return paths

        def add_node_target(raw, label, reachable_key, unreachable_key):
            target = str(raw).removeprefix("chunk:")
            report[label].append(target)
            paths = walk(target, set())
            if not paths:
                report[unreachable_key].append(target)
                return
            report[reachable_key].append(target)
            report["sources"][target] = len(paths)
            for path in paths:
                confirmed.extend(path)

        for raw in sorted(set(str(c) for c in supported_chunks)):
            target = raw.removeprefix("chunk:")
            add_node_target(target, "supported", "reachable", "unreachable")
        for raw in sorted(set(str(e) for e in supported_entities)):
            add_node_target(raw, "supported_entities", "reachable_entities",
                            "unreachable_entities")
        # Edge facts directly support the exact graph edge they cite. Resolve
        # the supplied edge string to its traversed edge ID before reinforcing.
        for cited in sorted(set(str(e) for e in supported_edges)):
            matched = False
            for path_edge in self.path:
                if path_edge.get("type") == "relation":
                    rendered = (f"from entity:{path_edge.get('from')} to entity:{path_edge.get('to')} "
                                f"with relations: {path_edge.get('documents', '')}")
                elif path_edge.get("type") == "connection":
                    rendered = f"from entity:{path_edge.get('from')} to chunk:{path_edge.get('to')}"
                else:
                    continue
                if rendered == cited and path_edge.get("id"):
                    confirmed.append((str(path_edge["type"]), str(path_edge["id"])))
                    report["supported_graph_edges"].append(cited)
                    matched = True
                    break
            if not matched:
                report["unmatched_graph_edges"].append(cited)
        confirmed = list(dict.fromkeys(confirmed))
        report["enhance_edges"] = len(confirmed)
        return confirmed, report

    def _make_c2_trace(self, enough):
        """Export current-question provenance for the external C2 round gate."""
        roots = {str(r["node"]) for r in self.b2_roots}

        def walk(node, seen):
            node = str(node)
            if node in roots:
                return [[]]
            if node in seen:
                return []
            result = []
            for arrival in self.b2_arrivals.get(node, []):
                for prefix in walk(arrival["parent"], seen | {node}):
                    result.append(prefix + [{
                        "type": arrival["edge_type"], "id": arrival["edge_id"],
                        "parent": arrival["parent"], "child": node,
                        "source": arrival["source"]}])
            return result

        paths = {}
        for raw in sorted({str(c).removeprefix("chunk:")
                           for c in self.b2_supported_chunks}):
            paths[raw] = walk(raw, set())
        paths_by_graph = {}
        for raw in sorted(getattr(self, "b2_supported_entities", set())):
            paths_by_graph[f"entity::{raw}"] = walk(str(raw), set())
        supported_graph_edges = []
        for cited in sorted(getattr(self, "b2_supported_edges", set())):
            for path_edge in self.path:
                if path_edge.get("type") == "relation":
                    rendered = (f"from entity:{path_edge.get('from')} to entity:{path_edge.get('to')} "
                                f"with relations: {path_edge.get('documents', '')}")
                elif path_edge.get("type") == "connection":
                    rendered = f"from entity:{path_edge.get('from')} to chunk:{path_edge.get('to')}"
                else:
                    continue
                if rendered == cited and path_edge.get("id"):
                    key = (str(path_edge["type"]), str(path_edge["id"]))
                    supported_graph_edges.append({"type": key[0], "id": key[1]})
                    paths_by_graph[f"{key[0]}::{key[1]}"] = [[{
                        "type": key[0], "id": key[1], "source": "graph_evidence"}]]
                    break
        graph_path_edges = {}
        for routes in paths_by_graph.values():
            for route in routes:
                for edge in route:
                    if edge.get("type") in ("relation", "connection") and edge.get("id"):
                        key = (str(edge["type"]), str(edge["id"]))
                        graph_path_edges[key] = {"type": key[0], "id": key[1]}
        supported_graph_edges = list(graph_path_edges.values())
        quick_entries = {}
        quick_routes = {}

        def walk_quick(node, seen):
            node = str(node)
            if node in roots:
                return [[]]
            if node in seen:
                return []
            result = []
            for arrival in self.b2_arrivals.get(node, []):
                if arrival["source"] not in ("quick_subgraph", "quick_anchor"):
                    continue
                for prefix in walk_quick(arrival["parent"], seen | {node}):
                    result.append(prefix + [{
                        "type": arrival["edge_type"], "id": arrival["edge_id"],
                        "parent": arrival["parent"], "child": node,
                        "source": arrival["source"]}])
            return result

        for raw in sorted({str(c).removeprefix("chunk:")
                           for c in self.b2_quick_chunks}):
            quick_entries[raw] = [{
                "type": str(a["edge_type"]), "id": str(a["edge_id"]),
                "parent": str(a["parent"]), "child": raw,
                "source": str(a["source"])}
                for a in self.b2_arrivals.get(raw, [])
                if a["source"] in ("quick_subgraph", "quick_anchor")]
            quick_routes[raw] = walk_quick(raw, set())
        return {
            "query": str(getattr(self, "query", "")), "enough": bool(enough),
            "structured_judge_valid": bool(getattr(
                self, "v7_structured_judge_valid", False)),
            "supported_chunks": sorted({str(c).removeprefix("chunk:")
                                         for c in self.b2_supported_chunks}),
            "supported_entities": sorted(getattr(self, "b2_supported_entities", set())),
            "supported_graph_edges": supported_graph_edges,
            "paths_by_graph": paths_by_graph,
            "paths_by_chunk": paths, "quick_chunks": sorted(
                {str(c).removeprefix("chunk:") for c in self.b2_quick_chunks}),
            "quick_chunk_entry_edges": quick_entries,
            "quick_chunk_routes": quick_routes,
            "roots": sorted(roots),
        }

    def _d1_unrouted_quick_chunk_edges(self, supported_chunks, confirmed_edges,
                                       supported_entities=(), supported_graph_edges=()):
        """Punish introductions of initial quick-subgraph chunks outside evidence routes.

        This consumes the existing quote-validated evidence marks and B2
        question-local provenance ledger; it performs no additional LLM call.
        """
        roots = {str(r["node"]) for r in self.b2_roots}
        route_nodes = set()
        route_edges = set()
        reachable = set()

        def walk(node, seen):
            node = str(node)
            if node in roots:
                return [({node}, set())]
            if node in seen:
                return []
            paths = []
            for arrival in self.b2_arrivals.get(node, []):
                parent = str(arrival["parent"])
                edge = (arrival["edge_type"], arrival["edge_id"])
                for nodes, edges in walk(parent, seen | {node}):
                    paths.append((nodes | {node}, edges | {edge}))
            return paths

        for raw in sorted(set(str(c).removeprefix("chunk:")
                              for c in supported_chunks)):
            paths = walk(raw, set())
            if paths:
                reachable.add(raw)
                for nodes, edges in paths:
                    route_nodes.update(nodes)
                    route_edges.update(edges)
        reachable_entities = set()
        for raw in sorted(set(map(str, supported_entities))):
            paths = walk(raw, set())
            if paths:
                reachable_entities.add(raw)
                for nodes, edges in paths:
                    route_nodes.update(nodes)
                    route_edges.update(edges)
        for cited in supported_graph_edges:
            for edge in self.path:
                if edge["type"] == "relation":
                    rendered = (f"from entity:{edge.get('from')} to entity:{edge.get('to')} "
                                f"with relations: {edge.get('documents', '')}")
                elif edge["type"] == "connection":
                    rendered = f"from entity:{edge.get('from')} to chunk:{edge.get('to')}"
                else:
                    continue
                if rendered == str(cited) and edge.get("id"):
                    route_edges.add((str(edge["type"]), str(edge["id"])))
                    break

        # A missing provenance route is an attribution failure, not evidence
        # that every quick-subgraph chunk was irrelevant. Fail closed here.
        if not reachable and not reachable_entities and not route_edges:
            return [], {
                "quick_chunks": len(self.b2_quick_chunks),
                "supported_chunks": sorted(set(str(c).removeprefix("chunk:")
                                                for c in supported_chunks)),
                "reachable_supported_chunks": [],
                "reachable_supported_entities": [],
                "protected_route_nodes": 0,
                "unrouted_quick_chunks": [],
                "candidate_arrivals": 0,
                "punish_edges": 0,
                "skip_reason": "no_reachable_supported_chunk",
            }

        candidates = sorted(self.b2_quick_chunks - route_nodes)
        punish = []
        source_records = 0
        for chunk_id in candidates:
            # Only penalize real quick-subgraph introduction edges into this
            # chunk; do not punish its ancestors or later guided expansions.
            for arrival in self.b2_arrivals.get(str(chunk_id), []):
                if arrival["source"] not in ("quick_subgraph", "quick_anchor"):
                    continue
                edge = (arrival["edge_type"], arrival["edge_id"])
                if edge in route_edges or edge in confirmed_edges:
                    continue
                source_records += 1
                punish.append(edge)
        punish = list(dict.fromkeys(punish))
        report = {
            "quick_chunks": len(self.b2_quick_chunks),
            "supported_chunks": sorted(set(str(c).removeprefix("chunk:")
                                            for c in supported_chunks)),
            "reachable_supported_chunks": sorted(reachable),
            "reachable_supported_entities": sorted(reachable_entities),
            "protected_route_nodes": len(route_nodes),
            "unrouted_quick_chunks": candidates,
            "candidate_arrivals": source_records,
            "punish_edges": len(punish),
        }
        return punish, report

    # ------------------------------------------------------------------
    # V16 question-level traversal. Fact planning never starts a separate
    # traversal; the main question owns the graph state and jump budget.
    # ------------------------------------------------------------------

    def get_query_ans_fact(self, query, do_update, search_keys, max_jumps):
        """Run one question-level traversal under the V16 shared budget.

        Fact/sub-question planning is advisory only. It never starts a second
        quick or full traversal and never receives a separate jump budget.
        The main traversal owns the single max_jumps budget, and its F1
        frontier scans every discovered node before declaring the graph stuck.
        """
        self.logger.info("V16: question-level traversal; sub-question reruns disabled")
        return self._get_query_ans_original(
            query, do_update, search_keys, max_jumps,
        )

    def _judge_subquestion(self, sub_q, entities, summaries, edges):
        from .fact_planner import SUBQUESTION_JUDGE_PROMPT, _extract_cot_ans
        raw = self.agent.generate_response("", [{"role": "user", "content":
            SUBQUESTION_JUDGE_PROMPT.format(
                sub_question=sub_q, entity_list=str(entities),
                chunk_summary=str(summaries)[:4000], edge_list=str(edges)[:2000])}])
        body = _extract_cot_ans(raw)
        m = re.search(r"\{.*\}", body, re.S)
        if m:
            try:
                obj = json.loads(m.group(0))
                enough_raw = str(obj.get("enough", "no")).strip().lower()
                enough = enough_raw in ("yes", "true", "1")
                return {"enough": enough,
                        "answer": str(obj.get("answer", "")),
                        "entity": str(obj.get("entity", ""))}
            except Exception:
                pass
        return {"enough": False, "answer": "", "entity": ""}


    def _v6_judge(self, query):
        from .fact_v6 import V6_JUDGE, parse_v6_judge
        raw = self.agent.generate_response("", [{"role": "user", "content": V6_JUDGE.format(
            query=query, entity_list=str(self.entity), edge_list=str(self.edge)[-3000:],
            chunk_summary=str(self.summary)[-3500:],
            fact_block="(the judge derives facts from the question)")}])
        result = parse_v6_judge(raw)
        if result is None:
            self.logger.warning("V6 judge unparseable — falling back to yes/no")
            enough_str = self.judge_sufficient_information()
            return enough_str.strip() == "yes", [], []
        self.logger.info("V6 judge: enough=" + str(result["enough"]) + ", solved=" + str(sum(1 for f in result["facts"] if f["status"]=="solved")) + ", gaps=" + str(len(result["gaps"])))
        return result["enough"], result["facts"], result["gaps"]

    def _judge_dispatch(self, query):
        """Unified judge entry: fact_v6 uses the extended judge (facts+gaps),
        every other mode keeps the original yes/no judge verbatim."""
        if self.traversal_mode == "fact_v7":
            from .fact_v7 import judge
            return judge(self, query)
        if self.traversal_mode == "fact_v6":
            enough, facts, gaps = self._v6_judge(query)
            self._v6_facts = facts
            self._v6_gaps = gaps
            return enough
        return self.judge_sufficient_information_ft() == "yes"

    def _v6_entry_correction(self, c_node):
        """FGAT v6 core: turn an unsatisfied judge into a retrieval action.
        (a) switch to a proven bound entity already in the graph but unvisited;
        (b) otherwise seed ONE instantiated gap question, merging its quick
        subgraph (edge weights still computed from the full question)."""
        for fact in self._v6_facts:
            ent = fact.get("entity", "")
            if (fact.get("status") == "solved" and ent and ent in self.entity
                    and ent not in self.c_node_list and ent != c_node):
                self.logger.info(f"V6 entry correction: switch to bound entity {ent}")
                self._v6_metrics["switches"] += 1
                return ent
        for gap in list(self._v6_gaps):
            if gap in self._v6_metrics["gap_tried"]:
                continue
            self._v6_metrics["gap_tried"].append(gap)
            gap_embedding = self.database.embedding.sentence_embedding(gap)
            found = self.database.entity_collection.query(query_embeddings=[gap_embedding], n_results=1)
            if not found["ids"][0]:
                continue
            seed = found["ids"][0][0]
            if seed in self.entity:
                continue
            self.logger.info(f"V6 entry correction: gap seed {seed} <- {gap}")
            self._v6_metrics["gap_entries"] += 1
            self._v6_extra_roots.append(seed)
            origin_entities, origin_chunks, origin_edges = self.database.quick_query(self.database_query, seed)
            for edge in origin_edges:
                if edge["type"] == "relation":
                    if edge["to"] not in self.entity:
                        self.edge.append(f"from entity:{edge['from']} to entity:{edge['to']} with relations: {edge['documents']}")
                        self.path.append({"from": edge['from'], "to": edge['to'], "id": edge['id'], "documents": edge['documents'], "type": "relation"})
                else:
                    if edge["to"] not in self.chunk:
                        self.edge.append(f"from entity:{edge['from']} to chunk:{edge['to']}")
                        self.path.append({"from": edge['from'], "to": edge['to'], "id": edge['id'], "type": "connection"})
            for origin_entity in origin_entities:
                if origin_entity not in self.entity:
                    self.entity.append(origin_entity)
                    self.out_dgree[origin_entity], _ = self.get_out_dgree(origin_entity)
            for origin_chunk in origin_chunks:
                if origin_chunk not in self.chunk:
                    self.chunk.append(origin_chunk)
                    self.summary[f"chunk:{origin_chunk}"] = self.get_chunk_summary(origin_chunk)
            if seed not in self.entity:
                self.entity.append(seed)
                self.out_dgree[seed], _ = self.get_out_dgree(seed)
            return seed
        return c_node

