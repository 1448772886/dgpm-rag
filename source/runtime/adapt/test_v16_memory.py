"""Memory lifecycle regression checks without a live model or database."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from adapt.v16_memory import commit_question_memory, plan_evidence_paths
from adapt.v12_memory import apply_round_d1


def edge(kind, name, parent="seed", child="anchor-1"):
    return {"type": kind, "id": name, "parent": parent, "child": child}


def record(**overrides):
    result = {
        "phase": "round1", "qi": 0, "query": "question", "enough": True,
        "structured_judge_valid": True, "supported_chunks": ["good"],
        "supported_entities": [], "supported_graph_edges": [],
        "paths_by_chunk": {"good": [[edge("relation", "shared"),
            edge("connection", "good", "anchor-1", "good")]]},
        "paths_by_graph": {}, "quick_chunks": ["good", "bad"],
        "quick_chunk_routes": {"bad": [[edge("relation", "shared"),
            edge("connection", "bad", "anchor-1", "bad")]]},
    }
    result.update(overrides)
    return result


def make_rag(**overrides):
    plan = {"query": "question", "update_requested": True,
            "enhance_edges": [("relation", "shared"), ("relation", "shared")],
            "punish_edges": [("relation", "shared"), ("connection", "bad")],
            "enhance_applied": False, "punish_applied": False, "receipt": None}
    plan.update(overrides)
    return SimpleNamespace(database=Mock(), preprocess=SimpleNamespace(
        path_finder=SimpleNamespace(v16_memory_plan=plan)))


class MemoryTests(unittest.TestCase):
    def test_merges_duplicates_and_protects_shared_ancestor(self):
        rag = make_rag()
        audit = commit_question_memory(rag, record())
        self.assertEqual(rag.database.method_calls, [
            unittest.mock.call.enhance_edge_weight("question", [
                ("connection", "good"), ("relation", "shared")]),
            unittest.mock.call.punish_edge_weight("question", [("connection", "bad")]),
        ])
        self.assertEqual(audit["protected_edges"], [["relation", "shared"]])
        self.assertFalse(set(map(tuple, audit["enhanced_edges"])) &
                         set(map(tuple, audit["punished_edges"])))

    def test_graph_only_evidence_is_enhanced_and_protected(self):
        rag = make_rag(enhance_edges=[], punish_edges=[("relation", "graph")])
        current = record(supported_chunks=[], quick_chunks=[], paths_by_chunk={},
                         supported_entities=["E"],
                         paths_by_graph={"entity::E": [[edge("relation", "graph")]]})
        commit_question_memory(rag, current)
        rag.database.enhance_edge_weight.assert_called_once_with(
            "question", [("relation", "graph")])
        rag.database.punish_edge_weight.assert_not_called()

    def test_two_evidence_chunks_share_edge_written_once(self):
        current = record(supported_chunks=["good", "second"], paths_by_chunk={
            "good": [[edge("relation", "shared")]],
            "second": [[edge("relation", "shared")], [edge("relation", "shared")]]})
        rag = make_rag(punish_edges=[])
        commit_question_memory(rag, current)
        rag.database.enhance_edge_weight.assert_called_once_with(
            "question", [("relation", "shared")])

    def test_repeated_commit_does_not_write_again(self):
        rag = make_rag()
        first = commit_question_memory(rag, record())
        self.assertIs(commit_question_memory(rag, record()), first)
        rag.database.enhance_edge_weight.assert_called_once()
        rag.database.punish_edge_weight.assert_called_once()

    def test_same_plan_cannot_be_reused_for_another_round(self):
        rag = make_rag()
        commit_question_memory(rag, record())
        with self.assertRaisesRegex(RuntimeError, "different question/round"):
            commit_question_memory(rag, record(phase="round2"))

    def test_fresh_and_retrieval_only_do_not_write(self):
        for current, rag in [(record(phase="fresh"), make_rag()),
                             (record(), make_rag(update_requested=False))]:
            commit_question_memory(rag, current)
            self.assertFalse(rag.database.method_calls)

    def test_missing_defer_or_wrong_query_refuses_commit(self):
        for plan in [None, {"query": "other question"}]:
            rag = make_rag()
            rag.preprocess.path_finder.v16_memory_plan = plan
            with self.assertRaises(RuntimeError):
                commit_question_memory(rag, record())
            self.assertFalse(rag.database.method_calls)

    def test_d1_explicit_enhancements_are_protected(self):
        rag = make_rag()
        audit = apply_round_d1(rag, record(paths_by_chunk={}),
                               enhanced_edges=[("relation", "shared")], write=False)
        self.assertEqual(audit["d1_applied"], [["connection", "bad"]])
        self.assertFalse(rag.database.method_calls)

    def test_invalid_evidence_does_not_create_new_enhancement(self):
        self.assertEqual(plan_evidence_paths(record(structured_judge_valid=False)), set())

    def test_retry_after_punishment_error_does_not_repeat_enhancement(self):
        rag = make_rag()
        rag.database.punish_edge_weight.side_effect = [RuntimeError("write failed"), None]
        with self.assertRaises(RuntimeError):
            commit_question_memory(rag, record())
        commit_question_memory(rag, record())
        rag.database.enhance_edge_weight.assert_called_once()


class RetrievalMemoryTests(unittest.TestCase):
    def test_real_traversal_defers_then_commits_and_resets_for_next_question(self):
        from adapt import test_fact_v7 as support
        pf = support.V7Tests().make_pf()
        pf.database.embedding.sentence_embedding.return_value = [1]
        pf.database.entity_collection.query.return_value = {"ids": [["A"]]}
        pf.database.quick_query.return_value = (["A"], [], [])
        pf.database.get_entity_edges.return_value = ([], {})
        pf.get_out_dgree = Mock(return_value=(0, []))
        pf.get_anchor_chunk_title = Mock(return_value=[])

        def judge(query):
            pf.b2_supported_chunks = {"good"}
            pf.b2_roots = [{"node": "A"}]
            pf.b2_arrivals = {"good": [{"parent": "A", "edge_type": "connection",
                "edge_id": "good", "source": "quick_subgraph"}]}
            pf.v7_structured_judge_valid = True
            return True

        pf._judge_dispatch = Mock(side_effect=judge)
        rag = SimpleNamespace(database=pf.database, preprocess=SimpleNamespace(path_finder=pf))
        with patch.dict("os.environ", REMIND_B1="2", REMIND_V16_DEFER_MEMORY="1",
                        REMIND_C2_DEFER_UPDATE="0", REMIND_C2_TRACE_OUT=""):
            pf._get_query_ans_original(pf.query, True, 1, 10)
            pf.database.enhance_edge_weight.assert_not_called()
            pf.database.punish_edge_weight.assert_not_called()
            commit_question_memory(rag, record(query=pf.query, quick_chunks=[],
                supported_chunks=["good"], paths_by_chunk={"good": [[edge("connection", "good")]]}))
            pf.database.enhance_edge_weight.assert_called_once_with(
                pf.query, [("connection", "good")])
            old_plan = pf.v16_memory_plan
            pf._get_query_ans_original(pf.query, True, 1, 10)
            self.assertIsNot(pf.v16_memory_plan, old_plan)
            self.assertIsNone(pf.v16_memory_plan["receipt"])


if __name__ == "__main__":
    unittest.main()
