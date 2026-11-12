import unittest
from unittest.mock import Mock

try:
    from adapt.v12_memory import apply_round_d1
except ModuleNotFoundError:  # direct execution from the adapt directory
    from v12_memory import apply_round_d1


def edge(kind, edge_id, parent, child):
    return {"type": kind, "id": edge_id, "parent": parent, "child": child}


def current_record(**overrides):
    current = {
        "phase": "round2",
        "query": "question",
        "enough": True,
        "structured_judge_valid": True,
        "quick_chunks": ["supported", "unsupported"],
        "supported_chunks": ["supported"],
        "supported_graph_edges": [],
        "paths_by_chunk": {},
        "paths_by_graph": {},
        "quick_chunk_routes": {
            "unsupported": [[
                edge("relation", "seed_anchor", "seed", "anchor-1"),
                edge("connection", "anchor_chunk", "anchor-1", "unsupported"),
            ]]
        },
    }
    current.update(overrides)
    return current


class V12D1Tests(unittest.TestCase):
    def setUp(self):
        self.rag = Mock()

    def test_punishes_back_through_anchor_ingress_without_history_registry(self):
        audit = apply_round_d1(self.rag, current_record())
        self.assertEqual(audit["d1_applied"], [
            ["connection", "anchor_chunk"], ["relation", "seed_anchor"]
        ])
        self.rag.database.punish_edge_weight.assert_called_once_with(
            "question", [("connection", "anchor_chunk"),
                         ("relation", "seed_anchor")])

    def test_all_actual_arrivals_to_anchor_are_considered(self):
        routes = [[
            edge("relation", "left_anchor", "left", "anchor-1"),
            edge("connection", "anchor_chunk_left", "anchor-1", "unsupported"),
        ], [
            edge("relation", "right_anchor", "right", "anchor-1"),
            edge("connection", "anchor_chunk_right", "anchor-1", "unsupported"),
        ]]
        audit = apply_round_d1(self.rag, current_record(
            quick_chunk_routes={"unsupported": routes}))
        self.assertEqual(set(map(tuple, audit["d1_applied"])), {
            ("relation", "left_anchor"),
            ("relation", "right_anchor"),
            ("connection", "anchor_chunk_left"),
            ("connection", "anchor_chunk_right"),
        })

    def test_supported_shared_ingress_is_protected(self):
        support_route = [
            edge("relation", "seed_anchor", "seed", "anchor-1"),
            edge("connection", "anchor_supported", "anchor-1", "supported"),
        ]
        audit = apply_round_d1(self.rag, current_record(
            paths_by_chunk={"supported": [support_route]}))
        self.assertEqual(audit["protected_edges"], [["relation", "seed_anchor"]])
        self.assertEqual(audit["d1_applied"], [["connection", "anchor_chunk"]])

    def test_supported_entity_path_also_protects_shared_ingress(self):
        entity_route = [[edge("relation", "seed_anchor", "seed", "anchor-1")]]
        audit = apply_round_d1(self.rag, current_record(
            supported_chunks=[],
            supported_entities=["anchor-1"],
            paths_by_graph={"entity::anchor-1": entity_route}))
        self.assertEqual(audit["protected_edges"], [["relation", "seed_anchor"]])
        self.assertEqual(audit["d1_applied"], [["connection", "anchor_chunk"]])

    def test_anchor_seed_falls_back_to_its_chunk_connection(self):
        route = [[edge("connection", "anchor_chunk", "anchor-seed", "unsupported")]]
        audit = apply_round_d1(self.rag, current_record(
            quick_chunk_routes={"unsupported": route}))
        self.assertEqual(audit["d1_applied"], [["connection", "anchor_chunk"]])

    def test_non_anchor_path_punishes_only_direct_chunk_ingress(self):
        route = [[
            edge("relation", "seed_middle", "seed", "middle"),
            edge("connection", "middle_chunk", "middle", "unsupported"),
        ]]
        audit = apply_round_d1(self.rag, current_record(
            quick_chunk_routes={"unsupported": route}))
        self.assertEqual(audit["d1_applied"], [["connection", "middle_chunk"]])

    def test_invalid_or_insufficient_judgment_fails_closed(self):
        for overrides, reason in (
            ({"structured_judge_valid": False}, "structured_judge_invalid"),
            ({"enough": False}, "answer_not_sufficient"),
        ):
            with self.subTest(reason=reason):
                rag = Mock()
                audit = apply_round_d1(rag, current_record(**overrides))
                self.assertEqual(audit["skip_reason"], reason)
                rag.database.punish_edge_weight.assert_not_called()


if __name__ == "__main__":
    unittest.main()
