"""Behavioral regression tests; no GPU, live API, or persistent database."""
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = '_v7_offline'
for suffix, path in [('', 'ReMindRag'), ('.generator', 'ReMindRag/generator'),
                     ('.utils', 'ReMindRag/utils')]:
    module = types.ModuleType(PACKAGE + suffix)
    module.__path__ = [str(ROOT / path)]
    sys.modules[module.__name__] = module
for suffix, name in [('.llms', 'AgentBase'), ('.database', 'ChromaDBManager')]:
    module = types.ModuleType(PACKAGE + suffix)
    setattr(module, name, object)
    sys.modules[module.__name__] = module
v7 = importlib.import_module(PACKAGE + '.generator.fact_v7')
pf_module = importlib.import_module(PACKAGE + '.generator.pathfinder')
spec = importlib.util.spec_from_file_location('mc_protocol', ROOT / 'adapt/mc_protocol.py')
mc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mc)


class V7Tests(unittest.TestCase):
    def make_pf(self, mode='fact_v7'):
        with patch.dict('os.environ', REMIND_TRAVERSAL=mode):
            pf = pf_module.PathFinder(Mock(), Mock(), 100, 10, None)
        pf.logger = Mock()
        pf.entity, pf.chunk, pf.edge, pf.path = ['A'], ['1'], [], []
        pf.out_dgree, pf.c_node_list = {'A': 1}, []
        pf.query = pf.database_query = 'Where was the successor of A born?'
        pf.summary = {'chunk:1': 'The successor of A is B.'}
        v7.initialize(pf)
        pf.v7_gap = dict(id='birthplace', question='Where was B born?', entity='B',
                         source_chunk='chunk:1', quote='The successor of A is B.')
        pf.database.entity_collection.get.return_value = {'ids': ['B']}
        pf.database.chunk_collection.get.return_value = {'documents': ['The successor of A is B.']}
        pf.database.strong_connection_threshold = .5
        return pf

    def test_prose_article_is_not_choice(self):
        self.assertEqual(mc.parse_choice('There is a problem. Answer: D'), -1)
        self.assertEqual(mc.parse_choice('{"choice":"D"}'), 3)
        self.assertEqual(mc.parse_choice('```cot-ans\nC\n```'), 2)

    def test_judge_sees_head_middle_tail_without_truncation(self):
        pf = self.make_pf()
        pf.summary = {'chunk:1': 'HEAD ' + 'x'*6000 + ' TAIL'}
        pf.edge = ['EDGE_HEAD' + 'e'*5000 + 'EDGE_TAIL']
        pf.agent.generate_response.return_value = '{"status":"answerable","enough":true,"gap":null}'
        self.assertTrue(v7.judge(pf, pf.query))
        prompt = pf.agent.generate_response.call_args.args[1][0]['content']
        for marker in ['HEAD', 'TAIL', 'EDGE_HEAD', 'EDGE_TAIL']:
            self.assertIn(marker, prompt)
        pf.agent.generate_response.assert_called_once()

    def test_query_entity_needs_no_already_retrieved_evidence(self):
        pf = self.make_pf()
        pf.query = 'Where was B born?'
        pf.summary = {}
        pf.v7_gap.update(quote='', source_chunk='')
        self.assertEqual(v7.eligible_entity(pf), 'B')
        pf.database.chunk_collection.get.assert_not_called()
        pf.query = 'Where was Babbage born?'
        self.assertIsNone(v7.eligible_entity(pf))

    def test_inferred_entity_without_binding_is_rejected(self):
        pf = self.make_pf()
        pf.v7_gap.update(quote='', source_chunk='')
        self.assertIsNone(v7.eligible_entity(pf))

    def test_evidence_cache_skips_backtracks_but_not_new_relation(self):
        pf = self.make_pf()
        pf.v7_gap.update(quote='', source_chunk='')
        pf.agent.generate_response.return_value = json.dumps(dict(
            status='missing_fact', enough=False, gap=pf.v7_gap))
        self.assertFalse(v7.judge(pf, pf.query))
        pf.c_node_list.append('A')
        self.assertFalse(v7.judge(pf, pf.query))
        self.assertEqual(pf.agent.generate_response.call_count, 1)
        pf.edge.append('B was born in C')
        v7.judge(pf, pf.query)
        self.assertEqual(pf.agent.generate_response.call_count, 2)
        pf.summary['chunk:2'] = 'B was born in C.'
        v7.judge(pf, pf.query)
        self.assertEqual(pf.agent.generate_response.call_count, 3)

    def test_graph_entities_and_edges_can_be_validated_as_evidence(self):
        pf = self.make_pf()
        pf.entity = ['A', 'B']
        pf.edge = ['from entity:A to entity:B with relations: successor of']
        pf.v7_gap.update(quote='', source_chunk='')
        pf.agent.generate_response.return_value = json.dumps(dict(
            status='answerable', enough=True, gap=None,
            known_facts=[
                dict(fact='B is the successor of A.', source_type='edge',
                     source_id=pf.edge[0], quote=''),
                dict(fact='A is a named graph entity.', source_type='entity',
                     source_id='A', quote=''),
                dict(fact='Unsupported claim.', source_type='entity',
                     source_id='not-in-graph', quote=''),
            ]))
        self.assertTrue(v7.judge(pf, pf.query))
        self.assertEqual({f['source_type'] for f in pf.v7_verified_facts},
                         {'edge', 'entity'})
        self.assertEqual(len(pf.v7_verified_facts), 2)

    def test_graph_evidence_marks_b2_entities_and_edges(self):
        pf = self.make_pf()
        pf.b2_supported_entities = set()
        pf.b2_supported_edges = set()
        pf.edge = ['from entity:A to entity:B with relations: successor of']
        pf.v7_gap.update(quote='', source_chunk='')
        pf.agent.generate_response.return_value = json.dumps(dict(
            status='answerable', enough=True, gap=None,
            known_facts=[
                dict(fact='B is A successor.', source_type='edge',
                     source_id=pf.edge[0], quote=''),
                dict(fact='A is an entity.', source_type='entity',
                     source_id='A', quote=''),
            ]))
        with patch.dict('os.environ', REMIND_B1='2'):
            self.assertTrue(v7.judge(pf, pf.query))
        self.assertEqual(pf.b2_supported_entities, {'A'})
        self.assertEqual(pf.b2_supported_edges, {pf.edge[0]})

    def test_b2_attributes_graph_support_to_verified_graph_path(self):
        pf = self.make_pf()
        edge_text = 'from entity:A to entity:B with relations: successor of'
        pf.entity = ['A', 'B']
        pf.edge = [edge_text]
        pf.path = [{'from': 'A', 'to': 'B', 'id': 'rel-1',
                    'documents': 'successor of', 'type': 'relation'}]
        pf.b2_roots = [{'node': 'A', 'source': 'root'}]
        pf.b2_arrivals = {'B': [{'parent': 'A', 'edge_type': 'relation',
                                 'edge_id': 'rel-1', 'source': 'guided_expansion'}]}
        confirmed, report = pf._b2_confirmed_paths([], {'B'}, {edge_text})
        self.assertEqual(confirmed, [('relation', 'rel-1')])
        self.assertEqual(report['reachable_entities'], ['B'])
        self.assertEqual(report['supported_graph_edges'], [edge_text])

    def test_explicit_mc_parser_is_gold_independent_and_unambiguous(self):
        for text, want in [('The answer is **B**.', 1), ('B. 1, 2, 3', 1),
                           ('There is a problem. Answer: D', 3),
                           ('A. first\nB. second\nC. third\nD. fourth', -1),
                           ('Answer: B\nFinal answer: D', -1),
                           ('There is a problem with option B.', -1)]:
            self.assertEqual(mc.parse_explicit_choice(text), want, text)

    def test_query_entry_executes_then_rejudges_with_new_evidence(self):
        pf = self.make_pf()
        pf.query = 'Where was B born?'
        pf.v7_gap.update(quote='', source_chunk='')
        pf.database.get_entity_edges.return_value = ([], dict(
            id='b2', **{'from': 'B', 'to': '2'}, documents='', weight=.9))
        pf.get_out_dgree = Mock(return_value=(1, []))
        pf.get_chunk_summary = Mock(return_value='B was born in C.')
        self.assertEqual(v7.enter(pf, 3), ('B', 2))
        self.assertIn('chunk:2', pf.summary)
        self.assertEqual(pf.v7_metrics['query_entries'], 1)
        self.assertEqual(pf.v7_metrics['entries'], 1)
        pf.agent.generate_response.return_value = '{"status":"answerable","enough":true,"gap":null}'
        self.assertTrue(v7.judge(pf, pf.query))
        self.assertIsNone(v7.enter(pf, 3))

    def test_binding_requires_real_source_and_matching_entity(self):
        pf = self.make_pf()
        self.assertEqual(v7.eligible_entity(pf), 'B')
        pf.v7_source_cache.clear()
        pf.database.chunk_collection.get.return_value = {'documents': ['No successor is given.']}
        self.assertIsNone(v7.eligible_entity(pf))

    def test_related_ingraph_entity_is_not_skipped(self):
        pf = self.make_pf()
        pf.entity.append('B')
        pf.out_dgree['B'] = 1
        self.assertEqual(v7.enter(pf, 3), ('B', 1))
        self.assertIsNone(v7.enter(pf, 3))
        pf.database.enhance_edge_weight.assert_not_called()

    def test_temp_evidence_ledger_is_query_scoped_and_accumulative(self):
        pf = self.make_pf()
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'attempt' / 'temporary_summary.json'
            pf.v7_temp_summary_path = str(target)
            v7.record_chunk_evidence(pf, '1', 'A was elected in 2020.')
            gap = dict(pf.v7_gap, quote='', source_chunk='')
            pf.agent.generate_response.return_value = json.dumps(dict(
                status='missing_fact', enough=False,
                known_facts=[dict(fact='A was elected in 2020.',
                                  source_chunk='chunk:1',
                                  quote='A was elected in 2020.')],
                gap=gap))
            self.assertFalse(v7.judge(pf, pf.query))
            self.assertEqual(len(pf.v7_verified_facts), 1)
            # The next judge may omit a previously verified fact; it survives.
            pf.summary['chunk:2'] = 'B was born in C.'
            pf.edge.append('B born in C')
            pf.agent.generate_response.return_value = json.dumps(dict(
                status='answerable', enough=True, known_facts=[], gap=None))
            self.assertTrue(v7.judge(pf, pf.query))
            self.assertEqual(len(pf.v7_verified_facts), 1)
            v7.finalize_temp_summary(pf)
            saved = json.loads(target.read_text(encoding='utf-8'))
            self.assertEqual(saved['status'], 'complete')
            self.assertEqual(saved['scope'], 'one_query_traversal')
            self.assertEqual(saved['source_evidence'][0]['source_chunk'], 'chunk:1')
            self.assertEqual(len(saved['verified_facts']), 1)

    def test_extra_dfs_obeys_threshold_and_total_budget(self):
        pf = self.make_pf()
        pf.database.get_entity_edges.return_value = ([dict(id='bc', **{'from':'B', 'to':'C'},
            documents='B knows C', weight=.9)], dict(id='b2', **{'from':'B','to':'2'},
            documents='B is described here', weight=.9))
        entities, chunks, edges, cost = v7.bounded_quick(pf, 'B', 2)
        self.assertEqual(cost, 2)
        self.assertEqual(entities, ['B'])
        self.assertEqual(chunks, ['2'])
        self.assertEqual(len(edges), 1)
        pf.database.get_entity_edges.return_value[1]['weight'] = .1
        entities, chunks, edges, cost = v7.bounded_quick(pf, 'B', 2)
        self.assertEqual(chunks, [])
        self.assertEqual(entities, ['B', 'C'])

    def test_selector_uses_gap_and_original_prompt(self):
        pf = self.make_pf()
        pf.agent.generate_response.return_value = '```cot-ans\nentity:B\n```'
        self.assertEqual(pf.find_next_node('A', [], '', [], []), 'entity:B')
        prompt = pf.agent.generate_response.call_args.args[1][0]['content']
        self.assertIn('Where was B born?', prompt)
        self.assertIn('Current unresolved requirement', prompt)
        self.assertIn('gap:entry', prompt)

    def test_v12a_no_gap_ablation_removes_gap_guidance_and_entry(self):
        pf = self.make_pf()
        with patch.dict('os.environ', REMIND_V12A_DISABLE_GAP='1'):
            suffix = v7.selector_suffix(pf)
            self.assertNotIn('Where was B born?', suffix)
            self.assertNotIn('gap:entry', suffix)
            self.assertIsNone(v7.eligible_entity(pf))
            self.assertIsNone(v7.enter(pf, 3))

    def test_original_selector_does_not_receive_v7_instructions(self):
        pf = self.make_pf('original')
        pf.agent.generate_response.return_value = '```cot-ans\nentity:B\n```'
        pf.find_next_node('A', [], '', [], [])
        prompt = pf.agent.generate_response.call_args.args[1][0]['content']
        self.assertNotIn('Current unresolved requirement', prompt)
        self.assertNotIn('gap:entry', prompt)

    def test_exhausted_region_rescues_then_judges_before_next_expansion(self):
        pf = self.make_pf()
        pf.database.embedding.sentence_embedding.return_value = [1]
        pf.database.entity_collection.query.return_value = {'ids': [['A']]}
        pf.database.quick_query.return_value = (['A'], [], [])
        pf.database.get_entity_edges.return_value = ([], {})
        pf.get_out_dgree = Mock(return_value=(0, []))
        decisions = iter([False, True])
        def judge(query):
            return next(decisions)
        pf._judge_dispatch = Mock(side_effect=judge)
        pf.get_anchor_chunk_title = Mock(return_value=[])
        pf.find_next_node_ft = Mock(side_effect=AssertionError('must rejudge before selecting'))
        with patch.object(v7, 'enter', return_value=('B', 1)) as enter:
            pf._get_query_ans_original(pf.query, False, 1, 10)
        enter.assert_called_once()
        self.assertEqual(pf._judge_dispatch.call_count, 2)
        pf.database.enhance_edge_weight.assert_not_called()


if __name__ == '__main__':
    unittest.main()
