"""Gap-conditioned original traversal. No gold access or persistent path memory."""
import json
import os
import re
from pathlib import Path

from .prompts import (v7_sufficiency_prompt, v12_sufficiency_prompt,
                      v12b_sufficiency_prompt,
                      v13_sufficiency_prompt, v14a_sufficiency_prompt)


def normalize(text):
    return " ".join(str(text).split()).casefold()


def normalize_quote(text):
    """Normalize typography-only quote differences, not lexical content."""
    text = str(text or '').translate(str.maketrans({
        '"': None, "'": None, '“': None, '”': None, '‘': None, '’': None,
        '「': None, '」': None, '『': None, '』': None,
    }))
    return normalize(text)


def contains_entity(text, entity):
    return bool(entity and re.search(r'(?<!\w)' + re.escape(normalize(entity)) +
                                    r'(?!\w)', normalize(text)))


_STOPWORDS = {
    'what', 'when', 'where', 'which', 'who', 'whom', 'whose', 'how',
    'many', 'much', 'does', 'did', 'was', 'were', 'are', 'the', 'and',
    'that', 'this', 'from', 'with', 'after', 'before', 'between', 'into',
    'about', 'does', 'have', 'has', 'had', 'for', 'following', 'according',
    'occur', 'occurred', 'number', 'days', 'day'
}


def _current_raw_excerpt(text, query, limit=5, max_chars=2400):
    """Select a small, query-linked excerpt from the newly entered chunk.

    The full chunk is deliberately not exposed during every judge call. This
    preserves exact dates/numbers while keeping the old summary as the main
    navigation state. It is local string filtering, so it adds no LLM call.
    """
    text = str(text or '').strip()
    if not text:
        return None
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', text)
                 if s.strip()]
    if not sentences:
        return text[:max_chars]
    terms = [t for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*", str(query).lower())
             if len(t) > 2 and t not in _STOPWORDS]
    ranked = []
    for idx, sentence in enumerate(sentences):
        lower = sentence.lower()
        overlap = sum(1 for term in set(terms) if term in lower)
        # Dates, numbers, and negation are high-value details that summaries
        # frequently omit, so retain them when they are query-linked.
        detail = len(re.findall(r'\b\d{1,4}\b|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b', lower))
        score = overlap * 4 + detail
        if any(k in lower for k in ('not ', 'no ', 'never ', 'without ')):
            score += 1
        ranked.append((score, -idx, sentence))
    chosen = [s for score, _, s in sorted(ranked, reverse=True) if score > 0][:limit]
    if not chosen:
        chosen = sentences[:min(limit, len(sentences))]
    chosen = sorted(chosen, key=lambda s: sentences.index(s))
    out = ' '.join(chosen)
    return out[:max_chars]


def initialize(pf):
    pf.v7_gap = None
    pf.v7_missing_items = []
    pf.v7_previous = None
    pf.v7_roots = []
    pf.v7_attempted = set()
    pf.v7_global_attempted = set()
    pf.v7_frontier_attempted = set()
    pf.v7_source_cache = {}
    pf.v7_steps = 0
    pf.v7_evidence_key = None
    pf.v7_enough = False
    pf.v7_last_gap_key = None
    pf.v7_last_state = None
    pf.v7_stalls = 0
    pf.v7_structured_judge_valid = False
    # Question-scoped evidence ledger. Source excerpts and quote-validated
    # facts live for the whole traversal; the attempt-specific file is never
    # loaded by another question or written back to the graph.
    pf.v7_evidence_ledger = []
    pf.v7_verified_facts = []
    # V11 keeps the exact first-read excerpt separately from the compact text
    # carried into later selector and sufficiency prompts.
    pf._v11_context_evidence = {}
    pf._v11_context_judged_chunks = set()
    pf._v11_post_compact_evidence_key = None
    pf._v12_context_evidence = {}
    pf._v12_context_judged_chunks = set()
    pf._v12_post_compact_evidence_key = None
    pf.v7_temp_summary_path = os.environ.get('REMIND_TEMP_SUMMARY_FILE', '')
    pf.v7_metrics = dict(judges=0, invalid_judges=0, selectors=0,
                         entries=0, switches=0, added_nodes=0, cached_judges=0,
                         query_entries=0, evidence_entries=0)


def _write_temp_summary(pf, complete=False):
    path = getattr(pf, 'v7_temp_summary_path', '')
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'scope': 'one_query_traversal',
        'status': 'complete' if complete else 'active',
        'query': pf.query,
        'source_evidence': pf.v7_evidence_ledger,
        'verified_facts': pf.v7_verified_facts,
        'unresolved_gap': pf.v7_gap,
        'unresolved_missing_items': getattr(pf, 'v7_missing_items', []),
    }
    temporary = target.with_suffix(target.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                         encoding='utf-8')
    temporary.replace(target)


def record_chunk_evidence(pf, chunk_id, evidence):
    """Keep exact retrieved evidence in the current question's ledger."""
    chunk_id = str(chunk_id)
    evidence = str(evidence or '')
    status = 'evidence_added' if evidence.strip() else 'no_new_evidence'
    context = getattr(pf, '_v12_context_evidence', {}).get(chunk_id, {})
    source_fields = ({'source_document': context.get('source', ''),
                      'sentence_spans': context.get('sentence_spans', []),
                      'selected_sentence_ids': context.get('selected_ids', []),
                      'context_dependencies': context.get('context_dependencies', []),
                      'compact_evidence': context.get('compact', evidence)}
                     if context else {})
    for row in pf.v7_evidence_ledger:
        if row['source_chunk'] == f'chunk:{chunk_id}':
            changed = row['evidence'] != evidence or row.get('status') != status
            changed = changed or any(row.get(key) != value
                                     for key, value in source_fields.items())
            if changed:
                row['evidence'] = evidence
                row['status'] = status
                row.update(source_fields)
                _write_temp_summary(pf)
            return
    pf.v7_evidence_ledger.append({
        'source_chunk': f'chunk:{chunk_id}',
        'evidence': evidence,
        'status': status,
        **source_fields,
    })
    _write_temp_summary(pf)


def finalize_temp_summary(pf):
    """Mark the question-local ledger complete; it remains audit-only."""
    _write_temp_summary(pf, complete=True)


def judge(pf, query):
    # Optional probe: keep the normal short summaries for history, but expose
    # the just-entered chunk verbatim to this sufficiency decision only.
    raw_current = None
    current_chunk_id = getattr(pf, '_judge_current_raw_chunk_id', None)
    if current_chunk_id is not None:
        fetched = pf.database.chunk_collection.get(
            ids=[str(current_chunk_id)], include=['documents'])
        documents = fetched.get('documents') or []
        if documents:
            mode = getattr(pf, '_judge_current_raw_mode', 'full')
            raw_current = (documents[0] if mode != 'selected' else
                           _current_raw_excerpt(documents[0], query))
    v12_enabled = getattr(pf, '_v12_context_compaction_enabled', lambda: False)()
    v11_enabled = getattr(pf, '_v11_context_compaction_enabled', lambda: False)()
    context_enabled = v12_enabled or v11_enabled
    if v12_enabled:
        first_use_chunks = pf._v12_context_chunks_not_judged()
        judge_summary = pf._v12_prompt_summary(for_judge=True)
    elif v11_enabled:
        first_use_chunks = pf._v11_context_chunks_not_judged()
        judge_summary = pf._v11_prompt_summary(for_judge=True)
    else:
        first_use_chunks, judge_summary = [], dict(pf.summary)
    if raw_current is not None:
        judge_summary[f'chunk:{current_chunk_id}'] = raw_current
    # Revisiting a node does not change the evidence or its sufficiency.
    # Include entities/relations as well as chunks: a new relation can resolve
    # a fact even when no new chunk has been read. Never truncate evidence.
    key = (pf._v11_evidence_state_key(query, judge_summary) if context_enabled else
           json.dumps([query, sorted(set(pf.entity)), sorted(set(pf.edge)),
                       sorted(set(pf.chunk)), judge_summary], sort_keys=True))
    if (key == pf.v7_evidence_key or
            (v11_enabled and key == pf._v11_post_compact_evidence_key) or
            (v12_enabled and key == pf._v12_post_compact_evidence_key)):
        pf.v7_metrics['cached_judges'] += 1
        pf._judge_current_raw_chunk_id = None
        if pf.v7_gap:
            pf.v7_stalls += 1
            pf.logger.warning('V7 no-progress gate: same evidence/gap after '
                              'selection; keep searching without re-judging')
        return pf.v7_enough
    # Use one concise, source-grounded V7 prompt. The original yes/no judge
    # remains untouched and is still the fallback if structured output fails.
    judge_template = (v14a_sufficiency_prompt
                      if os.environ.get('REMIND_V14A', '0') == '1' else
                      v13_sufficiency_prompt
                      if os.environ.get('REMIND_V13_PROMPT', '0') == '1' else
                      v12b_sufficiency_prompt
                      if os.environ.get('REMIND_V12B', '0') == '1' else
                      v12_sufficiency_prompt
                      if os.environ.get('REMIND_V12_PROMPT', '0') == '1'
                      else v7_sufficiency_prompt)
    prompt = judge_template.format(
        query=query,
        entity_list=json.dumps(pf.entity, ensure_ascii=False),
        edge_list=json.dumps(pf.edge, ensure_ascii=False),
        chunk_summary=json.dumps(judge_summary, ensure_ascii=False),
        verified_facts=json.dumps(pf.v7_verified_facts, ensure_ascii=False),
        previous_gap=json.dumps(pf.v7_previous, ensure_ascii=False),
        previous_missing_items=json.dumps(
            getattr(pf, 'v7_missing_items', []), ensure_ascii=False),
    )
    # The raw block belongs to this decision only. It must not leak into the
    # next selector or the next sufficiency check when we move to an entity.
    pf._judge_current_raw_chunk_id = None
    pf.v7_metrics['judges'] += 1
    pf.v7_structured_judge_valid = False
    raw = pf.agent.generate_response("", [{"role": "user", "content": prompt}])
    pf.logger.debug("V7 judge raw: " + raw)
    try:
        match = re.search(r"\{.*\}", raw, re.S)
        obj = json.loads(match.group() if match else raw)
        if type(obj.get('enough')) is not bool:
            raise ValueError('enough must be boolean')
        status = obj.get('status')
        if status not in ('answerable', 'missing_fact', 'conflicting_evidence'):
            raise ValueError('missing decision status')
        if obj['enough'] != (status == 'answerable'):
            raise ValueError('inconsistent decision status')
        gap = obj.get('gap')
        if not obj['enough'] and not (isinstance(gap, dict) and
                isinstance(gap.get('question'), str) and gap['question'].strip()):
            raise ValueError('missing unresolved gap')
        pf.v7_structured_judge_valid = True
        parsed_gap = None if obj['enough'] else {
            key: str(gap.get(key) or '') for key in
            ('id', 'question', 'entity', 'source_chunk', 'quote')}
        parsed_missing_items = []
        if os.environ.get('REMIND_V12B', '0') == '1' and not obj['enough']:
            raw_items = obj.get('missing_items')
            if not isinstance(raw_items, list) or not raw_items:
                raise ValueError('V12b requires a non-empty missing_items list')
            seen_missing = set()
            for index, item in enumerate(raw_items):
                if not isinstance(item, dict):
                    raise ValueError('V12b missing item must be an object')
                question_text = str(item.get('question') or '').strip()
                if not question_text:
                    raise ValueError('V12b missing item has no question')
                missing_item = {
                    'id': str(item.get('id') or f'missing_{index + 1}'),
                    'question': question_text,
                    'entity': str(item.get('entity') or ''),
                    'source_chunk': str(item.get('source_chunk') or ''),
                    'quote': str(item.get('quote') or ''),
                }
                item_key = normalize(missing_item['id'] + ' ' + question_text)
                if item_key not in seen_missing:
                    seen_missing.add(item_key)
                    parsed_missing_items.append(missing_item)
            # Grounded-entry is retained for the primary unresolved item.
            if parsed_gap is None:
                parsed_gap = dict(parsed_missing_items[0])
            obj['missing_items'] = parsed_missing_items
        elif os.environ.get('REMIND_V12B', '0') == '1':
            obj['missing_items'] = []
        # Carry facts between visits only when their citation and verbatim
        # support are present in the source-grounded evidence ledger.
        sources = {row['source_chunk']: row['evidence']
                   for row in pf.v7_evidence_ledger}
        supplied_entities = {str(value) for value in pf.entity}
        supplied_edges = {str(value) for value in pf.edge}
        accepted = []
        for fact in obj.get('known_facts', []):
            if not isinstance(fact, dict):
                continue
            source_type = str(fact.get('source_type') or '')
            source = str(fact.get('source_id') or fact.get('source_chunk') or '')
            quote = str(fact.get('quote') or '')
            if not source_type and source in sources:
                source_type = 'chunk'  # Backward-compatible chunk citation.
            valid_source = False
            if source_type == 'chunk':
                valid_source = (source in sources and quote and
                                normalize_quote(quote) in normalize_quote(sources[source]))
            elif source_type == 'entity':
                valid_source = source in supplied_entities and not quote
            elif source_type == 'edge':
                valid_source = source in supplied_edges and not quote
            if valid_source:
                item = {'fact': str(fact.get('fact') or '').strip(),
                        'source_type': source_type, 'source_id': source,
                        'quote': quote}
                if source_type == 'chunk':
                    item['source_chunk'] = source
                if item['fact'] and item not in accepted:
                    accepted.append(item)
        # B2 uses the same quote-validated facts as the judge.  A supported
        # chunk is marked directly; no second LLM evidence-claim call is made.
        if (getattr(pf, 'traversal_mode', '') == 'fact_v7' and
                os.environ.get('REMIND_B1', '0') == '2'):
            for item in accepted:
                source_type = item.get('source_type')
                source = str(item.get('source_id') or '')
                if source_type == 'chunk' and source:
                    pf.b2_supported_chunks.add(source.removeprefix('chunk:'))
                elif source_type == 'entity' and source:
                    pf.b2_supported_entities.add(source)
                elif source_type == 'edge' and source:
                    pf.b2_supported_edges.add(source)
            pf.logger.info('B2 evidence gate: mode=%s accepted_facts=%d supported_chunks=%s supported_entities=%s supported_edges=%s',
                           os.environ.get('REMIND_B1', '0'), len(accepted),
                           sorted(pf.b2_supported_chunks),
                           sorted(pf.b2_supported_entities),
                           sorted(pf.b2_supported_edges))
        # Facts are cumulative for the lifetime of this traversal. A judge
        # that omits a previously verified fact must not erase it; retaining
        # both source-grounded versions also makes contradictions auditable.
        merged = list(pf.v7_verified_facts)
        for item in accepted:
            if item not in merged:
                merged.append(item)
        pf.v7_verified_facts = merged
        _write_temp_summary(pf)
        # The model occasionally labels a supplied quote as a missing fact.
        # A source-grounded quote is a deterministic admission that this gap
        # is already covered; do not spend more traversal on it.
        if parsed_gap and parsed_gap['source_chunk'] and parsed_gap['quote']:
            # V11 may omit a self-contained chunk's neighbors from later
            # prompts, but quote validation must still see the full stored
            # first-read excerpt, which remains in pf.summary and the ledger.
            source = (pf.summary if context_enabled else judge_summary)
            source_text = str(source.get(parsed_gap['source_chunk'], ''))
            if normalize_quote(parsed_gap['quote']) in normalize_quote(source_text):
                pf.logger.info('V7 evidence gate: supplied gap quote is covered; '
                               'marking answerable')
                obj['enough'] = True
                obj['status'] = 'answerable'
                parsed_gap = None
        gap_key = None if parsed_gap is None else normalize(
            parsed_gap.get('id') or parsed_gap.get('question'))
        state = (tuple(sorted(set(pf.entity))), tuple(sorted(set(pf.chunk))),
                 tuple(sorted(set(pf.edge))))
        if gap_key and gap_key == pf.v7_last_gap_key and state == pf.v7_last_state:
            pf.v7_stalls += 1
        else:
            pf.v7_stalls = 0
        pf.v7_gap = parsed_gap
        pf.v7_missing_items = (parsed_missing_items
                               if os.environ.get('REMIND_V12B', '0') == '1'
                               else [])
        pf.v7_last_gap_key = gap_key
        pf.v7_last_state = state
        pf.v7_previous = pf.v7_gap
        pf.v7_evidence_key, pf.v7_enough = key, obj['enough']
        if v12_enabled:
            pf._v12_mark_context_judged(first_use_chunks)
            pf._v12_post_compact_evidence_key = pf._v11_evidence_state_key(
                query, pf._v12_prompt_summary(for_judge=True))
        if v11_enabled:
            pf._v11_mark_context_judged(first_use_chunks)
            pf._v11_post_compact_evidence_key = pf._v11_evidence_state_key(
                query, pf._v11_prompt_summary(for_judge=True))
        pf.logger.info('V7 judge: ' + json.dumps(obj, ensure_ascii=False))
        return obj['enough']
    except (ValueError, TypeError, AttributeError):
        pf.v7_metrics['invalid_judges'] += 1
        pf.v7_gap = None
        pf.v7_missing_items = []
        # Formatting failure cannot create an unsupported entry.
        fallback = pf.judge_sufficient_information_ft() == 'yes'
        if v12_enabled:
            pf._v12_mark_context_judged(first_use_chunks)
            pf._v12_post_compact_evidence_key = pf._v11_evidence_state_key(
                query, pf._v12_prompt_summary(for_judge=True))
        if v11_enabled:
            pf._v11_mark_context_judged(first_use_chunks)
            pf._v11_post_compact_evidence_key = pf._v11_evidence_state_key(
                query, pf._v11_prompt_summary(for_judge=True))
        return fallback


def eligible_entity(pf):
    if (os.environ.get('REMIND_V12A_DISABLE_GAP', '0') == '1' or
            os.environ.get('REMIND_V12A_SOFT_GAP', '0') == '1'):
        return None
    gap = pf.v7_gap
    if not gap:
        return None
    entity, quote, source = (gap.get(k, '') for k in ('entity', 'quote', 'source_chunk'))
    if not entity or entity.startswith(('chunk:', 'anchor-')):
        return None
    query_explicit = contains_entity(pf.query, entity)
    if not query_explicit:
        if not quote or source not in pf.summary:
            return None
        if normalize_quote(quote) not in normalize_quote(pf.summary[source]):
            return None
        if not contains_entity(quote, entity):
            return None
        # Text checks establish provenance, not a proof of semantic correctness.
        cid = source.removeprefix('chunk:')
        if cid not in pf.v7_source_cache:
            docs = pf.database.chunk_collection.get(ids=[cid], include=['documents'])
            pf.v7_source_cache[cid] = '\n'.join(docs.get('documents') or [])
        if not contains_entity(pf.v7_source_cache[cid], entity):
            return None
    found = pf.database.entity_collection.get(ids=[entity], include=['documents'])
    if entity not in found.get('ids', []):
        return None
    # Entity-level dedup is deliberately stricter than free-text gap dedup:
    # rephrasing a gap cannot repeatedly inject the same entrance.
    if entity in pf.v7_attempted:
        return None
    if entity in pf.entity and (entity in pf.c_node_list or pf.out_dgree.get(entity, 0) <= 0):
        return None
    return entity


def selector_suffix(pf):
    pf.v7_metrics['selectors'] += 1
    if os.environ.get('REMIND_V14A', '0') == '1':
        return ''  # V14a already supplies question, evidence, and secondary gap.
    suffix = '\nSEARCH STATE (always keep the original question in view):\n'
    if os.environ.get('REMIND_V12A_SOFT_GAP', '0') == '1':
        suffix += 'All listed local candidates remain eligible. Use the original question as the main ranking signal; the unresolved gap is only a weak secondary preference. Do not discard candidates or request a separate gap-entry jump.'
        if os.environ.get('REMIND_V12A_ANCHOR_REACH', '0') == '1':
            suffix += (' Candidates are ordered with a deterministic lookahead: '
                       'an anchor with an unread, query-matched chunk reachable '
                       'within the remaining jump budget is prioritized. It is '
                       'still an entity node; select it as entity:anchor-id.')
        if pf.v7_gap:
            suffix += '\nUnresolved requirement (weak ranking hint only):\n'
            suffix += json.dumps(pf.v7_gap, ensure_ascii=False)
        return suffix
    if os.environ.get('REMIND_V12A_DISABLE_GAP', '0') == '1':
        return suffix + ('Choose an admissible local expansion or backtrack based on the original question and accumulated evidence. '
                         'Do not use a separate unresolved-gap target or request a gap-entry jump.')
    checklist_mode = os.environ.get('REMIND_V12B', '0') == '1'
    if checklist_mode:
        suffix += json.dumps({
            'primary_gap': pf.v7_gap,
            'unresolved_fact_checklist': getattr(pf, 'v7_missing_items', []),
        }, ensure_ascii=False)
        suffix += ('\nChoose an admissible local expansion or backtrack likely '
                   'to add evidence for an unresolved item. Prefer the candidate '
                   'most likely to advance a specific item. This checklist is a '
                   'search hint and will be revised after the next evidence check.')
    else:
        suffix += 'Current unresolved requirement:\n'
        suffix += json.dumps(pf.v7_gap, ensure_ascii=False)
        suffix += '\nPrefer an admissible local expansion or backtrack that advances this requirement.'
    if eligible_entity(pf):
        target = 'primary checklist item' if checklist_mode else 'requirement'
        suffix += (f'\nOnly if no existing admissible expansion or backtrack can '
                   f'advance the {target}, output ```cot-ans\ngap:entry\n``` '
                   'to request its grounded entrance. Use this only as a fallback.')
    return suffix


def bounded_quick(pf, seed, budget):
    """Same connection-first threshold DFS/order as original; charge NEW nodes.

    Only additional entrances are bounded. The original initial quick graph
    is untouched. This explicitly accounts for v6's formerly uncharged work.
    """
    entities, chunks, edges = [seed], [], []
    seen_e, seen_c = set(pf.entity), set(pf.chunk)
    cost = int(seed not in seen_e)
    seen_e.add(seed)
    def visit(node):
        nonlocal cost
        if cost >= budget:
            return
        relations, connection = pf.database.get_entity_edges(pf.database_query, node)
        if (connection and connection['weight'] > pf.database.strong_connection_threshold
                and connection['to'] not in seen_c and cost < budget):
            seen_c.add(connection['to'])
            chunks.append(connection['to'])
            edges.append(dict(connection, type='connection'))
            cost += 1
        for relation in relations:
            if cost >= budget:
                break
            if (relation['to'] in seen_e or
                    relation['weight'] <= pf.database.strong_connection_threshold):
                continue
            seen_e.add(relation['to'])
            entities.append(relation['to'])
            edges.append(dict(relation, type='relation'))
            cost += 1
            visit(relation['to'])
    visit(seed)
    return entities, chunks, edges, cost


def enter(pf, budget):
    """Select one grounded gap entry for the shared main traversal.

    Entry selection uses the current full question/gap state. It does not run
    bounded_quick: the caller's traversal consumes the remaining jump budget
    and expands the selected node through the normal loop.
    """
    if (budget <= 0 or
            os.environ.get('REMIND_V12A_DISABLE_GAP', '0') == '1' or
            os.environ.get('REMIND_V12A_SOFT_GAP', '0') == '1'):
        return None
    seed = eligible_entity(pf)
    if not seed:
        return None
    pf.v7_attempted.add(seed)
    source_type = 'query' if contains_entity(pf.query, seed) else 'evidence'
    pf.v7_metrics[source_type + '_entries'] += 1
    pf.logger.info('V7 entry provenance: %s entity=%s (main traversal)',
                   source_type, seed)
    if seed in pf.entity:
        pf.v7_metrics['switches'] += 1
        pf.logger.info('V7 grounded in-graph switch: %s', seed)
        return seed, 1

    # The entity is already in the KG but not in the current local frontier.
    # Register it as a root and let the normal traversal inspect its edges.
    # No quick-subgraph expansion or extra jump budget is charged here.
    found = pf.database.entity_collection.get(ids=[seed], include=['documents'])
    if seed not in found.get('ids', []):
        return None
    pf.v7_roots.append(seed)
    pf._b2_record_root(seed, 'gap_entry')
    pf.v7_metrics['entries'] += 1
    pf.v7_metrics['added_nodes'] += 1
    pf.entity.append(seed)
    pf.out_dgree[seed], _ = pf.get_out_dgree(seed)
    pf.logger.info('V7 gap entry registered for main traversal: %s', seed)
    return seed, 1


def enter_global(pf, budget):
    """A5 continuation fallback: seed a fresh weighted DFS from the whole KG."""
    if os.environ.get('REMIND_A5_GLOBAL_FALLBACK', '1') == '0':
        return None
    if budget <= 1:
        return None
    gap = getattr(pf, 'v7_gap', None) or {}
    search_text = str((gap.get('question') if
                       os.environ.get('REMIND_V12A_DISABLE_GAP', '0') != '1' and
                       os.environ.get('REMIND_V12A_SOFT_GAP', '0') != '1'
                       else None) or pf.query)
    embedding = pf.database.embedding.sentence_embedding(search_text)
    count = pf.database.entity_collection.count()
    if not count:
        return None
    candidates = pf.database.entity_collection.query(
        query_embeddings=[embedding], n_results=min(count, 64))
    for seed in (candidates.get('ids') or [[]])[0]:
        if seed in pf.entity or seed in pf.v7_global_attempted:
            continue
        pf.v7_global_attempted.add(seed)
        entities, chunks, edges, cost = bounded_quick(pf, seed, budget)
        # A seed that cannot reach a new graph node is not useful; continue to
        # the next nearest entity rather than repeatedly entering it.
        if cost <= 1 or not (chunks or len(entities) > 1):
            continue
        pf.v7_roots.append(seed)
        pf._b2_record_root(seed, 'global_gap_entry')
        pf.v7_metrics['global_entries'] = pf.v7_metrics.get('global_entries', 0) + 1
        pf.v7_metrics['added_nodes'] += cost
        for edge in edges:
            if edge['type'] == 'relation':
                pf.edge.append(f"from entity:{edge['from']} to entity:{edge['to']} with relations: {edge['documents']}")
            else:
                pf.edge.append(f"from entity:{edge['from']} to chunk:{edge['to']}")
            pf.path.append({k: edge[k] for k in ('from', 'to', 'id', 'type', 'documents')})
            pf._b2_record_arrival(pf.path[-1], 'global_gap_entry')
        pf.entity.extend(e for e in entities if e not in pf.entity)
        pf.chunk.extend(c for c in chunks if c not in pf.chunk)
        for entity in pf.entity:
            pf.out_dgree[entity], _ = pf.get_out_dgree(entity)
        for cid in chunks:
            pf.summary[f'chunk:{cid}'] = pf.get_chunk_summary(cid)
        pf.logger.info('A5 global fallback entry: ' + json.dumps(
            dict(seed=seed, cost=cost, chunks=chunks, gap=gap), ensure_ascii=False))
        return seed, cost
    return None
