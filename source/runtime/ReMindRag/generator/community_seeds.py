"""Deterministic community-aware seed selection for fresh retrieval."""

from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import tempfile


_CACHE_VERSION = 2
_CHUNK_PROJECTION_WEIGHT = 0.25


def _read_all(collection, include=("metadatas",), batch_size=5000):
    rows = []
    count = collection.count()
    for offset in range(0, count, batch_size):
        result = collection.get(
            limit=batch_size, offset=offset, include=list(include))
        rows.extend(zip(result.get("ids", []),
                        result.get("metadatas", [])))
    return rows


def _community_data(database, cache_path=None):
    if cache_path and os.path.isfile(cache_path):
        try:
            with open(cache_path, encoding="utf-8") as stream:
                cached = json.load(stream)
            if cached.get("version") == _CACHE_VERSION:
                return cached["entities"], cached["anchors"], cached["count"]
        except (OSError, ValueError, KeyError, TypeError):
            pass

    entity_rows = _read_all(database.entity_collection)
    entities = sorted(
        str(entity_id) for entity_id, _ in entity_rows
        if not str(entity_id).startswith("anchor-"))
    entity_set = set(entities)
    relation_rows = _read_all(database.relation_collection)

    edge_weights = Counter()
    anchor_neighbors = defaultdict(list)
    for _, metadata in relation_rows:
        if not metadata:
            continue
        subject = str(metadata.get("subject_entity_id", ""))
        object_ = str(metadata.get("object_entity_id", ""))
        if subject.startswith("anchor-") and object_ in entity_set:
            anchor_neighbors[subject].append(object_)
        elif object_.startswith("anchor-") and subject in entity_set:
            anchor_neighbors[object_].append(subject)
        elif subject in entity_set and object_ in entity_set and subject != object_:
            edge_weights[tuple(sorted((subject, object_)))] += 1.0

    # Anchors are omitted as vertices so they cannot become artificial hubs.
    # Their entity memberships are instead projected as weak, normalized
    # co-occurrence edges, preserving within-chunk locality.
    for neighbors in anchor_neighbors.values():
        members = sorted(set(neighbors))
        if len(members) < 2:
            continue
        pair_weight = _CHUNK_PROJECTION_WEIGHT / (len(members) - 1)
        for index, left in enumerate(members):
            for right in members[index + 1:]:
                edge_weights[(left, right)] += pair_weight

    if entities:
        import igraph as ig
        import leidenalg

        index = {entity: position for position, entity in enumerate(entities)}
        sorted_edges = sorted(edge_weights)
        graph = ig.Graph(
            n=len(entities),
            edges=[(index[left], index[right]) for left, right in sorted_edges],
            directed=False)
        graph.es["weight"] = [edge_weights[edge] for edge in sorted_edges]
        graph.vs["name"] = entities
        if graph.ecount():
            partition = leidenalg.find_partition(
                graph,
                leidenalg.RBConfigurationVertexPartition,
                weights="weight",
                seed=17,
                n_iterations=-1)
            entity_communities = {
                entity: int(partition.membership[index[entity]])
                for entity in entities
            }
        else:
            entity_communities = {
                entity: position for position, entity in enumerate(entities)
            }
    else:
        entity_communities = {}

    anchor_communities = {}
    for anchor, neighbors in anchor_neighbors.items():
        counts = Counter(entity_communities[node] for node in neighbors
                         if node in entity_communities)
        if counts:
            # Stable tie-breaking keeps repeated runs reproducible.
            anchor_communities[anchor] = min(
                counts, key=lambda community: (-counts[community], community))

    community_count = len(set(entity_communities.values()))
    if cache_path:
        cache = {
            "version": _CACHE_VERSION,
            "entities": entity_communities,
            "anchors": anchor_communities,
            "count": community_count,
        }
        cache_dir = os.path.dirname(os.path.abspath(cache_path))
        os.makedirs(cache_dir, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=".community-seeds-", suffix=".json", dir=cache_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(cache, stream, ensure_ascii=False, sort_keys=True)
            os.replace(temporary, cache_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return entity_communities, anchor_communities, community_count


def choose_community_seeds(candidates, communities, seed_count=2):
    """Keep the top candidate, then the highest-ranked distinct community."""
    candidates = list(candidates)
    if seed_count <= 0 or len(candidates) < seed_count:
        return candidates[:seed_count], {
            "selected": False,
            "reason": "insufficient_candidates",
        }

    first = candidates[0]
    first_community = communities.get(first)
    if first_community is None:
        return candidates[:seed_count], {
            "selected": False,
            "reason": "top_candidate_unassigned",
            "first": first,
        }

    for candidate in candidates[1:]:
        candidate_community = communities.get(candidate)
        if (candidate_community is not None and
                candidate_community != first_community):
            return [first, candidate], {
                "selected": True,
                "first": first,
                "first_community": first_community,
                "second": candidate,
                "second_community": candidate_community,
            }

    return candidates[:seed_count], {
        "selected": False,
        "reason": "no_distinct_community_candidate",
        "first": first,
        "first_community": first_community,
    }


def get_community_seed_map(database, cache_path=None):
    entities, anchors, count = _community_data(database, cache_path)
    return {**entities, **anchors}, count
