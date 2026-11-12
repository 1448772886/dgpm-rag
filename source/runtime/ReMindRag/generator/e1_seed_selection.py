"""Deterministic E1 selection of the second initial entity seed."""
from collections import deque
import math


def score_second_seed(candidates, query_distances, graph, alpha=0.33):
    """Keep rank 1 fixed and choose rank 2..5 by relevance and graph distance.

    `query_distances` are the vector-store distances in retrieval-rank order.
    Graph distances are unweighted, undirected shortest-path hop counts.
    If a candidate is disconnected, the formula is undefined and the caller
    can retain the original top-two seeds; ties favor the higher retrieval
    rank.
    """
    if len(candidates) < 5 or len(query_distances) < 5:
        raise ValueError("five ranked candidates and distances are required")

    first = candidates[0]
    hop = {first: 0}
    queue = deque([first])
    while queue:
        node = queue.popleft()
        for neighbor in graph.get(node, ()):
            if neighbor not in hop:
                hop[neighbor] = hop[node] + 1
                queue.append(neighbor)

    l1, l5 = float(query_distances[0]), float(query_distances[4])
    raw_distances = [hop.get(node, math.inf) for node in candidates[1:5]]
    if any(math.isinf(distance) for distance in raw_distances):
        raise ValueError("a rank-2..5 candidate is disconnected from rank 1")
    max_distance = max(raw_distances)
    scored = []
    for rank, (node, li, distance) in enumerate(
            zip(candidates[1:5], query_distances[1:5], raw_distances), start=2):
        relevance = (float(l5) - float(li)) / (l5 - l1) if l5 > l1 else 1.0
        normalized_distance = distance / max_distance if max_distance else 0.0
        score = alpha * relevance + (1.0 - alpha) * normalized_distance
        scored.append({
            "rank": rank, "id": node, "query_distance": float(li),
            "graph_distance": distance,
            "relevance": relevance, "normalized_distance": normalized_distance,
            "score": score,
        })

    winner = max(scored, key=lambda row: (row["score"], -row["rank"]))
    audit = [{
        "rank": 1, "id": first, "query_distance": float(query_distances[0]),
        "graph_distance": 0, "score": None,
    }, *scored]
    return first, winner["id"], audit
