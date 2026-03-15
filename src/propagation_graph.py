"""
Narrative propagation graph builder for the Narrative Forensics Tool — Phase 4.

Tracks how narrative phrases spread across a set of documents and builds
a directed propagation graph (CLAUDE.md §11).

Graph model:
  node  = document (source_id)
  edge  = shared narrative phrase cluster; weight = number of shared phrases

From this graph, we can identify:
  - which documents are narrative hubs (high out-degree)
  - which phrases propagate most widely
  - the direction of propagation (earlier → later documents)

Temporal tracking uses document order in the input list as a proxy for
chronological order when explicit timestamps are unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PropagationEdge:
    source: str             # document that introduced the phrase
    target: str             # document where the phrase re-appears
    shared_phrases: List[str]
    weight: int             # number of shared phrases

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "weight": self.weight,
            "shared_phrases": self.shared_phrases[:10],
        }


@dataclass
class PropagationNode:
    doc_id: str
    in_degree: int
    out_degree: int
    unique_phrases: int
    is_hub: bool

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "in_degree": self.in_degree,
            "out_degree": self.out_degree,
            "unique_phrases": self.unique_phrases,
            "is_hub": self.is_hub,
        }


@dataclass
class PropagationResult:
    source_ids: List[str]
    nodes: List[PropagationNode] = field(default_factory=list)
    edges: List[PropagationEdge] = field(default_factory=list)
    top_propagating_phrases: List[str] = field(default_factory=list)
    network_density: float = 0.0

    @property
    def hub_documents(self) -> List[str]:
        return [n.doc_id for n in self.nodes if n.is_hub]

    def to_dict(self) -> dict:
        return {
            "source_ids": self.source_ids,
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "network_density": round(self.network_density, 4),
            "hub_documents": self.hub_documents,
            "top_propagating_phrases": self.top_propagating_phrases[:20],
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
        }

    def summary(self) -> str:
        return (
            f"Documents       : {len(self.nodes)}\n"
            f"Propagation edges: {len(self.edges)}\n"
            f"Network density : {self.network_density:.4f}\n"
            f"Hub documents   : {', '.join(self.hub_documents) or 'none'}\n"
            f"Top phrases     : {', '.join(self.top_propagating_phrases[:5])}"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_propagation_graph(phrase_results: list) -> PropagationResult:
    """
    Build a narrative propagation graph from ordered PhraseExtractionResults.

    Args:
        phrase_results: List of PhraseExtractionResult objects in document
                        order (chronological when possible).

    Returns:
        PropagationResult with nodes, edges, and hub detection.
    """
    source_ids = [r.source_id for r in phrase_results]
    result = PropagationResult(source_ids=source_ids)

    if len(phrase_results) < 2:
        return result

    # Top phrases per document as sets for fast intersection
    doc_phrase_sets: List[set] = []
    doc_phrase_lists: List[List[str]] = []
    for res in phrase_results:
        top = {p.text for p in res.top_phrases(60)}
        doc_phrase_sets.append(top)
        doc_phrase_lists.append(list(top))

    # Build directed edges: doc[i] → doc[j] for i < j (earlier → later)
    edges: List[PropagationEdge] = []
    in_degree  = {sid: 0 for sid in source_ids}
    out_degree = {sid: 0 for sid in source_ids}

    for i in range(len(phrase_results)):
        for j in range(i + 1, len(phrase_results)):
            shared = sorted(doc_phrase_sets[i] & doc_phrase_sets[j])
            if not shared:
                continue
            src = source_ids[i]
            tgt = source_ids[j]
            edges.append(PropagationEdge(
                source=src,
                target=tgt,
                shared_phrases=shared,
                weight=len(shared),
            ))
            out_degree[src] += 1
            in_degree[tgt]  += 1

    edges.sort(key=lambda e: e.weight, reverse=True)

    # Identify hub documents (top-33% by out-degree)
    max_out = max(out_degree.values()) if out_degree else 0
    hub_threshold = max(1, int(max_out * 0.67))

    nodes = [
        PropagationNode(
            doc_id=sid,
            in_degree=in_degree[sid],
            out_degree=out_degree[sid],
            unique_phrases=len(doc_phrase_sets[i]),
            is_hub=out_degree[sid] >= hub_threshold,
        )
        for i, sid in enumerate(source_ids)
    ]
    nodes.sort(key=lambda n: n.out_degree, reverse=True)

    # Top propagating phrases: appear in the most documents
    from collections import Counter
    phrase_doc_count: Counter = Counter()
    for phrase_set in doc_phrase_sets:
        for phrase in phrase_set:
            phrase_doc_count[phrase] += 1
    top_phrases = [p for p, _ in phrase_doc_count.most_common(30) if phrase_doc_count[p] > 1]

    # Network density: actual edges / possible directed edges
    n = len(phrase_results)
    max_edges = n * (n - 1) / 2
    density = len(edges) / max_edges if max_edges > 0 else 0.0

    result.nodes = nodes
    result.edges = edges
    result.top_propagating_phrases = top_phrases
    result.network_density = density
    return result


def build_propagation_graph_with_networkx(phrase_results: list) -> tuple:
    """
    Same as build_propagation_graph but also returns a networkx DiGraph
    for callers that want raw graph access (e.g. future visualisation).

    Returns: (PropagationResult, nx.DiGraph or None)
    """
    prop_result = build_propagation_graph(phrase_results)
    try:
        import networkx as nx  # type: ignore
        G = nx.DiGraph()
        for node in prop_result.nodes:
            G.add_node(node.doc_id, **node.to_dict())
        for edge in prop_result.edges:
            G.add_edge(edge.source, edge.target, weight=edge.weight,
                       phrases=edge.shared_phrases[:5])
        return prop_result, G
    except ImportError:
        return prop_result, None
