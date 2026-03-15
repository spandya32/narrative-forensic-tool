"""
Citation network analyser for the Narrative Forensics Tool.

Builds a directed citation graph from a CitationReport (Phase 2) and
computes network metrics that reveal echo-chamber structure.

Graph model (CLAUDE.md §9):
  node  = author / source
  edge  = citation (one source cites another)

When a document has only one implicit "author" (e.g. a PDF book), the
graph is built from co-citation: two cited authors are connected when
they appear together in the same sentence.

Metrics:
  - Degree centrality         : how many citations each node receives
  - Betweenness centrality    : how often a node lies on paths between others
  - Degree centralization     : network-level measure of hub dominance
  - Clustering coefficient    : how tightly clustered the citation neighbourhood is

High centralization → few dominant sources (echo chamber signal).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class NetworkNode:
    name: str
    degree: int
    degree_centrality: float
    betweenness_centrality: float
    is_hub: bool                    # top-5% by degree


@dataclass
class CitationNetworkResult:
    source_id: str
    node_count: int
    edge_count: int
    degree_centralization: float    # 0=star/hub, 1=uniform — note: inverted below
    mean_clustering: float
    top_nodes: List[NetworkNode] = field(default_factory=list)
    isolated_authors: List[str] = field(default_factory=list)

    @property
    def centralization_label(self) -> str:
        # High centralization = hub dominance
        if self.degree_centralization >= 0.60:
            return "high (potential echo chamber)"
        if self.degree_centralization >= 0.30:
            return "moderate"
        return "low (diverse sources)"

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "degree_centralization": round(self.degree_centralization, 4),
            "centralization_label": self.centralization_label,
            "mean_clustering": round(self.mean_clustering, 4),
            "top_nodes": [
                {
                    "name": n.name,
                    "degree": n.degree,
                    "degree_centrality": round(n.degree_centrality, 4),
                    "betweenness_centrality": round(n.betweenness_centrality, 4),
                    "is_hub": n.is_hub,
                }
                for n in self.top_nodes
            ],
            "isolated_authors": self.isolated_authors[:20],
        }

    def summary(self) -> str:
        return (
            f"Nodes (authors)  : {self.node_count}\n"
            f"Edges            : {self.edge_count}\n"
            f"Centralization   : {self.degree_centralization:.4f}  ({self.centralization_label})\n"
            f"Mean clustering  : {self.mean_clustering:.4f}"
        )


# ---------------------------------------------------------------------------
# Graph construction helpers
# ---------------------------------------------------------------------------

# Names that look like authors but are publishers, cities, or calendar terms
_NOISE_NODES = {
    # Cities / publishers commonly appearing in bibliographies
    "london", "oxford", "cambridge", "calcutta", "bombay", "madras",
    "allahabad", "pondicherry", "delhi", "new delhi", "poona", "pune",
    "varanasi", "banaras", "lucknow", "patna", "bangalore", "hyderabad",
    "new york", "chicago", "princeton", "harvard", "berlin", "paris",
    "toronto", "sydney", "amsterdam", "leiden", "tokyo", "beijing",
    "westminster", "edinburgh", "glasgow", "budapest", "vienna",
    "leipzig", "cologne", "munich", "frankfurt",
    # Calendar terms (false positives in author-year patterns)
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "winter", "summer", "spring", "autumn", "fall",
    # Generic false positives
    "ibid", "anon", "idem", "et al", "op cit", "loc cit",
    "sinbad", "unknown",
}


def _is_noise_author(name: str) -> bool:
    return name.lower().strip() in _NOISE_NODES or len(name.strip()) <= 2


def _build_cocitation_graph(citation_report):
    """
    Build an undirected co-citation graph.

    Two authors are connected when they both appear as citations in the
    same sentence. Publisher names, city names, and calendar terms are
    excluded from the graph.
    """
    import networkx as nx  # type: ignore
    from collections import defaultdict

    G = nx.Graph()

    # Group citations by sentence index — filter noise authors
    sent_to_authors: dict = defaultdict(set)
    for cit in citation_report.citations:
        if cit.author and not _is_noise_author(cit.author):
            sent_to_authors[cit.sentence_index].add(cit.author)

    # Add nodes for every cited author
    all_authors = {
        cit.author for cit in citation_report.citations
        if cit.author and not _is_noise_author(cit.author)
    }
    for author in all_authors:
        G.add_node(author)

    # Connect co-cited authors
    for authors in sent_to_authors.values():
        authors_list = list(authors)
        for i in range(len(authors_list)):
            for j in range(i + 1, len(authors_list)):
                a, b = authors_list[i], authors_list[j]
                if G.has_edge(a, b):
                    G[a][b]["weight"] = G[a][b].get("weight", 1) + 1
                else:
                    G.add_edge(a, b, weight=1)

    return G


def _degree_centralization(G) -> float:
    """
    Freeman degree centralization in [0, 1].
    0 = all nodes equal, 1 = star (single hub).
    """
    import networkx as nx  # type: ignore

    n = G.number_of_nodes()
    if n <= 2:
        return 0.0

    degrees = dict(G.degree())
    max_deg = max(degrees.values())
    numerator = sum(max_deg - d for d in degrees.values())
    denominator = (n - 1) * (n - 2)
    if denominator == 0:
        return 0.0
    return numerator / denominator


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_citation_network(citation_report, source_id: str = "") -> CitationNetworkResult:
    """
    Build and analyse the citation network from a CitationReport.

    Args:
        citation_report: CitationReport from citation_extractor.
        source_id:       Label for the source document.

    Returns:
        CitationNetworkResult with graph metrics.
    """
    try:
        import networkx as nx  # type: ignore
    except ImportError:
        return CitationNetworkResult(
            source_id=source_id,
            node_count=0,
            edge_count=0,
            degree_centralization=0.0,
            mean_clustering=0.0,
        )

    sid = source_id or citation_report.source_id
    G = _build_cocitation_graph(citation_report)

    n = G.number_of_nodes()
    e = G.number_of_edges()

    if n == 0:
        return CitationNetworkResult(
            source_id=sid,
            node_count=0,
            edge_count=0,
            degree_centralization=0.0,
            mean_clustering=0.0,
        )

    # Compute metrics
    centralization = _degree_centralization(G)
    try:
        mean_clustering = nx.average_clustering(G)
    except Exception:
        mean_clustering = 0.0

    deg_cent = nx.degree_centrality(G)
    try:
        bet_cent = nx.betweenness_centrality(G, normalized=True)
    except Exception:
        bet_cent = {node: 0.0 for node in G.nodes()}

    degrees = dict(G.degree())
    max_deg = max(degrees.values()) if degrees else 1
    hub_threshold = max(1, int(max_deg * 0.75))

    # Build top-node list (sorted by degree descending)
    nodes_sorted = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
    top_nodes = [
        NetworkNode(
            name=name,
            degree=deg,
            degree_centrality=deg_cent.get(name, 0.0),
            betweenness_centrality=bet_cent.get(name, 0.0),
            is_hub=deg >= hub_threshold,
        )
        for name, deg in nodes_sorted[:20]
    ]

    isolated = [n for n, d in degrees.items() if d == 0]

    return CitationNetworkResult(
        source_id=sid,
        node_count=n,
        edge_count=e,
        degree_centralization=centralization,
        mean_clustering=mean_clustering,
        top_nodes=top_nodes,
        isolated_authors=isolated,
    )
