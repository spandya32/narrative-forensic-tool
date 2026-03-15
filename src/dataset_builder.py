"""
Dataset builder for the Narrative Forensics Tool — Phase 5.

Parses all analysis markdown reports in a directory, extracts structured
metrics from each, computes cross-document statistics, and generates
an INDEX.md comparison table.

Metric fields extracted per document:
  - title, author, year, source_type
  - claims_count, citation_count, evidence_density
  - hedging_index_pct
  - entropy_score, entropy_normalised, entropy_label
  - network_nodes, network_edges, degree_centralization
  - actors_count, causal_claims, compression_ratio
  - top_phrases (list)
  - filename
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DocumentRecord:
    filename: str
    filepath: str

    # Metadata
    title: str = ""
    author: str = ""
    year: str = ""
    source_type: str = ""

    # Evidence density
    claims_count: int = 0
    citation_count: int = 0
    evidence_density: float = 0.0
    density_label: str = ""

    # Hedging
    hedging_index_pct: float = 0.0

    # Narrative entropy
    entropy_score: float = 0.0
    entropy_normalised: float = 0.0
    entropy_label: str = ""
    entropy_clusters: int = 0

    # Citation network
    network_nodes: int = 0
    network_edges: int = 0
    degree_centralization: float = 0.0
    centralization_label: str = ""

    # Compression
    actors_count: int = 0
    causal_claims_count: int = 0
    compression_ratio: float = 0.0

    # Propagation
    top_phrases: List[str] = field(default_factory=list)
    phrase_cluster_count: int = 0

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "title": self.title,
            "author": self.author,
            "year": self.year,
            "source_type": self.source_type,
            "claims_count": self.claims_count,
            "citation_count": self.citation_count,
            "evidence_density": self.evidence_density,
            "density_label": self.density_label,
            "hedging_index_pct": self.hedging_index_pct,
            "entropy_score": self.entropy_score,
            "entropy_normalised": self.entropy_normalised,
            "entropy_label": self.entropy_label,
            "entropy_clusters": self.entropy_clusters,
            "network_nodes": self.network_nodes,
            "network_edges": self.network_edges,
            "degree_centralization": self.degree_centralization,
            "actors_count": self.actors_count,
            "causal_claims_count": self.causal_claims_count,
            "compression_ratio": self.compression_ratio,
            "top_phrases": self.top_phrases[:5],
            "phrase_cluster_count": self.phrase_cluster_count,
        }


@dataclass
class DatasetIndex:
    records: List[DocumentRecord] = field(default_factory=list)
    generated_at: str = ""

    @property
    def count(self) -> int:
        return len(self.records)

    def mean(self, attr: str) -> float:
        vals = [getattr(r, attr) for r in self.records if getattr(r, attr, 0)]
        return sum(vals) / len(vals) if vals else 0.0

    def max_by(self, attr: str) -> Optional[DocumentRecord]:
        return max(self.records, key=lambda r: getattr(r, attr, 0), default=None)

    def min_by(self, attr: str) -> Optional[DocumentRecord]:
        vals = [r for r in self.records if getattr(r, attr, 0) > 0]
        return min(vals, key=lambda r: getattr(r, attr, 0), default=None)

    def sorted_by(self, attr: str, reverse: bool = True) -> List[DocumentRecord]:
        return sorted(self.records, key=lambda r: getattr(r, attr, 0), reverse=reverse)


# ---------------------------------------------------------------------------
# Markdown parser
# ---------------------------------------------------------------------------

def _extract(pattern: str, text: str, group: int = 1, default="") -> str:
    m = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
    return m.group(group).strip() if m else default


def _extract_float(pattern: str, text: str, default: float = 0.0) -> float:
    raw = _extract(pattern, text, default="")
    try:
        return float(raw)
    except (ValueError, TypeError):
        return default


def _extract_int(pattern: str, text: str, default: int = 0) -> int:
    raw = _extract(pattern, text, default="")
    try:
        return int(raw)
    except (ValueError, TypeError):
        return default


def parse_report(filepath: str) -> DocumentRecord:
    """Parse a single analysis markdown report and return a DocumentRecord."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return DocumentRecord(filename=os.path.basename(filepath), filepath=filepath)

    filename = os.path.basename(filepath)
    rec = DocumentRecord(filename=filename, filepath=filepath)

    # Skip index files
    if filename.lower() == "index.md":
        return rec

    # Metadata
    rec.title       = _extract(r"^Title:\s*(.+)$", text)
    rec.author      = _extract(r"^Author:\s*(.+)$", text)
    rec.year        = _extract(r"^Year:\s*(.+)$", text)
    rec.source_type = _extract(r"^Source Type:\s*(.+)$", text)

    # Evidence density
    rec.claims_count    = _extract_int(r"^Claims Count:\s*(\d+)", text)
    rec.citation_count  = _extract_int(r"^Citation Count:\s*(\d+)", text)
    rec.evidence_density = _extract_float(r"Evidence Density Score:\s*([0-9.]+)", text)
    rec.density_label   = _extract(r"Evidence Density Score:\s*[0-9.]+\s+\((.+?)\)", text)

    # Hedging
    rec.hedging_index_pct = _extract_float(r"Hedging Index:\s*\*\*([0-9.]+)%\*\*", text)

    # Narrative entropy
    rec.entropy_score      = _extract_float(r"Entropy Score:\s*\*\*([0-9.]+)\*\*", text)
    rec.entropy_normalised = _extract_float(r"Normalised:\s*\*\*([0-9.]+)\*\*", text)
    rec.entropy_label      = _extract(r"Normalised:\s*\*\*[0-9.]+\*\*\s*\((.+?)\)", text)
    m_clusters = re.search(r"Clusters:\s*(\d+)\s*\|.*?Sentences analysed", text)
    rec.entropy_clusters   = int(m_clusters.group(1)) if m_clusters else 0

    # Citation network
    m_net = re.search(r"Nodes \(authors\):\s*(\d+)\s*\|+\s*Edges:\s*(\d+)", text)
    if m_net:
        rec.network_nodes = int(m_net.group(1))
        rec.network_edges = int(m_net.group(2))
    rec.degree_centralization = _extract_float(
        r"Degree Centralization:\s*\*\*([0-9.]+)\*\*", text
    )
    rec.centralization_label = _extract(
        r"Degree Centralization:\s*\*\*[0-9.]+\*\*\s*\((.+?)\)", text
    )

    # Compression
    rec.actors_count       = _extract_int(r"^Actors Detected:\s*(\d+)", text)
    rec.causal_claims_count = _extract_int(r"^Causal Claims Detected:\s*(\d+)", text)
    rec.compression_ratio  = _extract_float(r"Compression Ratio:\s*\*\*([0-9.]+)\*\*", text)

    # Top phrases
    phrase_section = re.search(
        r"### Key Narrative Phrases\n(.*?)(?=\n###|\n---|\Z)", text, re.DOTALL
    )
    if phrase_section:
        phrases = re.findall(r"- `([^`]+)`", phrase_section.group(1))
        rec.top_phrases = phrases[:10]

    # Phrase cluster count
    m_pc = re.search(r"### Phrase Clusters \((\d+) total\)", text)
    rec.phrase_cluster_count = int(m_pc.group(1)) if m_pc else 0

    return rec


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------

def build_index(analysis_dir: str) -> DatasetIndex:
    """
    Scan an analysis directory and parse all markdown reports.

    Args:
        analysis_dir: Path to directory containing .md report files.

    Returns:
        DatasetIndex with all parsed DocumentRecords.
    """
    idx = DatasetIndex(generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"))
    if not os.path.isdir(analysis_dir):
        return idx

    for fname in sorted(os.listdir(analysis_dir)):
        if not fname.endswith(".md"):
            continue
        if fname.lower() in ("index.md", "example_book_analysis.md"):
            continue
        fpath = os.path.join(analysis_dir, fname)
        rec = parse_report(fpath)
        if rec.title or rec.claims_count:
            idx.records.append(rec)

    return idx


# ---------------------------------------------------------------------------
# INDEX.md generator
# ---------------------------------------------------------------------------

def generate_index_markdown(idx: DatasetIndex) -> str:
    """Render DatasetIndex as an INDEX.md document."""
    lines: List[str] = []
    lines.append("# Narrative Forensics — Analysis Index\n")
    lines.append(f"Generated: {idx.generated_at}  |  Documents: {idx.count}\n")
    lines.append("\n---\n")

    # Summary comparison table
    lines.append("## Document Comparison\n")
    lines.append(
        "| Document | Type | Claims | Citations | Density | Hedging% | Entropy | Actors | Centralization |"
    )
    lines.append(
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"
    )
    for r in sorted(idx.records, key=lambda x: x.title):
        short = r.title[:40] if r.title else r.filename[:40]
        lines.append(
            f"| {short} | {r.source_type} "
            f"| {r.claims_count:,} | {r.citation_count:,} "
            f"| {r.evidence_density:.3f} | {r.hedging_index_pct:.1f}% "
            f"| {r.entropy_normalised:.3f} | {r.actors_count} "
            f"| {r.degree_centralization:.3f} |"
        )
    lines.append("")

    # Cross-dataset statistics
    lines.append("\n---\n")
    lines.append("## Cross-Dataset Statistics\n")

    lines.append("### Evidence & Citations\n")
    lines.append(f"- Mean evidence density: **{idx.mean('evidence_density'):.4f}**")
    lines.append(f"- Mean hedging index: **{idx.mean('hedging_index_pct'):.1f}%**")
    lines.append(f"- Mean citation count: **{idx.mean('citation_count'):.0f}**")
    hi = idx.max_by("hedging_index_pct")
    lo = idx.min_by("hedging_index_pct")
    if hi:
        lines.append(f"- Most hedged: **{hi.title or hi.filename}** ({hi.hedging_index_pct:.1f}%)")
    if lo:
        lines.append(f"- Least hedged: **{lo.title or lo.filename}** ({lo.hedging_index_pct:.1f}%)")
    hd = idx.max_by("evidence_density")
    if hd:
        lines.append(f"- Highest evidence density: **{hd.title or hd.filename}** ({hd.evidence_density:.4f})")
    lines.append("")

    lines.append("### Narrative Entropy\n")
    lines.append(f"- Mean entropy (normalised): **{idx.mean('entropy_normalised'):.4f}**")
    me = idx.max_by("entropy_normalised")
    le = idx.min_by("entropy_normalised")
    if me:
        lines.append(f"- Most diverse narrative: **{me.title or me.filename}** ({me.entropy_normalised:.4f})")
    if le:
        lines.append(f"- Most consolidated narrative: **{le.title or le.filename}** ({le.entropy_normalised:.4f})")
    lines.append("")

    lines.append("### Citation Networks\n")
    lines.append(f"- Mean network nodes: **{idx.mean('network_nodes'):.0f}**")
    mn = idx.max_by("degree_centralization")
    if mn:
        lines.append(
            f"- Highest centralization (echo chamber risk): **{mn.title or mn.filename}** "
            f"({mn.degree_centralization:.4f})"
        )
    lines.append("")

    lines.append("### Narrative Compression\n")
    lines.append(f"- Mean actors detected: **{idx.mean('actors_count'):.0f}**")
    lines.append(f"- Mean causal claims: **{idx.mean('causal_claims_count'):.0f}**")
    mc = idx.max_by("compression_ratio")
    if mc and mc.compression_ratio > 0:
        lines.append(
            f"- Least compressed (most diverse actors): **{mc.title or mc.filename}** "
            f"(ratio {mc.compression_ratio:.4f})"
        )
    lines.append("")

    # Per-document detail cards
    lines.append("\n---\n")
    lines.append("## Per-Document Profiles\n")
    for r in sorted(idx.records, key=lambda x: x.title):
        lines.append(f"### {r.title or r.filename}\n")
        lines.append(f"- **File**: `{r.filename}`")
        lines.append(f"- **Author**: {r.author or 'Unknown'}")
        lines.append(f"- **Evidence Density**: {r.evidence_density:.4f}  ({r.density_label})")
        lines.append(f"- **Hedging Index**: {r.hedging_index_pct:.1f}%")
        lines.append(f"- **Narrative Entropy**: {r.entropy_normalised:.4f}  ({r.entropy_label})")
        lines.append(
            f"- **Citation Network**: {r.network_nodes} nodes, {r.network_edges} edges, "
            f"centralization {r.degree_centralization:.4f}"
        )
        lines.append(
            f"- **Compression**: {r.actors_count} actors / {r.causal_claims_count} causal claims"
        )
        if r.top_phrases:
            lines.append(f"- **Top Phrases**: {', '.join('`' + p + '`' for p in r.top_phrases[:5])}")
        lines.append("")

    lines.append("\n---\n")
    lines.append("_Generated by Narrative Forensics Tool_")
    return "\n".join(lines)


def save_index(idx: DatasetIndex, output_path: str) -> None:
    """Write INDEX.md to disk."""
    md = generate_index_markdown(idx)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)
