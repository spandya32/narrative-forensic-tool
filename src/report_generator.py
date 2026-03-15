"""
Report generator for the Narrative Forensics Tool — Phase 2.

Assembles the output of all Phase 2 analysis modules into a structured
Markdown report that follows the DATA_SCHEMA.md format exactly.

Inputs (all optional — sections are omitted if data is not provided):
  - DocumentMetadata        (Phase 1 metadata_extractor)
  - PreprocessedDocument    (Phase 1 text_preprocessor)
  - EvidenceDensityResult   (Phase 2 evidence_density)
  - HedgingResult           (Phase 2 hedging_detector)
  - CitationReport          (Phase 2 citation_extractor)
  - SentimentResult         (Phase 2 sentiment_analyzer)
  - DiffResult              (Phase 2 diff_engine)    — optional, for versioned docs

The report is written to the analysis/ folder.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _section(title: str, body: str) -> str:
    return f"\n---\n\n# {title}\n\n{body}\n"


def _bullet_list(items) -> str:
    if not items:
        return "_None detected._"
    return "\n".join(f"- {i}" for i in items)


def _score_bar(score: float, width: int = 20) -> str:
    """ASCII progress bar for a 0–1 score."""
    filled = round(score * width)
    return f"[{'█' * filled}{'░' * (width - filled)}] {score:.2%}"


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _build_metadata_section(meta) -> str:
    return (
        f"Title: {meta.title or 'Unknown'}\n"
        f"Author: {meta.author or 'Unknown'}\n"
        f"Year: {meta.year or 'Unknown'}\n"
        f"Source Type: {meta.source_type or 'Unknown'}"
    )


def _build_key_claims_section(preprocessed) -> str:
    claims = [
        preprocessed.sentences[i]
        for i in preprocessed.claim_sentence_indices[:10]
    ]
    if not claims:
        return "_No claim sentences detected._"
    return _bullet_list(s[:200] for s in claims)


def _build_evidence_density_section(density_result) -> str:
    return (
        f"Claims Count: {density_result.claim_count}\n"
        f"Citation Count: {density_result.citation_count}\n\n"
        f"Evidence Density Score: {density_result.density:.4f}  "
        f"({density_result.density_label})\n\n"
        f"{_score_bar(min(density_result.density, 1.0))}\n\n"
        f"Evidence Density = citations / claims"
    )


def _build_hedging_section(hedging_result) -> str:
    top_terms = hedging_result.top_terms(15)
    term_lines = _bullet_list(f"`{t}` × {c}" for t, c in top_terms)

    samples = hedging_result.hedged_sentences[:5]
    sample_lines = "\n\n".join(
        f"> _{s.sentence[:200]}_  \n> **Terms**: {', '.join(s.matched_terms[:5])}"
        for s in samples
    ) or "_No examples._"

    return (
        f"Detected Terms:\n\n{term_lines}\n\n"
        f"Hedging Index: **{hedging_result.hedging_index_pct:.1f}%** "
        f"({hedging_result.hedged_count} of {hedging_result.total_sentences} sentences)\n\n"
        f"{_score_bar(hedging_result.hedging_index)}\n\n"
        f"### Sample Hedged Sentences\n\n{sample_lines}"
    )


def _build_framing_section(diff_result=None) -> str:
    if diff_result is None:
        return "_Framing change detection requires two document versions. Use `nft compare` or `nft timeline`._"

    if not diff_result.modified:
        return "_No significant framing changes detected between the two versions._"

    lines = []
    for change in diff_result.modified[:8]:
        lines.append(
            f"**Original phrase:**\n> {change.text_a[:200]}\n\n"
            f"**Alternative phrasing detected:**\n> {change.text_b[:200]}\n\n"
            f"Similarity score: `{change.similarity:.2f}`\n"
        )
    return "\n---\n".join(lines)


def _build_citation_network_section(citation_report) -> str:
    authors = citation_report.unique_authors[:20]
    author_lines = _bullet_list(authors) if authors else "_No named authors detected._"

    by_type = citation_report.by_type()
    type_lines = "\n".join(f"- {k}: {v}" for k, v in sorted(by_type.items()))

    cluster_detected = "Yes" if len(authors) <= 5 and len(authors) > 0 else "No"

    return (
        f"Total Citations: {citation_report.citation_count}\n\n"
        f"Main cited authors:\n\n{author_lines}\n\n"
        f"Citation types:\n\n{type_lines}\n\n"
        f"Citation cluster detection: {cluster_detected}"
    )


def _build_narrative_entropy_section(entropy_result=None) -> str:
    if entropy_result is None:
        return (
            "Entropy Score: _Not calculated._\n\n"
            "Estimate diversity of explanations in the text.\n\n"
            "Low entropy suggests narrative consolidation."
        )
    clusters = "\n".join(
        f"- Cluster {c.cluster_id + 1} ({c.size} sentences, p={c.probability:.2%}): "
        f"_{c.representative_sentence[:120]}_"
        for c in entropy_result.clusters[:6]
    ) or "_No clusters detected._"

    return (
        f"Entropy Score: **{entropy_result.entropy:.4f}**\n\n"
        f"Normalised: **{entropy_result.normalised_entropy:.4f}** ({entropy_result.label})\n\n"
        f"{_score_bar(entropy_result.normalised_entropy)}\n\n"
        f"Clusters: {entropy_result.n_clusters}  |  "
        f"Sentences analysed: {entropy_result.sentences_used}  |  "
        f"Embedder: {entropy_result.embedder}\n\n"
        f"### Explanation Clusters\n\n{clusters}"
    )


def _build_compression_section(compression_result=None, preprocessed=None) -> str:
    if compression_result is not None:
        actors_block = _bullet_list(compression_result.actor_list[:20]) \
            if compression_result.actor_list else "_None detected._"
        return (
            f"Actors Detected: {compression_result.actors_count}\n\n"
            f"Causal Claims Detected: {compression_result.causal_claims_count}\n\n"
            f"Events Detected: {compression_result.events_count}\n\n"
            f"Compression Ratio: **{compression_result.compression_ratio:.4f}**\n\n"
            f"_{compression_result.compression_label}_\n\n"
            f"Compression Ratio = actors / causal claims\n\n"
            f"### Actors\n\n{actors_block}"
        )
    # Fallback (Phase 1 style)
    if preprocessed is None:
        return "_Not calculated._"
    from text_preprocessor import extract_unique_entities
    actors = extract_unique_entities(preprocessed, labels=["PERSON", "ORG"])
    actor_count = len(actors)
    causal_count = preprocessed.claim_count
    ratio = round(actor_count / causal_count, 4) if causal_count > 0 else 0.0
    return (
        f"Actors Detected: {actor_count}\n\n"
        f"Causal Claims Detected: {causal_count}\n\n"
        f"Compression Ratio: {ratio:.4f}\n\n"
        f"Compression Ratio = actors / causal claims\n\n"
        f"_(Low ratio may indicate narrative oversimplification.)_"
    )


def _build_framing_detail_section(framing_result=None) -> str:
    if framing_result is None or not framing_result.shifts:
        return "_No significant framing shifts detected._"

    lines = []
    for s in framing_result.shifts[:8]:
        entity_tag = f"**Entity: {s.entity}**  \n" if s.entity else ""
        lines.append(
            f"{entity_tag}"
            f"Original phrase:\n> {s.text_a[:200]}\n\n"
            f"Alternative phrasing detected:\n> {s.text_b[:200]}\n\n"
            f"Shift score: `{s.shift_score:.4f}`  |  "
            f"Similarity: `{s.similarity:.4f}`"
        )

    meta = (
        f"Embedder: {framing_result.embedder}  |  "
        f"Shifts detected: {framing_result.shift_count}  |  "
        f"Mean shift score: {framing_result.mean_shift_score:.4f}\n\n"
        f"Intra-document framing variance: {framing_result.intra_variance:.4f}"
    )
    return meta + "\n\n---\n\n" + "\n\n---\n\n".join(lines)


def _build_citation_network_detail_section(network_result=None) -> str:
    if network_result is None:
        return "_Not calculated._"

    hub_lines = "\n".join(
        f"- **{n.name}** — degree: {n.degree}, "
        f"centrality: {n.degree_centrality:.3f}, "
        f"betweenness: {n.betweenness_centrality:.3f}"
        + ("  ⟵ hub" if n.is_hub else "")
        for n in network_result.top_nodes[:10]
    ) or "_No authors detected._"

    return (
        f"Nodes (authors): {network_result.node_count}  |  "
        f"Edges: {network_result.edge_count}\n\n"
        f"Degree Centralization: **{network_result.degree_centralization:.4f}** "
        f"({network_result.centralization_label})\n\n"
        f"Mean Clustering Coefficient: {network_result.mean_clustering:.4f}\n\n"
        f"### Top Cited Authors\n\n{hub_lines}"
    )


def _build_sentiment_section(sentiment_result) -> str:
    if not sentiment_result.entity_profiles:
        return "_No named entities with sufficient mentions for sentiment analysis._"

    profile_lines = []
    for p in sentiment_result.entity_profiles[:15]:
        bar = _score_bar((p.mean_score + 1) / 2, width=10)   # remap [-1,1] → [0,1]
        profile_lines.append(
            f"- **{p.entity_text}** ({p.entity_label}) — "
            f"tone: `{p.tone}`, score: `{p.mean_score:+.3f}`, "
            f"mentions: {p.mention_count}"
        )

    asym_lines = []
    for pair in sentiment_result.asymmetry_pairs[:5]:
        asym_lines.append(
            f"- **{pair['entity_a']}** (score `{pair['score_a']:+.3f}`) vs "
            f"**{pair['entity_b']}** (score `{pair['score_b']:+.3f}`) — "
            f"gap: `{pair['difference']:.3f}`, favoured: **{pair['favoured']}**"
        )

    asym_block = (
        "### Asymmetric Skepticism Signals\n\n" + "\n".join(asym_lines)
        if asym_lines
        else "### Asymmetric Skepticism Signals\n\n_No significant asymmetry detected._"
    )

    scorer_note = f"_(Scorer: {sentiment_result.scorer})_"
    return (
        f"Entity Sentiment Summary {scorer_note}:\n\n"
        + "\n".join(profile_lines)
        + f"\n\n{asym_block}"
    )


def _build_propagation_section(
    phrase_result=None,
    clusters=None,
    mutation_result=None,
    propagation_result=None,
    citation_report=None,
) -> str:
    parts = []

    # Top narrative phrases
    if phrase_result and phrase_result.phrases:
        top = phrase_result.top_phrases(15)
        phrase_lines = _bullet_list(
            f"`{p.text}` — freq: {p.frequency}, TF-IDF: {p.tfidf_score:.4f}"
            for p in top
        )
        parts.append(f"### Key Narrative Phrases\n\n{phrase_lines}")
    elif citation_report and citation_report.citations:
        narrative_refs = [
            c.raw_text[:120]
            for c in citation_report.citations
            if c.citation_type == "narrative"
        ][:10]
        if narrative_refs:
            parts.append(f"### Key Narrative Phrases\n\n{_bullet_list(narrative_refs)}")

    # Phrase clusters
    if clusters and clusters.n_clusters > 0:
        cluster_lines = "\n".join(
            f"- **Cluster {c.cluster_id + 1}** ({c.size} phrases): `{c.centroid_phrase}`"
            for c in clusters.clusters[:8]
        )
        parts.append(f"### Phrase Clusters ({clusters.n_clusters} total)\n\n{cluster_lines}")

    # Mutation chains
    if mutation_result and mutation_result.chains:
        chain_lines = []
        for chain in mutation_result.chains[:5]:
            steps = " → ".join(
                [chain.seed_phrase] + [s.phrase_to for s in chain.steps]
            )
            chain_lines.append(
                f"- `{steps}`  \n"
                f"  Total drift: `{chain.total_drift:.4f}` ({chain.mutation_label})"
            )
        parts.append(
            f"### Mutation Chains  "
            f"({mutation_result.chain_count} detected, "
            f"{mutation_result.high_mutation_count} high-mutation)\n\n"
            + "\n".join(chain_lines)
        )

    # Propagation graph
    if propagation_result and propagation_result.edges:
        parts.append(
            f"### Propagation Graph\n\n"
            f"{propagation_result.summary()}"
        )

    return "\n\n".join(parts) if parts else "_No propagation data available._"


def _build_diff_section(diff_result) -> str:
    if diff_result is None:
        return "_Version comparison not performed._"
    return (
        f"{diff_result.summary()}\n\n"
        f"### Sample Added Sentences\n\n"
        + _bullet_list(s[:160] for s in diff_result.added[:5])
        + "\n\n### Sample Removed Sentences\n\n"
        + _bullet_list(s[:160] for s in diff_result.removed[:5])
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_report(
    meta,
    preprocessed,
    # Phase 2
    density_result=None,
    hedging_result=None,
    citation_report=None,
    sentiment_result=None,
    diff_result=None,
    # Phase 3
    framing_result=None,
    entropy_result=None,
    network_result=None,
    compression_result=None,
    # Phase 4
    phrase_result=None,
    phrase_clusters=None,
    mutation_result=None,
    propagation_result=None,
    notes: str = "",
) -> str:
    """
    Assemble a full Markdown analysis report conforming to DATA_SCHEMA.md.

    All parameters except meta and preprocessed are optional.

    Returns the report as a string.
    """
    parts = ["# Document Metadata\n"]
    parts.append(_build_metadata_section(meta))

    parts.append(_section("Key Claims", _build_key_claims_section(preprocessed)))

    # --- Phase 2 sections ---
    parts.append(_section(
        "Evidence Density",
        _build_evidence_density_section(density_result) if density_result else "_Not calculated._",
    ))
    parts.append(_section(
        "Hedging Language",
        _build_hedging_section(hedging_result) if hedging_result else "_Not calculated._",
    ))

    # --- Phase 3 sections ---
    parts.append(_section(
        "Framing Patterns",
        _build_framing_detail_section(framing_result),
    ))
    parts.append(_section(
        "Citation Network",
        _build_citation_network_detail_section(network_result)
        if network_result else (
            _build_citation_network_section(citation_report)
            if citation_report else "_Not calculated._"
        ),
    ))
    parts.append(_section(
        "Narrative Entropy",
        _build_narrative_entropy_section(entropy_result),
    ))
    parts.append(_section(
        "Narrative Compression",
        _build_compression_section(compression_result, preprocessed),
    ))
    parts.append(_section(
        "Sentiment Asymmetry",
        _build_sentiment_section(sentiment_result) if sentiment_result else "_Not calculated._",
    ))
    parts.append(_section(
        "Narrative Propagation Indicators",
        _build_propagation_section(
            phrase_result=phrase_result,
            clusters=phrase_clusters,
            mutation_result=mutation_result,
            propagation_result=propagation_result,
            citation_report=citation_report,
        ),
    ))

    if diff_result is not None:
        parts.append(_section("Version Diff Summary", _build_diff_section(diff_result)))

    footer = (
        f"Generated by Narrative Forensics Tool  \n"
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )
    if notes:
        footer = f"{notes}\n\n{footer}"
    parts.append(_section("Notes", footer))

    return "\n".join(parts)


def save_report(report_text: str, output_path: str) -> None:
    """Write the report string to a file, creating parent directories as needed."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"[NFT] Report saved → {output_path}")
