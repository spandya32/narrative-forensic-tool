#!/usr/bin/env python3
"""
Narrative Forensics Tool (NFT) — Command Line Interface

Commands:
  analyze-wikipedia URL        Download and analyze a Wikipedia article
  analyze-pdf FILE             Extract and analyze a PDF document
  compare FILE1 FILE2          Compare two documents
  timeline URL                 Fetch Wikipedia revision history
  build-index [DIR]            Build cross-document index from analysis reports

Usage:
  python src/cli.py analyze-wikipedia https://en.wikipedia.org/wiki/Ayodhya
  python src/cli.py analyze-pdf "data/pdfs/History_of_India_Vol_I.pdf" --save analysis/history.md
  python src/cli.py compare "data/pdfs/book_a.pdf" "data/pdfs/book_b.pdf"
  python src/cli.py timeline https://en.wikipedia.org/wiki/Ayodhya
  python src/cli.py build-index analysis --output analysis/INDEX.md
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_section(title: str, content: str, width: int = 80) -> None:
    bar = "─" * width
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)
    print(content)


def _truncate(text: str, max_chars: int = 800) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n… [{len(text) - max_chars} chars omitted]"


def _detect_source_type(path: str) -> str:
    return "pdf" if os.path.splitext(path)[1].lower() == ".pdf" else "text"


def _load_text(path: str) -> str:
    if _detect_source_type(path) == "pdf":
        from pdf_extractor import extract_pdf
        doc = extract_pdf(path)
        if doc.error:
            print(f"[ERROR] {doc.error}", file=sys.stderr)
            sys.exit(1)
        return doc.full_text
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)


def _run_pipeline(preprocessed, source_id: str, sections=None) -> dict:
    """Run Phase 2 + Phase 3 analysis modules."""
    from citation_extractor import extract_citations
    from hedging_detector import detect_hedging
    from evidence_density import calculate_density, calculate_section_densities
    from sentiment_analyzer import analyze_sentiment
    from framing_detector import detect_intra_framing
    from narrative_entropy import calculate_entropy
    from citation_network import build_citation_network
    from compression_detector import detect_compression
    from phrase_extractor import extract_phrases
    from phrase_clusterer import cluster_phrases
    from salience_analyzer import analyze_salience
    from bias_detector import detect_bias
    from llm_analyzer import analyze_with_llm

    print("[NFT] Running analysis …")
    citations   = extract_citations(preprocessed.sentences, source_id=source_id)
    hedging     = detect_hedging(preprocessed.sentences, source_id=source_id)
    density     = calculate_density(preprocessed, source_id=source_id)
    if sections:
        density.section_densities = calculate_section_densities(sections, source_id=source_id)
    sentiment   = analyze_sentiment(preprocessed)
    framing     = detect_intra_framing(preprocessed, source_id=source_id)
    entropy     = calculate_entropy(preprocessed, source_id=source_id)
    network     = build_citation_network(citations, source_id=source_id)
    compression = detect_compression(preprocessed, source_id=source_id)
    phrases     = extract_phrases(preprocessed.sentences, source_id=source_id)
    clusters    = cluster_phrases(phrases)
    salience    = analyze_salience(preprocessed, source_id=source_id)
    bias        = detect_bias(preprocessed, source_id=source_id)
    llm         = analyze_with_llm(bias, salience_result=salience, source_id=source_id)

    return {
        "citations": citations, "hedging": hedging,
        "density": density,     "sentiment": sentiment,
        "framing": framing,     "entropy": entropy,
        "network": network,     "compression": compression,
        "phrases": phrases,     "clusters": clusters,
        "salience": salience,   "bias": bias,
        "llm": llm,
    }


def _print_results(p: dict) -> None:
    d, h, c, s = p["density"], p["hedging"], p["citations"], p["sentiment"]
    fr, en, net, comp = p["framing"], p["entropy"], p["network"], p["compression"]
    phrases, clusters, sal = p["phrases"], p["clusters"], p["salience"]
    bias, llm = p["bias"], p["llm"]

    _print_section(
        "EVIDENCE DENSITY",
        f"Claims    : {d.claim_count}\n"
        f"Citations : {d.citation_count}\n"
        f"Score     : {d.density:.4f}  ({d.density_label})",
    )
    _print_section(
        "HEDGING LANGUAGE",
        f"Index     : {h.hedging_index_pct:.1f}%  "
        f"({h.hedged_count} of {h.total_sentences} sentences)\n"
        f"Top terms : {', '.join(t for t, _ in h.top_terms(8))}",
    )
    _print_section(
        "CITATIONS",
        f"Total     : {c.citation_count}\n"
        f"By type   : {c.by_type()}\n"
        f"Authors   : {', '.join(c.unique_authors[:10]) or 'none detected'}",
    )
    _print_section(
        "NARRATIVE ENTROPY",
        f"Score     : {en.entropy:.4f}  (normalised: {en.normalised_entropy:.4f})\n"
        f"Label     : {en.label}\n"
        f"Clusters  : {en.n_clusters}",
    )
    _print_section(
        "NARRATIVE COMPRESSION",
        f"Actors    : {comp.actors_count}\n"
        f"Causal    : {comp.causal_claims_count}\n"
        f"Ratio     : {comp.compression_ratio:.4f}  ({comp.compression_label})",
    )
    if net.node_count > 0:
        _print_section(
            "CITATION NETWORK",
            f"Authors   : {net.node_count}\n"
            f"Edges     : {net.edge_count}\n"
            f"Centralization : {net.degree_centralization:.4f}  ({net.centralization_label})",
        )
    if fr.shifts:
        top = fr.shifts[0]
        _print_section(
            f"TOP FRAMING SHIFT  (score={top.shift_score:.4f})",
            f"  A: {top.text_a[:120]}\n"
            f"  B: {top.text_b[:120]}",
        )
    if phrases.phrases:
        top = phrases.top_phrases(8)
        _print_section(
            "NARRATIVE PHRASES",
            f"Extracted : {phrases.phrase_count}  |  Clusters: {clusters.n_clusters}\n"
            + "\n".join(f"  `{p.text}`  (freq={p.frequency})" for p in top),
        )
    _print_section(
        "BIAS PATTERNS",
        f"Minimization instances  : {bias.minimization_count}  "
        f"(high severity: {bias.high_severity_count})\n"
        f"Scope minimizer hits    : {bias.scope_minimizer_count}\n"
        f"Scrutiny variance       : {bias.scrutiny_variance:.2f}\n"
        f"Asymmetric pairs        : {len(bias.asymmetry_pairs)}"
        + (
            "\n\nTop asymmetric pair:\n  "
            + bias.asymmetry_pairs[0].signal
            if bias.asymmetry_pairs else ""
        )
        + (
            "\n\nTop minimization:\n  " + bias.minimization_instances[0].sentence[:200]
            if bias.minimization_instances else ""
        ),
    )
    if llm.available:
        _print_section(
            f"LLM BIAS ANALYSIS  (model: {llm.model_used})",
            (f"Overall: {llm.overall_assessment}\n\n" if llm.overall_assessment else "")
            + "\n\n".join(
                f"[{f.severity.upper()}] {f.manipulation_type}: {f.explanation}\n"
                f"  > {f.passage[:150]}"
                for f in llm.findings[:4]
            ) or "_No manipulation detected in flagged passages._",
        )
    elif llm.error and "not set" not in llm.error:
        print(f"[WARN] LLM analysis: {llm.error}", file=sys.stderr)
    _print_section(
        "INFORMATION HIERARCHY",
        f"Lede inversion      : {sal.lede_inversion_score:+.4f}  ({sal.lede_inversion_label})\n"
        f"Evidence position   : {sal.mean_evidentiary_position:.1%} through document\n"
        f"Framing position    : {sal.mean_framing_position:.1%} through document\n"
        f"Buried key facts    : {len(sal.buried_facts)}\n"
        f"High-pressure claims: {sal.high_pressure_claim_count}\n"
        f"Max laundering depth: {sal.max_laundering_depth}\n"
        f"Framing-before-fact : {sal.framing_before_fact_count} sequences",
    )
    if s.entity_profiles:
        lines = [
            f"  {p.entity_text:30s}  [{p.entity_label}]  "
            f"tone={p.tone:8s}  score={p.mean_score:+.3f}  mentions={p.mention_count}"
            for p in s.entity_profiles[:12]
        ]
        _print_section(f"ENTITY SENTIMENT  (scorer: {s.scorer})", "\n".join(lines))
        if s.asymmetry_pairs:
            a = s.asymmetry_pairs[0]
            _print_section(
                "TOP ASYMMETRY SIGNAL",
                f"  {a['entity_a']}  ({a['score_a']:+.3f})  vs  "
                f"{a['entity_b']}  ({a['score_b']:+.3f})\n"
                f"  Gap: {a['difference']:.3f}  —  Favoured: {a['favoured']}",
            )


def _save_report(output_path: str, meta, preprocessed, p: dict | None) -> None:
    from report_generator import generate_report, save_report
    report = generate_report(
        meta=meta,
        preprocessed=preprocessed,
        density_result=p["density"]         if p else None,
        hedging_result=p["hedging"]         if p else None,
        citation_report=p["citations"]      if p else None,
        sentiment_result=p["sentiment"]     if p else None,
        framing_result=p["framing"]         if p else None,
        entropy_result=p["entropy"]         if p else None,
        network_result=p["network"]         if p else None,
        compression_result=p["compression"] if p else None,
        phrase_result=p["phrases"]          if p else None,
        phrase_clusters=p["clusters"]       if p else None,
        salience_result=p["salience"]       if p else None,
        bias_result=p["bias"]               if p else None,
        llm_result=p["llm"]                 if p else None,
    )
    save_report(report, output_path)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_analyze_wikipedia(args: argparse.Namespace) -> int:
    from wikipedia_fetcher import fetch_article
    from text_preprocessor import preprocess, extract_unique_entities
    from metadata_extractor import extract_wikipedia_metadata

    print(f"[NFT] Fetching: {args.url}")
    article = fetch_article(args.url)
    if article.error:
        print(f"[ERROR] {article.error}", file=sys.stderr)
        return 1

    print("[NFT] Preprocessing …")
    preprocessed = preprocess(article.current_text, source_id=args.url)
    meta = extract_wikipedia_metadata(article)
    p2 = _run_pipeline(preprocessed, args.url, sections=article.sections)

    _print_section("METADATA", meta.to_markdown())
    _print_section(
        "DOCUMENT STATISTICS",
        f"Sentences  : {preprocessed.sentence_count}\n"
        f"Sections   : {len(article.sections)}\n"
        f"References : {len(article.references)}\n"
        f"NER        : {'enabled' if preprocessed.ner_available else 'disabled (install spaCy)'}",
    )

    if preprocessed.entities:
        top = extract_unique_entities(preprocessed, labels=["PERSON", "ORG", "GPE"])[:15]
        _print_section(
            "TOP ENTITIES",
            "\n".join(f"  {e['text']!r:40s}  [{e['label']}]  ×{e['frequency']}" for e in top),
        )

    _print_results(p2)
    _print_section("TEXT PREVIEW", _truncate(article.current_text))

    if args.save:
        _save_report(args.save, meta, preprocessed, p2)
    return 0


def cmd_analyze_pdf(args: argparse.Namespace) -> int:
    from pdf_extractor import extract_pdf
    from text_preprocessor import preprocess, extract_unique_entities
    from metadata_extractor import extract_pdf_metadata

    if not os.path.isfile(args.file):
        print(f"[ERROR] File not found: {args.file}", file=sys.stderr)
        return 1

    print(f"[NFT] Extracting: {args.file}")
    extracted = extract_pdf(args.file)
    if extracted.error:
        print(f"[ERROR] {extracted.error}", file=sys.stderr)
        return 1

    print(f"[NFT] Preprocessing {extracted.total_pages} pages …")
    preprocessed = preprocess(extracted.full_text, source_id=args.file)
    first_page = extracted.pages[0].text if extracted.pages else ""
    meta = extract_pdf_metadata(args.file, first_page_text=first_page)
    p2 = _run_pipeline(preprocessed, args.file)

    _print_section("METADATA", meta.to_markdown())
    _print_section(
        "DOCUMENT STATISTICS",
        f"Pages      : {extracted.total_pages}\n"
        f"Characters : {len(extracted.full_text):,}\n"
        f"Sentences  : {preprocessed.sentence_count}\n"
        f"NER        : {'enabled' if preprocessed.ner_available else 'disabled (install spaCy)'}",
    )

    if preprocessed.entities:
        top = extract_unique_entities(preprocessed, labels=["PERSON", "ORG", "GPE"])[:15]
        _print_section(
            "TOP ENTITIES",
            "\n".join(f"  {e['text']!r:40s}  [{e['label']}]  ×{e['frequency']}" for e in top),
        )

    _print_results(p2)
    _print_section("TEXT PREVIEW", _truncate(extracted.full_text))

    if args.save:
        _save_report(args.save, meta, preprocessed, p2)
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    from text_preprocessor import preprocess, extract_unique_entities
    from diff_engine import diff_texts
    from citation_extractor import extract_citations, compare_citations
    from hedging_detector import detect_hedging, compare_hedging
    from evidence_density import calculate_density, compare_density
    from framing_detector import detect_inter_framing
    from compression_detector import detect_compression, compare_compression

    print(f"[NFT] Loading '{args.file1}' …")
    text_a = _load_text(args.file1)
    print(f"[NFT] Loading '{args.file2}' …")
    text_b = _load_text(args.file2)

    print("[NFT] Preprocessing …")
    doc_a = preprocess(text_a, source_id=args.file1)
    doc_b = preprocess(text_b, source_id=args.file2)

    print("[NFT] Comparing …")
    diff    = diff_texts(text_a, text_b,
                         source_a=os.path.basename(args.file1),
                         source_b=os.path.basename(args.file2),
                         sentences_a=doc_a.sentences,
                         sentences_b=doc_b.sentences)
    cit_a   = extract_citations(doc_a.sentences, source_id=args.file1)
    cit_b   = extract_citations(doc_b.sentences, source_id=args.file2)
    compare_citations(cit_a, cit_b)
    hed_d   = compare_hedging(
                  detect_hedging(doc_a.sentences),
                  detect_hedging(doc_b.sentences))
    den_d   = compare_density(
                  calculate_density(doc_a),
                  calculate_density(doc_b))
    mod_pairs = [(c.text_a, c.text_b) for c in diff.modified[:50]]
    framing = detect_inter_framing(
                  doc_a.sentences, doc_b.sentences,
                  source_a=os.path.basename(args.file1),
                  source_b=os.path.basename(args.file2),
                  modified_pairs=mod_pairs or None)
    comp_d  = compare_compression(
                  detect_compression(doc_a),
                  detect_compression(doc_b))

    ents_a = {e["text"].lower() for e in extract_unique_entities(doc_a)}
    ents_b = {e["text"].lower() for e in extract_unique_entities(doc_b)}

    name_a, name_b = os.path.basename(args.file1), os.path.basename(args.file2)

    _print_section(
        "STRUCTURAL COMPARISON",
        f"{'':38s}  {name_a:>14s}  {name_b:>14s}\n"
        f"{'Sentences':38s}  {doc_a.sentence_count:>14d}  {doc_b.sentence_count:>14d}\n"
        f"{'Claim sentences':38s}  {doc_a.claim_count:>14d}  {doc_b.claim_count:>14d}\n"
        f"{'Citation sentences':38s}  {doc_a.citation_count:>14d}  {doc_b.citation_count:>14d}\n"
        f"{'Named entities':38s}  {len(ents_a):>14d}  {len(ents_b):>14d}",
    )
    _print_section("DIFF SUMMARY", diff.summary())

    if diff.modified:
        lines = []
        for c in diff.modified[:3]:
            lines.append(f"  A: {c.text_a[:100]}\n  B: {c.text_b[:100]}\n  similarity: {c.similarity:.2f}\n")
        _print_section("SAMPLE MODIFIED SENTENCES", "\n".join(lines))

    _print_section(
        "NARRATIVE SIGNAL CHANGES",
        f"Hedging index      : {hed_d['hedging_index_a']:.1f}% → {hed_d['hedging_index_b']:.1f}%  "
        f"({hed_d['direction']})\n"
        f"Evidence density   : {den_d['density_a']:.4f} → {den_d['density_b']:.4f}  "
        f"({den_d['trend']})\n"
        f"Citations added    : {len(cit_b.added)}\n"
        f"Citations removed  : {len(cit_a.removed)}\n"
        f"Persistence        : {cit_a.persistence_score:.2%}\n"
        f"Compression ratio  : {comp_d['ratio_a']:.4f} → {comp_d['ratio_b']:.4f}  "
        f"({comp_d['label_b']})\n"
        f"Framing shifts     : {framing.shift_count}  "
        f"(mean score: {framing.mean_shift_score:.4f})",
    )

    if framing.shifts:
        top = framing.shifts[0]
        _print_section(
            f"TOP FRAMING SHIFT  (score={top.shift_score:.4f})",
            f"  A: {top.text_a[:120]}\n  B: {top.text_b[:120]}",
        )

    only_a = sorted(ents_a - ents_b)[:25]
    only_b = sorted(ents_b - ents_a)[:25]
    if only_a:
        _print_section(f"ENTITIES ONLY IN A  ({name_a})",
                       "\n".join(f"  - {e}" for e in only_a))
    if only_b:
        _print_section(f"ENTITIES ONLY IN B  ({name_b})",
                       "\n".join(f"  - {e}" for e in only_b))
    return 0


def cmd_build_index(args: argparse.Namespace) -> int:
    from dataset_builder import build_index, save_index

    analysis_dir = args.dir
    output = args.output

    print(f"[NFT] Scanning: {analysis_dir}")
    idx = build_index(analysis_dir)

    if idx.count == 0:
        print("[WARN] No analysis reports found.", file=sys.stderr)
        return 1

    print(f"[NFT] Parsed {idx.count} documents")

    for r in sorted(idx.records, key=lambda x: x.title):
        print(
            f"  {(r.title or r.filename)[:45]:45s}  "
            f"density={r.evidence_density:.3f}  "
            f"hedging={r.hedging_index_pct:.1f}%  "
            f"entropy={r.entropy_normalised:.3f}  "
            f"actors={r.actors_count}"
        )

    save_index(idx, output)
    print(f"[NFT] Index saved → {output}")
    return 0


def cmd_timeline(args: argparse.Namespace) -> int:
    from wikipedia_fetcher import fetch_article

    print(f"[NFT] Fetching revision history: {args.url}")
    article = fetch_article(args.url, fetch_revisions=args.limit)
    if article.error:
        print(f"[ERROR] {article.error}", file=sys.stderr)
        return 1
    if not article.revisions:
        print("[WARN] No revision history available.")
        return 0

    _print_section(
        "TIMELINE",
        f"Article  : {article.title}\n"
        f"URL      : {article.url}\n"
        f"Revisions: {len(article.revisions)}",
    )
    print(f"\n  {'Timestamp':25s}  {'RevID':10s}  {'User':20s}  Comment")
    print(f"  {'─'*25}  {'─'*10}  {'─'*20}  {'─'*30}")
    for rev in article.revisions:
        comment = (rev.comment or "").replace("\n", " ")
        comment = comment[:60] + "…" if len(comment) > 60 else comment
        print(f"  {rev.timestamp:25s}  {rev.revid:<10d}  {(rev.user or '')[:20]:20s}  {comment}")
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nft",
        description="Narrative Forensics Tool — detect structural signals of narrative manipulation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              nft analyze-wikipedia https://en.wikipedia.org/wiki/Ayodhya
              nft analyze-pdf "data/pdfs/History_of_India_Vol_I.pdf" --save analysis/history.md
              nft compare "data/pdfs/book_a.pdf" "data/pdfs/book_b.pdf"
              nft timeline https://en.wikipedia.org/wiki/Ayodhya
              nft build-index analysis --output analysis/INDEX.md
        """),
    )

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    p_wiki = sub.add_parser("analyze-wikipedia", help="Download and analyze a Wikipedia article")
    p_wiki.add_argument("url", help="Wikipedia article URL")
    p_wiki.add_argument("--save", metavar="FILE", help="Save report to FILE")
    p_wiki.set_defaults(func=cmd_analyze_wikipedia)

    p_pdf = sub.add_parser("analyze-pdf", help="Extract and analyze a PDF document")
    p_pdf.add_argument("file", help="Path to the PDF file")
    p_pdf.add_argument("--save", metavar="FILE", help="Save report to FILE")
    p_pdf.set_defaults(func=cmd_analyze_pdf)

    p_cmp = sub.add_parser("compare", help="Compare two documents")
    p_cmp.add_argument("file1", help="First document")
    p_cmp.add_argument("file2", help="Second document")
    p_cmp.set_defaults(func=cmd_compare)

    p_tl = sub.add_parser("timeline", help="Fetch Wikipedia revision history")
    p_tl.add_argument("url", help="Wikipedia article URL")
    p_tl.add_argument("--limit", type=int, default=20, metavar="N",
                      help="Number of revisions (default: 20)")
    p_tl.set_defaults(func=cmd_timeline)

    p_idx = sub.add_parser("build-index", help="Build cross-document dataset index from analysis reports")
    p_idx.add_argument("dir", nargs="?", default="analysis",
                       help="Directory containing .md analysis reports (default: analysis)")
    p_idx.add_argument("--output", metavar="FILE", default="analysis/INDEX.md",
                       help="Output path for INDEX.md (default: analysis/INDEX.md)")
    p_idx.set_defaults(func=cmd_build_index)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    src_dir = os.path.dirname(os.path.abspath(__file__))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
