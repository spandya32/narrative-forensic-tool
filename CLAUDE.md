# CLAUDE.md

## Project: Narrative Forensics Tool (NFT)

### Goal

Create a Linux-based command line tool that analyzes text sources
(Wikipedia articles, books, PDFs, and other documents) to detect
structural signals of narrative manipulation, bias, framing shifts, and
source evolution.

The system should not attempt to determine truth. Instead, it identifies
patterns such as:

-   narrative shifts
-   framing changes
-   hedging language
-   citation disappearance
-   evidence density changes
-   asymmetric skepticism
-   narrative propagation
-   narrative mutation
-   citation echo chambers

Supported inputs:

1.  Wikipedia articles
2.  PDF books
3.  Plain text documents
4.  Multiple documents for comparison

The tool should run locally on Linux.

------------------------------------------------------------------------

# System Architecture

Input Sources │ ▼ Text Extraction Layer (Wikipedia / PDF / Text) │ ▼
Preprocessing Layer (tokenization, NER, citation parsing) │ ▼ Analysis
Modules • Version Diff Analysis • Citation Tracking • Hedging Language
Detection • Framing Change Detection • Evidence Density Scoring •
Sentiment Asymmetry • Narrative Entropy • Citation Network
Centralization • Narrative Compression Detection • Narrative Propagation
Analysis │ ▼ Narrative Timeline Builder │ ▼ Report Generator (CLI +
JSON)

------------------------------------------------------------------------

# Supported Commands

Analyze Wikipedia article

nft analyze-wikipedia URL

Analyze PDF

nft analyze-pdf book.pdf

Compare multiple texts

nft compare file1 file2

Build narrative timeline

nft timeline URL

------------------------------------------------------------------------

# Algorithms

## 1. Version Diff Analysis

Detect textual changes between versions of documents.

Algorithms: - Myers diff algorithm - Longest Common Subsequence (LCS)

Python library: difflib

Outputs: - added sentences - removed sentences - modified sentences

------------------------------------------------------------------------

## 2. Named Entity Recognition

Extract key entities: - historians - organizations - institutions -
authors

Library: spaCy

Entities should be tracked across versions to detect appearance or
disappearance.

------------------------------------------------------------------------

## 3. Citation Extraction

Detect references including: - URLs - books - academic citations -
authors

Approach: - regex extraction - entity recognition - reference parsing

Outputs: - sources added - sources removed - source persistence score

------------------------------------------------------------------------

## 4. Framing Change Detection

Detect semantic shifts in how events are described.

Example: Version A: "temple ruins were discovered"

Version B: "structures interpreted by some as temple remains"

Algorithm: - sentence embeddings - cosine similarity

Model: sentence-transformers

Output: framing shift score

------------------------------------------------------------------------

## 5. Hedging Language Detection

Detect linguistic uncertainty.

Examples: allegedly purportedly interpreted as some scholars claim it is
believed

Approach: dictionary matching + POS tagging

Output: Hedging Index

------------------------------------------------------------------------

## 6. Evidence Density Scoring

Measure evidence support in a text.

Formula: Evidence Density = citations / claim sentences

Process: 1. detect claim sentences 2. detect citation sentences 3.
compute ratio

------------------------------------------------------------------------

## 7. Sentiment Asymmetry

Measure tone toward entities.

Process: 1. detect entity 2. evaluate sentiment around entity 3. compute
entity sentiment distribution

Goal: detect asymmetric skepticism toward specific groups.

------------------------------------------------------------------------

# Advanced Algorithms

## 8. Narrative Entropy

Measure diversity of explanations.

Formula: Entropy = - Σ (p_i log p_i)

Where p_i represents probability of each explanation cluster.

Low entropy indicates narrative consolidation.

------------------------------------------------------------------------

## 9. Citation Network Centralization

Build citation graph.

node = author edge = citation

Library: networkx

Metrics: degree centralization betweenness centrality

High centralization may indicate source echo chambers.

------------------------------------------------------------------------

## 10. Narrative Compression Detection

Detect oversimplification of complex events.

Metrics: actors count causal claims count events count

Compression Ratio = unique actors / causal claims

Low ratio may indicate narrative oversimplification.

------------------------------------------------------------------------

## 11. Narrative Propagation Analysis

Track how narrative phrases spread across texts.

Steps: 1. extract narrative phrases using n-grams 2. filter using TF-IDF
3. cluster phrases using embeddings 4. track phrase frequency over time
5. build propagation graph

Libraries: scikit-learn sentence-transformers pandas networkx

Output: Narrative Propagation Report

------------------------------------------------------------------------

## 12. Narrative Mutation Detection

Detect gradual wording changes.

Example: temple ruins → temple-like remains → structures interpreted as
temple

Algorithm: embedding comparison over time

Metric: cosine distance between phrase embeddings.

------------------------------------------------------------------------

# Narrative Timeline

Example output:

## Year Narrative Change

2015 original claim appears 2018 citation added 2022 source removed 2024
hedging language introduced

------------------------------------------------------------------------

# Example Output Report

Narrative Analysis Report

Framing Shift: 2019 → "temple ruins discovered" 2024 → "structures
interpreted as temple remains"

Evidence Density: Earlier: 0.42 Current: 0.18

Hedging Index: Earlier: 4% Current: 17%

Citation Network Centralization: 0.79 (high)

Narrative Entropy: 0.36 (low diversity of explanations)

Narrative Propagation: Detected across academic papers → news →
Wikipedia.

------------------------------------------------------------------------

# Technology Stack

Language: Python 3.10+

Libraries: spaCy sentence-transformers scikit-learn networkx
pdfminer.six pymupdf requests beautifulsoup4 pandas difflib

Optional AI integration: Claude API OpenAI API

------------------------------------------------------------------------

# AI Integration

AI should be used for: - summarizing narrative shifts - explaining
detected structural patterns - historiographical comparison

Statistical algorithms must produce the base signals.

------------------------------------------------------------------------

# Performance Targets

The tool should: - analyze Wikipedia article in under 30 seconds -
analyze PDF under 50MB in under 60 seconds - run entirely on Linux
locally

------------------------------------------------------------------------

# Development Roadmap

Version 1: text diff citation extraction hedging detection evidence
density basic report generation

Version 2: narrative entropy citation network analysis framing detection

Version 3: narrative propagation mutation detection visualization
dashboards

------------------------------------------------------------------------

# End Goal

Create a Narrative Forensics Tool capable of auditing narrative
structure in: - Wikipedia articles - historical books - research texts -
media narratives

The system should help researchers, historians, journalists, and
analysts detect structural signals of narrative manipulation or bias.

Claude must not modify folder structure, schema files, or repository architecture unless explicitly instructed by the user.