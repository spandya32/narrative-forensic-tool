# Development Tasks

This document defines the development roadmap for the Narrative Forensics Tool.

Claude should complete tasks **phase by phase** and avoid implementing everything at once.

---

# Phase 1: Infrastructure

Goal: build the basic system capable of reading documents.

Tasks:

- Implement PDF text extraction
- Implement Wikipedia article downloader
- Implement CLI interface
- Create basic text preprocessing
- Create document metadata extractor

Output:

Clean structured text for analysis.

---

# Phase 2: Basic Narrative Analysis

Goal: implement foundational analysis modules.

Tasks:

- Text diff engine
- Citation extraction
- Hedging language detection
- Evidence density calculation
- Named entity extraction
- Basic sentiment analysis

Output:

Basic narrative structure report.

---

# Phase 3: Structural Narrative Analysis

Goal: detect deeper narrative patterns.

Tasks:

- Framing change detection
- Narrative entropy calculation
- Citation network analysis
- Narrative compression detection
- Entity sentiment asymmetry

Output:

Expanded narrative report.

---

# Phase 4: Narrative Propagation Analysis

Goal: track how narratives spread across texts.

Tasks:

- Phrase extraction using n-grams
- TF-IDF filtering
- Phrase embedding clustering
- Temporal narrative tracking
- Narrative mutation detection
- Narrative propagation graph

Output:

Narrative propagation report.

---

# Phase 5: Information Hierarchy Analysis

Goal: detect lede burial and architectural narrative manipulation.

Tasks:

- Positional salience scoring (key facts buried late in document)
- Contextual pressure indexing (hedge/attribution density around claims)
- Framing-before-fact sequence detection
- Lede inversion measurement (framing position vs evidence position)
- Attribution laundering depth detection

Output:

Information Hierarchy section in each report.

---

# Phase 6: Dataset Construction

Goal: build a structured dataset of narrative analyses.

Tasks:

- generate markdown reports for each PDF
- store results in analysis folder
- build analysis index (INDEX.md)
- compute metrics across dataset

Output:

Structured dataset for research.

---

# Phase 7: LLM-Assisted Bias Detection

Goal: detect malicious intent and systematic bias using statistical + LLM analysis.

Tasks:

- Concession-reversal minimization detection ("although X, he was generally tolerant")
- Scope minimizer detection ("few temples", "only brahmins were affected")
- Asymmetric scrutiny scoring per entity/group
- LLM interpretation layer via Claude API (optional, requires API key)

Output:

Bias Patterns section in each report. LLM findings when API key is set.

---

# Phase 8: Visualization (Future)

Possible future tasks:

- citation network visualization
- narrative propagation graphs
- entropy dashboards
- comparison dashboards
