# Analysis File Schema

Each analyzed document generates a Markdown report following this structure.
Consistency allows cross-document dataset analysis.

## Index Reference Guide

This section describes each metric, what it measures, and how to interpret it.

### Evidence Density Score
Formula: `citations / claim sentences`
| Score | Label | Interpretation |
|---|---|---|
| > 0.30 | strong | Well-evidenced — most claims have citation support |
| 0.15–0.30 | moderate | Acceptable evidence support |
| 0.05–0.15 | weak | Many unsupported claims |
| < 0.05 | poor | Very few citations relative to claims made |
Note: footnote-heavy books score low here regardless of actual citation count.

### Hedging Index
Formula: `sentences with hedge words / total sentences × 100`
| Score | Interpretation |
|---|---|
| < 5% | Very assertive — may be overconfident |
| 5–15% | Normal academic range |
| 15–25% | High epistemic caution — contested subject or honest uncertainty |
| > 25% | Possible deliberate vagueness or avoidance of falsifiable claims |
Watch for: **selective hedging** — high hedging around claims the author dislikes,
low hedging around claims they favor.

### Narrative Entropy (Normalised)
Formula: Shannon entropy over explanation clusters, normalised to [0,1]
| Score | Label | Interpretation |
|---|---|---|
| > 0.85 | high diversity | Many different explanatory frameworks — no single agenda |
| 0.60–0.85 | moderate diversity | Some consolidation around preferred explanations |
| 0.40–0.60 | low diversity | Narrative gravitating toward single framework |
| < 0.40 | consolidated narrative | Single dominant explanation repeatedly reinforced |

### Citation Network Centralization (Freeman Degree)
Formula: sum of degree deviations from max / theoretical maximum
| Score | Label | Interpretation |
|---|---|---|
| < 0.30 | low | Diverse sources — many authors cited roughly equally |
| 0.30–0.60 | moderate | Some hub sources dominate |
| > 0.60 | high | Echo chamber signal — 1–2 sources dominate all citations |

### Compression Ratio
Formula: `unique actors / causal claims`
| Score | Interpretation |
|---|---|
| > 1.5 | Low compression — many actors cause events (realistic) |
| 0.5–1.5 | Moderate compression |
| < 0.5 | High compression — few actors cause everything (oversimplification signal) |
| 0.0 | Indeterminate (NER unavailable or no causal claims) |

### Lede Inversion Score
Formula: `mean position of framing sentences − mean position of evidentiary sentences`
| Score | Interpretation |
|---|---|
| > +0.10 | Normal — evidence appears before interpretation |
| −0.05 to +0.10 | Neutral / mixed |
| −0.05 to −0.15 | Mild inversion — framing slightly precedes evidence |
| < −0.15 | Strong inversion — interpretive frames precede key facts (manipulation signal) |

### Minimization Count
Counts "concession-reversal" patterns: "although X committed atrocity, he was generally tolerant"
| Count | Interpretation |
|---|---|
| 0 | No detected minimization |
| 1–5 | Low — possibly stylistic |
| 6–15 | Moderate — pattern worth examining |
| > 15 | High — systematic minimization signal |
High-severity instances have both harm words AND reversal language in the same sentence.

### Scrutiny Variance
Measures how unevenly analytical attention is distributed across named entities.
Higher variance = some groups receive far more critical scrutiny than others.
Compare scrutiny scores per entity in the Entity Scrutiny Profiles table.

---

# Document Metadata

Title:
Author:
Year:
Source Type: (Book / Article / Wikipedia)

---

# Key Claims

List the main claims made in the text.

Example format:

- Claim 1
- Claim 2
- Claim 3

---

# Evidence Density

Claims Count:
Citation Count:

Evidence Density Score:

Evidence Density = citations / claims

---

# Hedging Language

Detected Terms:

List hedging words detected in the text.

Examples:

- allegedly
- purportedly
- interpreted as
- some scholars claim
- it is believed

Hedging Index:

Percentage of sentences containing hedging language.

---

# Framing Patterns

Provide examples of framing language.

Example:

Original phrase:

Alternative phrasing detected:

Interpretation:

---

# Citation Network

Main cited authors:

List major authors referenced in the document.

Citation cluster detection:

Yes / No

---

# Narrative Entropy

Entropy Score:

Estimate diversity of explanations in the text.

Low entropy suggests narrative consolidation.

---

# Narrative Compression

Actors Detected:

Causal Claims Detected:

Compression Ratio:

Compression Ratio = actors / causal claims

---

# Sentiment Asymmetry

Entity Sentiment Summary:

List tone toward major actors.

---

# Narrative Propagation Indicators

Key narrative phrases detected:

Example:

- phrase 1
- phrase 2
- phrase 3

---

# Bias Patterns

Detects three structural bias mechanisms:

**1. Concession-Reversal Minimization**
Pattern: "although X committed atrocity, he was generally tolerant"
Acknowledges a harm then immediately negates its significance.

**2. Scope Minimizers**
Pattern: "desecrated few temples", "only brahmins were affected"
Quantifiers that shrink the scale or scope of harmful events.

**3. Asymmetric Scrutiny**
Measures analytical attention per entity/group.
High variance = some groups receive far more critical examination than others.

LLM Interpretation: (requires ANTHROPIC_API_KEY)
Provides qualitative analysis of top-flagged passages.

---

# Information Hierarchy

Detects "lede burial" — key facts that are technically present and cited,
but architecturally hidden by surrounding context.

Lede Inversion Score:
  Positive = framing appears after evidence (neutral/normal)
  Negative = framing appears before evidence (manipulation signal)

Mean Evidentiary Position:
  Where cited claim sentences appear on average (0% = top, 100% = bottom)

Mean Framing Position:
  Where attribution/framing sentences appear on average

Buried key facts detected:
  Claim sentences that are late in the document AND under high contextual pressure

High-pressure claim sentences:
  Claims surrounded by dense hedging and attribution language

Max attribution laundering depth:
  Deepest nesting of "X said that Y found that Z claimed that..."

Framing-before-fact sequences:
  Count of cases where interpretive framing directly precedes a factual claim

## Algorithms Applied

1. Positional Salience Scoring  — key cited facts appearing late in document
2. Contextual Pressure Index    — hedge/attribution density around each claim
3. Framing-Before-Fact          — attribution frames directly preceding claims
4. Lede Inversion               — evidentiary position vs framing position
5. Attribution Laundering Depth — nested attribution chains on single facts

---

# Notes

Optional observations.
