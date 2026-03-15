# Analysis File Schema

Each analyzed document must generate a markdown report following this exact structure.

Consistency is required to allow later dataset analysis.

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
