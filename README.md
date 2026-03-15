# Narrative Forensics Tool

## Overview

Narrative Forensics Tool (NFT) is a Linux-based command line system designed to analyze texts and detect structural signals of narrative bias, framing shifts, and source evolution.

The tool does **not determine historical truth**. Instead, it identifies patterns that may indicate narrative construction or reframing within texts.

Supported input sources:

- Wikipedia articles
- Books in PDF format
- Academic papers
- Plain text documents
- Collections of documents for comparative analysis

The system analyzes structural features such as:

- narrative shifts across versions
- framing changes in descriptions
- hedging and uncertainty language
- citation disappearance or addition
- evidence density changes
- asymmetric skepticism
- narrative propagation
- narrative mutation
- citation echo chambers
- narrative compression
- narrative entropy

---

## Project Philosophy

The goal of this project is to build a **computational historiography tool** capable of assisting researchers, journalists, and analysts in auditing narratives.

The system identifies **structural signals** rather than making ideological judgments.

---

## Workflow

The project uses a staged pipeline:

PDF / Wikipedia Article  
↓  
Text Extraction  
↓  
Narrative Analysis  
↓  
Structured Markdown Report  
↓  
Dataset Construction  
↓  
Algorithm Development

---

## Repository Structure
Narrative Forensic Tool/
│
├── CLAUDE.md
├── README.md
├── TASKS.md
├── DATA_SCHEMA.md
│
├── prompts/
│ └── analysis_prompt.md
│
├── data/
│ ├── pdf/
│ └── wikipedia/
│
├── analysis/
│
└── src/



---

## Folder Roles

### data/pdf
Contains PDF books and research papers.

### data/wikipedia
Contains exported Wikipedia article text or revision snapshots.

### analysis
Stores generated markdown analysis reports for each document.

### prompts
Contains prompts used by Claude for analysis tasks.

### src
Contains Python modules implementing the system.

---

## Design Principles

1. Algorithms should produce measurable signals.
2. AI should assist interpretation but not replace statistical analysis.
3. The system must run locally on Linux.
4. The dataset of analyzed documents should grow over time.
5. Folder structure and schema must remain stable.

---

## Important Rule

Claude must **not redesign the architecture or folder structure** unless explicitly instructed.

---

## Long-Term Goal

Build a research-grade tool capable of auditing narrative structures in:

- history books
- political analysis
- journalism
- encyclopedia articles
- academic literature
