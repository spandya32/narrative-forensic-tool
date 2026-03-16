"""
LLM interpretation layer for the Narrative Forensics Tool.

Uses the Claude API to provide deep qualitative analysis of passages
already flagged by the statistical modules (bias_detector, salience_analyzer).

The statistical layer always runs first — the LLM interprets only the
top-flagged candidates, keeping API costs low and analysis focused.

Setup:
  Set ANTHROPIC_API_KEY environment variable, or pass api_key directly.

If no API key is set, this module returns empty results gracefully —
the rest of the tool continues to work without it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LLMFinding:
    passage: str
    manipulation_type: str     # "minimization" | "scope_reduction" | "framing_bias" |
                               # "asymmetric_scrutiny" | "attribution_laundering"
    severity: str              # "high" | "medium" | "low"
    explanation: str           # LLM's explanation of the bias mechanism
    source_type: str           # which statistical module flagged it

    def to_dict(self) -> dict:
        return {
            "passage": self.passage[:300],
            "manipulation_type": self.manipulation_type,
            "severity": self.severity,
            "explanation": self.explanation,
            "source_type": self.source_type,
        }


@dataclass
class LLMAnalysisResult:
    source_id: str
    model_used: str = ""
    findings: List[LLMFinding] = field(default_factory=list)
    overall_assessment: str = ""
    error: str = ""

    @property
    def available(self) -> bool:
        return bool(self.model_used) and not self.error

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "model_used": self.model_used,
            "available": self.available,
            "finding_count": len(self.findings),
            "overall_assessment": self.overall_assessment,
            "findings": [f.to_dict() for f in self.findings],
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert in narrative analysis, historiography, and the detection of \
systematic bias in academic and journalistic texts.

You analyze passages of text that have been pre-flagged by statistical algorithms \
as potentially containing narrative manipulation. Your task is to:

1. Identify the specific bias mechanism in each passage (if present)
2. Explain HOW the bias operates structurally (not ideologically)
3. Rate severity: high / medium / low
4. Be precise and avoid overclaiming — some flagged passages may be neutral

You are analysing STRUCTURAL patterns, not making ideological judgements. \
Your goal is to help researchers identify manipulation techniques.

Output JSON only. No prose outside the JSON.
"""

_ANALYSIS_PROMPT = """\
Analyse these passages for structural narrative bias. Each was flagged by a \
statistical algorithm. For each passage return a JSON object.

Passages to analyse:
{passages}

Return a JSON array where each element has:
  "passage_index": integer (0-based)
  "manipulation_type": one of: "minimization" | "scope_reduction" | "framing_bias" | \
"asymmetric_scrutiny" | "attribution_laundering" | "none"
  "severity": "high" | "medium" | "low" | "none"
  "explanation": one sentence explaining the bias mechanism, or "No manipulation detected"
  "key_phrase": the specific phrase that carries the bias (under 10 words)

Also return a final element with index -1:
  "passage_index": -1
  "overall_assessment": 2-3 sentence summary of the document's bias patterns
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_with_llm(
    bias_result,
    salience_result=None,
    source_id: str = "",
    api_key: Optional[str] = None,
    max_passages: int = 8,
) -> LLMAnalysisResult:
    """
    Send top-flagged passages to Claude API for qualitative bias analysis.

    Args:
        bias_result:     BiasDetectionResult from bias_detector.
        salience_result: SalienceResult from salience_analyzer (optional).
        source_id:       Document label.
        api_key:         Anthropic API key (falls back to ANTHROPIC_API_KEY env var).
        max_passages:    Max passages to send to API (keeps cost low).

    Returns:
        LLMAnalysisResult. If no API key, returns empty result gracefully.
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return LLMAnalysisResult(
            source_id=source_id,
            error="ANTHROPIC_API_KEY not set — LLM analysis skipped",
        )

    try:
        import anthropic
    except ImportError:
        return LLMAnalysisResult(
            source_id=source_id,
            error="anthropic package not installed — run: pip install anthropic",
        )

    # Collect top candidate passages from statistical modules
    candidates: List[dict] = []

    # High-severity minimizations first
    for m in bias_result.minimization_instances:
        if m.severity in ("high", "medium"):
            candidates.append({
                "text": m.sentence,
                "source": "minimization",
                "severity": m.severity,
            })

    # Scope minimizer sentences
    for s in bias_result.scope_minimizer_sentences[:3]:
        candidates.append({
            "text": s["sentence"],
            "source": "scope_minimizer",
            "severity": "medium",
        })

    # Buried facts under high contextual pressure
    if salience_result:
        for f in salience_result.buried_facts[:3]:
            if f.contextual_pressure > 0.05 or f.attribution_depth >= 2:
                candidates.append({
                    "text": f.sentence,
                    "source": "burial_high_pressure",
                    "severity": "medium",
                })

    # Deduplicate by text
    seen: set = set()
    unique_candidates = []
    for c in candidates:
        key_text = c["text"][:100]
        if key_text not in seen:
            seen.add(key_text)
            unique_candidates.append(c)

    unique_candidates = unique_candidates[:max_passages]

    if not unique_candidates:
        return LLMAnalysisResult(
            source_id=source_id,
            error="No candidate passages to analyse",
        )

    # Build prompt
    passage_block = "\n\n".join(
        f"[{i}] ({c['source']}, {c['severity']} priority):\n{c['text'][:400]}"
        for i, c in enumerate(unique_candidates)
    )

    prompt = _ANALYSIS_PROMPT.format(passages=passage_block)

    # Call API
    try:
        client = anthropic.Anthropic(api_key=key)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
    except Exception as exc:
        return LLMAnalysisResult(source_id=source_id, error=str(exc))

    # Parse JSON response
    import json
    import re

    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return LLMAnalysisResult(
            source_id=source_id,
            model_used="claude-sonnet-4-6",
            error=f"JSON parse error in LLM response: {raw[:200]}",
        )

    findings: List[LLMFinding] = []
    overall = ""

    for item in data:
        idx = item.get("passage_index", -1)
        if idx == -1:
            overall = item.get("overall_assessment", "")
            continue
        if idx >= len(unique_candidates):
            continue
        if item.get("manipulation_type", "none") == "none":
            continue

        findings.append(LLMFinding(
            passage=unique_candidates[idx]["text"],
            manipulation_type=item.get("manipulation_type", "unknown"),
            severity=item.get("severity", "low"),
            explanation=item.get("explanation", ""),
            source_type=unique_candidates[idx]["source"],
        ))

    # Sort high severity first
    findings.sort(key=lambda f: {"high": 0, "medium": 1, "low": 2}.get(f.severity, 3))

    return LLMAnalysisResult(
        source_id=source_id,
        model_used="claude-sonnet-4-6",
        findings=findings,
        overall_assessment=overall,
    )
