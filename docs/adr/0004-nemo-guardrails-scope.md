# ADR-0004: NeMo Guardrails as input/output/retrieval rails only

**Status:** Accepted (2026-11-18)

## Context
NeMo Guardrails prefers to own the whole LLM loop and adds latency (often an extra LLM call per rail).
We own retrieval and generation ourselves (ADR-0002).

## Decision
Use NeMo Guardrails only to **wrap** our pipeline, behind a `Guardrails` Protocol:
- **Input rails:** jailbreak / prompt-injection detection; off-topic check.
- **Retrieval rail:** strip instruction-like content from retrieved chunks (indirect injection —
  documents are untrusted input).
- **Output rails:** groundedness self-check against retrieved chunks; financial-advice policy.

**Advice policy:** answer factual questions about documents; decline personalized investment
advice with a standard message while still offering the relevant facts.

Rails are toggleable per environment (off by default in local dev and unit tests).

## Consequences
Extra latency and cost when enabled; measured in M8 traces.
