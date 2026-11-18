# ADR-0003: DeepEval for generation evals, own code for retrieval metrics

**Status:** Accepted (2026-11-18)

## Context
Every retrieval/generation change must be measured. Candidates: Ragas, DeepEval, custom harness.

## Decision
- **Retrieval metrics** (recall@k, MRR) are computed by our own code against hand-labeled gold
  chunks/pages — simple and exact.
- **Generation metrics** use **DeepEval** (pytest-native): faithfulness, answer relevancy,
  contextual precision/recall, plus a G-Eval metric for citation correctness, and a refusal check
  for unanswerable questions.
- Judge model: **Claude Opus** (stronger than, and different from, the Sonnet generator).
- Golden set: ~50 questions over 5–10 committed 10-K fixtures, in `evals/`.
- Evals run on demand / nightly (`pytest -m eval`), not on every commit.
- Ragas may be used later only for synthetic test-set generation.

## Consequences
Results are stored per run for comparison; M4+ changes must show their delta vs the M3 baseline.
