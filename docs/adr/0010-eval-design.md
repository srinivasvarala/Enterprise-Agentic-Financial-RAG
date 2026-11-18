# ADR-0010: M3 eval design: pinned real filings, span-based gold, trimmed metrics

**Status:** Accepted (2026-11-18). Amends ADR-0003 where noted; the rest of ADR-0003 stands.

## Context
ADR-0003 chose DeepEval for generation and our own retrieval metrics, over about 50 questions on
5-10 committed 10-K fixtures. Working through M3 showed that committing real filings is heavy, that
chunk-id labels break whenever chunking changes (M4 changes it by design), and that the parser
differs between macOS and the Linux image CI and production use.

## Decision
- **Fixtures.** Five real SEC 10-Ks (Apple, Microsoft, Alphabet, Walmart, Nike; most recent
  complete fiscal year at fetch time), fetched by script into a gitignored cache. A committed
  manifest pins each by accession number and sha256. The fetcher identifies itself with a generic
  client name, never a personal contact. This replaces "committed fixtures".
- **Gold labels** are Gold Spans: document, page range and a verbatim span for text answers, or a
  row label plus figure for table answers. A Hit is a retrieved chunk containing the Gold Span.
  A script checks every Gold Span appears on its stated page of the parsed document.
- **Golden Set**: about 50 questions in `evals/golden/*.jsonl`: 20 text, 12 table, 6 multi-span
  (including calculation questions, answered per ADR-0009), 8 unanswerable, 4 company-ambiguity.
  Investment-advice questions wait for the guardrails milestone. A seeded sample of 12 plus all 8
  unanswerable questions is reviewed by a human; at least 11 of 12 and 8 of 8 must be right.
- **Metrics.** Retrieval: recall@8, recall@100 (before the rerank) and MRR. Pipeline, no LLM:
  false-refusal rate on answerable questions, correct-refusal rate on unanswerable ones,
  `unverifiable` rate, citation hit rate. DeepEval with a Claude Opus judge: faithfulness, answer
  relevancy and answer correctness (G-Eval against the reference answer). Dropped from ADR-0003:
  contextual precision/recall (our retrieval metrics cover it) and the G-Eval citation judge (the
  validator proves quotes are verbatim; faithfulness covers whether they support the sentence).
- **DeepEval** lives in its own `evals` dependency group, installed only for evals. Its telemetry
  is opted out in code, with a test that fails if that opt-out is missing.
- **Where it runs.** Inside the Linux image under a compose `eval` profile, with its own tenant,
  collection and Qdrant collection, so results reflect the parser CI and production use and never
  mix with dev documents. Each run records the image digest and model ids.
- **Index lifecycle.** `make eval-index` builds the index through the real ingestion pipeline,
  idempotently, and records a fingerprint (manifest hash, chunking settings, embedder and sparse
  model ids, parser version) and the expected point count in `evals/results/index.json`.
  `make eval` and `make eval-retrieval` check both and refuse when stale; they never re-ingest.
  The profile runs with 6 GB and `make eval-index` refuses to start below 8 GB for the Docker VM.
- **Results and gating.** Run files go to `evals/results/` (gitignored). `evals/baseline.json` is
  committed; `make eval-compare` prints the delta table for PR descriptions. M3 is report-only.
  Regression tolerance is set later from the run-to-run noise of three baseline runs.
- **Cost.** Nothing paid runs until a 5-question dry run has reported its measured cost and a
  per-run cap (proposed $20) is approved.

## Considered options
- Fictional generated filings: deterministic and trivially labelled, but too small and too regular
  to show what M4's contextual chunking is meant to fix.
- Chunk-id gold labels: simpler, but invalid after any chunking change.
- A CI gate in M3: would flap or be too loose before the noise is known.

## Consequences
A one-time, slow and memory-hungry index build; an explicit rebuild whenever chunking changes.
Gold labels are authored by the agent and reviewed by a human on a sample.
