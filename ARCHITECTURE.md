# Architecture

FinRAG is a retrieval-augmented question-answering platform over financial documents
(primarily SEC filings). Every answer is grounded in retrieved evidence and carries
validated, source-level citations.

## Goals

- Accurate, evidence-based answers with page/section-level citations; refuse when evidence is insufficient.
- Hybrid retrieval (dense + BM25) with reranking and contextual chunking.
- Tenant-isolated, secure document management.
- Measured quality: every retrieval/generation change is evaluated against a golden set.
- Observable: every pipeline stage is traced.
- Extensible toward multi-turn chat, agentic tool use, GraphRAG, a UI and cloud deployment.

## Non-goals (v1)

- Arithmetic on financial figures (numbers are quoted, never computed — see ADR-0006).
- Multi-turn conversation, automatic filter inference, UI, cloud/Kubernetes deployment.
- Personalized investment advice (declined by policy).

## System overview

```
                 ┌──────────────── FastAPI (finrag.api) ────────────────┐
  client ──key──▶│ auth/tenancy ─▶ guardrails(in) ─▶ retrieval ─▶ gen ─▶ guardrails(out) ─▶ SSE/JSON
                 │      │                               │                                      │
                 │      └─ POST /documents ─▶ S3 blob + Postgres row ─▶ enqueue job            │
                 └──────────────────────────────────────────────────────────────────────────────┘
                                                        │
  Procrastinate worker (finrag.jobs):  parse ─▶ chunk ─▶ contextualize ─▶ embed (dense+BM25) ─▶ Qdrant upsert

  Postgres  : source of truth — tenants, users, API keys, collections, documents, versions, chunks, jobs
  Qdrant    : derived search index — dense + sparse(BM25) vectors, payload incl. tenant_id; rebuildable
  S3        : original files (SeaweedFS locally)
  Phoenix   : OpenTelemetry traces (LLM, retrieval, ingestion spans)
```

## Pipelines

### Ingestion
1. `POST /documents` stores the original in S3, creates a `document` row (status `queued`),
   dedupes by SHA-256 per tenant, and enqueues a job.
2. **Parse** — Docling (PDF, DOCX), openpyxl (XLSX), plain loader (TXT). OCR off by default.
3. **Chunk** — structure-aware: split on section headings, ~500–800 tokens with small overlap.
   Tables are standalone Markdown chunks with caption + section path; large tables split by
   row groups with the header repeated. Excel: per sheet, row blocks with header repeated.
4. **Contextualize** — Claude Haiku writes a 1–2 sentence context blurb per chunk from the full
   document (contextual retrieval), using prompt caching. The blurb is prepended for embedding and BM25.
5. **Metadata** — company, ticker/CIK, form type, fiscal year/period, filing date (header parse or
   LLM extraction; user overrides on upload).
6. **Index** — Voyage dense embeddings + FastEmbed `Qdrant/bm25` sparse vectors upserted to Qdrant.
   Status transitions `queued → parsing → chunking → embedding → ready | failed`.

A changed re-upload creates a new version and atomically swaps the document's chunks.
Delete removes blob, rows and vectors.

### Query
1. Authenticate API key → tenant, role, permitted collections.
2. Input rails (jailbreak/injection, off-topic).
3. Hybrid retrieval in Qdrant: top-50 dense + top-50 BM25, mandatory `tenant_id` + collection +
   metadata filters, fused server-side with RRF (k=60).
4. Rerank with a bge cross-encoder → top 8.
5. Retrieval rail strips instruction-like content from chunks (documents are untrusted).
6. Generate with Claude Sonnet; structured output `{answer, citations[]}`. Every factual sentence
   cites `[n]`; each citation resolves to document, page(s), section path and a quoted span.
7. Validate: every `[n]` maps to a retrieved chunk; flag uncited sentences; refuse if evidence is insufficient.
8. Output rails (groundedness self-check, advice policy).

All k-values and thresholds are configuration, tuned via the eval harness.

## Module layout

```
src/finrag/
  api/          FastAPI routes, auth dependencies, SSE
  ingestion/    parsers, chunker, contextualizer, metadata extractor
  indexing/     embedder, Qdrant VectorStore, BM25 sparse encoder
  retrieval/    hybrid search, reranker, filters
  generation/   prompt building, answer + citation validation
  guardrails/   NeMo Guardrails adapter
  storage/      Postgres models (SQLAlchemy + Alembic), S3 blob store
  jobs/         Procrastinate tasks
  core/         config, interfaces (Protocols), tenancy, telemetry
evals/          golden set, DeepEval tests, retrieval metrics CLI
tests/          unit/ + integration/
docs/adr/       architecture decision records
```

Every external provider (LLM, embedder, reranker, vector store, blob store, guardrails) sits
behind a `typing.Protocol` in `finrag.core.interfaces`. Concrete implementations are wired at a
single composition root. Unit tests use in-memory fakes.

## Security model

- Per-tenant API keys (hashed). Roles: `admin` (upload/delete/manage keys), `reader` (query).
- Documents live in collections; readers are granted collections.
- Tenant isolation is enforced in one place — the `VectorStore` adapter refuses any query without
  a tenant filter — and proven by integration tests.
- Retrieved document text is treated as untrusted input (indirect prompt injection).

## Milestones

| | Milestone | Scope |
|---|---|---|
| M0 | Skeleton | uv project, Compose (Postgres, Qdrant, SeaweedFS, Phoenix), config, CI, `/health`, docs |
| M1 | Ingest TXT + PDF | upload, blob store, Docling, structure-aware chunking, embed, Qdrant index, job status |
| M2 | Query | hybrid search, RRF, rerank, cited generation + validation, SSE |
| M3 | Eval baseline | golden set over 10-K fixtures, retrieval metrics + DeepEval |
| M4 | Contextual chunking + metadata | Haiku context, metadata extraction + filters, measured vs M3 |
| M5 | DOCX + XLSX, lifecycle | versioning, dedupe, delete |
| M6 | Auth & tenancy hardening | tenants, collections, roles, keys, isolation tests |
| M7 | Guardrails | NeMo input/output/retrieval rails, advice policy |
| M8 | Observability polish | OTel spans per stage, cost/latency in Phoenix |
| Later | | multi-turn, agentic tools, GraphRAG, UI, cloud deploy |

## Decisions

See [`docs/adr/`](docs/adr/).
