# ADR-0001: Qdrant for dense + BM25 hybrid search

**Status:** Accepted (2026-11-18)

## Context
We need dense vector search plus true BM25 lexical search, fused and filtered by tenant.
Options considered: Postgres `tsvector` (not real BM25), ParadeDB `pg_search` + pgvector,
OpenSearch/Elasticsearch, Pinecone, Qdrant.

## Decision
Use **Qdrant**, one collection per environment, with a named dense vector (Voyage) and a named
sparse vector (FastEmbed `Qdrant/bm25`, collection configured with `modifier=idf`). Hybrid queries
use Qdrant's Query API with prefetch (dense top-50, sparse top-50) and server-side RRF fusion.
`tenant_id` is a tenant-indexed payload field and is mandatory on every query.

Postgres remains the **source of truth**; Qdrant is a derived index that a `reindex` command can
rebuild from Postgres.

## Why not Pinecone
Hosted-only: local development, CI and committed fixtures would depend on a paid cloud service.
It can be added later as another `VectorStore` adapter.

## Consequences
- Two stores must be kept consistent; ingestion writes Postgres first, then Qdrant, and document
  status only becomes `ready` after the Qdrant upsert succeeds.
- BM25 term weighting is computed client-side (FastEmbed) with IDF server-side — close to, not
  byte-identical with, classic BM25. Acceptable; validated by evals.
