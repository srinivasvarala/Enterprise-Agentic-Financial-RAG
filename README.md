# Enterprise-Agentic-Financial-RAG
An Enterprise-Grade Financial Document Intelligence Platform Powered by RAG, GraphRAG, and Agentic AI

Evidence-based question answering over financial documents (SEC filings first) using hybrid search
(dense + BM25), reranking, contextual chunking and validated source-level citations.

- Design: [`ARCHITECTURE.md`](ARCHITECTURE.md)
- Decisions: [`docs/adr/`](docs/adr/)
- Status: **M0 — skeleton**

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Docker.

```bash
make install   # deps + pre-commit hooks
make up        # postgres, qdrant, seaweedfs (S3), phoenix
make dev       # API on http://localhost:8000  →  GET /health
make lint test
```

| Service | URL |
|---|---|
| API | http://localhost:8000/docs |
| Qdrant dashboard | http://localhost:6333/dashboard |
| Phoenix traces | http://localhost:6006 |
| S3 (SeaweedFS) | http://localhost:8333 |
