# ADR-0005: Pluggable model providers with hosted defaults

**Status:** Accepted (2026-11-18)

## Decision
All models sit behind Protocols (ADR-0002). Defaults:

| Role | Default |
|---|---|
| Answer generation | Claude Sonnet (`claude-sonnet-5-5`) |
| Contextual chunk blurbs, metadata extraction | Claude Haiku (`claude-haiku-5-5`) with prompt caching |
| Eval judge | Claude Opus (`claude-opus-5-5`) |
| Dense embeddings | Voyage |
| BM25 sparse vectors | FastEmbed `Qdrant/bm25` (local) |
| Reranker | bge cross-encoder (local) |

Model IDs are configuration, not code. A fully local stack (Ollama, bge/e5 embeddings) can be
added as alternative adapters.
