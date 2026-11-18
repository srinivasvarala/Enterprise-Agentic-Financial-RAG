# ADR-0002: Framework-free core behind Protocols

**Status:** Accepted (2026-11-18)

## Context
LangChain / LlamaIndex offer ready-made RAG pipelines but hide retrieval and prompt details,
churn frequently, and make citation logic and tenant filtering harder to reason about and test.

## Decision
The core is plain Python. Each external dependency (LLM, embedder, sparse encoder, reranker,
vector store, blob store, guardrails) is a `typing.Protocol` in `finrag.core.interfaces`, with
concrete adapters calling vendor SDKs directly (Anthropic, Voyage, Qdrant, ...). Everything is
wired at a single composition root. Unit tests use in-memory fakes.

## Consequences
- More code to write up front; full control over retrieval, prompts and citations.
- NeMo Guardrails pulls in LangChain transitively; that is contained inside the guardrails adapter.
- LangGraph may be adopted at the agentic milestone, wrapping our modules as tools.
