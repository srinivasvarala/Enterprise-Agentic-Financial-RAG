# Enterprise-Agentic-Financial-RAG
An Enterprise-Grade Financial Document Intelligence Platform Powered by RAG, GraphRAG, and Agentic AI

Evidence-based question answering over financial documents (SEC filings first) using hybrid search
(dense + BM25), reranking, contextual chunking and validated source-level citations.

- Design: [`ARCHITECTURE.md`](ARCHITECTURE.md)
- Decisions: [`docs/adr/`](docs/adr/)
- Status: **M1: ingestion** (upload, parse, chunk, embed, index)

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Docker.

```bash
make install   # deps + pre-commit hooks (for development)
make up        # builds the app image; starts postgres, qdrant, seaweedfs, phoenix, migrate, api, worker
make lint test
```

`make up` blocks until every service is healthy. The first build downloads the Python wheels and
~505 MB of Docling models plus the FastEmbed BM25 files, all baked into the image (containers run
with `HF_HUB_OFFLINE=1`, so a job never downloads models); later builds are cached.
`make logs` tails the api and worker, `make down` stops the stack.

### Embeddings: Voyage key or offline mode

Dense embeddings come from Voyage (ADR-0005). Put your key in `.env` (or the host environment):

```bash
echo 'FINRAG_VOYAGE_API_KEY=your-key' >> .env
make up
```

Without a key the api and worker refuse to start. To try the stack without one, use the offline
hash backends (deterministic, not semantically meaningful; good for checking the plumbing):

```bash
FINRAG_EMBEDDER_BACKEND=hash FINRAG_SPARSE_BACKEND=hash make up
```

### Ingest a document

```bash
# upload a PDF or plain text file: 202 with the Document and its id
curl -F file=@filing.pdf localhost:8000/documents

# poll until "status" is "ready" (it passes through queued and processing; "failed" carries an error)
curl localhost:8000/documents/<id>

# the Chunks that were indexed (limit/offset paging)
curl 'localhost:8000/documents/<id>/chunks?limit=5'
```

Chunks are indexed in the Qdrant collection `finrag_chunks`. To count the points for a Document:

```bash
curl -s localhost:6333/collections/finrag_chunks/points/count \
  -H 'content-type: application/json' \
  -d '{"exact": true, "filter": {"must": [{"key": "document_id", "match": {"value": "<id>"}}]}}'
```

### Ask a question

```bash
# streams Server-Sent Events: progress (retrieving, generating, validating), then one answer
curl -N localhost:8000/query -H 'content-type: application/json' \
  -d '{"question": "What were cash and cash equivalents at year end?"}'
```

The `answer` event carries `text`, `citations` (document, pages, section path, quoted span) and
`refusal_reason`, which is `null` for a verified answer and otherwise `insufficient_evidence`,
`unverifiable` or `out_of_policy`. Text is sent only after every citation has been verified
(ADR-0008). If answering fails once the stream has started, a single `error` event ends it.

Answers come from Claude: set `FINRAG_ANTHROPIC_API_KEY` in `.env`. The bge reranker is baked into
the image at build time (the containers run offline); `make check-image` proves it loads. To exercise the endpoint offline, use
`FINRAG_GENERATOR_BACKEND=extractive FINRAG_RERANKER_BACKEND=overlap` (it quotes the best-matching
chunk; not a real model).

### Memory

The worker runs Docling (layout and table models) on the CPU: roughly 1 GB for a short PDF and more
for a full 10-K, one document at a time. Docker Desktop's default VM can be too small. Raise it
under Settings > Resources > Memory to **6 GB or more**. The worker container is capped at 3 GB
(override with `FINRAG_WORKER_MEM_LIMIT=4g`); if a PDF job dies with exit code 137 in `make logs`,
the cap or the VM ran out of memory. Plain-text uploads need very little.

### Build the eval index

The M3 evaluation runs against the five pinned 10-K filings, in the Linux `eval` image, in its own
Tenant, Collection, Qdrant collection (`finrag_eval_chunks`) and bucket, so the dev stack's
documents and the eval's never mix (ADR-0010). With the stack up (`make up`):

```bash
make eval-fetch    # download the filings (SEC EDGAR)
make eval-render   # render them to PDF in the eval image, offline
make eval-index    # ingest the PDFs into the eval tenant; a second run adds nothing
```

`make eval-index` ingests the smallest PDF first and prints each filing's duration and peak
memory. The real filings need `FINRAG_VOYAGE_API_KEY` (in `.env` or the host environment; the offline
hash backends would make the index meaningless) and a Docker VM with at least
6 GB of memory; the container itself is capped at 6 GB. The judged metrics later need
`FINRAG_ANTHROPIC_API_KEY`. The judge library (DeepEval) is installed only in the `eval` image.

### Running the API outside Docker

```bash
docker compose up -d --wait postgres qdrant seaweedfs phoenix
make migrate
make dev                              # API on http://localhost:8000
uv run python -m finrag.jobs.worker   # worker, in a second terminal
```

| Service | URL |
|---|---|
| API | http://localhost:8000/docs |
| Qdrant dashboard | http://localhost:6333/dashboard |
| Phoenix traces | http://localhost:6006 |
| S3 (SeaweedFS) | http://localhost:8333 |
