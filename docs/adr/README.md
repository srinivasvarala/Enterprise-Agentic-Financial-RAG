# Architecture Decision Records

| ADR | Decision |
|---|---|
| [0001](0001-qdrant-for-hybrid-search.md) | Qdrant for dense + BM25 hybrid search (not Pinecone) |
| [0002](0002-framework-free-core.md) | Framework-free core behind Protocols |
| [0003](0003-deepeval-for-evaluation.md) | DeepEval for generation evals; own retrieval metrics |
| [0004](0004-nemo-guardrails-scope.md) | NeMo Guardrails as wrap-around rails only |
| [0005](0005-model-providers.md) | Pluggable model providers with hosted defaults |
| [0006](0006-quote-numbers-never-compute.md) | v1 quotes numbers, never computes them |
| [0007](0007-storage-and-jobs.md) | Postgres system of record, S3 blobs, Procrastinate jobs |
| [0008](0008-buffer-then-validate-answers.md) | Buffer, validate, then stream; unverifiable answers are refused |
| [0009](0009-calculation-questions-are-answered.md) | Calculation questions are answered with cited figures; three Refusal reasons |

New ADRs: copy the shape of an existing one, number sequentially, never edit an accepted ADR's
decision — supersede it with a new ADR instead.
