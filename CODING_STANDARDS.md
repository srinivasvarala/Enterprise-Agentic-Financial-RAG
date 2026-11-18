# Coding standards

Judgement calls the reviewer enforces. Anything a tool can check is a check, not a rule here.

**Enforced by tooling** (`make lint`, pre-commit, CI): formatting, types (mypy strict), reading
config only through `Settings` (ruff `TID251`), no LangChain/LlamaIndex outside `guardrails/`
(ruff `TID251`), `core` importing no feature module (import-linter).

## Tenant scope

- Every database and vector-store query carries `tenant_id`. A new query ships with a
  cross-tenant test showing the other Tenant sees nothing.
- A Document owned by another Tenant is "not found", never "forbidden".
- Reuse the shared guards in `core/tenancy.py` instead of copying the check.

## Vocabulary

- Names in code, tests, error messages and docs follow `GLOSSARY.md`, including its `_Avoid_`
  lists. A Document is never a "file" in a message a user reads (`filename` and the HTTP `file`
  field are HTTP/OS terms and stay).

## Failures an admin can read

- Text stored in `failure_reason` or returned by the API names the stage and what to do next.
  It carries no provider internals: no hostnames, URLs or raw exception text. Log the detail
  with tenant, document and job ids.
- A `ParseError` message may be shown as written; it describes the Document, not the system.

## Background work

- Every Document reaches a terminal status. A job is idempotent, so re-running it is always safe,
  and anything left in a non-terminal status by a crash or a failed enqueue is recoverable by the
  reconciler.
- Order writes so `ready` means searchable: Postgres, then the vector store, then the status.

## Providers and wiring

- Each external provider sits behind a `Protocol` in `core/interfaces.py`; adapters live in their
  own module; only the composition root wires them. Unit tests use the in-memory fakes.
- A constructor does not repeat a default that `Settings` owns.

## Parsers

- Layout-model output differs by platform (macOS vs the Linux CPU-torch image). Normalise it in
  a pure function with unit tests over hand-built blocks; real-Docling tests alone are not enough.

## Evidence and numbers

- Retrieved text is untrusted input; it never alters instructions.
- Answers quote numbers from evidence and never compute them (ADR-0006).
- A retrieval or generation change reports eval deltas against the current baseline.
