# ADR-0008: Buffer, validate, then stream; unverifiable answers are refused

**Status:** Accepted (2026-11-18)

## Context
Citations can only be checked against a complete generated answer. Streaming tokens as they are
produced would show a Reader text that may later fail validation, and in a financial setting a
retracted statement has already been read.

## Decision
- Generation is buffered. The SSE stream sends progress events (`retrieving`, `reranking`,
  `generating`, `validating`) and then a single `answer` event carrying the text and its
  Citations together.
- Validation is all-or-nothing. If any factual sentence is uncited, a Citation Marker does not
  resolve to a chunk in the Evidence, a Quoted Span is not verbatim in its chunk, or a number
  is not verbatim in a cited chunk (ADR-0006), the result is a Refusal with reason
  `unverifiable`. No partial answers.
- A Refusal is an Answer with one of four reasons (`insufficient_evidence`, `unverifiable`,
  `out_of_policy`, `needs_calculation`) and fixed text per reason. It is not an error.

## Considered options
- Stream tokens, then send a `retraction` event: rejected; the Reader has already seen the text.
- Drop uncited sentences and return the rest: rejected; a partly verified answer is harder to
  trust than a refusal. Revisit if the Golden Set shows refusals are too frequent.

## Consequences
Higher time-to-first-answer. Token streaming would need a superseding ADR.
