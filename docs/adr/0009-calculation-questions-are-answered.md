# ADR-0009: Calculation questions are answered with cited figures, not refused

**Status:** Accepted (2026-11-18). Amends the Refusal reasons in ADR-0008.

## Context
ADR-0006 says a question that needs calculation is answered with the cited source figures and a
note that computation is not performed. ADR-0008 and the glossary nevertheless listed
`needs_calculation` as a Refusal reason, which nothing would ever produce.

## Decision
`needs_calculation` is not a Refusal reason. A Refusal has one of three reasons:
`insufficient_evidence`, `unverifiable`, `out_of_policy`. A calculation question is an ordinary
Answer: the generator quotes the cited figures and says it does not compute. Validation still
requires every number to appear verbatim in a cited chunk, so a computed figure is refused as
`unverifiable`.

## Consequences
A Reader asking for year-over-year growth gets the two cited figures instead of a refusal.
