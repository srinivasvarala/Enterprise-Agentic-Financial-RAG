# ADR-0006: v1 quotes numbers, never computes them

**Status:** Accepted (2026-11-18)

## Context
LLMs doing mental arithmetic on financial figures (YoY growth, ratios) produce plausible but
wrong numbers — the worst failure mode for this domain.

## Decision
In v1 the generator may only state numbers that appear verbatim in cited evidence. Questions that
require calculation are answered with the cited source figures and a note that computation is not
performed. Calculation arrives with the agentic milestone as a deterministic tool (calculator /
sandboxed Python) whose inputs are cited figures.
