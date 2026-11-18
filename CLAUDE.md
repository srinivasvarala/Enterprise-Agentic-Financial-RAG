# CLAUDE.md

FinRAG: enterprise RAG over financial documents.

- Design: `ARCHITECTURE.md` and `docs/adr/`. Propose a superseding ADR before contradicting one.
- Names: `GLOSSARY.md`.
- Review rules: `CODING_STANDARDS.md`. Read it before touching tenant scoping, failure messages,
  background jobs or provider adapters.
- Commands: `make help`. Run `make ci` after every push to a PR branch.
- Dependencies: `uv add`. Build core logic test-first (chunking, fusion, citation validation,
  tenancy).

## Agent skills

### Issue tracker

Issues and specs live in this repo's GitHub Issues, managed via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `GLOSSARY.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
