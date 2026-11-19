# ADR-0011: Render the pinned filings to PDF with wkhtmltopdf in the eval image

**Status:** Accepted (2026-11-18). Refines ADR-0010 (fixtures and where evals run).

## Context
The five pinned filings are inline-XBRL HTML: the SEC serves 10-Ks as HTML. Ingestion accepts
only PDF and plain text, Docling refused to open the raw files (they start with an XML prolog),
and HTML has no pages, while Gold Spans and Citations are page-based (ADR-0010). The owner chose
to render each filing to PDF and ingest it through the unchanged PDF path, so the evaluation
exercises the same Docling PDF and table parsing production uses.

## Decision
- Render each cached filing to PDF inside a Linux `eval` Docker target (`app-base` plus the
  renderer). `make eval-render` does it offline, idempotently, and writes `evals/cache/render.json`
  (renderer name and version, per filing: HTML sha256, page count, render seconds). The index
  fingerprint (#24) reads that record.
- Renderer: **wkhtmltopdf 0.12.6** (Debian bookworm package `0.12.6-2+b1`, pinned in the
  Dockerfile), JavaScript disabled, run with `--network none`. It is installed only in the `eval`
  target; the default (runtime) image and the CI docker-build job do not change.
- PDF bytes are not reproducible; a re-render must give identical extracted text per page
  (`make check-eval-image` checks this).
- A PDF is up to date when the HTML hash and the renderer name and version match the record.

## Spike (Apple 10-K, 1.6 MB HTML, Linux container, no network)
| Renderer | Pages | Render s | Peak MB | Part/Item headings | Tables |
|---|---|---|---|---|---|
| wkhtmltopdf 0.12.6 | 66 | 0.9 | 99 | Part I, Item 1, Item 1A found as separate headings | statements and cover tables survive |
| Chromium 156 (headless) | 91 | 2.4 | 713 | same | same |
| WeasyPrint 68 | 128 | 9.8 | 357 | merged into one heading ("PART I Item 1. Business ...") | survive |

Microsoft (8.5 MB HTML): wkhtmltopdf 102 pages, 1.5 s, 144 MB; Chromium 143 pages, 4.7 s, 933 MB.
Docling parsed 8-page and 2-page slices of each PDF (the statements pages gave the full income
statement as a Markdown table). Parsing a whole filing PDF was killed (exit 137) in the 4 GB
Docker VM available during the spike, so full-document parse memory is for #23 to measure.

## Considered options
- **Chromium headless.** Most faithful and maintained, but 7x the memory, 2.5x the time, a far
  larger image, an apt pin that breaks when Debian drops the version, and 38% more pages, which
  costs Docling memory and time downstream. It made a letter-spaced cover page text on Microsoft.
- **WeasyPrint.** Pins cleanly with uv, but it doubled the page count, was slowest, and merged the
  Part and Item headings the chunker relies on.
- **An HTML parser (ingest the HTML directly).** Needs a new ingestion path and has no pages, so
  page-based Gold Spans and Citations would not exist, and the eval would not exercise production's
  PDF and table parsing.
- **Investor-site PDFs.** Not pinnable by accession number and checksum, and not the filed document.

## Consequences
- wkhtmltopdf's upstream is archived (2023). Accepted: it runs only on checksum-pinned filings,
  with JavaScript off and no network, in an eval-only image. Revisit (Chromium is the fallback)
  if a future filing renders badly.
- The Dockerfile gains a stage ordering: `app-base`, then `eval`, then `runtime` last so it stays
  the default target.
- The pinned checksums in `evals/manifest.json` did not match the SEC's current bytes when this
  was built (see the ticket); rendering does not depend on that, but `make eval-fetch` does.
