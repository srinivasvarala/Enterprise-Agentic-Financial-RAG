# FinRAG

Evidence-based question answering over financial documents. Every answer is grounded in
retrieved evidence from a tenant's own documents and carries verifiable citations.

## Access

**Tenant**:
An organization whose documents, users and API keys are fully isolated from every other tenant.
_Avoid_: Customer, org, account, workspace

**Collection**:
A named group of documents within a tenant; the unit readers are granted access to.
_Avoid_: Folder, library, corpus, project

**Admin**:
A tenant role that may upload and delete documents and manage API keys.

**Reader**:
A tenant role that may only ask questions over the collections it has been granted.
_Avoid_: Viewer, end user

## Documents

**Document**:
A single uploaded file (PDF, Word, Excel or text) belonging to one collection.
_Avoid_: File, asset, upload

**Filing**:
A document that is a regulatory submission, such as a 10-K, 10-Q or 8-K.
_Avoid_: Report, statement (when the regulatory form is meant)

**Document Version**:
One specific content of a document; re-uploading changed content creates a new version that replaces the previous one.
_Avoid_: Revision, copy

**Document Metadata**:
The facts identifying what a document is about: company, ticker, form type, fiscal period and filing date.
_Avoid_: Tags, attributes

**Section Path**:
The chain of headings under which a piece of content sits, e.g. "Part II › Item 7. MD&A › Liquidity".
_Avoid_: Breadcrumb, location

## Retrieval

**Chunk**:
A contiguous, structure-respecting piece of a document version that can be retrieved and cited on its own.
_Avoid_: Passage, segment, snippet, node

**Table Chunk**:
A chunk that holds one table (or one row group of a large table) with its caption and header.

**Chunk Context**:
A short generated description situating a chunk within its whole document, stored alongside the chunk.
_Avoid_: Summary, blurb, preamble

**Evidence**:
The set of chunks retrieved for a question and given to the model to answer from.
_Avoid_: Context (ambiguous with Chunk Context), sources, results

## Answers

**Answer**:
The system's response to a question, made only of statements supported by the evidence.
_Avoid_: Response, completion, output

**Citation**:
A link from a statement in an answer to the chunk that supports it, resolving to document, page, section path and quoted span.
_Avoid_: Reference, source, footnote

**Quoted Span**:
The exact text inside a cited chunk that supports a statement.
_Avoid_: Highlight, excerpt

**Refusal**:
An answer that declines to answer because the evidence is insufficient or the question falls outside policy.
_Avoid_: Fallback, error, "I don't know"

**Investment Advice**:
A recommendation on what a specific person should buy, sell or hold; always declined by policy.

**Rail**:
A policy check applied to a question, the evidence or an answer, which may block or alter it.
_Avoid_: Filter, guard, moderation

## Evaluation

**Golden Set**:
The curated questions, with known answers and the chunks or pages that support them, against which quality is measured.
_Avoid_: Test set, benchmark, ground truth

**Baseline**:
The recorded evaluation results that later changes are compared against.
