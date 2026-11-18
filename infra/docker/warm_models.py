"""Build-time model download: parse one PDF, encode one string and rerank one pair, so every model
file is cached.

Run by the Dockerfile with HF_HOME and FINRAG_FASTEMBED_CACHE_DIR set to the baked paths, so the
running containers (HF_HUB_OFFLINE=1) never download anything.
"""

import sys
from pathlib import Path
from uuid import UUID

from finrag.core.config import get_settings
from finrag.core.domain import EvidenceChunk
from finrag.indexing.bm25 import FastEmbedBm25Encoder
from finrag.ingestion.docling_parser import DoclingParser
from finrag.retrieval.rerankers import BgeReranker

pdf = Path(sys.argv[1])
parsed = DoclingParser(ocr_enabled=False).parse(pdf.read_bytes())
print(f"docling warmed: {len(parsed.blocks)} blocks")

settings = get_settings()
FastEmbedBm25Encoder(settings.fastembed_bm25_model, settings.fastembed_cache_dir).encode_documents(
    ["warm up"]
)
print("fastembed warmed")

chunk = EvidenceChunk(UUID(int=1), UUID(int=2), "warm up", (), 1, 1)
BgeReranker.from_settings(settings).score("warm up", [chunk])
print(f"reranker warmed: {settings.reranker_model}")
