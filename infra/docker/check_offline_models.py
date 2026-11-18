"""Run inside the built image: every model a question needs must load with no network.

    docker run --rm -i <image> python - < infra/docker/check_offline_models.py

The image sets HF_HUB_OFFLINE=1, so a model that was not baked in at build time fails here, not on
a user's first question.
"""

from uuid import UUID

from finrag.core.config import get_settings
from finrag.core.domain import EvidenceChunk
from finrag.indexing.bm25 import FastEmbedBm25Encoder
from finrag.retrieval.rerankers import BgeReranker

settings = get_settings()

chunk = EvidenceChunk(UUID(int=1), UUID(int=2), "Cash was $1,234 million.", (), 1, 1)
(score,) = BgeReranker.from_settings(settings).score("how much cash was there?", [chunk])
assert 0.0 < score < 1.0, score
print(f"reranker ok: {settings.reranker_model} scored {score:.3f}")

query = FastEmbedBm25Encoder.from_settings(settings).encode_query("cash")
assert query.indices, "empty BM25 query vector"
print(f"bm25 query ok: {settings.fastembed_bm25_model}")
