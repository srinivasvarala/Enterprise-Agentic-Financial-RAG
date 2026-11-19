"""Retrieval: hybrid search, chunk loading and reranking produce the Evidence for a question."""

from dataclasses import dataclass

from finrag.core.config import Settings
from finrag.core.domain import EvidenceChunk, Principal
from finrag.core.interfaces import ChunkRepository, Embedder, Reranker, SparseEncoder, VectorStore


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    chunk: EvidenceChunk
    score: float  # the reranker's score


@dataclass(frozen=True, slots=True)
class RetrievalTrace:
    """What retrieval did, for evaluation: the candidates before the rerank and the Evidence."""

    candidates: tuple[ScoredChunk, ...]  # search (fusion) order, each with its rerank score
    reranked: tuple[ScoredChunk, ...]  # the same candidates, best rerank score first
    evidence: tuple[EvidenceChunk, ...]  # what `retrieve` returns


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        sparse_encoder: SparseEncoder,
        vector_store: VectorStore,
        chunks: ChunkRepository,
        reranker: Reranker,
        settings: Settings,
    ) -> None:
        self._embedder = embedder
        self._sparse = sparse_encoder
        self._store = vector_store
        self._chunks = chunks
        self._reranker = reranker
        self._settings = settings

    def retrieve(self, principal: Principal, question: str) -> tuple[EvidenceChunk, ...]:
        """The Evidence for `question`: the best reranked chunks of the Principal's Collection."""
        return self.retrieve_traced(principal, question).evidence

    def retrieve_traced(self, principal: Principal, question: str) -> RetrievalTrace:
        """`retrieve`, also returning the pre-rerank candidates and scores (for evaluation)."""
        s = self._settings
        hits = self._store.search(
            principal.tenant_id,
            principal.collection_id,
            self._embedder.embed_query(question),
            self._sparse.encode_query(question),
            dense_k=s.retrieve_dense_k,
            sparse_k=s.retrieve_sparse_k,
            rrf_k=s.rrf_k,
            limit=s.retrieve_dense_k + s.retrieve_sparse_k,
        )
        loaded = {
            c.chunk_id: c
            for c in self._chunks.get_many(
                principal.tenant_id, principal.collection_id, [h.chunk_id for h in hits]
            )
        }
        candidates = [loaded[h.chunk_id] for h in hits if h.chunk_id in loaded]
        scores = self._reranker.score(question, candidates)
        scored = tuple(ScoredChunk(c, sc) for c, sc in zip(candidates, scores, strict=True))
        reranked = tuple(sorted(scored, key=lambda item: -item.score))
        relevant = [item.chunk for item in reranked if item.score >= s.min_rerank_score]
        return RetrievalTrace(scored, reranked, tuple(relevant[: s.rerank_top_n]))
