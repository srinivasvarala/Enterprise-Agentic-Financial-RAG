"""Retrieval: hybrid search, chunk loading and reranking produce the Evidence for a question."""

from finrag.core.config import Settings
from finrag.core.domain import EvidenceChunk, Principal
from finrag.core.interfaces import ChunkRepository, Embedder, Reranker, SparseEncoder, VectorStore


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
        ranked = sorted(zip(candidates, scores, strict=True), key=lambda pair: -pair[1])
        relevant = [chunk for chunk, score in ranked if score >= s.min_rerank_score]
        return tuple(relevant[: s.rerank_top_n])
