"""Retrieval: hybrid search, chunk loading and reranking produce the Evidence for a question."""

from uuid import UUID, uuid4, uuid5

from fakes import FixedReranker, InMemoryChunkRepository

from finrag.core.config import Settings
from finrag.core.domain import ChunkKind, EvidenceChunk, Principal, VectorPoint
from finrag.indexing.in_memory import HashEmbedder, HashSparseEncoder, InMemoryVectorStore
from finrag.retrieval.retriever import Retriever

TENANT = uuid4()
COLLECTION = uuid4()
PRINCIPAL = Principal(tenant_id=TENANT, collection_id=COLLECTION)


class World:
    """An indexed Collection plus a Retriever wired to in-memory adapters."""

    def __init__(self, rerank_top_n: int = 8, min_rerank_score: float = 0.0) -> None:
        self.embedder = HashEmbedder()
        self.sparse = HashSparseEncoder()
        self.vectors = InMemoryVectorStore()
        self.chunks = InMemoryChunkRepository()
        self.reranker = FixedReranker()
        self.retriever = Retriever(
            embedder=self.embedder,
            sparse_encoder=self.sparse,
            vector_store=self.vectors,
            chunks=self.chunks,
            reranker=self.reranker,
            settings=Settings(  # type: ignore[call-arg]
                _env_file=None, rerank_top_n=rerank_top_n, min_rerank_score=min_rerank_score
            ),
        )

    def index(
        self, text: str, tenant_id: UUID = TENANT, collection_id: UUID = COLLECTION
    ) -> EvidenceChunk:
        chunk_id = uuid4()
        document_id = uuid4()
        chunk = EvidenceChunk(chunk_id, document_id, text, ("Part II",), 1, 1)
        self.chunks.add(tenant_id, collection_id, chunk)
        self.vectors.upsert(
            tenant_id,
            [
                VectorPoint(
                    id=uuid5(chunk_id, "point"),
                    dense=self.embedder.embed_documents([text])[0],
                    sparse=self.sparse.encode_documents([text])[0],
                    tenant_id=tenant_id,
                    collection_id=collection_id,
                    document_id=document_id,
                    document_version_id=uuid4(),
                    chunk_id=chunk_id,
                    kind=ChunkKind.TEXT,
                    page_start=1,
                    page_end=1,
                )
            ],
        )
        return chunk


def test_evidence_is_the_top_chunks_in_rerank_order() -> None:
    world = World(rerank_top_n=2)
    a = world.index("cash and cash equivalents were high")
    b = world.index("cash flow from operations rose")
    c = world.index("cash held in foreign subsidiaries")
    world.reranker.scores = {a.chunk_id: 0.2, b.chunk_id: 0.9, c.chunk_id: 0.5}

    evidence = world.retriever.retrieve(PRINCIPAL, "how much cash did the company hold")

    assert [e.chunk_id for e in evidence] == [b.chunk_id, c.chunk_id]


def test_a_chunk_found_by_search_but_missing_from_the_system_of_record_is_dropped() -> None:
    world = World()
    kept = world.index("cash and cash equivalents were high")
    deleted = world.index("cash flow from operations rose")
    world.chunks.remove(TENANT, COLLECTION, deleted.chunk_id)
    world.reranker.scores = {kept.chunk_id: 0.4, deleted.chunk_id: 0.9}

    evidence = world.retriever.retrieve(PRINCIPAL, "how much cash did the company hold")

    assert [e.chunk_id for e in evidence] == [kept.chunk_id]


def test_nothing_indexed_means_no_evidence() -> None:
    world = World()

    assert world.retriever.retrieve(PRINCIPAL, "how much cash did the company hold") == ()


def test_chunks_scoring_below_the_relevance_threshold_are_not_evidence() -> None:
    world = World(min_rerank_score=0.5)
    relevant = world.index("cash and cash equivalents were high")
    irrelevant = world.index("cash flow from operations rose")
    world.reranker.scores = {relevant.chunk_id: 0.8, irrelevant.chunk_id: 0.1}

    evidence = world.retriever.retrieve(PRINCIPAL, "how much cash did the company hold")

    assert [e.chunk_id for e in evidence] == [relevant.chunk_id]


def test_when_every_chunk_scores_below_the_threshold_there_is_no_evidence() -> None:
    world = World(min_rerank_score=0.5)
    chunk = world.index("cash and cash equivalents were high")
    world.reranker.scores = {chunk.chunk_id: 0.1}

    assert world.retriever.retrieve(PRINCIPAL, "how much cash did the company hold") == ()


def test_chunks_of_another_tenant_or_collection_are_never_evidence() -> None:
    world = World()
    mine = world.index("cash and cash equivalents were high")
    other_tenant = world.index("cash held by a rival", tenant_id=uuid4())
    other_collection = world.index("cash held in another collection", collection_id=uuid4())
    world.reranker.scores = {
        mine.chunk_id: 0.1,
        other_tenant.chunk_id: 0.9,
        other_collection.chunk_id: 0.9,
    }

    evidence = world.retriever.retrieve(PRINCIPAL, "how much cash did the company hold")

    assert [e.chunk_id for e in evidence] == [mine.chunk_id]
