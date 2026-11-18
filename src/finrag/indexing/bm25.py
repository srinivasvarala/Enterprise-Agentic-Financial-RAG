"""FastEmbed `Qdrant/bm25` sparse encoder; pairs with a Qdrant sparse vector using IDF."""

from collections.abc import Callable, Iterable, Sequence
from typing import Any, Protocol

from finrag.core.config import Settings
from finrag.core.domain import SparseVector


class _SparseModel(Protocol):
    def embed(self, documents: Iterable[str]) -> Iterable[Any]: ...

    def query_embed(self, query: str) -> Iterable[Any]: ...


def _load_fastembed(model_name: str, cache_dir: str | None) -> _SparseModel:
    from fastembed import SparseTextEmbedding

    return SparseTextEmbedding(model_name=model_name, cache_dir=cache_dir)


def _to_sparse(embedding: Any) -> SparseVector:
    return SparseVector(
        indices=tuple(int(i) for i in embedding.indices),
        values=tuple(float(v) for v in embedding.values),
    )


class FastEmbedBm25Encoder:
    """Loads the model lazily: first use downloads stopword/stemmer files, startup never does."""

    def __init__(
        self,
        model_name: str = "Qdrant/bm25",
        cache_dir: str | None = None,
        model_factory: Callable[[], _SparseModel] | None = None,
    ) -> None:
        self._factory = model_factory or (lambda: _load_fastembed(model_name, cache_dir))
        self._model: _SparseModel | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> "FastEmbedBm25Encoder":
        return cls(settings.fastembed_bm25_model, settings.fastembed_cache_dir)

    def encode_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        return [_to_sparse(e) for e in self._load().embed(list(texts))]

    def encode_query(self, text: str) -> SparseVector:
        return _to_sparse(next(iter(self._load().query_embed(text))))

    def _load(self) -> _SparseModel:
        if self._model is None:
            self._model = self._factory()
        return self._model
