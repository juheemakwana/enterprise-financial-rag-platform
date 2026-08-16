"""Tools for creating semantic vectors from document chunks."""

from typing import Protocol, Sequence

from sentence_transformers import SentenceTransformer

from financial_rag.chunking import EMBEDDING_MODEL_NAME
from financial_rag.models import DocumentChunk, EmbeddedChunk


EMBEDDING_DIMENSIONS = 384
DEFAULT_EMBEDDING_BATCH_SIZE = 32
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class EmbeddingModel(Protocol):
    """Small model interface required by the embedding pipeline."""

    def encode(
        self,
        sentences: Sequence[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        show_progress_bar: bool,
        convert_to_numpy: bool,
    ): ...


def load_embedding_model(
    model_name: str = EMBEDDING_MODEL_NAME,
    device: str = "cpu",
) -> SentenceTransformer:
    """Load the selected Sentence Transformers embedding model."""
    return SentenceTransformer(model_name, device=device)


def embed_chunks(
    chunks: list[DocumentChunk],
    model: EmbeddingModel,
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
) -> list[EmbeddedChunk]:
    """Create normalized BGE vectors while retaining chunk provenance."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not chunks:
        return []

    vectors = model.encode(
        [chunk.content for chunk in chunks],
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    if len(vectors) != len(chunks):
        raise ValueError("Embedding model returned an unexpected vector count")

    embedded_chunks = []
    for chunk, vector in zip(chunks, vectors, strict=True):
        values = tuple(float(value) for value in vector)
        if len(values) != EMBEDDING_DIMENSIONS:
            raise ValueError(
                "Embedding vector must contain "
                f"{EMBEDDING_DIMENSIONS} dimensions"
            )
        embedded_chunks.append(EmbeddedChunk(chunk=chunk, vector=values))

    return embedded_chunks


def embed_query(
    question: str,
    model: EmbeddingModel,
) -> tuple[float, ...]:
    """Create one normalized BGE query vector for semantic retrieval."""
    if not question.strip():
        raise ValueError("question must not be empty")

    vectors = model.encode(
        [QUERY_INSTRUCTION + question.strip()],
        batch_size=1,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    if len(vectors) != 1:
        raise ValueError("Embedding model must return one query vector")

    values = tuple(float(value) for value in vectors[0])
    if len(values) != EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"Query vector must contain {EMBEDDING_DIMENSIONS} dimensions"
        )
    return values
