"""Qdrant storage tools for embedded SEC filing chunks."""

from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from financial_rag.embeddings import EMBEDDING_DIMENSIONS
from financial_rag.models import (
    DocumentChunk,
    EmbeddedChunk,
    FilingMetadata,
    SearchResult,
)


DEFAULT_COLLECTION_NAME = "sec_filing_chunks"


def open_local_vector_store(storage_path: Path) -> QdrantClient:
    """Open a persistent local Qdrant database without running a server."""
    storage_path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(storage_path))


def ensure_collection(
    client: QdrantClient,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> None:
    """Create the cosine-similarity collection when it does not exist."""
    if client.collection_exists(collection_name):
        collection = client.get_collection(collection_name)
        vector_config = collection.config.params.vectors
        if vector_config.size != EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"Collection vector size must be {EMBEDDING_DIMENSIONS}"
            )
        if vector_config.distance != models.Distance.COSINE:
            raise ValueError("Collection must use cosine distance")
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=EMBEDDING_DIMENSIONS,
            distance=models.Distance.COSINE,
        ),
    )


def build_chunk_payload(chunk: DocumentChunk) -> dict:
    """Convert a document chunk into searchable Qdrant metadata."""
    return {
        "chunk_id": chunk.chunk_id,
        "sequence": chunk.sequence,
        "chunk_type": chunk.chunk_type,
        "content": chunk.content,
        "token_count": chunk.token_count,
        "source_sequences": list(chunk.source_sequences),
        "ticker": chunk.filing.ticker,
        "company_name": chunk.filing.company_name,
        "cik": chunk.filing.cik,
        "form": chunk.filing.form,
        "filing_date": chunk.filing.filing_date,
        "accession_number": chunk.filing.accession_number,
        "source_url": chunk.filing.source_url,
        "local_path": str(chunk.filing.local_path),
    }


def point_id_for_chunk(chunk_id: str) -> str:
    """Create a stable Qdrant-compatible UUID for one chunk ID."""
    return str(uuid5(NAMESPACE_URL, chunk_id))


def upsert_embedded_chunks(
    client: QdrantClient,
    embedded_chunks: list[EmbeddedChunk],
    collection_name: str = DEFAULT_COLLECTION_NAME,
    batch_size: int = 64,
) -> int:
    """Insert or update embedded chunks and return the processed count."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    ensure_collection(client, collection_name)

    for start in range(0, len(embedded_chunks), batch_size):
        batch = embedded_chunks[start : start + batch_size]
        points = []

        for embedded_chunk in batch:
            if len(embedded_chunk.vector) != EMBEDDING_DIMENSIONS:
                raise ValueError(
                    "Embedding vector must contain "
                    f"{EMBEDDING_DIMENSIONS} dimensions"
                )
            points.append(
                models.PointStruct(
                    id=point_id_for_chunk(embedded_chunk.chunk.chunk_id),
                    vector=list(embedded_chunk.vector),
                    payload=build_chunk_payload(embedded_chunk.chunk),
                )
            )

        if points:
            client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True,
            )

    return len(embedded_chunks)


def _chunk_from_payload(payload: dict) -> DocumentChunk:
    """Rebuild one document chunk from stored Qdrant payload fields."""
    filing = FilingMetadata(
        ticker=payload["ticker"],
        company_name=payload.get("company_name", payload["ticker"]),
        cik=payload["cik"],
        form=payload["form"],
        filing_date=payload["filing_date"],
        accession_number=payload["accession_number"],
        source_url=payload["source_url"],
        local_path=Path(payload["local_path"]),
    )
    return DocumentChunk(
        chunk_id=payload["chunk_id"],
        sequence=payload["sequence"],
        chunk_type=payload["chunk_type"],
        content=payload["content"],
        token_count=payload["token_count"],
        source_sequences=tuple(payload["source_sequences"]),
        filing=filing,
    )


def search_similar_chunks(
    client: QdrantClient,
    query_vector: tuple[float, ...],
    limit: int = 5,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    ticker: str | None = None,
    form: str | None = None,
) -> list[SearchResult]:
    """Return the closest chunks, optionally filtered by ticker and form."""
    if len(query_vector) != EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"Query vector must contain {EMBEDDING_DIMENSIONS} dimensions"
        )
    if limit <= 0:
        raise ValueError("limit must be positive")

    conditions = []
    if ticker:
        conditions.append(
            models.FieldCondition(
                key="ticker",
                match=models.MatchValue(value=ticker.upper()),
            )
        )
    if form:
        conditions.append(
            models.FieldCondition(
                key="form",
                match=models.MatchValue(value=form.upper()),
            )
        )
    query_filter = models.Filter(must=conditions) if conditions else None

    response = client.query_points(
        collection_name=collection_name,
        query=list(query_vector),
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    return [
        SearchResult(
            score=float(point.score),
            chunk=_chunk_from_payload(point.payload),
        )
        for point in response.points
    ]
