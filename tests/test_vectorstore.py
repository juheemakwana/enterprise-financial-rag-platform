from pathlib import Path

from qdrant_client import QdrantClient, models

from financial_rag.embeddings import EMBEDDING_DIMENSIONS
from financial_rag.models import DocumentChunk, EmbeddedChunk, FilingMetadata
from financial_rag.vectorstore import (
    DEFAULT_COLLECTION_NAME,
    ensure_collection,
    point_id_for_chunk,
    search_similar_chunks,
    upsert_embedded_chunks,
)


def make_embedded_chunks() -> list[EmbeddedChunk]:
    filing = FilingMetadata(
        ticker="NVDA",
        company_name="NVIDIA CORP",
        cik="0001045810",
        form="10-K",
        filing_date="2026-02-25",
        accession_number="0001045810-26-000021",
        source_url="https://www.sec.gov/Archives/example/nvda.htm",
        local_path=Path("data/raw/NVDA_10-K_2026-02-25.html"),
    )
    chunks = [
        DocumentChunk(
            chunk_id="NVDA_10-K_2026-02-25-00000",
            sequence=0,
            chunk_type="text",
            content="NVIDIA reported strong data center revenue.",
            token_count=8,
            source_sequences=(0, 1),
            filing=filing,
        ),
        DocumentChunk(
            chunk_id="NVDA_10-K_2026-02-25-00001",
            sequence=1,
            chunk_type="table",
            content="| Year | Revenue |\n| --- | --- |\n| 2026 | 130,500 |",
            token_count=15,
            source_sequences=(2,),
            filing=filing,
        ),
    ]
    vectors = [
        (1.0, *([0.0] * (EMBEDDING_DIMENSIONS - 1))),
        (0.0, 1.0, *([0.0] * (EMBEDDING_DIMENSIONS - 2))),
    ]
    return [
        EmbeddedChunk(chunk=chunk, vector=tuple(vector))
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]


def test_ensure_collection_creates_384_dimension_cosine_collection() -> None:
    client = QdrantClient(":memory:")

    ensure_collection(client)

    collection = client.get_collection(DEFAULT_COLLECTION_NAME)
    vector_config = collection.config.params.vectors
    assert vector_config.size == EMBEDDING_DIMENSIONS
    assert vector_config.distance == models.Distance.COSINE


def test_upsert_embedded_chunks_preserves_payload_without_duplicates() -> None:
    client = QdrantClient(":memory:")
    embedded_chunks = make_embedded_chunks()

    first_count = upsert_embedded_chunks(client, embedded_chunks, batch_size=1)
    second_count = upsert_embedded_chunks(client, embedded_chunks, batch_size=1)

    assert first_count == 2
    assert second_count == 2
    collection = client.get_collection(DEFAULT_COLLECTION_NAME)
    assert collection.points_count == 2

    point = client.retrieve(
        collection_name=DEFAULT_COLLECTION_NAME,
        ids=[point_id_for_chunk("NVDA_10-K_2026-02-25-00000")],
        with_payload=True,
    )[0]
    assert point.payload["ticker"] == "NVDA"
    assert point.payload["company_name"] == "NVIDIA CORP"
    assert point.payload["chunk_type"] == "text"
    assert point.payload["source_sequences"] == [0, 1]
    assert point.payload["accession_number"] == "0001045810-26-000021"
    assert point.payload["source_url"].startswith("https://www.sec.gov/")


def test_search_similar_chunks_returns_ranked_typed_results() -> None:
    client = QdrantClient(":memory:")
    embedded_chunks = make_embedded_chunks()
    upsert_embedded_chunks(client, embedded_chunks)

    query_vector = embedded_chunks[0].vector
    results = search_similar_chunks(
        client,
        query_vector,
        limit=2,
        ticker="nvda",
        form="10-k",
    )

    assert len(results) == 2
    assert results[0].chunk == embedded_chunks[0].chunk
    assert results[0].score == 1.0
    assert results[1].chunk == embedded_chunks[1].chunk
    assert results[0].chunk.filing.source_url.startswith("https://www.sec.gov/")
