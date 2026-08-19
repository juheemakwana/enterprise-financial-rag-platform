from dataclasses import replace
from pathlib import Path

import pytest

from qdrant_client import QdrantClient, models

from financial_rag.embeddings import EMBEDDING_DIMENSIONS
from financial_rag.models import DocumentChunk, EmbeddedChunk, FilingMetadata
from financial_rag.vectorstore import (
    DEFAULT_COLLECTION_NAME,
    ensure_collection,
    list_filtered_chunks,
    point_id_for_chunk,
    replace_filing_chunks,
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


def test_list_filtered_chunks_loads_all_matching_payloads() -> None:
    client = QdrantClient(":memory:")
    embedded_chunks = make_embedded_chunks()
    upsert_embedded_chunks(client, embedded_chunks)

    chunks = list_filtered_chunks(
        client,
        ticker="nvda",
        form="10-k",
        batch_size=1,
    )

    assert sorted(chunks, key=lambda chunk: chunk.chunk_id) == sorted(
        [item.chunk for item in embedded_chunks],
        key=lambda chunk: chunk.chunk_id,
    )


def test_replace_filing_chunks_removes_only_matching_accession() -> None:
    client = QdrantClient(":memory:")
    original_chunks = make_embedded_chunks()
    other_filing = replace(
        original_chunks[0].chunk.filing,
        filing_date="2025-02-26",
        accession_number="0001045810-25-000050",
        local_path=Path("data/raw/NVDA_10-K_2025-02-26.html"),
    )
    other_chunk = EmbeddedChunk(
        chunk=replace(
            original_chunks[0].chunk,
            chunk_id="NVDA_10-K_2025-02-26-00000",
            filing=other_filing,
        ),
        vector=original_chunks[0].vector,
    )
    replacement = EmbeddedChunk(
        chunk=replace(
            original_chunks[0].chunk,
            chunk_id="NVDA_10-K_2026-02-25-row-00000",
            content="Company: NVIDIA CORP. Revenue: $215,938 million.",
        ),
        vector=original_chunks[0].vector,
    )
    upsert_embedded_chunks(client, [*original_chunks, other_chunk])

    count = replace_filing_chunks(client, [replacement])

    assert count == 1
    collection = client.get_collection(DEFAULT_COLLECTION_NAME)
    assert collection.points_count == 2
    remaining_chunks = list_filtered_chunks(client, ticker="NVDA", form="10-K")
    assert {chunk.chunk_id for chunk in remaining_chunks} == {
        "NVDA_10-K_2025-02-26-00000",
        "NVDA_10-K_2026-02-25-row-00000",
    }


def test_replace_filing_chunks_rejects_mixed_accessions_before_delete() -> None:
    client = QdrantClient(":memory:")
    original_chunks = make_embedded_chunks()
    upsert_embedded_chunks(client, original_chunks)
    different_filing = replace(
        original_chunks[0].chunk.filing,
        accession_number="different-accession",
    )
    mixed_chunk = replace(
        original_chunks[0],
        chunk=replace(original_chunks[0].chunk, filing=different_filing),
    )

    with pytest.raises(ValueError, match="share one accession number"):
        replace_filing_chunks(client, [original_chunks[0], mixed_chunk])

    assert client.get_collection(DEFAULT_COLLECTION_NAME).points_count == 2
