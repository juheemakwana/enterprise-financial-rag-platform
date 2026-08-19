from pathlib import Path

import numpy as np
import pytest

from financial_rag.embeddings import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    EMBEDDING_DIMENSIONS,
    QUERY_INSTRUCTION,
    embed_chunks,
    embed_query,
    load_embedding_model,
)
from financial_rag.models import DocumentChunk, FilingMetadata


class FakeEmbeddingModel:
    def __init__(self, dimensions: int = EMBEDDING_DIMENSIONS) -> None:
        self.dimensions = dimensions
        self.captured_call = {}

    def encode(self, sentences, **kwargs):
        self.captured_call["sentences"] = sentences
        self.captured_call.update(kwargs)
        return np.ones((len(sentences), self.dimensions), dtype=np.float32)


def make_chunks() -> list[DocumentChunk]:
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
    return [
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


def test_embed_chunks_retains_chunks_and_requests_normalized_vectors() -> None:
    chunks = make_chunks()
    model = FakeEmbeddingModel()

    embedded_chunks = embed_chunks(chunks, model)

    assert [item.chunk for item in embedded_chunks] == chunks
    assert all(len(item.vector) == EMBEDDING_DIMENSIONS for item in embedded_chunks)
    assert model.captured_call["sentences"] == [
        chunk.content for chunk in chunks
    ]
    assert model.captured_call["batch_size"] == DEFAULT_EMBEDDING_BATCH_SIZE
    assert model.captured_call["normalize_embeddings"] is True
    assert model.captured_call["show_progress_bar"] is False
    assert model.captured_call["convert_to_numpy"] is True


def test_embed_chunks_rejects_wrong_vector_dimensions() -> None:
    with pytest.raises(ValueError, match="384 dimensions"):
        embed_chunks(make_chunks(), FakeEmbeddingModel(dimensions=10))


def test_load_embedding_model_uses_selected_name_and_device(monkeypatch) -> None:
    captured_call = {}
    expected_model = FakeEmbeddingModel()

    def fake_sentence_transformer(model_name, *, device):
        captured_call["model_name"] = model_name
        captured_call["device"] = device
        return expected_model

    monkeypatch.setattr(
        "financial_rag.embeddings.SentenceTransformer",
        fake_sentence_transformer,
    )

    result = load_embedding_model()

    assert result is expected_model
    assert captured_call == {
        "model_name": "BAAI/bge-small-en-v1.5",
        "device": "cpu",
    }


def test_embed_query_adds_bge_instruction_and_normalizes_vector() -> None:
    model = FakeEmbeddingModel()

    vector = embed_query("How did NVIDIA revenue change?", model)

    assert len(vector) == EMBEDDING_DIMENSIONS
    assert model.captured_call["sentences"] == [
        QUERY_INSTRUCTION + "How did NVIDIA revenue change?"
    ]
    assert model.captured_call["batch_size"] == 1
    assert model.captured_call["normalize_embeddings"] is True


def test_embed_query_rejects_empty_question() -> None:
    with pytest.raises(ValueError, match="question must not be empty"):
        embed_query("   ", FakeEmbeddingModel())
