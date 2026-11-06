import json
from pathlib import Path

import pytest

from financial_rag.chunking import (
    DEFAULT_CHUNK_TOKENS,
    DEFAULT_OVERLAP_TOKENS,
    EMBEDDING_MODEL_NAME,
    MAX_MODEL_TOKENS,
    chunk_document_elements,
    chunk_table_element,
    chunk_text_elements,
    count_tokens,
    load_chunks_jsonl,
    load_tokenizer,
    write_chunks_jsonl,
)
from financial_rag.models import DocumentChunk, DocumentElement, FilingMetadata


class FakeTokenizer:
    def __init__(self) -> None:
        self.token_to_id = {}
        self.id_to_token = {}

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        verbose: bool = False,
    ) -> list[int]:
        assert add_special_tokens is False
        assert verbose is False
        token_ids = []
        for token in text.split():
            if token not in self.token_to_id:
                token_id = len(self.token_to_id)
                self.token_to_id[token] = token_id
                self.id_to_token[token_id] = token
            token_ids.append(self.token_to_id[token])
        return token_ids

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
    ) -> str:
        assert skip_special_tokens is True
        return " ".join(self.id_to_token[token_id] for token_id in token_ids)


def make_filing_metadata() -> FilingMetadata:
    return FilingMetadata(
        ticker="NVDA",
        company_name="NVIDIA CORP",
        cik="0001045810",
        form="10-K",
        filing_date="2026-02-25",
        accession_number="0001045810-26-000021",
        source_url="https://www.sec.gov/Archives/example/nvda.htm",
        local_path=Path("data/raw/NVDA_10-K_2026-02-25.html"),
    )


def test_chunk_settings_fit_embedding_model_limit() -> None:
    assert EMBEDDING_MODEL_NAME == "BAAI/bge-small-en-v1.5"
    assert DEFAULT_CHUNK_TOKENS == 400
    assert DEFAULT_OVERLAP_TOKENS == 80
    assert DEFAULT_CHUNK_TOKENS < MAX_MODEL_TOKENS


def test_count_tokens_uses_selected_tokenizer() -> None:
    tokenizer = FakeTokenizer()

    result = count_tokens("Revenue increased during the year", tokenizer)

    assert result == 5


def test_load_tokenizer_uses_embedding_model(monkeypatch) -> None:
    captured_call = {}
    expected_tokenizer = FakeTokenizer()

    def fake_from_pretrained(model_name):
        captured_call["model_name"] = model_name
        return expected_tokenizer

    monkeypatch.setattr(
        "financial_rag.chunking.AutoTokenizer.from_pretrained",
        fake_from_pretrained,
    )

    result = load_tokenizer()

    assert result is expected_tokenizer
    assert captured_call["model_name"] == EMBEDDING_MODEL_NAME


def test_chunk_text_elements_combines_blocks_with_overlap() -> None:
    elements = [
        DocumentElement(0, "text", "alpha beta"),
        DocumentElement(1, "text", "gamma delta"),
        DocumentElement(2, "text", "epsilon zeta"),
    ]

    chunks = chunk_text_elements(
        elements=elements,
        filing=make_filing_metadata(),
        tokenizer=FakeTokenizer(),
        chunk_tokens=5,
        overlap_tokens=2,
    )

    assert [chunk.content for chunk in chunks] == [
        "alpha beta\n\ngamma delta",
        "gamma delta\n\nepsilon zeta",
    ]
    assert [chunk.source_sequences for chunk in chunks] == [
        (0, 1),
        (1, 2),
    ]
    assert all(chunk.token_count <= 5 for chunk in chunks)


def test_chunk_text_elements_splits_one_oversized_block() -> None:
    elements = [
        DocumentElement(
            7,
            "text",
            "one two three four five six seven eight nine",
        )
    ]

    chunks = chunk_text_elements(
        elements=elements,
        filing=make_filing_metadata(),
        tokenizer=FakeTokenizer(),
        chunk_tokens=4,
        overlap_tokens=1,
    )

    assert [chunk.content for chunk in chunks] == [
        "one two three four",
        "four five six seven",
        "seven eight nine",
    ]
    assert [chunk.source_sequences for chunk in chunks] == [(7,), (7,), (7,)]
    assert [chunk.token_count for chunk in chunks] == [4, 4, 3]


def test_chunk_table_element_adds_context_to_small_table_row() -> None:
    element = DocumentElement(
        12,
        "table",
        "| Year | Revenue |\n| --- | --- |\n| 2026 | 130,500 |",
    )

    chunks = chunk_table_element(
        element=element,
        filing=make_filing_metadata(),
        tokenizer=FakeTokenizer(),
        chunk_tokens=20,
        starting_sequence=4,
    )

    assert len(chunks) == 1
    assert chunks[0].sequence == 4
    assert chunks[0].chunk_id == "NVDA_10-K_2026-02-25-00004"
    assert chunks[0].content == (
        "Company: NVIDIA CORP (NVDA)\n"
        "Form: 10-K\n\n"
        "Table context:\n"
        "Year | Revenue\n\n"
        "Data row:\n"
        "2026 | 130,500"
    )
    assert chunks[0].source_sequences == (12,)


def test_chunk_table_element_creates_one_chunk_per_row() -> None:
    element = DocumentElement(
        12,
        "table",
        "| Year | Revenue |\n"
        "| --- | --- |\n"
        "| 2026 | 130,500 |\n"
        "| 2025 | 60,900 |",
    )

    chunks = chunk_table_element(
        element=element,
        filing=make_filing_metadata(),
        tokenizer=FakeTokenizer(),
        chunk_tokens=25,
    )

    assert [chunk.content for chunk in chunks] == [
        "Company: NVIDIA CORP (NVDA)\nForm: 10-K\n\n"
        "Table context:\nYear | Revenue\n\nData row:\n2026 | 130,500",
        "Company: NVIDIA CORP (NVDA)\nForm: 10-K\n\n"
        "Table context:\nYear | Revenue\n\nData row:\n2025 | 60,900",
    ]
    assert [chunk.source_sequences for chunk in chunks] == [(12,), (12,)]
    assert all(chunk.token_count <= 25 for chunk in chunks)


def test_chunk_table_element_repeats_multirow_financial_context() -> None:
    element = DocumentElement(
        483,
        "table",
        "|  |  | Year Ended |  |  |\n"
        "| --- | --- | --- | --- | --- |\n"
        "|  | Jan 25, 2026 | Jan 26, 2025 | $ Change | % Change |\n"
        "|  | ($ in millions) |  |  |  |\n"
        "| Research and development | $ | 18,497 | $ | 12,914 | 43 | % |\n"
        "| Sales, general and administrative | $ | 4,579 | $ | 3,491 | 31 | % |",
    )

    chunks = chunk_table_element(
        element=element,
        filing=make_filing_metadata(),
        tokenizer=FakeTokenizer(),
        chunk_tokens=50,
    )

    assert [chunk.content for chunk in chunks] == [
        "Company: NVIDIA CORP (NVDA)\n"
        "Form: 10-K\n\n"
        "Table context:\n"
        "Year Ended\n"
        "Jan 25, 2026 | Jan 26, 2025 | $ Change | % Change\n"
        "($ in millions)\n\n"
        "Data row:\n"
        "Research and development | $18,497 | $12,914 | 43%",
        "Company: NVIDIA CORP (NVDA)\n"
        "Form: 10-K\n\n"
        "Table context:\n"
        "Year Ended\n"
        "Jan 25, 2026 | Jan 26, 2025 | $ Change | % Change\n"
        "($ in millions)\n\n"
        "Data row:\n"
        "Sales, general and administrative | $4,579 | $3,491 | 31%",
    ]
    assert [chunk.source_sequences for chunk in chunks] == [(483,), (483,)]


def test_chunk_table_element_promotes_labeled_unit_row_to_context() -> None:
    element = DocumentElement(
        899,
        "table",
        "|  | Year Ended |  |\n"
        "| --- | --- | --- |\n"
        "|  | 2026 | 2025 |\n"
        "| Revenue by Geography: | (In millions) |  |\n"
        "| United States | 149,617 | 77,482 |\n"
        "| Total revenue | 215,938 | 130,497 |",
    )

    chunks = chunk_table_element(
        element=element,
        filing=make_filing_metadata(),
        tokenizer=FakeTokenizer(),
        chunk_tokens=50,
    )

    assert len(chunks) == 2
    assert all(
        "Revenue by Geography: | (In millions)" in chunk.content
        for chunk in chunks
    )
    assert "Data row:\nTotal revenue | 215,938 | 130,497" in (
        chunks[1].content
    )


def test_chunk_document_elements_copies_adjacent_unit_note_to_table() -> None:
    elements = [
        DocumentElement(608, "text", "(In millions, except per share data)"),
        DocumentElement(
            609,
            "table",
            "|  | Year Ended |  |\n"
            "| --- | --- | --- |\n"
            "|  | 2026 | 2025 |\n"
            "| Revenue | 215,938 | 130,497 |\n"
            "| Net income | 120,067 | 72,880 |",
        ),
    ]

    chunks = chunk_document_elements(
        elements=elements,
        filing=make_filing_metadata(),
        tokenizer=FakeTokenizer(),
        chunk_tokens=50,
        overlap_tokens=10,
    )
    table_chunks = [chunk for chunk in chunks if chunk.chunk_type == "table"]

    assert len(table_chunks) == 2
    assert all(
        "(In millions, except per share data)" in chunk.content
        for chunk in table_chunks
    )
    assert table_chunks[0].source_sequences == (609,)


def test_chunk_table_element_preserves_single_row_table() -> None:
    element = DocumentElement(
        3,
        "table",
        "| Annual report | ☒ |\n| --- | --- |",
    )

    chunks = chunk_table_element(
        element=element,
        filing=make_filing_metadata(),
        tokenizer=FakeTokenizer(),
        chunk_tokens=20,
    )

    assert len(chunks) == 1
    assert chunks[0].content == (
        "Company: NVIDIA CORP (NVDA)\n"
        "Form: 10-K\n\n"
        "Table row:\n"
        "Annual report | ☒"
    )
    assert chunks[0].source_sequences == (3,)


def test_chunk_document_elements_preserves_text_and_table_order() -> None:
    elements = [
        DocumentElement(0, "text", "Revenue increased during the year."),
        DocumentElement(1, "text", "Management discussed strong demand."),
        DocumentElement(
            2,
            "table",
            "| Year | Revenue |\n| --- | --- |\n| 2026 | 130,500 |",
        ),
        DocumentElement(3, "text", "Management expects continued growth."),
    ]

    chunks = chunk_document_elements(
        elements=elements,
        filing=make_filing_metadata(),
        tokenizer=FakeTokenizer(),
        chunk_tokens=20,
        overlap_tokens=4,
    )

    assert [chunk.sequence for chunk in chunks] == [0, 1, 2]
    assert [chunk.chunk_type for chunk in chunks] == [
        "text",
        "table",
        "text",
    ]
    assert [chunk.source_sequences for chunk in chunks] == [
        (0, 1),
        (2,),
        (3,),
    ]
    assert [chunk.chunk_id for chunk in chunks] == [
        "NVDA_10-K_2026-02-25-00000",
        "NVDA_10-K_2026-02-25-00001",
        "NVDA_10-K_2026-02-25-00002",
    ]


def test_write_chunks_jsonl_preserves_content_and_provenance(tmp_path) -> None:
    filing = make_filing_metadata()
    chunks = [
        DocumentChunk(
            chunk_id="NVDA_10-K_2026-02-25-00000",
            sequence=0,
            chunk_type="text",
            content="NVIDIA reports financial results.",
            token_count=6,
            source_sequences=(0, 1),
            filing=filing,
        )
    ]
    output_path = tmp_path / "chunks" / "chunks.jsonl"

    result = write_chunks_jsonl(chunks, output_path)
    record = json.loads(output_path.read_text(encoding="utf-8"))

    assert result == output_path
    assert record["chunk_id"] == "NVDA_10-K_2026-02-25-00000"
    assert record["source_sequences"] == [0, 1]
    assert record["filing"]["ticker"] == "NVDA"
    assert record["filing"]["company_name"] == "NVIDIA CORP"
    assert record["filing"]["accession_number"] == (
        "0001045810-26-000021"
    )
    assert record["filing"]["local_path"] == (
        "data/raw/NVDA_10-K_2026-02-25.html"
    )

    restored_chunks = load_chunks_jsonl(output_path)
    assert restored_chunks == chunks


def test_load_chunks_jsonl_rejects_duplicate_chunk_ids(tmp_path) -> None:
    filing = make_filing_metadata()
    chunks = [
        DocumentChunk(
            chunk_id="duplicate-id",
            sequence=0,
            chunk_type="text",
            content="First chunk",
            token_count=2,
            source_sequences=(0,),
            filing=filing,
        ),
        DocumentChunk(
            chunk_id="duplicate-id",
            sequence=1,
            chunk_type="text",
            content="Second chunk",
            token_count=2,
            source_sequences=(1,),
            filing=filing,
        ),
    ]
    output_path = tmp_path / "chunks.jsonl"
    write_chunks_jsonl(chunks, output_path)

    with pytest.raises(ValueError, match="Chunk IDs must be unique"):
        load_chunks_jsonl(output_path)
