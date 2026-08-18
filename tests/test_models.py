from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from financial_rag.models import DocumentChunk, FilingMetadata


def test_document_chunk_keeps_content_and_filing_provenance() -> None:
    filing = FilingMetadata(
        ticker="NVDA",
        company_name="NVIDIA CORP",
        cik="0001045810",
        form="10-K",
        filing_date="2026-02-25",
        accession_number="0001045810-26-000021",
        source_url="https://www.sec.gov/Archives/example/nvda-20260125.htm",
        local_path=Path("data/raw/NVDA_10-K_2026-02-25.html"),
    )
    chunk = DocumentChunk(
        chunk_id="NVDA_10-K_2026-02-25-00000",
        sequence=0,
        chunk_type="text",
        content="NVIDIA reports financial results.",
        token_count=6,
        source_sequences=(0, 1),
        filing=filing,
    )

    assert chunk.filing.ticker == "NVDA"
    assert chunk.filing.company_name == "NVIDIA CORP"
    assert chunk.source_sequences == (0, 1)
    assert chunk.content == "NVIDIA reports financial results."

    with pytest.raises(FrozenInstanceError):
        chunk.content = "Changed evidence"
