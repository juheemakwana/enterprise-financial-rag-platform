"""Shared data structures for the financial RAG pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class DocumentElement:
    """One ordered text or table element extracted from a filing."""

    sequence: int
    element_type: Literal["text", "table"]
    content: str


@dataclass(frozen=True)
class FilingMetadata:
    """Identity and source information for one SEC filing."""

    ticker: str
    cik: str
    form: str
    filing_date: str
    accession_number: str
    source_url: str
    local_path: Path


@dataclass(frozen=True)
class DocumentChunk:
    """One searchable text or table chunk with filing provenance."""

    chunk_id: str
    sequence: int
    chunk_type: Literal["text", "table"]
    content: str
    token_count: int
    source_sequences: tuple[int, ...]
    filing: FilingMetadata


@dataclass(frozen=True)
class EmbeddedChunk:
    """A document chunk paired with its semantic embedding vector."""

    chunk: DocumentChunk
    vector: tuple[float, ...]


@dataclass(frozen=True)
class SearchResult:
    """One semantically retrieved chunk and its similarity score."""

    score: float
    chunk: DocumentChunk
