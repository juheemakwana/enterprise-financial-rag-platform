"""Shared data structures for the financial RAG pipeline."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class DocumentElement:
    """One ordered text or table element extracted from a filing."""

    sequence: int
    element_type: Literal["text", "table"]
    content: str
