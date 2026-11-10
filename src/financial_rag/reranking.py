"""Cross-encoder reranking for retrieved filing evidence."""

from typing import Protocol, Sequence

import torch
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder

from financial_rag.chunking import TABLE_UNIT_PATTERN
from financial_rag.models import SearchResult
from financial_rag.retrieval import (
    DEFAULT_HYBRID_CANDIDATE_LIMIT,
    hybrid_search_chunks,
)
from financial_rag.vectorstore import DEFAULT_COLLECTION_NAME


DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
DEFAULT_RERANKER_MAX_LENGTH = 512
DEFAULT_RERANK_CANDIDATE_LIMIT = 25
DATA_ROW_MARKER = "Data row:\n"


class PairScorer(Protocol):
    """Something that scores question-and-passage pairs."""

    def predict(
        self,
        sentences: list[tuple[str, str]],
    ) -> Sequence[float]: ...


def _prefer_unit_preserving_table_rows(
    results: list[SearchResult],
) -> list[SearchResult]:
    """Collapse duplicate table rows while retaining explicit unit context."""
    selected_results = []
    position_by_row = {}

    for result in results:
        chunk = result.chunk
        if chunk.chunk_type != "table" or DATA_ROW_MARKER not in chunk.content:
            selected_results.append(result)
            continue

        context, data_row = chunk.content.rsplit(DATA_ROW_MARKER, maxsplit=1)
        normalized_row = " ".join(data_row.casefold().split())
        row_key = (chunk.filing.accession_number, normalized_row)
        existing_position = position_by_row.get(row_key)
        if existing_position is None:
            position_by_row[row_key] = len(selected_results)
            selected_results.append(result)
            continue

        existing_result = selected_results[existing_position]
        existing_context = existing_result.chunk.content.rsplit(
            DATA_ROW_MARKER,
            maxsplit=1,
        )[0]
        if (
            TABLE_UNIT_PATTERN.search(context)
            and not TABLE_UNIT_PATTERN.search(existing_context)
        ):
            selected_results[existing_position] = result

    return selected_results


def load_reranker(
    model_name: str = DEFAULT_RERANKER_MODEL,
    device: str | None = None,
) -> CrossEncoder:
    """Load the cached cross-encoder on Apple MPS when available."""
    selected_device = device
    if selected_device is None:
        selected_device = "mps" if torch.backends.mps.is_available() else "cpu"

    return CrossEncoder(
        model_name,
        device=selected_device,
        max_length=DEFAULT_RERANKER_MAX_LENGTH,
        local_files_only=True,
    )


def rerank_search_results(
    question: str,
    results: list[SearchResult],
    scorer: PairScorer,
    limit: int = 5,
) -> list[SearchResult]:
    """Rescore retrieved chunks and return the most relevant evidence."""
    if not question.strip():
        raise ValueError("question must not be empty")
    if limit <= 0:
        raise ValueError("limit must be positive")
    if not results:
        return []

    selected_results = _prefer_unit_preserving_table_rows(results)
    pairs = [
        (question, result.chunk.content)
        for result in selected_results
    ]
    scores = list(scorer.predict(pairs))
    if len(scores) != len(selected_results):
        raise ValueError("scorer must return one score per search result")

    reranked = [
        SearchResult(score=float(score), chunk=result.chunk)
        for result, score in zip(selected_results, scores, strict=True)
    ]
    return sorted(reranked, key=lambda result: -result.score)[:limit]


def hybrid_search_and_rerank(
    client: QdrantClient,
    question: str,
    query_vector: tuple[float, ...],
    scorer: PairScorer,
    limit: int = 5,
    rerank_candidate_limit: int = DEFAULT_RERANK_CANDIDATE_LIMIT,
    hybrid_candidate_limit: int = DEFAULT_HYBRID_CANDIDATE_LIMIT,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    ticker: str | None = None,
    form: str | None = None,
) -> list[SearchResult]:
    """Retrieve a broad hybrid candidate set, then rerank its best results."""
    if rerank_candidate_limit < limit:
        raise ValueError("rerank_candidate_limit must be at least limit")
    if hybrid_candidate_limit < rerank_candidate_limit:
        raise ValueError(
            "hybrid_candidate_limit must be at least rerank_candidate_limit"
        )

    candidates = hybrid_search_chunks(
        client=client,
        question=question,
        query_vector=query_vector,
        limit=rerank_candidate_limit,
        candidate_limit=hybrid_candidate_limit,
        collection_name=collection_name,
        ticker=ticker,
        form=form,
    )
    return rerank_search_results(
        question=question,
        results=candidates,
        scorer=scorer,
        limit=limit,
    )
