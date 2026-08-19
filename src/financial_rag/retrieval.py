"""Hybrid semantic and lexical retrieval for filing chunks."""

import math
import re
from collections import Counter

from qdrant_client import QdrantClient

from financial_rag.models import DocumentChunk, SearchResult
from financial_rag.vectorstore import (
    DEFAULT_COLLECTION_NAME,
    list_filtered_chunks,
    search_similar_chunks,
)


DEFAULT_HYBRID_CANDIDATE_LIMIT = 100
DEFAULT_RRF_K = 60


def _lexical_tokens(text: str) -> list[str]:
    """Create lowercase word-and-number tokens for local lexical search."""
    return re.findall(r"[a-z0-9]+", text.casefold())


def rank_chunks_lexically(
    question: str,
    chunks: list[DocumentChunk],
) -> list[SearchResult]:
    """Rank chunks with BM25 using only local text and no new dependency."""
    query_terms = tuple(dict.fromkeys(_lexical_tokens(question)))
    if not query_terms or not chunks:
        return []

    tokenized_chunks = [_lexical_tokens(chunk.content) for chunk in chunks]
    document_count = len(tokenized_chunks)
    average_length = sum(map(len, tokenized_chunks)) / document_count
    document_frequency = Counter(
        token
        for tokens in tokenized_chunks
        for token in set(tokens)
    )
    k1 = 1.5
    length_normalization = 0.75
    ranked_results = []

    for chunk, tokens in zip(chunks, tokenized_chunks, strict=True):
        term_frequency = Counter(tokens)
        score = 0.0
        for term in query_terms:
            frequency = term_frequency[term]
            if frequency == 0:
                continue
            inverse_document_frequency = math.log(
                1
                + (
                    document_count
                    - document_frequency[term]
                    + 0.5
                )
                / (document_frequency[term] + 0.5)
            )
            denominator = frequency + k1 * (
                1
                - length_normalization
                + length_normalization * len(tokens) / average_length
            )
            score += (
                inverse_document_frequency
                * frequency
                * (k1 + 1)
                / denominator
            )

        if score > 0:
            ranked_results.append(SearchResult(score=score, chunk=chunk))

    return sorted(
        ranked_results,
        key=lambda result: (-result.score, result.chunk.chunk_id),
    )


def fuse_ranked_results(
    semantic_results: list[SearchResult],
    lexical_results: list[SearchResult],
    limit: int,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[SearchResult]:
    """Combine two rankings using reciprocal-rank fusion."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")

    semantic_ranks = {
        result.chunk.chunk_id: rank
        for rank, result in enumerate(semantic_results, start=1)
    }
    lexical_ranks = {
        result.chunk.chunk_id: rank
        for rank, result in enumerate(lexical_results, start=1)
    }
    chunks_by_id = {
        result.chunk.chunk_id: result.chunk
        for result in [*semantic_results, *lexical_results]
    }

    fused_results = []
    for chunk_id, chunk in chunks_by_id.items():
        score = 0.0
        if chunk_id in semantic_ranks:
            score += 1 / (rrf_k + semantic_ranks[chunk_id])
        if chunk_id in lexical_ranks:
            score += 1 / (rrf_k + lexical_ranks[chunk_id])
        fused_results.append(SearchResult(score=score, chunk=chunk))

    return sorted(
        fused_results,
        key=lambda result: (
            -result.score,
            semantic_ranks.get(result.chunk.chunk_id, math.inf),
            lexical_ranks.get(result.chunk.chunk_id, math.inf),
            result.chunk.chunk_id,
        ),
    )[:limit]


def hybrid_search_chunks(
    client: QdrantClient,
    question: str,
    query_vector: tuple[float, ...],
    limit: int = 5,
    candidate_limit: int = DEFAULT_HYBRID_CANDIDATE_LIMIT,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    ticker: str | None = None,
    form: str | None = None,
) -> list[SearchResult]:
    """Return RRF-fused semantic and BM25 filing evidence."""
    if not question.strip():
        raise ValueError("question must not be empty")
    if limit <= 0:
        raise ValueError("limit must be positive")
    if candidate_limit < limit:
        raise ValueError("candidate_limit must be at least limit")

    semantic_results = search_similar_chunks(
        client=client,
        query_vector=query_vector,
        limit=candidate_limit,
        collection_name=collection_name,
        ticker=ticker,
        form=form,
    )
    filtered_chunks = list_filtered_chunks(
        client=client,
        collection_name=collection_name,
        ticker=ticker,
        form=form,
    )
    lexical_results = rank_chunks_lexically(question, filtered_chunks)[
        :candidate_limit
    ]
    return fuse_ranked_results(
        semantic_results=semantic_results,
        lexical_results=lexical_results,
        limit=limit,
    )
