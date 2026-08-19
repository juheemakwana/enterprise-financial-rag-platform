from pathlib import Path

from financial_rag.models import DocumentChunk, FilingMetadata, SearchResult
from financial_rag.retrieval import (
    fuse_ranked_results,
    hybrid_search_chunks,
    rank_chunks_lexically,
)


def make_chunk(chunk_id: str, content: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        sequence=int(chunk_id[-1]),
        chunk_type="table",
        content=content,
        token_count=len(content.split()),
        source_sequences=(int(chunk_id[-1]),),
        filing=FilingMetadata(
            ticker="NVDA",
            company_name="NVIDIA CORP",
            cik="0001045810",
            form="10-K",
            filing_date="2026-02-25",
            accession_number="0001045810-26-000021",
            source_url="https://www.sec.gov/Archives/example/nvda.htm",
            local_path=Path("data/raw/NVDA_10-K_2026-02-25.html"),
        ),
    )


def test_rank_chunks_lexically_prioritizes_exact_financial_terms() -> None:
    correct = make_chunk(
        "chunk-1",
        "Net income for 2026 was $120,067 million.",
    )
    unrelated = make_chunk(
        "chunk-2",
        "Customer advances generated other income.",
    )

    results = rank_chunks_lexically(
        "What was net income in 2026?",
        [unrelated, correct],
    )

    assert results[0].chunk == correct
    assert results[0].score > results[1].score


def test_fuse_ranked_results_rewards_agreement() -> None:
    first = make_chunk("chunk-1", "First")
    agreed = make_chunk("chunk-2", "Agreed")
    third = make_chunk("chunk-3", "Third")
    semantic = [
        SearchResult(0.9, first),
        SearchResult(0.8, agreed),
        SearchResult(0.7, third),
    ]
    lexical = [
        SearchResult(8.0, agreed),
        SearchResult(7.0, third),
        SearchResult(6.0, first),
    ]

    results = fuse_ranked_results(semantic, lexical, limit=3)

    assert results[0].chunk == agreed
    assert results[0].score > results[1].score


def test_hybrid_search_combines_filtered_rankings(monkeypatch) -> None:
    semantic_first = make_chunk("chunk-1", "Revenue increased.")
    agreed = make_chunk("chunk-2", "Net income was $120,067 million.")
    captured_calls = {}

    def fake_semantic_search(**kwargs):
        captured_calls["semantic"] = kwargs
        return [
            SearchResult(0.9, semantic_first),
            SearchResult(0.8, agreed),
        ]

    def fake_list_chunks(**kwargs):
        captured_calls["list"] = kwargs
        return [semantic_first, agreed]

    monkeypatch.setattr(
        "financial_rag.retrieval.search_similar_chunks",
        fake_semantic_search,
    )
    monkeypatch.setattr(
        "financial_rag.retrieval.list_filtered_chunks",
        fake_list_chunks,
    )

    results = hybrid_search_chunks(
        client="fake-client",
        question="What was net income?",
        query_vector=tuple([0.0] * 384),
        limit=2,
        candidate_limit=10,
        ticker="NVDA",
        form="10-K",
    )

    assert results[0].chunk == agreed
    assert captured_calls["semantic"]["limit"] == 10
    assert captured_calls["semantic"]["ticker"] == "NVDA"
    assert captured_calls["list"]["form"] == "10-K"
