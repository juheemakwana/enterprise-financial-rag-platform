from pathlib import Path

from financial_rag.models import DocumentChunk, FilingMetadata, SearchResult
from financial_rag.reranking import (
    DEFAULT_RERANK_CANDIDATE_LIMIT,
    DEFAULT_RERANKER_MAX_LENGTH,
    DEFAULT_RERANKER_MODEL,
    hybrid_search_and_rerank,
    load_reranker,
    rerank_search_results,
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


class FakeScorer:
    def predict(self, sentences: list[tuple[str, str]]) -> list[float]:
        assert sentences == [
            (
                "What was NVIDIA's net income?",
                "NVIDIA opened a new office.",
            ),
            (
                "What was NVIDIA's net income?",
                "NVIDIA's net income was $120,067 million.",
            ),
        ]
        return [-4.0, 8.0]


def test_load_reranker_uses_cached_model_and_mps(monkeypatch) -> None:
    captured_call = {}
    expected_model = FakeScorer()

    def fake_cross_encoder(model_name: str, **kwargs: object) -> FakeScorer:
        captured_call["model_name"] = model_name
        captured_call.update(kwargs)
        return expected_model

    monkeypatch.setattr(
        "financial_rag.reranking.torch.backends.mps.is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "financial_rag.reranking.CrossEncoder",
        fake_cross_encoder,
    )

    model = load_reranker()

    assert model is expected_model
    assert captured_call == {
        "model_name": DEFAULT_RERANKER_MODEL,
        "device": "mps",
        "max_length": DEFAULT_RERANKER_MAX_LENGTH,
        "local_files_only": True,
    }


def test_rerank_search_results_promotes_the_more_relevant_chunk() -> None:
    unrelated = make_chunk("chunk-1", "NVIDIA opened a new office.")
    correct = make_chunk(
        "chunk-2",
        "NVIDIA's net income was $120,067 million.",
    )
    initial_results = [
        SearchResult(score=0.04, chunk=unrelated),
        SearchResult(score=0.03, chunk=correct),
    ]

    reranked = rerank_search_results(
        question="What was NVIDIA's net income?",
        results=initial_results,
        scorer=FakeScorer(),
        limit=1,
    )

    assert reranked == [SearchResult(score=8.0, chunk=correct)]


def test_reranking_prefers_unit_context_for_duplicate_table_rows() -> None:
    unitless = make_chunk(
        "chunk-1",
        "Table context:\nYear Ended 2026\n\n"
        "Data row:\nNet income | $120,067",
    )
    unit_preserving = make_chunk(
        "chunk-2",
        "Table context:\nYear Ended 2026 | (In millions)\n\n"
        "Data row:\nNet income | $120,067",
    )

    class RecordingScorer:
        def __init__(self) -> None:
            self.pairs = []

        def predict(self, sentences):
            self.pairs = sentences
            return [7.0]

    scorer = RecordingScorer()
    reranked = rerank_search_results(
        question="What was net income?",
        results=[
            SearchResult(score=0.04, chunk=unitless),
            SearchResult(score=0.03, chunk=unit_preserving),
        ],
        scorer=scorer,
        limit=1,
    )

    assert reranked == [SearchResult(score=7.0, chunk=unit_preserving)]
    assert scorer.pairs == [
        ("What was net income?", unit_preserving.content)
    ]


def test_hybrid_search_and_rerank_uses_a_broad_candidate_set(
    monkeypatch,
) -> None:
    unrelated = make_chunk("chunk-1", "NVIDIA opened a new office.")
    correct = make_chunk(
        "chunk-2",
        "NVIDIA's net income was $120,067 million.",
    )
    captured_call = {}

    def fake_hybrid_search(**kwargs):
        captured_call.update(kwargs)
        return [
            SearchResult(score=0.04, chunk=unrelated),
            SearchResult(score=0.03, chunk=correct),
        ]

    monkeypatch.setattr(
        "financial_rag.reranking.hybrid_search_chunks",
        fake_hybrid_search,
    )

    results = hybrid_search_and_rerank(
        client="fake-client",
        question="What was NVIDIA's net income?",
        query_vector=tuple([0.0] * 384),
        scorer=FakeScorer(),
        limit=1,
        ticker="NVDA",
        form="10-K",
    )

    assert results == [SearchResult(score=8.0, chunk=correct)]
    assert captured_call["limit"] == DEFAULT_RERANK_CANDIDATE_LIMIT
    assert captured_call["candidate_limit"] == 100
    assert captured_call["ticker"] == "NVDA"
    assert captured_call["form"] == "10-K"
