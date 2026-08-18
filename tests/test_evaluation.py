import json
from types import SimpleNamespace

import pytest

from financial_rag.evaluation import (
    RetrievalCase,
    calculate_case_metrics,
    load_retrieval_cases,
    run_retrieval_evaluation,
)


def test_calculate_case_metrics_for_result_at_rank_two() -> None:
    metrics = calculate_case_metrics(
        retrieved_chunk_ids=["wrong-chunk", "correct-chunk", "another-chunk"],
        accepted_evidence_chunk_ids=("correct-chunk",),
    )

    assert metrics.first_relevant_rank == 2
    assert metrics.hit_at_1 == 0
    assert metrics.hit_at_3 == 1
    assert metrics.hit_at_5 == 1
    assert metrics.reciprocal_rank == 0.5


def test_calculate_case_metrics_when_no_result_is_relevant() -> None:
    metrics = calculate_case_metrics(
        retrieved_chunk_ids=["wrong-chunk"],
        accepted_evidence_chunk_ids=("correct-chunk",),
    )

    assert metrics.first_relevant_rank is None
    assert metrics.hit_at_1 == 0
    assert metrics.hit_at_3 == 0
    assert metrics.hit_at_5 == 0
    assert metrics.reciprocal_rank == 0.0


def test_run_retrieval_evaluation(monkeypatch) -> None:
    cases = [
        RetrievalCase(
            case_id="revenue",
            question="What was revenue?",
            ticker="NVDA",
            form="10-K",
            expected_answer="$215.9 billion",
            category="financial_table",
            accepted_evidence_chunk_ids=("revenue-chunk",),
            notes="Test case.",
        ),
        RetrievalCase(
            case_id="liquidity",
            question="What was liquidity?",
            ticker="NVDA",
            form="10-K",
            expected_answer="$62.6 billion",
            category="financial_table",
            accepted_evidence_chunk_ids=("liquidity-chunk",),
            notes="Test case.",
        ),
    ]
    returned_ids = {
        "What was revenue?": ["wrong-chunk", "revenue-chunk"],
        "What was liquidity?": ["wrong-chunk"],
    }

    def fake_embed_query(question, model):
        return question

    def fake_search_similar_chunks(
        client, query_vector, limit, ticker, form
    ):
        assert client == "fake-client"
        assert limit == 5
        assert ticker == "NVDA"
        assert form == "10-K"
        return [
            SimpleNamespace(chunk=SimpleNamespace(chunk_id=chunk_id))
            for chunk_id in returned_ids[query_vector]
        ]

    monkeypatch.setattr(
        "financial_rag.embeddings.embed_query", fake_embed_query
    )
    monkeypatch.setattr(
        "financial_rag.vectorstore.search_similar_chunks",
        fake_search_similar_chunks,
    )

    summary = run_retrieval_evaluation(
        cases=cases,
        model="fake-model",
        client="fake-client",
        retrieval_limit=5,
    )

    assert summary.results[0].metrics.first_relevant_rank == 2
    assert summary.results[1].metrics.first_relevant_rank is None
    assert summary.hit_at_1 == 0.0
    assert summary.hit_at_3 == 0.5
    assert summary.hit_at_5 == 0.5
    assert summary.mean_reciprocal_rank == 0.25
    assert summary.mean_latency_ms >= 0


def test_load_retrieval_cases(tmp_path) -> None:
    dataset_path = tmp_path / "cases.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "case_id": "nvda_revenue",
                "question": "What was NVIDIA's revenue?",
                "ticker": "nvda",
                "form": "10-k",
                "expected_answer": "$215.9 billion",
                "category": "financial_table",
                "accepted_evidence_chunk_ids": ["chunk-1", "chunk-2"],
                "notes": "Either table is acceptable.",
            }
        ),
        encoding="utf-8",
    )

    cases = load_retrieval_cases(dataset_path)

    assert len(cases) == 1
    assert cases[0].case_id == "nvda_revenue"
    assert cases[0].ticker == "NVDA"
    assert cases[0].form == "10-K"
    assert cases[0].accepted_evidence_chunk_ids == ("chunk-1", "chunk-2")


def test_load_retrieval_cases_rejects_invalid_json(tmp_path) -> None:
    dataset_path = tmp_path / "cases.jsonl"
    dataset_path.write_text("not valid JSON\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Line 1: invalid JSON"):
        load_retrieval_cases(dataset_path)
