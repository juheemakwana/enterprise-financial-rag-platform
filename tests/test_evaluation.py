import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from financial_rag.evaluation import (
    RetrievalCase,
    calculate_case_metrics,
    load_retrieval_cases,
    run_retrieval_evaluation,
)


def test_calculate_case_metrics_for_result_at_rank_two() -> None:
    case = RetrievalCase(
        case_id="revenue",
        question="What was revenue?",
        ticker="NVDA",
        form="10-K",
        expected_answer="$215.9 billion",
        category="financial_table",
        accepted_source_sequences=(436,),
        accepted_evidence_phrase_groups=(("revenue", "215,938"),),
        notes="Test case.",
    )
    metrics = calculate_case_metrics(
        retrieved_chunks=[
            SimpleNamespace(
                source_sequences=(999,),
                content="Revenue was 215,938.",
            ),
            SimpleNamespace(
                source_sequences=(436,),
                content="Revenue was 215,938.",
            ),
        ],
        case=case,
    )

    assert metrics.first_relevant_rank == 2
    assert metrics.hit_at_1 == 0
    assert metrics.hit_at_3 == 1
    assert metrics.hit_at_5 == 1
    assert metrics.reciprocal_rank == 0.5


def test_calculate_case_metrics_when_no_result_is_relevant() -> None:
    case = RetrievalCase(
        case_id="revenue",
        question="What was revenue?",
        ticker="NVDA",
        form="10-K",
        expected_answer="$215.9 billion",
        category="financial_table",
        accepted_source_sequences=(436,),
        accepted_evidence_phrase_groups=(("revenue", "215,938"),),
        notes="Test case.",
    )
    metrics = calculate_case_metrics(
        retrieved_chunks=[
            SimpleNamespace(
                source_sequences=(436,),
                content="Revenue amount is missing.",
            )
        ],
        case=case,
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
            accepted_source_sequences=(436,),
            accepted_evidence_phrase_groups=(("revenue", "215.9 billion"),),
            notes="Test case.",
        ),
        RetrievalCase(
            case_id="liquidity",
            question="What was liquidity?",
            ticker="NVDA",
            form="10-K",
            expected_answer="$62.6 billion",
            category="financial_table",
            accepted_source_sequences=(498,),
            accepted_evidence_phrase_groups=(("cash", "62.6 billion"),),
            notes="Test case.",
        ),
    ]
    returned_chunks = {
        "What was revenue?": [
            SimpleNamespace(
                chunk_id="wrong-chunk",
                source_sequences=(999,),
                content="Unrelated revenue.",
            ),
            SimpleNamespace(
                chunk_id="revenue-chunk",
                source_sequences=(436,),
                content="Revenue was $215.9 billion.",
            ),
        ],
        "What was liquidity?": [
            SimpleNamespace(
                chunk_id="wrong-chunk",
                source_sequences=(999,),
                content="Unrelated cash.",
            )
        ],
    }

    def fake_embed_query(question, model):
        return question

    def fake_hybrid_search_chunks(
        client, question, query_vector, limit, ticker, form
    ):
        assert client == "fake-client"
        assert question == query_vector
        assert limit == 5
        assert ticker == "NVDA"
        assert form == "10-K"
        return [
            SimpleNamespace(chunk=chunk)
            for chunk in returned_chunks[query_vector]
        ]

    monkeypatch.setattr(
        "financial_rag.embeddings.embed_query", fake_embed_query
    )
    monkeypatch.setattr(
        "financial_rag.retrieval.hybrid_search_chunks",
        fake_hybrid_search_chunks,
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
                "accepted_answer_phrase_groups": [
                    ["215.9 billion"],
                    ["215,938 million"],
                ],
                "category": "financial_table",
                "accepted_source_sequences": [436, 609],
                "accepted_evidence_phrase_groups": [
                    ["revenue", "215,938"],
                    ["revenue", "215.9 billion"],
                ],
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
    assert cases[0].accepted_source_sequences == (436, 609)
    assert cases[0].accepted_evidence_phrase_groups == (
        ("revenue", "215,938"),
        ("revenue", "215.9 billion"),
    )
    assert cases[0].accepted_answer_phrase_groups == (
        ("215.9 billion",),
        ("215,938 million",),
    )


def test_load_retrieval_cases_rejects_invalid_json(tmp_path) -> None:
    dataset_path = tmp_path / "cases.jsonl"
    dataset_path.write_text("not valid JSON\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Line 1: invalid JSON"):
        load_retrieval_cases(dataset_path)


def test_saved_cases_all_define_accepted_answer_phrases() -> None:
    cases = load_retrieval_cases(
        Path("evals/datasets/retrieval_cases.jsonl")
    )

    assert len(cases) == 6
    assert all(case.accepted_answer_phrase_groups for case in cases)
    phrases_by_case = {
        case.case_id: case.accepted_answer_phrase_groups
        for case in cases
    }
    assert phrases_by_case["nvda_fy2026_net_income"] == (
        ("120.067 billion",),
        ("120,067 million",),
    )
    assert phrases_by_case["nvda_2026_liquidity"] == (
        ("62.556 billion",),
        ("62,556 million",),
        ("62.6 billion",),
    )
