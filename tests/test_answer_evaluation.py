from dataclasses import replace
from pathlib import Path

from financial_rag.answer_evaluation import calculate_answer_case_metrics
from financial_rag.answer_evaluation import run_answer_evaluation
from financial_rag.evaluation import RetrievalCase
from financial_rag.generation import GroundedAnswer, NO_EVIDENCE_ANSWER
from financial_rag.grounding import GroundedContext, GroundedEvidence
from financial_rag.models import (
    DocumentChunk,
    FilingMetadata,
    SearchResult,
)


def make_revenue_case() -> RetrievalCase:
    return RetrievalCase(
        case_id="nvda_revenue",
        question="What was NVIDIA's revenue?",
        ticker="NVDA",
        form="10-K",
        expected_answer="$215.9 billion",
        category="table_and_narrative",
        accepted_source_sequences=(436, 437),
        accepted_evidence_phrase_groups=(
            ("revenue", "215,938"),
            ("revenue", "215.9 billion"),
        ),
        notes="Accept exact or rounded revenue.",
        accepted_answer_phrase_groups=(
            ("215.9 billion",),
            ("215,938 million",),
        ),
    )


def make_answer(
    text: str,
    evidence_text: str,
    source_sequences: tuple[int, ...],
) -> GroundedAnswer:
    evidence = GroundedEvidence(
        citation_number=1,
        text=evidence_text,
        chunk_id="revenue-chunk",
        retrieval_score=0.032,
        source_sequences=source_sequences,
        ticker="NVDA",
        company_name="NVIDIA CORP",
        form="10-K",
        filing_date="2026-02-25",
        sec_url="https://www.sec.gov/Archives/example/nvda.htm",
    )
    return GroundedAnswer(
        text=text,
        cited_numbers=(1,),
        context=GroundedContext(
            question="What was NVIDIA's revenue?",
            evidence=(evidence,),
        ),
    )


def test_answer_metrics_pass_for_correct_answer_and_citation() -> None:
    answer = make_answer(
        text="NVIDIA's revenue was $215.9 billion [1].",
        evidence_text="Revenue for fiscal 2026 was $215.9 billion.",
        source_sequences=(437,),
    )

    metrics = calculate_answer_case_metrics(answer, make_revenue_case())

    assert metrics.answer_correct == 1
    assert metrics.abstained == 0
    assert metrics.citation_present == 1
    assert metrics.citation_correct == 1
    assert metrics.grounded_pass == 1


def test_answer_metrics_fail_grounding_for_unrelated_citation() -> None:
    answer = make_answer(
        text="NVIDIA's revenue was $215.9 billion [1].",
        evidence_text="This chunk discusses a different financial topic.",
        source_sequences=(999,),
    )

    metrics = calculate_answer_case_metrics(answer, make_revenue_case())

    assert metrics.answer_correct == 1
    assert metrics.abstained == 0
    assert metrics.citation_present == 1
    assert metrics.citation_correct == 0
    assert metrics.grounded_pass == 0


def test_answer_metrics_record_safe_abstention() -> None:
    answer = make_answer(
        text=NO_EVIDENCE_ANSWER,
        evidence_text="This chunk does not contain the requested fact.",
        source_sequences=(999,),
    )
    answer = replace(answer, cited_numbers=())

    metrics = calculate_answer_case_metrics(answer, make_revenue_case())

    assert metrics.answer_correct == 0
    assert metrics.abstained == 1
    assert metrics.citation_present == 0
    assert metrics.citation_correct == 0
    assert metrics.grounded_pass == 0


def test_run_answer_evaluation_continues_after_invalid_citation(
    monkeypatch,
) -> None:
    valid_case = make_revenue_case()
    invalid_case = replace(
        valid_case,
        case_id="invalid_citation",
        question="Return an invalid citation.",
    )
    filing = FilingMetadata(
        ticker="NVDA",
        company_name="NVIDIA CORP",
        cik="0001045810",
        form="10-K",
        filing_date="2026-02-25",
        accession_number="0001045810-26-000021",
        source_url="https://www.sec.gov/Archives/example/nvda.htm",
        local_path=Path("data/raw/NVDA_10-K_2026-02-25.html"),
    )
    chunk = DocumentChunk(
        chunk_id="revenue-chunk",
        sequence=1,
        chunk_type="text",
        content="Revenue for fiscal 2026 was $215.9 billion.",
        token_count=10,
        source_sequences=(437,),
        filing=filing,
    )

    monkeypatch.setattr(
        "financial_rag.embeddings.embed_query",
        lambda question, model: (0.0,),
    )
    monkeypatch.setattr(
        "financial_rag.retrieval.hybrid_search_chunks",
        lambda **kwargs: [SearchResult(score=0.032, chunk=chunk)],
    )

    class FakeAnswerModel:
        def generate(self, prompt: str) -> str:
            if "Return an invalid citation." in prompt:
                return "Revenue was $215.9 billion [2]."
            return "Revenue was $215.9 billion [1]."

    summary = run_answer_evaluation(
        cases=[valid_case, invalid_case],
        embedding_model="fake-embedding-model",
        client="fake-client",
        answer_model=FakeAnswerModel(),
        retrieval_limit=1,
    )

    assert len(summary.results) == 2
    assert summary.results[0].metrics.grounded_pass == 1
    assert summary.results[0].error is None
    assert summary.results[1].metrics.grounded_pass == 0
    assert summary.results[1].answer_text == (
        "Revenue was $215.9 billion [2]."
    )
    assert summary.results[1].cited_numbers == (2,)
    assert "unavailable citations: [2]" in summary.results[1].error
    assert summary.answer_accuracy == 1.0
    assert summary.abstention_rate == 0.0
    assert summary.citation_presence_rate == 1.0
    assert summary.citation_correctness_rate == 0.5
    assert summary.grounded_pass_rate == 0.5
    assert summary.mean_latency_ms >= 0


def test_run_answer_evaluation_uses_optional_reranker(monkeypatch) -> None:
    case = make_revenue_case()
    filing = FilingMetadata(
        ticker="NVDA",
        company_name="NVIDIA CORP",
        cik="0001045810",
        form="10-K",
        filing_date="2026-02-25",
        accession_number="0001045810-26-000021",
        source_url="https://www.sec.gov/Archives/example/nvda.htm",
        local_path=Path("data/raw/NVDA_10-K_2026-02-25.html"),
    )
    chunk = DocumentChunk(
        chunk_id="revenue-chunk",
        sequence=1,
        chunk_type="text",
        content="Revenue for fiscal 2026 was $215.9 billion.",
        token_count=10,
        source_sequences=(437,),
        filing=filing,
    )
    reranker = object()
    captured_call = {}

    monkeypatch.setattr(
        "financial_rag.embeddings.embed_query",
        lambda question, model: (0.0,),
    )

    def fake_reranked_search(**kwargs):
        captured_call.update(kwargs)
        return [SearchResult(score=8.0, chunk=chunk)]

    monkeypatch.setattr(
        "financial_rag.reranking.hybrid_search_and_rerank",
        fake_reranked_search,
    )

    class FakeAnswerModel:
        def generate(self, prompt: str) -> str:
            return "Revenue was $215.9 billion [1]."

    summary = run_answer_evaluation(
        cases=[case],
        embedding_model="fake-embedding-model",
        client="fake-client",
        answer_model=FakeAnswerModel(),
        retrieval_limit=5,
        reranker=reranker,
    )

    assert summary.grounded_pass_rate == 1.0
    assert captured_call["scorer"] is reranker
    assert captured_call["limit"] == 5
    assert captured_call["ticker"] == "NVDA"
    assert captured_call["form"] == "10-K"
