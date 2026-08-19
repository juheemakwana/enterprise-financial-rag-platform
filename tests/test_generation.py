from pathlib import Path

import pytest

from financial_rag.generation import (
    NO_EVIDENCE_ANSWER,
    generate_grounded_answer,
)
from financial_rag.models import DocumentChunk, FilingMetadata, SearchResult


class FakeAnswerModel:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.prompts = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answer


def make_search_result() -> SearchResult:
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
        content="Revenue for fiscal year 2026 was $215.9 billion.",
        token_count=10,
        source_sequences=(437,),
        filing=filing,
    )
    return SearchResult(score=0.032, chunk=chunk)


def test_generate_grounded_answer_calls_model_and_validates_answer() -> None:
    model = FakeAnswerModel(
        "NVIDIA's fiscal 2026 revenue was $215.9 billion [1]."
    )

    answer = generate_grounded_answer(
        question="What was NVIDIA's revenue in fiscal 2026?",
        results=[make_search_result()],
        model=model,
    )

    assert answer.text == (
        "NVIDIA's fiscal 2026 revenue was $215.9 billion [1]."
    )
    assert answer.cited_numbers == (1,)
    assert answer.context.evidence[0].chunk_id == "revenue-chunk"
    assert len(model.prompts) == 1
    assert "using only the supplied evidence" in model.prompts[0]
    assert "Revenue for fiscal year 2026 was $215.9 billion." in model.prompts[0]


def test_generate_grounded_answer_rejects_model_citation_not_in_context() -> None:
    model = FakeAnswerModel("Revenue was $215.9 billion [2].")

    with pytest.raises(ValueError, match=r"unavailable citations: \[2\]"):
        generate_grounded_answer(
            question="What was NVIDIA's revenue?",
            results=[make_search_result()],
            model=model,
        )


def test_generate_grounded_answer_abstains_without_calling_model() -> None:
    model = FakeAnswerModel("This answer must not be used.")

    answer = generate_grounded_answer(
        question="What was an unavailable fact?",
        results=[],
        model=model,
    )

    assert answer.text == NO_EVIDENCE_ANSWER
    assert answer.cited_numbers == ()
    assert answer.context.evidence == ()
    assert model.prompts == []
