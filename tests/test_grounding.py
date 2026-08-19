from pathlib import Path

import pytest

from financial_rag.grounding import (
    build_grounded_context,
    format_grounded_prompt,
    validate_answer_citations,
)
from financial_rag.models import DocumentChunk, FilingMetadata, SearchResult


def make_search_result(
    chunk_id: str,
    content: str,
    score: float,
) -> SearchResult:
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
        chunk_id=chunk_id,
        sequence=1,
        chunk_type="text",
        content=content,
        token_count=len(content.split()),
        source_sequences=(437, 438),
        filing=filing,
    )
    return SearchResult(score=score, chunk=chunk)


def test_build_grounded_context_numbers_unique_ranked_evidence() -> None:
    first = make_search_result(
        "revenue-chunk",
        "Revenue for fiscal year 2026 was $215.9 billion.",
        0.032,
    )
    duplicate = make_search_result(
        "revenue-chunk",
        "Duplicate content should not appear.",
        0.030,
    )
    second = make_search_result(
        "income-chunk",
        "Net income was $120,067 million.",
        0.029,
    )

    context = build_grounded_context(
        "  What was NVIDIA's revenue?  ",
        [first, duplicate, second],
        max_evidence=2,
    )

    assert context.question == "What was NVIDIA's revenue?"
    assert [item.citation_label for item in context.evidence] == ["[1]", "[2]"]
    assert [item.chunk_id for item in context.evidence] == [
        "revenue-chunk",
        "income-chunk",
    ]
    assert context.evidence[0].retrieval_score == 0.032
    assert context.evidence[0].source_sequences == (437, 438)
    assert context.evidence[0].company_name == "NVIDIA CORP"


def test_format_grounded_prompt_includes_rules_and_provenance() -> None:
    context = build_grounded_context(
        "What was NVIDIA's revenue?",
        [
            make_search_result(
                "revenue-chunk",
                "Revenue for fiscal year 2026 was $215.9 billion.",
                0.032,
            )
        ],
    )

    prompt = format_grounded_prompt(context)

    assert "using only the supplied evidence" in prompt
    assert "Question:\nWhat was NVIDIA's revenue?" in prompt
    assert "[1]" in prompt
    assert "Company: NVIDIA CORP (NVDA)" in prompt
    assert "Chunk ID: revenue-chunk" in prompt
    assert "Source elements: 437, 438" in prompt
    assert "SEC source: https://www.sec.gov/Archives/example/nvda.htm" in prompt
    assert "Revenue for fiscal year 2026 was $215.9 billion." in prompt


def test_format_grounded_prompt_handles_no_evidence() -> None:
    context = build_grounded_context("Unknown question?", [])

    prompt = format_grounded_prompt(context)

    assert prompt.endswith("Evidence:\nNo evidence was retrieved.")


def test_build_grounded_context_rejects_empty_question() -> None:
    with pytest.raises(ValueError, match="question must not be empty"):
        build_grounded_context("  ", [])


def test_validate_answer_citations_returns_unique_numbers() -> None:
    context = build_grounded_context(
        "What were revenue and net income?",
        [
            make_search_result("revenue-chunk", "Revenue evidence.", 0.032),
            make_search_result("income-chunk", "Income evidence.", 0.031),
        ],
    )

    cited_numbers = validate_answer_citations(
        "Revenue was reported [1], and net income was reported [2] [1].",
        context,
    )

    assert cited_numbers == (1, 2)


def test_validate_answer_citations_rejects_unavailable_number() -> None:
    context = build_grounded_context(
        "What was revenue?",
        [make_search_result("revenue-chunk", "Revenue evidence.", 0.032)],
    )

    with pytest.raises(ValueError, match=r"unavailable citations: \[2\]"):
        validate_answer_citations("Revenue was reported [2].", context)


def test_validate_answer_citations_requires_a_citation_when_evidence_exists() -> None:
    context = build_grounded_context(
        "What was revenue?",
        [make_search_result("revenue-chunk", "Revenue evidence.", 0.032)],
    )

    with pytest.raises(ValueError, match="must cite at least one"):
        validate_answer_citations("Revenue was reported.", context)


def test_validate_answer_citations_allows_uncited_no_evidence_response() -> None:
    context = build_grounded_context("Unknown question?", [])

    cited_numbers = validate_answer_citations(
        "I cannot answer from the supplied evidence.",
        context,
    )

    assert cited_numbers == ()
