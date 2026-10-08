"""Grounding and citation context for evidence-based answers."""

import re
from dataclasses import dataclass

from financial_rag.models import SearchResult


ABSTENTION_ANSWER = "I cannot answer from the supplied evidence."
GROUNDING_INSTRUCTIONS = """Answer the question using only the supplied evidence.
Cite each factual claim with its evidence number, such as [1].
Do not cite an evidence number that is not supplied.
If the evidence is insufficient, respond exactly with this sentence and no citation:
I cannot answer from the supplied evidence."""


@dataclass(frozen=True)
class GroundedEvidence:
    """One numbered evidence item with complete filing provenance."""

    citation_number: int
    text: str
    chunk_id: str
    retrieval_score: float
    source_sequences: tuple[int, ...]
    ticker: str
    company_name: str
    form: str
    filing_date: str
    sec_url: str

    @property
    def citation_label(self) -> str:
        """Return the bracketed citation label shown to an answer model."""
        return f"[{self.citation_number}]"


@dataclass(frozen=True)
class GroundedContext:
    """A user question paired with numbered SEC filing evidence."""

    question: str
    evidence: tuple[GroundedEvidence, ...]


def build_grounded_context(
    question: str,
    results: list[SearchResult],
    max_evidence: int = 5,
) -> GroundedContext:
    """Convert ranked search results into unique, numbered evidence."""
    normalized_question = question.strip()
    if not normalized_question:
        raise ValueError("question must not be empty")
    if max_evidence <= 0:
        raise ValueError("max_evidence must be positive")

    evidence = []
    seen_chunk_ids = set()
    for result in results:
        chunk = result.chunk
        if chunk.chunk_id in seen_chunk_ids:
            continue
        evidence.append(
            GroundedEvidence(
                citation_number=len(evidence) + 1,
                text=chunk.content,
                chunk_id=chunk.chunk_id,
                retrieval_score=result.score,
                source_sequences=chunk.source_sequences,
                ticker=chunk.filing.ticker,
                company_name=chunk.filing.company_name,
                form=chunk.filing.form,
                filing_date=chunk.filing.filing_date,
                sec_url=chunk.filing.source_url,
            )
        )
        seen_chunk_ids.add(chunk.chunk_id)
        if len(evidence) == max_evidence:
            break

    return GroundedContext(
        question=normalized_question,
        evidence=tuple(evidence),
    )


def format_grounded_prompt(context: GroundedContext) -> str:
    """Format a strict evidence-only prompt for any future answer model."""
    lines = [
        GROUNDING_INSTRUCTIONS,
        "",
        "Question:",
        context.question,
        "",
        "Evidence:",
    ]
    if not context.evidence:
        lines.append("No evidence was retrieved.")
        return "\n".join(lines)

    for item in context.evidence:
        lines.extend(
            [
                "",
                item.citation_label,
                f"Company: {item.company_name} ({item.ticker})",
                f"Filing: {item.form}, filed {item.filing_date}",
                f"Chunk ID: {item.chunk_id}",
                "Source elements: "
                + ", ".join(str(value) for value in item.source_sequences),
                f"SEC source: {item.sec_url}",
                "Text:",
                item.text,
            ]
        )

    return "\n".join(lines)


def validate_answer_citations(
    answer: str,
    context: GroundedContext,
) -> tuple[int, ...]:
    """Validate answer citation labels and return unique cited numbers."""
    if not answer.strip():
        raise ValueError("answer must not be empty")

    cited_numbers = extract_answer_citations(answer)
    allowed_numbers = {
        item.citation_number for item in context.evidence
    }
    invalid_numbers = tuple(
        number for number in cited_numbers if number not in allowed_numbers
    )
    if invalid_numbers:
        labels = ", ".join(f"[{number}]" for number in invalid_numbers)
        raise ValueError(f"Answer contains unavailable citations: {labels}")
    if (
        context.evidence
        and not cited_numbers
        and not is_abstention_answer(answer)
    ):
        raise ValueError("Answer must cite at least one supplied evidence item")

    return cited_numbers


def extract_answer_citations(answer: str) -> tuple[int, ...]:
    """Return unique bracketed citation numbers without validating them."""
    return tuple(
        dict.fromkeys(
            int(value) for value in re.findall(r"\[(\d+)\]", answer)
        )
    )


def is_abstention_answer(answer: str) -> bool:
    """Return whether text is exactly the canonical evidence abstention."""
    normalized_answer = " ".join(answer.casefold().split()).rstrip(".")
    normalized_abstention = ABSTENTION_ANSWER.casefold().rstrip(".")
    return normalized_answer == normalized_abstention
