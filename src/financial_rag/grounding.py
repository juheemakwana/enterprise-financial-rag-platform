"""Grounding and citation context for evidence-based answers."""

from dataclasses import dataclass

from financial_rag.models import SearchResult


GROUNDING_INSTRUCTIONS = """Answer the question using only the supplied evidence.
Cite each factual claim with its evidence number, such as [1].
Do not cite an evidence number that is not supplied.
If the evidence is insufficient, say that you cannot answer from the supplied evidence."""


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
