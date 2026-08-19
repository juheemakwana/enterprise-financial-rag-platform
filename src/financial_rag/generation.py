"""Provider-independent grounded answer generation."""

from dataclasses import dataclass
from typing import Protocol

from financial_rag.grounding import (
    GroundedContext,
    build_grounded_context,
    format_grounded_prompt,
    validate_answer_citations,
)
from financial_rag.models import SearchResult


NO_EVIDENCE_ANSWER = "I cannot answer from the supplied evidence."


class AnswerModel(Protocol):
    """Small interface required from any local or API answer model."""

    def generate(self, prompt: str) -> str: ...


@dataclass(frozen=True)
class GroundedAnswer:
    """A validated answer paired with its complete grounding context."""

    text: str
    cited_numbers: tuple[int, ...]
    context: GroundedContext


def generate_grounded_answer(
    question: str,
    results: list[SearchResult],
    model: AnswerModel,
    max_evidence: int = 5,
) -> GroundedAnswer:
    """Generate an evidence-only answer and validate all citations."""
    context = build_grounded_context(
        question=question,
        results=results,
        max_evidence=max_evidence,
    )
    if not context.evidence:
        return GroundedAnswer(
            text=NO_EVIDENCE_ANSWER,
            cited_numbers=(),
            context=context,
        )

    prompt = format_grounded_prompt(context)
    answer_text = model.generate(prompt).strip()
    cited_numbers = validate_answer_citations(answer_text, context)
    return GroundedAnswer(
        text=answer_text,
        cited_numbers=cited_numbers,
        context=context,
    )
