"""Provider-independent grounded answer generation."""

from dataclasses import dataclass
from typing import Protocol

from financial_rag.grounding import (
    ABSTENTION_ANSWER,
    GroundedContext,
    build_grounded_context,
    extract_answer_citations,
    format_grounded_prompt,
    validate_answer_citations,
)
from financial_rag.models import SearchResult


NO_EVIDENCE_ANSWER = ABSTENTION_ANSWER


class AnswerModel(Protocol):
    """Small interface required from any local or API answer model."""

    def generate(self, prompt: str) -> str: ...


class AnswerValidationError(ValueError):
    """A rejected model answer retained for evaluation and debugging."""

    def __init__(self, message: str, answer_text: str) -> None:
        super().__init__(message)
        self.answer_text = answer_text
        self.cited_numbers = extract_answer_citations(answer_text)


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
    try:
        cited_numbers = validate_answer_citations(answer_text, context)
    except ValueError as error:
        raise AnswerValidationError(str(error), answer_text) from error
    return GroundedAnswer(
        text=answer_text,
        cited_numbers=cited_numbers,
        context=context,
    )
