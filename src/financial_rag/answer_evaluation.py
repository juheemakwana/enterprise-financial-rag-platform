"""Deterministic quality checks for generated grounded answers."""

from dataclasses import dataclass
from statistics import mean
from time import perf_counter

from qdrant_client import QdrantClient

from financial_rag.embeddings import EmbeddingModel
from financial_rag.evaluation import RetrievalCase, evidence_matches_case
from financial_rag.generation import (
    AnswerModel,
    AnswerValidationError,
    GroundedAnswer,
)
from financial_rag.grounding import is_abstention_answer


@dataclass(frozen=True)
class AnswerCaseMetrics:
    """Binary quality scores for one generated answer."""

    answer_correct: int
    abstained: int
    citation_present: int
    citation_correct: int
    grounded_pass: int


@dataclass(frozen=True)
class AnswerEvaluationResult:
    """Answer, audit details, and metrics for one evaluation case."""

    case_id: str
    question: str
    retrieved_chunk_ids: tuple[str, ...]
    answer_text: str
    cited_numbers: tuple[int, ...]
    metrics: AnswerCaseMetrics
    latency_ms: float
    error: str | None


@dataclass(frozen=True)
class AnswerEvaluationSummary:
    """Individual answer results and average quality metrics."""

    results: tuple[AnswerEvaluationResult, ...]
    answer_accuracy: float
    abstention_rate: float
    citation_presence_rate: float
    citation_correctness_rate: float
    grounded_pass_rate: float
    mean_latency_ms: float


def calculate_answer_case_metrics(
    answer: GroundedAnswer,
    case: RetrievalCase,
) -> AnswerCaseMetrics:
    """Score answer text and its cited evidence against one saved case."""
    answer_correct = _answer_text_is_correct(answer.text, case)
    abstained = int(is_abstention_answer(answer.text))

    citation_present = int(bool(answer.cited_numbers))
    cited_evidence = tuple(
        evidence
        for evidence in answer.context.evidence
        if evidence.citation_number in answer.cited_numbers
    )
    citation_correct = int(
        bool(cited_evidence)
        and len(cited_evidence) == len(answer.cited_numbers)
        and all(
            evidence_matches_case(
                content=evidence.text,
                source_sequences=evidence.source_sequences,
                case=case,
            )
            for evidence in cited_evidence
        )
    )
    grounded_pass = int(bool(answer_correct and citation_correct))

    return AnswerCaseMetrics(
        answer_correct=answer_correct,
        abstained=abstained,
        citation_present=citation_present,
        citation_correct=citation_correct,
        grounded_pass=grounded_pass,
    )


def _answer_text_is_correct(answer_text: str, case: RetrievalCase) -> int:
    """Return whether answer text contains one accepted phrase group."""
    normalized_answer = " ".join(answer_text.casefold().split())
    return int(
        any(
            all(
                " ".join(phrase.casefold().split()) in normalized_answer
                for phrase in phrase_group
            )
            for phrase_group in case.accepted_answer_phrase_groups
        )
    )


def run_answer_evaluation(
    cases: list[RetrievalCase],
    embedding_model: EmbeddingModel,
    client: QdrantClient,
    answer_model: AnswerModel,
    retrieval_limit: int = 5,
) -> AnswerEvaluationSummary:
    """Retrieve, generate, and score an answer for every saved case."""
    if not cases:
        raise ValueError("At least one answer evaluation case is required")
    if retrieval_limit <= 0:
        raise ValueError("retrieval_limit must be positive")

    missing_answer_rules = [
        case.case_id
        for case in cases
        if not case.accepted_answer_phrase_groups
    ]
    if missing_answer_rules:
        missing_ids = ", ".join(missing_answer_rules)
        raise ValueError(
            f"Answer evaluation cases need accepted answer phrases: {missing_ids}"
        )

    from financial_rag.embeddings import embed_query
    from financial_rag.generation import generate_grounded_answer
    from financial_rag.retrieval import hybrid_search_chunks

    results = []
    zero_metrics = AnswerCaseMetrics(
        answer_correct=0,
        abstained=0,
        citation_present=0,
        citation_correct=0,
        grounded_pass=0,
    )

    for case in cases:
        started_at = perf_counter()
        query_vector = embed_query(case.question, embedding_model)
        search_results = hybrid_search_chunks(
            client=client,
            question=case.question,
            query_vector=query_vector,
            limit=retrieval_limit,
            ticker=case.ticker,
            form=case.form,
        )
        retrieved_chunk_ids = tuple(
            result.chunk.chunk_id for result in search_results
        )

        try:
            answer = generate_grounded_answer(
                question=case.question,
                results=search_results,
                model=answer_model,
                max_evidence=retrieval_limit,
            )
            metrics = calculate_answer_case_metrics(answer, case)
            answer_text = answer.text
            cited_numbers = answer.cited_numbers
            error = None
        except AnswerValidationError as generation_error:
            answer_text = generation_error.answer_text
            cited_numbers = generation_error.cited_numbers
            metrics = AnswerCaseMetrics(
                answer_correct=_answer_text_is_correct(answer_text, case),
                abstained=0,
                citation_present=int(bool(cited_numbers)),
                citation_correct=0,
                grounded_pass=0,
            )
            error = str(generation_error)
        except (RuntimeError, ValueError) as generation_error:
            metrics = zero_metrics
            answer_text = ""
            cited_numbers = ()
            error = str(generation_error)

        latency_ms = (perf_counter() - started_at) * 1_000
        results.append(
            AnswerEvaluationResult(
                case_id=case.case_id,
                question=case.question,
                retrieved_chunk_ids=retrieved_chunk_ids,
                answer_text=answer_text,
                cited_numbers=cited_numbers,
                metrics=metrics,
                latency_ms=latency_ms,
                error=error,
            )
        )

    return AnswerEvaluationSummary(
        results=tuple(results),
        answer_accuracy=mean(
            result.metrics.answer_correct for result in results
        ),
        abstention_rate=mean(
            result.metrics.abstained for result in results
        ),
        citation_presence_rate=mean(
            result.metrics.citation_present for result in results
        ),
        citation_correctness_rate=mean(
            result.metrics.citation_correct for result in results
        ),
        grounded_pass_rate=mean(
            result.metrics.grounded_pass for result in results
        ),
        mean_latency_ms=mean(result.latency_ms for result in results),
    )
