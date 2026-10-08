"""Dataset tools for evaluating SEC filing retrieval."""

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter

from qdrant_client import QdrantClient

from financial_rag.embeddings import EmbeddingModel
from financial_rag.models import DocumentChunk


@dataclass(frozen=True)
class RetrievalCase:
    """One question with stable rules describing accepted filing evidence."""

    case_id: str
    question: str
    ticker: str
    form: str
    expected_answer: str
    category: str
    accepted_source_sequences: tuple[int, ...]
    accepted_evidence_phrase_groups: tuple[tuple[str, ...], ...]
    notes: str
    accepted_answer_phrase_groups: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True)
class RetrievalCaseMetrics:
    """Retrieval scores calculated for one evaluation question."""

    first_relevant_rank: int | None
    hit_at_1: int
    hit_at_3: int
    hit_at_5: int
    reciprocal_rank: float


@dataclass(frozen=True)
class RetrievalEvaluationResult:
    """Metrics and retrieved chunk IDs for one evaluation case."""

    case_id: str
    question: str
    retrieved_chunk_ids: tuple[str, ...]
    metrics: RetrievalCaseMetrics
    latency_ms: float


@dataclass(frozen=True)
class RetrievalEvaluationSummary:
    """Individual results and average metrics for one evaluation run."""

    results: tuple[RetrievalEvaluationResult, ...]
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    mean_reciprocal_rank: float
    mean_latency_ms: float


def calculate_case_metrics(
    retrieved_chunks: list[DocumentChunk],
    case: RetrievalCase,
) -> RetrievalCaseMetrics:
    """Calculate rank-based metrics for one retrieval case."""
    first_relevant_rank = next(
        (
            rank
            for rank, chunk in enumerate(retrieved_chunks, start=1)
            if chunk_matches_case(chunk, case)
        ),
        None,
    )

    def hit_at(k: int) -> int:
        return int(
            first_relevant_rank is not None and first_relevant_rank <= k
        )

    reciprocal_rank = (
        0.0 if first_relevant_rank is None else 1 / first_relevant_rank
    )
    return RetrievalCaseMetrics(
        first_relevant_rank=first_relevant_rank,
        hit_at_1=hit_at(1),
        hit_at_3=hit_at(3),
        hit_at_5=hit_at(5),
        reciprocal_rank=reciprocal_rank,
    )


def chunk_matches_case(chunk: DocumentChunk, case: RetrievalCase) -> bool:
    """Return whether a chunk contains accepted source-backed evidence."""
    return evidence_matches_case(
        content=chunk.content,
        source_sequences=chunk.source_sequences,
        case=case,
    )


def evidence_matches_case(
    content: str,
    source_sequences: tuple[int, ...],
    case: RetrievalCase,
) -> bool:
    """Return whether text and source sequences match accepted evidence."""
    accepted_sources = set(case.accepted_source_sequences)
    if not accepted_sources.intersection(source_sequences):
        return False

    normalized_content = " ".join(content.casefold().split())
    return any(
        all(
            " ".join(phrase.casefold().split()) in normalized_content
            for phrase in phrase_group
        )
        for phrase_group in case.accepted_evidence_phrase_groups
    )


def run_retrieval_evaluation(
    cases: list[RetrievalCase],
    model: EmbeddingModel,
    client: QdrantClient,
    retrieval_limit: int = 50,
) -> RetrievalEvaluationSummary:
    """Search for every case and summarize rank-based retrieval quality."""
    if not cases:
        raise ValueError("At least one retrieval case is required")
    if retrieval_limit <= 0:
        raise ValueError("retrieval_limit must be positive")

    from financial_rag.embeddings import embed_query
    from financial_rag.retrieval import hybrid_search_chunks

    results = []
    for case in cases:
        started_at = perf_counter()
        query_vector = embed_query(case.question, model)
        search_results = hybrid_search_chunks(
            client=client,
            question=case.question,
            query_vector=query_vector,
            limit=retrieval_limit,
            ticker=case.ticker,
            form=case.form,
        )
        latency_ms = (perf_counter() - started_at) * 1_000
        retrieved_chunk_ids = tuple(
            result.chunk.chunk_id for result in search_results
        )
        metrics = calculate_case_metrics(
            retrieved_chunks=[result.chunk for result in search_results],
            case=case,
        )
        results.append(
            RetrievalEvaluationResult(
                case_id=case.case_id,
                question=case.question,
                retrieved_chunk_ids=retrieved_chunk_ids,
                metrics=metrics,
                latency_ms=latency_ms,
            )
        )

    return RetrievalEvaluationSummary(
        results=tuple(results),
        hit_at_1=mean(result.metrics.hit_at_1 for result in results),
        hit_at_3=mean(result.metrics.hit_at_3 for result in results),
        hit_at_5=mean(result.metrics.hit_at_5 for result in results),
        mean_reciprocal_rank=mean(
            result.metrics.reciprocal_rank for result in results
        ),
        mean_latency_ms=mean(result.latency_ms for result in results),
    )


def _required_text(record: dict, field: str, line_number: int) -> str:
    """Read one required, non-empty text field from a dataset record."""
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Line {line_number}: {field} must be non-empty text"
        )
    return value.strip()


def _phrase_groups(
    record: dict,
    field: str,
    line_number: int,
    required: bool,
) -> tuple[tuple[str, ...], ...]:
    """Read and validate required or optional phrase groups."""
    value = record.get(field)
    if value is None and not required:
        return ()
    if (
        not isinstance(value, list)
        or (required and not value)
        or not all(
            isinstance(group, list)
            and group
            and all(isinstance(phrase, str) and phrase.strip() for phrase in group)
            for group in value
        )
    ):
        requirement = "non-empty " if required else ""
        raise ValueError(
            f"Line {line_number}: {field} must be a {requirement}list "
            "of non-empty text lists"
        )
    return tuple(
        tuple(phrase.strip() for phrase in group)
        for group in value
    )


def load_retrieval_cases(input_path: Path) -> list[RetrievalCase]:
    """Load and validate retrieval cases from a JSON Lines file."""
    cases = []
    seen_case_ids = set()

    with input_path.open(encoding="utf-8") as dataset_file:
        for line_number, line in enumerate(dataset_file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Line {line_number}: invalid JSON"
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"Line {line_number}: each case must be a JSON object"
                )

            case_id = _required_text(record, "case_id", line_number)
            if case_id in seen_case_ids:
                raise ValueError(f"Case IDs must be unique: {case_id}")

            source_sequences = record.get("accepted_source_sequences")
            if (
                not isinstance(source_sequences, list)
                or not source_sequences
                or not all(
                    isinstance(sequence, int)
                    and not isinstance(sequence, bool)
                    and sequence >= 0
                    for sequence in source_sequences
                )
            ):
                raise ValueError(
                    f"Line {line_number}: accepted source sequences "
                    "must be a non-empty list of non-negative integers"
                )

            evidence_phrase_groups = _phrase_groups(
                record=record,
                field="accepted_evidence_phrase_groups",
                line_number=line_number,
                required=True,
            )
            answer_phrase_groups = _phrase_groups(
                record=record,
                field="accepted_answer_phrase_groups",
                line_number=line_number,
                required=False,
            )

            cases.append(
                RetrievalCase(
                    case_id=case_id,
                    question=_required_text(record, "question", line_number),
                    ticker=_required_text(record, "ticker", line_number).upper(),
                    form=_required_text(record, "form", line_number).upper(),
                    expected_answer=_required_text(
                        record, "expected_answer", line_number
                    ),
                    category=_required_text(record, "category", line_number),
                    accepted_source_sequences=tuple(source_sequences),
                    accepted_evidence_phrase_groups=evidence_phrase_groups,
                    notes=_required_text(record, "notes", line_number),
                    accepted_answer_phrase_groups=answer_phrase_groups,
                )
            )
            seen_case_ids.add(case_id)

    if not cases:
        raise ValueError("Retrieval dataset must contain at least one case")

    return cases
