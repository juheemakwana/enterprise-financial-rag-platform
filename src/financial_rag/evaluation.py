"""Dataset tools for evaluating SEC filing retrieval."""

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter

from qdrant_client import QdrantClient

from financial_rag.embeddings import EmbeddingModel


@dataclass(frozen=True)
class RetrievalCase:
    """One question with its expected answer and accepted evidence chunks."""

    case_id: str
    question: str
    ticker: str
    form: str
    expected_answer: str
    category: str
    accepted_evidence_chunk_ids: tuple[str, ...]
    notes: str


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
    retrieved_chunk_ids: list[str],
    accepted_evidence_chunk_ids: tuple[str, ...],
) -> RetrievalCaseMetrics:
    """Calculate rank-based metrics for one retrieval case."""
    if not accepted_evidence_chunk_ids:
        raise ValueError("At least one accepted evidence chunk ID is required")

    accepted_ids = set(accepted_evidence_chunk_ids)
    first_relevant_rank = next(
        (
            rank
            for rank, chunk_id in enumerate(retrieved_chunk_ids, start=1)
            if chunk_id in accepted_ids
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
    from financial_rag.vectorstore import search_similar_chunks

    results = []
    for case in cases:
        started_at = perf_counter()
        query_vector = embed_query(case.question, model)
        search_results = search_similar_chunks(
            client=client,
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
            retrieved_chunk_ids=list(retrieved_chunk_ids),
            accepted_evidence_chunk_ids=case.accepted_evidence_chunk_ids,
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

            evidence_ids = record.get("accepted_evidence_chunk_ids")
            if (
                not isinstance(evidence_ids, list)
                or not evidence_ids
                or not all(
                    isinstance(chunk_id, str) and chunk_id.strip()
                    for chunk_id in evidence_ids
                )
            ):
                raise ValueError(
                    f"Line {line_number}: accepted evidence IDs "
                    "must be a non-empty list of text values"
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
                    accepted_evidence_chunk_ids=tuple(
                        chunk_id.strip() for chunk_id in evidence_ids
                    ),
                    notes=_required_text(record, "notes", line_number),
                )
            )
            seen_case_ids.add(case_id)

    if not cases:
        raise ValueError("Retrieval dataset must contain at least one case")

    return cases
