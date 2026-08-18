from financial_rag.cli import main
from financial_rag.evaluation import (
    RetrievalCase,
    RetrievalCaseMetrics,
    RetrievalEvaluationResult,
    RetrievalEvaluationSummary,
)
from financial_rag.models import DocumentChunk, FilingMetadata, SearchResult


def make_filing_metadata(local_path):
    return FilingMetadata(
        ticker="NVDA",
        company_name="NVIDIA CORP",
        cik="0001045810",
        form="10-K",
        filing_date="2026-02-25",
        accession_number="0001045810-26-000021",
        source_url="https://www.sec.gov/Archives/example/nvda.htm",
        local_path=local_path,
    )


def test_cli_prints_cik(monkeypatch, capsys) -> None:
    monkeypatch.setenv(
        "SEC_USER_AGENT",
        "Test Project contact@test.invalid",
    )
    monkeypatch.setattr(
        "financial_rag.cli.get_cik_for_ticker",
        lambda ticker, user_agent: "0001045810",
    )

    main(["NVDA"])

    output = capsys.readouterr().out
    assert output == "NVDA CIK: 0001045810\n"


def test_cli_downloads_latest_filing(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.setenv(
        "SEC_USER_AGENT",
        "Test Project contact@test.invalid",
    )
    captured_call = {}
    expected_path = tmp_path / "raw" / "NVDA_10-K_2025-02-26.html"

    def fake_ingest(ticker, form, user_agent, output_directory):
        captured_call["ticker"] = ticker
        captured_call["form"] = form
        captured_call["user_agent"] = user_agent
        captured_call["output_directory"] = output_directory
        return make_filing_metadata(expected_path)

    monkeypatch.setattr(
        "financial_rag.cli.ingest_latest_filing",
        fake_ingest,
    )

    main(
        [
            "NVDA",
            "--download",
            "--form",
            "10-K",
            "--output-directory",
            str(tmp_path / "raw"),
        ]
    )

    assert captured_call["ticker"] == "NVDA"
    assert captured_call["form"] == "10-K"
    assert captured_call["output_directory"] == tmp_path / "raw"
    assert capsys.readouterr().out == f"Saved filing: {expected_path}\n"


def test_cli_downloads_and_parses_filing(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.setenv(
        "SEC_USER_AGENT",
        "Test Project contact@test.invalid",
    )
    saved_path = tmp_path / "raw" / "NVDA_10-K_2026-02-25.html"
    parsed_parent = tmp_path / "parsed"
    parsed_directory = parsed_parent / saved_path.stem
    captured_call = {}

    monkeypatch.setattr(
        "financial_rag.cli.ingest_latest_filing",
        lambda **kwargs: make_filing_metadata(saved_path),
    )

    def fake_parse(input_path, output_directory):
        captured_call["input_path"] = input_path
        captured_call["output_directory"] = output_directory
        return [
            output_directory / "text" / "00000.md",
            output_directory / "text" / "00001.md",
            output_directory / "tables" / "00002.md",
        ]

    monkeypatch.setattr(
        "financial_rag.cli.parse_filing_file",
        fake_parse,
    )

    main(
        [
            "NVDA",
            "--download",
            "--parse",
            "--parsed-output-directory",
            str(parsed_parent),
        ]
    )

    assert captured_call["input_path"] == saved_path
    assert captured_call["output_directory"] == parsed_directory
    assert capsys.readouterr().out == (
        f"Saved filing: {saved_path}\n"
        f"Parsed filing: 2 text blocks, 1 tables -> {parsed_directory}\n"
    )


def test_cli_downloads_parses_and_chunks_filing(
    monkeypatch,
    capsys,
    tmp_path,
) -> None:
    monkeypatch.setenv(
        "SEC_USER_AGENT",
        "Test Project contact@test.invalid",
    )
    saved_path = tmp_path / "raw" / "NVDA_10-K_2026-02-25.html"
    parsed_parent = tmp_path / "parsed"
    parsed_directory = parsed_parent / saved_path.stem
    chunks_parent = tmp_path / "chunks"
    chunks_path = chunks_parent / saved_path.stem / "chunks.jsonl"
    tokenizer = object()
    elements = [object(), object()]
    chunks = [object(), object(), object()]
    captured_write = {}

    monkeypatch.setattr(
        "financial_rag.cli.ingest_latest_filing",
        lambda **kwargs: make_filing_metadata(saved_path),
    )
    monkeypatch.setattr(
        "financial_rag.cli.parse_filing_file",
        lambda **kwargs: [
            parsed_directory / "text" / "00000.md",
            parsed_directory / "tables" / "00001.md",
        ],
    )
    monkeypatch.setattr(
        "financial_rag.cli.load_parsed_elements",
        lambda input_directory: elements,
    )
    monkeypatch.setattr(
        "financial_rag.cli.load_tokenizer",
        lambda: tokenizer,
    )
    monkeypatch.setattr(
        "financial_rag.cli.chunk_document_elements",
        lambda **kwargs: chunks,
    )

    def fake_write(chunk_records, output_path):
        captured_write["chunks"] = chunk_records
        captured_write["output_path"] = output_path
        return output_path

    monkeypatch.setattr(
        "financial_rag.cli.write_chunks_jsonl",
        fake_write,
    )

    main(
        [
            "NVDA",
            "--download",
            "--parse",
            "--chunk",
            "--parsed-output-directory",
            str(parsed_parent),
            "--chunks-output-directory",
            str(chunks_parent),
        ]
    )

    assert captured_write["chunks"] == chunks
    assert captured_write["output_path"] == chunks_path
    assert capsys.readouterr().out == (
        f"Saved filing: {saved_path}\n"
        f"Parsed filing: 1 text blocks, 1 tables -> {parsed_directory}\n"
        f"Chunked filing: 3 chunks -> {chunks_path}\n"
    )


def test_cli_searches_local_qdrant_and_prints_cited_evidence(
    monkeypatch,
    capsys,
    tmp_path,
) -> None:
    filing = make_filing_metadata(
        tmp_path / "raw" / "NVDA_10-K_2026-02-25.html"
    )
    chunk = DocumentChunk(
        chunk_id="NVDA_10-K_2026-02-25-00134",
        sequence=134,
        chunk_type="text",
        content="Revenue for fiscal year 2026 was $215.9 billion.",
        token_count=12,
        source_sequences=(437, 438),
        filing=filing,
    )
    expected_results = [SearchResult(score=0.8019, chunk=chunk)]
    model = object()
    query_vector = tuple([0.0] * 384)
    captured_search = {}

    class FakeClient:
        closed = False

        def close(self):
            self.closed = True

    client = FakeClient()

    monkeypatch.setattr(
        "financial_rag.cli.load_embedding_model",
        lambda: model,
    )
    monkeypatch.setattr(
        "financial_rag.cli.embed_query",
        lambda question, selected_model: query_vector,
    )
    monkeypatch.setattr(
        "financial_rag.cli.open_local_vector_store",
        lambda storage_path: client,
    )

    def fake_search(**kwargs):
        captured_search.update(kwargs)
        return expected_results

    monkeypatch.setattr(
        "financial_rag.cli.search_similar_chunks",
        fake_search,
    )

    main(
        [
            "NVDA",
            "--search",
            "What was revenue in fiscal 2026?",
            "--limit",
            "1",
            "--qdrant-directory",
            str(tmp_path / "qdrant"),
        ]
    )

    assert captured_search["client"] is client
    assert captured_search["query_vector"] == query_vector
    assert captured_search["limit"] == 1
    assert captured_search["ticker"] == "NVDA"
    assert captured_search["form"] == "10-K"
    assert client.closed is True
    assert capsys.readouterr().out == (
        "Search results for: What was revenue in fiscal 2026?\n"
        "1. score=0.8019 type=text "
        "chunk=NVDA_10-K_2026-02-25-00134\n"
        "   Revenue for fiscal year 2026 was $215.9 billion.\n"
        "   SEC source: "
        "https://www.sec.gov/Archives/example/nvda.htm\n"
    )


def test_cli_evaluates_local_retrieval(
    monkeypatch,
    capsys,
    tmp_path,
) -> None:
    case = RetrievalCase(
        case_id="nvda_revenue",
        question="What was NVIDIA's revenue?",
        ticker="NVDA",
        form="10-K",
        expected_answer="$215.9 billion",
        category="financial_table",
        accepted_source_sequences=(436,),
        accepted_evidence_phrase_groups=(("revenue", "215.9 billion"),),
        notes="Test case.",
    )
    result = RetrievalEvaluationResult(
        case_id=case.case_id,
        question=case.question,
        retrieved_chunk_ids=("revenue-chunk",),
        metrics=RetrievalCaseMetrics(
            first_relevant_rank=1,
            hit_at_1=1,
            hit_at_3=1,
            hit_at_5=1,
            reciprocal_rank=1.0,
        ),
        latency_ms=12.3,
    )
    summary = RetrievalEvaluationSummary(
        results=(result,),
        hit_at_1=1.0,
        hit_at_3=1.0,
        hit_at_5=1.0,
        mean_reciprocal_rank=1.0,
        mean_latency_ms=12.3,
    )
    model = object()
    captured_call = {}

    class FakeClient:
        closed = False

        def close(self):
            self.closed = True

    client = FakeClient()

    monkeypatch.setattr(
        "financial_rag.cli.load_retrieval_cases",
        lambda dataset_path: [case],
    )
    monkeypatch.setattr(
        "financial_rag.cli.load_embedding_model",
        lambda: model,
    )
    monkeypatch.setattr(
        "financial_rag.cli.open_local_vector_store",
        lambda storage_path: client,
    )

    def fake_run_retrieval_evaluation(**kwargs):
        captured_call.update(kwargs)
        return summary

    monkeypatch.setattr(
        "financial_rag.cli.run_retrieval_evaluation",
        fake_run_retrieval_evaluation,
    )

    main(
        [
            "NVDA",
            "--evaluate",
            "--evaluation-dataset",
            str(tmp_path / "cases.jsonl"),
            "--evaluation-limit",
            "25",
            "--qdrant-directory",
            str(tmp_path / "qdrant"),
        ]
    )

    assert captured_call == {
        "cases": [case],
        "model": model,
        "client": client,
        "retrieval_limit": 25,
    }
    assert client.closed is True
    assert capsys.readouterr().out == (
        "Retrieval evaluation: NVDA 10-K\n"
        "- nvda_revenue: rank=1, RR=1.000, latency=12.3 ms\n"
        "Hit@1: 1.000\n"
        "Hit@3: 1.000\n"
        "Hit@5: 1.000\n"
        "MRR: 1.000\n"
        "Mean latency: 12.3 ms\n"
    )


def test_cli_indexes_chunks_and_closes_qdrant(
    monkeypatch,
    capsys,
    tmp_path,
) -> None:
    monkeypatch.setenv(
        "SEC_USER_AGENT",
        "Test Project contact@test.invalid",
    )
    saved_path = tmp_path / "raw" / "NVDA_10-K_2026-02-25.html"
    parsed_directory = tmp_path / "parsed" / saved_path.stem
    chunks_path = tmp_path / "chunks" / saved_path.stem / "chunks.jsonl"
    qdrant_directory = tmp_path / "qdrant"
    chunks = [object(), object()]
    embedded_chunks = [object(), object()]
    model = object()

    class FakeClient:
        closed = False

        def close(self):
            self.closed = True

    client = FakeClient()

    monkeypatch.setattr(
        "financial_rag.cli.ingest_latest_filing",
        lambda **kwargs: make_filing_metadata(saved_path),
    )
    monkeypatch.setattr(
        "financial_rag.cli.parse_filing_file",
        lambda **kwargs: [parsed_directory / "text" / "00000.md"],
    )
    monkeypatch.setattr(
        "financial_rag.cli.load_parsed_elements",
        lambda input_directory: [object()],
    )
    monkeypatch.setattr(
        "financial_rag.cli.load_tokenizer",
        lambda: object(),
    )
    monkeypatch.setattr(
        "financial_rag.cli.chunk_document_elements",
        lambda **kwargs: chunks,
    )
    monkeypatch.setattr(
        "financial_rag.cli.write_chunks_jsonl",
        lambda chunk_records, output_path: output_path,
    )
    monkeypatch.setattr(
        "financial_rag.cli.load_embedding_model",
        lambda: model,
    )
    monkeypatch.setattr(
        "financial_rag.cli.embed_chunks",
        lambda chunk_records, selected_model: embedded_chunks,
    )
    monkeypatch.setattr(
        "financial_rag.cli.open_local_vector_store",
        lambda storage_path: client,
    )
    monkeypatch.setattr(
        "financial_rag.cli.upsert_embedded_chunks",
        lambda selected_client, records: len(records),
    )

    main(
        [
            "NVDA",
            "--download",
            "--parse",
            "--chunk",
            "--index",
            "--output-directory",
            str(tmp_path / "raw"),
            "--parsed-output-directory",
            str(tmp_path / "parsed"),
            "--chunks-output-directory",
            str(tmp_path / "chunks"),
            "--qdrant-directory",
            str(qdrant_directory),
        ]
    )

    assert client.closed is True
    assert capsys.readouterr().out == (
        f"Saved filing: {saved_path}\n"
        f"Parsed filing: 1 text blocks, 0 tables -> {parsed_directory}\n"
        f"Chunked filing: 2 chunks -> {chunks_path}\n"
        f"Indexed filing: 2 vectors -> {qdrant_directory}\n"
    )
