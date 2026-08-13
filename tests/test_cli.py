from financial_rag.models import FilingMetadata
from financial_rag.cli import main


def make_filing_metadata(local_path):
    return FilingMetadata(
        ticker="NVDA",
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
