from financial_rag.cli import main


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
        return expected_path

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
        lambda **kwargs: saved_path,
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
