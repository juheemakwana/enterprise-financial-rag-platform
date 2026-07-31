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
