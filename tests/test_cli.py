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
