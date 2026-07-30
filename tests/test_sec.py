from financial_rag.sec import fetch_company_tickers, find_cik, get_cik_for_ticker


def test_find_cik_returns_zero_padded_cik() -> None:
    companies = {
        "0": {
            "cik_str": 1045810,
            "ticker": "NVDA",
            "title": "NVIDIA CORP",
        }
    }

    result = find_cik(companies, ticker="NVDA")

    assert result == "0001045810"


def test_fetch_company_tickers_sends_sec_identity(monkeypatch) -> None:
    expected_data = {
        "0": {
            "cik_str": 1045810,
            "ticker": "NVDA",
            "title": "NVIDIA CORP",
        }
    }
    captured_request = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return expected_data

    def fake_get(url, *, headers, timeout):
        captured_request["url"] = url
        captured_request["headers"] = headers
        captured_request["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("financial_rag.sec.requests.get", fake_get)

    result = fetch_company_tickers("Example Project contact@example.com")

    assert result == expected_data
    assert captured_request["url"] == "https://www.sec.gov/files/company_tickers.json"
    assert captured_request["headers"]["User-Agent"] == "Example Project contact@example.com"
    assert captured_request["timeout"] == 30


def test_get_cik_for_ticker_combines_fetch_and_search(monkeypatch) -> None:
    companies = {
        "0": {
            "cik_str": 1045810,
            "ticker": "NVDA",
            "title": "NVIDIA CORP",
        }
    }

    monkeypatch.setattr(
        "financial_rag.sec.fetch_company_tickers",
        lambda user_agent: companies,
    )

    result = get_cik_for_ticker(
        ticker="NVDA",
        user_agent="Example Project contact@example.com",
    )

    assert result == "0001045810"
