from financial_rag.sec import (
    build_filing_url,
    download_filing_html,
    fetch_company_submissions,
    fetch_company_tickers,
    find_cik,
    find_latest_filing,
    get_cik_for_ticker,
    ingest_latest_filing,
    save_filing_html,
)


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


def test_fetch_company_submissions_builds_cik_url(monkeypatch) -> None:
    expected_data = {
        "cik": "0001045810",
        "name": "NVIDIA CORP",
        "filings": {"recent": {}},
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

    result = fetch_company_submissions(
        cik="0001045810",
        user_agent="Test Project contact@test.invalid",
    )

    assert result == expected_data
    assert captured_request["url"] == (
        "https://data.sec.gov/submissions/CIK0001045810.json"
    )
    assert captured_request["headers"]["User-Agent"] == (
        "Test Project contact@test.invalid"
    )
    assert captured_request["timeout"] == 30


def test_find_latest_filing_selects_requested_form() -> None:
    submissions = {
        "filings": {
            "recent": {
                "form": ["10-Q", "10-K", "8-K"],
                "accessionNumber": [
                    "0001045810-25-000100",
                    "0001045810-25-000050",
                    "0001045810-25-000010",
                ],
                "filingDate": ["2025-05-29", "2025-02-26", "2025-01-10"],
                "reportDate": ["2025-04-27", "2025-01-26", "2025-01-09"],
                "primaryDocument": [
                    "nvda-20250427.htm",
                    "nvda-20250126.htm",
                    "nvda-8k.htm",
                ],
            }
        }
    }

    result = find_latest_filing(submissions, form="10-K")

    assert result == {
        "form": "10-K",
        "accession_number": "0001045810-25-000050",
        "filing_date": "2025-02-26",
        "report_date": "2025-01-26",
        "primary_document": "nvda-20250126.htm",
    }


def test_build_filing_url_formats_sec_archive_path() -> None:
    result = build_filing_url(
        cik="0001045810",
        accession_number="0001045810-25-000050",
        primary_document="nvda-20250126.htm",
    )

    assert result == (
        "https://www.sec.gov/Archives/edgar/data/1045810/"
        "000104581025000050/nvda-20250126.htm"
    )


def test_download_filing_html_returns_validated_response_text(monkeypatch) -> None:
    filing_url = "https://www.sec.gov/Archives/example-filing.htm"
    captured_request = {}

    class FakeResponse:
        text = "<html><body>Example 10-K</body></html>"

        def raise_for_status(self) -> None:
            captured_request["status_checked"] = True

    def fake_get(url, *, headers, timeout):
        captured_request["url"] = url
        captured_request["headers"] = headers
        captured_request["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("financial_rag.sec.requests.get", fake_get)

    result = download_filing_html(
        filing_url,
        user_agent="Test Project contact@test.invalid",
    )

    assert result == "<html><body>Example 10-K</body></html>"
    assert captured_request["url"] == filing_url
    assert captured_request["headers"]["User-Agent"] == (
        "Test Project contact@test.invalid"
    )
    assert captured_request["timeout"] == 30
    assert captured_request["status_checked"] is True


def test_save_filing_html_creates_predictable_raw_file(tmp_path) -> None:
    output_directory = tmp_path / "raw"

    saved_path = save_filing_html(
        html="<html><body>Example 10-K</body></html>",
        output_directory=output_directory,
        ticker="NVDA",
        form="10-K",
        filing_date="2025-02-26",
    )

    assert saved_path == output_directory / "NVDA_10-K_2025-02-26.html"
    assert saved_path.read_text(encoding="utf-8") == (
        "<html><body>Example 10-K</body></html>"
    )


def test_ingest_latest_filing_connects_complete_download_flow(
    monkeypatch,
    tmp_path,
) -> None:
    companies = {
        "0": {
            "cik_str": 1045810,
            "ticker": "NVDA",
            "title": "NVIDIA CORP",
        }
    }
    submissions = {
        "filings": {
            "recent": {
                "form": ["10-K"],
                "accessionNumber": ["0001045810-25-000050"],
                "filingDate": ["2025-02-26"],
                "reportDate": ["2025-01-26"],
                "primaryDocument": ["nvda-20250126.htm"],
            }
        }
    }
    captured_download = {}

    monkeypatch.setattr(
        "financial_rag.sec.fetch_company_tickers",
        lambda user_agent: companies,
    )
    monkeypatch.setattr(
        "financial_rag.sec.fetch_company_submissions",
        lambda cik, user_agent: submissions,
    )

    def fake_download(filing_url, user_agent):
        captured_download["url"] = filing_url
        return "<html><body>NVIDIA 10-K</body></html>"

    monkeypatch.setattr(
        "financial_rag.sec.download_filing_html",
        fake_download,
    )

    filing = ingest_latest_filing(
        ticker="NVDA",
        form="10-K",
        user_agent="Test Project contact@test.invalid",
        output_directory=tmp_path / "raw",
    )

    assert filing.ticker == "NVDA"
    assert filing.cik == "0001045810"
    assert filing.form == "10-K"
    assert filing.filing_date == "2025-02-26"
    assert filing.accession_number == "0001045810-25-000050"
    assert filing.local_path.name == "NVDA_10-K_2025-02-26.html"
    assert filing.local_path.read_text(encoding="utf-8") == (
        "<html><body>NVIDIA 10-K</body></html>"
    )
    expected_url = (
        "https://www.sec.gov/Archives/edgar/data/1045810/"
        "000104581025000050/nvda-20250126.htm"
    )
    assert captured_download["url"] == expected_url
    assert filing.source_url == expected_url
