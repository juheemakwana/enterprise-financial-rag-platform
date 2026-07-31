"""Utilities for working with SEC EDGAR company data."""

from pathlib import Path

import requests


COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
COMPANY_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
FILING_ARCHIVE_URL = (
    "https://www.sec.gov/Archives/edgar/data/"
    "{cik}/{accession_number}/{primary_document}"
)


def fetch_company_tickers(user_agent: str, timeout: int = 30) -> dict:
    """Download the SEC mapping of company tickers to CIK numbers."""
    response = requests.get(
        COMPANY_TICKERS_URL,
        headers={
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def fetch_company_submissions(
    cik: str,
    user_agent: str,
    timeout: int = 30,
) -> dict:
    """Download one company's SEC submission history."""
    padded_cik = cik.zfill(10)
    url = COMPANY_SUBMISSIONS_URL.format(cik=padded_cik)

    response = requests.get(
        url,
        headers={
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def find_cik(companies: dict, ticker: str) -> str:
    """Return the SEC CIK for a ticker as a zero-padded 10-digit string."""
    requested_ticker = ticker.upper()

    for company in companies.values():
        if company["ticker"].upper() == requested_ticker:
            return str(company["cik_str"]).zfill(10)

    raise LookupError(f"Ticker not found: {ticker}")


def get_cik_for_ticker(ticker: str, user_agent: str) -> str:
    """Fetch the SEC company mapping and return the CIK for one ticker."""
    companies = fetch_company_tickers(user_agent)
    return find_cik(companies, ticker)


def find_latest_filing(submissions: dict, form: str = "10-K") -> dict:
    """Return metadata for the newest filing matching an SEC form type."""
    recent = submissions["filings"]["recent"]
    requested_form = form.upper()

    for index, filing_form in enumerate(recent["form"]):
        if filing_form == requested_form:
            return {
                "form": filing_form,
                "accession_number": recent["accessionNumber"][index],
                "filing_date": recent["filingDate"][index],
                "report_date": recent["reportDate"][index],
                "primary_document": recent["primaryDocument"][index],
            }

    raise LookupError(f"No recent {requested_form} filing found")


def build_filing_url(
    cik: str,
    accession_number: str,
    primary_document: str,
) -> str:
    """Build the official SEC archive URL for a filing document."""
    return FILING_ARCHIVE_URL.format(
        cik=int(cik),
        accession_number=accession_number.replace("-", ""),
        primary_document=primary_document,
    )


def download_filing_html(
    filing_url: str,
    user_agent: str,
    timeout: int = 30,
) -> str:
    """Download and return the raw HTML for an SEC filing."""
    response = requests.get(
        filing_url,
        headers={
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.text


def save_filing_html(
    html: str,
    output_directory: Path,
    ticker: str,
    form: str,
    filing_date: str,
) -> Path:
    """Save raw SEC filing HTML and return the created file path."""
    output_directory.mkdir(parents=True, exist_ok=True)
    filename = f"{ticker.upper()}_{form.upper()}_{filing_date}.html"
    output_path = output_directory / filename
    output_path.write_text(html, encoding="utf-8")
    return output_path


def ingest_latest_filing(
    ticker: str,
    form: str,
    user_agent: str,
    output_directory: Path,
) -> Path:
    """Download and save the newest SEC filing of a requested form."""
    cik = get_cik_for_ticker(ticker=ticker, user_agent=user_agent)
    submissions = fetch_company_submissions(cik=cik, user_agent=user_agent)
    filing = find_latest_filing(submissions=submissions, form=form)
    filing_url = build_filing_url(
        cik=cik,
        accession_number=filing["accession_number"],
        primary_document=filing["primary_document"],
    )
    html = download_filing_html(filing_url, user_agent=user_agent)
    return save_filing_html(
        html=html,
        output_directory=output_directory,
        ticker=ticker,
        form=filing["form"],
        filing_date=filing["filing_date"],
    )
