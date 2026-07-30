"""Utilities for working with SEC EDGAR company data."""

import requests


COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"


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
