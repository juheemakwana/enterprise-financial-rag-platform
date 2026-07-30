"""Command-line interface for the financial RAG project."""

import argparse
import os

from dotenv import load_dotenv

from financial_rag.sec import get_cik_for_ticker


def main(arguments: list[str] | None = None) -> None:
    """Look up and print the SEC CIK for a company ticker."""
    parser = argparse.ArgumentParser(
        description="Look up a company's SEC CIK number.",
    )
    parser.add_argument("ticker", help="Company ticker, for example NVDA")
    args = parser.parse_args(arguments)

    load_dotenv()
    user_agent = os.getenv("SEC_USER_AGENT", "").strip()

    if not user_agent or "example.com" in user_agent:
        raise SystemExit(
            "Set SEC_USER_AGENT in .env to your real name and contact email."
        )

    ticker = args.ticker.upper()
    cik = get_cik_for_ticker(ticker=ticker, user_agent=user_agent)
    print(f"{ticker} CIK: {cik}")


if __name__ == "__main__":
    main()
