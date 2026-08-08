"""Command-line interface for the financial RAG project."""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from financial_rag.parser import parse_filing_file
from financial_rag.sec import get_cik_for_ticker, ingest_latest_filing


def main(arguments: list[str] | None = None) -> None:
    """Look up and print the SEC CIK for a company ticker."""
    parser = argparse.ArgumentParser(
        description="Look up a company's SEC CIK number.",
    )
    parser.add_argument("ticker", help="Company ticker, for example NVDA")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download and save the latest filing",
    )
    parser.add_argument(
        "--form",
        default="10-K",
        help="SEC filing form to download (default: 10-K)",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("data/raw"),
        help="Directory for downloaded filing HTML",
    )
    parser.add_argument(
        "--parse",
        action="store_true",
        help="Parse the filing after downloading it",
    )
    parser.add_argument(
        "--parsed-output-directory",
        type=Path,
        default=Path("data/parsed"),
        help="Parent directory for parsed filing content",
    )
    args = parser.parse_args(arguments)

    if args.parse and not args.download:
        parser.error("--parse requires --download")

    load_dotenv()
    user_agent = os.getenv("SEC_USER_AGENT", "").strip()

    if not user_agent or "example.com" in user_agent:
        raise SystemExit(
            "Set SEC_USER_AGENT in .env to your real name and contact email."
        )

    ticker = args.ticker.upper()

    if args.download:
        saved_path = ingest_latest_filing(
            ticker=ticker,
            form=args.form.upper(),
            user_agent=user_agent,
            output_directory=args.output_directory,
        )
        print(f"Saved filing: {saved_path}")

        if args.parse:
            parsed_directory = args.parsed_output_directory / saved_path.stem
            parsed_paths = parse_filing_file(
                input_path=saved_path,
                output_directory=parsed_directory,
            )
            text_count = sum(
                path.parent.name == "text" for path in parsed_paths
            )
            table_count = sum(
                path.parent.name == "tables" for path in parsed_paths
            )
            print(
                f"Parsed filing: {text_count} text blocks, "
                f"{table_count} tables -> {parsed_directory}"
            )
        return

    cik = get_cik_for_ticker(ticker=ticker, user_agent=user_agent)
    print(f"{ticker} CIK: {cik}")


if __name__ == "__main__":
    main()
