# Enterprise Financial RAG Platform

## Project goal

Build a system that answers questions about SEC financial filings using retrieved
evidence and citations instead of relying only on an LLM's memory.

## What I will learn

- How to collect financial filings from the SEC EDGAR API
- How to extract and clean text and financial tables
- How tokenization and document chunking work
- How embeddings and vector search retrieve relevant evidence
- How to generate grounded answers with citations
- How to evaluate accuracy, latency, and cost

## Current status

SEC filing ingestion is working. The project can:

- Resolve a stock ticker to its SEC CIK number
- Find the latest filing of a requested form, such as `10-K`
- Build the official SEC archive URL
- Download and save the raw filing HTML
- Validate the workflow with automated tests

Document parsing, chunking, retrieval, answer generation, and evaluation are
planned for later stages.

## Requirements

- Python 3.12
- A real name and monitored contact email for the SEC `User-Agent`

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the project and its development tools:

```bash
python -m pip install -e '.[dev]'
```

Copy the safe environment template:

```bash
cp .env.example .env
```

Edit `.env` and replace the placeholder with your real SEC identity. The private
`.env` file is ignored by Git.

## Usage

Look up a company's SEC CIK:

```bash
python -m financial_rag.cli NVDA
```

Download the latest 10-K:

```bash
python -m financial_rag.cli NVDA --download --form 10-K
```

Raw filings are saved under `data/raw/` and excluded from Git because financial
filings can be large.

## Tests

Run the complete test suite:

```bash
python -m pytest
```
