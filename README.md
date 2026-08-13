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
- Remove scripts and hidden machine-only XBRL metadata
- Extract readable text and financial tables in document order
- Convert HTML tables into Markdown
- Save parsed text and tables as auditable local files
- Count tokens with the `BAAI/bge-small-en-v1.5` tokenizer
- Create ordered, citation-aware text and table chunks
- Save chunks and SEC provenance as JSON Lines
- Validate the workflow with automated tests

Embedding, retrieval, answer generation, and evaluation are planned for later
stages.

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
financial-rag NVDA
```

Download the latest 10-K:

```bash
financial-rag NVDA --download --form 10-K
```

Download and parse the latest 10-K in one command:

```bash
financial-rag NVDA --download --parse
```

Download, parse, and chunk the latest 10-K in one command:

```bash
financial-rag NVDA --download --parse --chunk
```

Raw filings are saved under `data/raw/`. Parsed filings are saved in a
filing-specific folder under `data/parsed/`, for example:

```text
data/parsed/NVDA_10-K_2026-02-25/
├── text/
│   ├── 00000.md
│   └── 00001.md
└── tables/
    ├── 00003.md
    └── 00005.md
```

The numbers preserve the elements' original order in the filing. Generated data
is excluded from Git because SEC filings and derived output can be large.

Chunk records are saved under a filing-specific directory:

```text
data/chunks/NVDA_10-K_2026-02-25/chunks.jsonl
```

Each JSON line contains the chunk content, type, token count, original element
sequences, ticker, CIK, form, filing date, accession number, official SEC URL,
and local source path.

## Chunking strategy

The selected embedding model, `BAAI/bge-small-en-v1.5`, accepts at most 512
tokens. Chunks target 400 tokens so model-added tokens and future prefixes have
room within that limit.

- Consecutive text elements are combined up to 400 tokens.
- Text chunks reuse up to 80 tokens of preceding context when whole elements fit.
- An individual oversized text block is split into exact token windows with
  80-token overlap.
- Tables remain intact when they fit within 400 tokens.
- Oversized tables are split by data rows, and their Markdown header is repeated
  in every resulting chunk.
- Text, tables, chunk IDs, and source sequence numbers retain document order.

The first chunking run downloads the BGE tokenizer from Hugging Face. PyTorch and
the embedding model weights are not required until the embedding stage.

## Tests

Run the complete test suite:

```bash
python -m pytest
```
