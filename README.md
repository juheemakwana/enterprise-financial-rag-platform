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
- Create normalized 384-dimensional BGE embeddings locally
- Persist vectors and searchable SEC metadata in local Qdrant
- Retrieve ranked evidence with ticker and filing-form filters
- Evaluate retrieval with a version-controlled question-and-evidence dataset
- Measure Hit@1, Hit@3, Hit@5, mean reciprocal rank, and latency
- Validate the workflow with automated tests

Answer generation and answer-quality evaluation are planned for later stages.

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

Run the complete ingestion and vector-indexing pipeline:

```bash
financial-rag NVDA --download --parse --chunk --index
```

Search the local vector index and return cited evidence:

```bash
financial-rag NVDA \
  --search "What was NVIDIA revenue in fiscal 2026?" \
  --limit 5
```

Search is local and does not require an SEC request. Use `--form 10-Q` to filter
for a different filing form and `--qdrant-directory` to select another local
Qdrant database.

Evaluate retrieval against the saved NVDA questions:

```bash
financial-rag NVDA --evaluate
```

Evaluation is also local and does not require an SEC request. By default, it
loads `evals/datasets/retrieval_cases.jsonl` and checks the first 50 retrieved
chunks for each question. Use `--evaluation-limit` to change that depth or
`--evaluation-dataset` to select another JSON Lines dataset.

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

## Embeddings and vector search

`BAAI/bge-small-en-v1.5` converts each chunk into a normalized 384-dimensional
vector. The project currently runs embedding inference on CPU. The first
`--index` run downloads the model weights from Hugging Face; later runs reuse the
local cache.

Qdrant runs in local on-disk mode at `data/qdrant/`, so Docker, a Qdrant server,
cloud account, and API key are not required. The collection uses cosine
similarity. Every vector stores the original chunk content and these payload
fields:

- Chunk ID, type, sequence, token count, and source element sequences
- Ticker, CIK, form, filing date, and accession number
- Official SEC URL and local source path

Chunk IDs are converted into stable UUIDs before insertion. Running `--index`
again updates the same points instead of creating duplicates.

For search, the question is embedded with BGE's retrieval instruction. Qdrant
returns the closest chunks with cosine scores and official SEC citations. These
are ranked evidence results, not a generated final answer. Optional reranking
can be added before answer generation.

## Retrieval evaluation

The tracked dataset at `evals/datasets/retrieval_cases.jsonl` currently contains
six questions about NVIDIA's fiscal 2026 10-K. Each JSON line records:

- A unique case ID and question
- Ticker and filing form filters
- The expected factual answer
- One or more accepted evidence chunk IDs
- A category and human-readable notes

Multiple evidence chunks can be accepted because the same fact may appear in a
financial table and again as rounded narrative text.

The evaluation command reports these metrics:

- **Rank:** the position of the first accepted evidence chunk. Rank 1 is best.
- **Hit@1:** 1 when accepted evidence is first; otherwise 0.
- **Hit@3:** 1 when accepted evidence appears within the first three results.
- **Hit@5:** 1 when accepted evidence appears within the first five results.
- **Reciprocal rank (RR):** `1 / rank`. Rank 2 gives `1 / 2 = 0.5`.
- **Mean reciprocal rank (MRR):** the average RR across all questions.
- **Latency:** time used to embed one question and search Qdrant.

The overall Hit@K values are averages across the dataset. For example, Hit@5 of
`0.667` means four of six questions found accepted evidence within the first
five results. `rank=not found` means no accepted evidence appeared within the
configured evaluation limit; its RR is 0.

## Tests

Run the complete test suite:

```bash
python -m pytest
```
