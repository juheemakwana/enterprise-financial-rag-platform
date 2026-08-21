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
- Retrieve hybrid-ranked evidence with ticker and filing-form filters
- Evaluate retrieval with a version-controlled question-and-evidence dataset
- Measure Hit@1, Hit@3, Hit@5, mean reciprocal rank, and latency
- Generate evidence-only answers locally with Gemma 4
- Validate generated citation numbers against the supplied evidence
- Print the exact SEC source and chunk behind every cited number
- Evaluate answer values, citations, abstentions, grounding, and latency
- Validate the workflow with automated tests

The current answer-quality results are a small baseline, not a final performance
claim. Broader filings, companies, question types, and unanswerable cases are
planned for later stages.

## Requirements

- Python 3.12
- A real name and monitored contact email for the SEC `User-Agent`
- About 10 GiB of free disk space for the local Gemma model

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

Download the local instruction-tuned answer model once:

```bash
hf download google/gemma-4-E2B-it
```

The model is stored in the current user's Hugging Face cache, not in the project
or Git repository. A Hugging Face login, Ollama, and a paid model API are not
required for this model.

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

Generate an answer from the retrieved SEC evidence:

```bash
financial-rag NVDA \
  --answer "What was NVIDIA's revenue in fiscal 2026?" \
  --limit 5
```

`--answer` first performs the same hybrid retrieval as `--search`. It then
numbers the retrieved evidence, asks local Gemma to answer using only that
evidence, validates the citation labels, and prints the SEC URL and chunk ID for
each cited number. The model is loaded once per command, so the first answer in
a process has a noticeable startup delay.

Evaluate retrieval against the saved NVDA questions:

```bash
financial-rag NVDA --evaluate
```

Evaluation is also local and does not require an SEC request. By default, it
loads `evals/datasets/retrieval_cases.jsonl` and checks the first 50 retrieved
chunks for each question. Use `--evaluation-limit` to change that depth or
`--evaluation-dataset` to select another JSON Lines dataset.

Evaluate generated answers and their citations:

```bash
financial-rag NVDA --evaluate-answers --limit 5
```

This command loads BGE and Gemma once, then reuses them for every matching case.
`--limit` controls how many retrieved chunks are supplied as numbered evidence
for each answer.

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
- Company name, ticker, CIK, form, filing date, and accession number
- Official SEC URL and local source path

Chunk IDs are converted into stable UUIDs before insertion. Running `--index`
again removes points for that exact SEC accession before inserting its current
chunks. Other filings remain untouched, and obsolete chunks cannot remain in
the collection after a chunking-strategy change.

For search, the question is embedded with BGE's retrieval instruction. Qdrant
returns semantic candidates while a local BM25 calculation ranks exact words
and numbers. Reciprocal-rank fusion combines both lists, and the final results
include official SEC citations. These are ranked evidence results, not a
generated final answer.

The current BM25 implementation scans chunks matching the ticker and form from
local Qdrant. This is appropriate for the learning-scale local corpus. A larger
enterprise deployment would move lexical ranking to a persistent search engine
such as OpenSearch or Elasticsearch.

## Retrieval evaluation

The tracked dataset at `evals/datasets/retrieval_cases.jsonl` currently contains
six questions about NVIDIA's fiscal 2026 10-K. Each JSON line records:

- A unique case ID and question
- Ticker and filing form filters
- The expected factual answer
- Accepted original source sequence numbers
- Phrase groups that must appear in accepted factual evidence
- A category and human-readable notes

Source sequences and factual phrase groups remain stable when chunk IDs change.
Multiple forms can be accepted because the same fact may appear in a financial
table and again as rounded narrative text.

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

## Answer evaluation

The answer evaluator reuses the tracked retrieval cases and adds accepted answer
phrase groups. It records the generated answer, retrieved chunk IDs, citations,
validation errors, and end-to-end latency for every case. A failed case remains
in the denominator instead of disappearing from the averages.

The reported metrics are:

- **Answer accuracy:** the answer contains an accepted factual value.
- **Abstention rate:** the model uses the standard evidence-insufficient reply.
- **Citation presence:** the response contains at least one citation label.
- **Citation correctness:** every cited item matches accepted SEC evidence.
- **Grounded pass rate:** both the answer value and its citation are correct.
- **Latency:** time used for retrieval and answer generation. The first case also
  includes model startup time.

Baseline 1 uses the six NVIDIA fiscal 2026 questions and five evidence chunks per
answer:

| Metric | Baseline 1 |
|---|---:|
| Answer accuracy | 0.667 |
| Abstention rate | 0.167 |
| Citation presence | 0.833 |
| Citation correctness | 0.667 |
| Grounded pass rate | 0.667 |
| Mean latency | 3.02 seconds |

This is a diagnostic baseline, not a final evaluation. Net-income retrieval did
not place accepted evidence within the first five chunks, so Gemma safely
abstained. For research and development, accepted evidence was retrieved, but
Gemma selected the narrower capitalized-expenditure value instead of total R&D
expense. The next evaluation stages will add retrieval reranking, unanswerable
cases, more companies and filings, per-category results, saved run reports, and
regression tracking.

## Tests

Run the complete test suite:

```bash
python -m pytest
```
