"""Tools for creating embedding-sized chunks from parsed filings."""

import json
from pathlib import Path
from typing import Protocol

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from financial_rag.models import DocumentChunk, DocumentElement, FilingMetadata


EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
MAX_MODEL_TOKENS = 512
DEFAULT_CHUNK_TOKENS = 400
DEFAULT_OVERLAP_TOKENS = 80


class Tokenizer(Protocol):
    """Small tokenizer interface needed by the chunking functions."""

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        verbose: bool = False,
    ) -> list[int]: ...

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
    ) -> str: ...


def load_tokenizer(
    model_name: str = EMBEDDING_MODEL_NAME,
) -> PreTrainedTokenizerBase:
    """Load the tokenizer used by the selected embedding model."""
    return AutoTokenizer.from_pretrained(model_name)


def count_tokens(text: str, tokenizer: Tokenizer) -> int:
    """Count text tokens without model-added special tokens."""
    return len(
        tokenizer.encode(
            text,
            add_special_tokens=False,
            verbose=False,
        )
    )


def _split_oversized_text(
    text: str,
    source_sequence: int,
    tokenizer: Tokenizer,
    chunk_tokens: int,
    overlap_tokens: int,
) -> list[tuple[int, str]]:
    """Split one long element into overlapping tokenizer-sized segments."""
    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
        verbose=False,
    )
    step = chunk_tokens - overlap_tokens
    segments = []

    for start in range(0, len(token_ids), step):
        window = token_ids[start : start + chunk_tokens]
        content = tokenizer.decode(
            window,
            skip_special_tokens=True,
        ).strip()
        if content:
            segments.append((source_sequence, content))
        if start + chunk_tokens >= len(token_ids):
            break

    return segments


def chunk_text_elements(
    elements: list[DocumentElement],
    filing: FilingMetadata,
    tokenizer: Tokenizer,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    starting_sequence: int = 0,
) -> list[DocumentChunk]:
    """Combine ordered text elements into token-limited searchable chunks."""
    if chunk_tokens <= 0 or chunk_tokens > MAX_MODEL_TOKENS:
        raise ValueError(
            f"chunk_tokens must be between 1 and {MAX_MODEL_TOKENS}"
        )
    if overlap_tokens < 0 or overlap_tokens >= chunk_tokens:
        raise ValueError("overlap_tokens must be smaller than chunk_tokens")

    segments = []
    for element in elements:
        if element.element_type != "text":
            raise ValueError("chunk_text_elements accepts only text elements")

        if count_tokens(element.content, tokenizer) <= chunk_tokens:
            segments.append((element.sequence, element.content))
        else:
            segments.extend(
                _split_oversized_text(
                    text=element.content,
                    source_sequence=element.sequence,
                    tokenizer=tokenizer,
                    chunk_tokens=chunk_tokens,
                    overlap_tokens=overlap_tokens,
                )
            )

    chunks = []
    current_segments = []

    def render(items: list[tuple[int, str]]) -> str:
        return "\n\n".join(content for _, content in items)

    def append_chunk(items: list[tuple[int, str]]) -> None:
        content = render(items)
        sequence = starting_sequence + len(chunks)
        chunks.append(
            DocumentChunk(
                chunk_id=f"{filing.local_path.stem}-{sequence:05d}",
                sequence=sequence,
                chunk_type="text",
                content=content,
                token_count=count_tokens(content, tokenizer),
                source_sequences=tuple(
                    dict.fromkeys(source for source, _ in items)
                ),
                filing=filing,
            )
        )

    for segment in segments:
        proposed_segments = [*current_segments, segment]
        if current_segments and count_tokens(render(proposed_segments), tokenizer) > (
            chunk_tokens
        ):
            append_chunk(current_segments)

            retained_segments = []
            for previous_segment in reversed(current_segments):
                candidate = [previous_segment, *retained_segments]
                if count_tokens(render(candidate), tokenizer) <= overlap_tokens:
                    retained_segments = candidate
                else:
                    break

            current_segments = retained_segments
            proposed_segments = [*current_segments, segment]
            if count_tokens(render(proposed_segments), tokenizer) > chunk_tokens:
                current_segments = []

        current_segments.append(segment)

    if current_segments:
        append_chunk(current_segments)

    return chunks


def chunk_table_element(
    element: DocumentElement,
    filing: FilingMetadata,
    tokenizer: Tokenizer,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    starting_sequence: int = 0,
) -> list[DocumentChunk]:
    """Create one searchable chunk per table row with repeated context."""
    if element.element_type != "table":
        raise ValueError("chunk_table_element accepts only a table element")
    if chunk_tokens <= 0 or chunk_tokens > MAX_MODEL_TOKENS:
        raise ValueError(
            f"chunk_tokens must be between 1 and {MAX_MODEL_TOKENS}"
        )

    def parse_cells(line: str) -> list[str]:
        return [cell.strip() for cell in line.strip().strip("|").split("|")]

    def is_separator(cells: list[str]) -> bool:
        non_empty_cells = [cell for cell in cells if cell]
        return bool(non_empty_cells) and all(
            set(cell) <= {"-", ":"} for cell in non_empty_cells
        )

    def compact(cells: list[str]) -> str:
        values = [cell for cell in cells if cell]
        compacted = []
        index = 0
        while index < len(values):
            value = values[index]
            if value in {"$", "€", "£"} and index + 1 < len(values):
                compacted.append(value + values[index + 1])
                index += 2
                continue
            if value == "%" and compacted:
                compacted[-1] += "%"
                index += 1
                continue
            compacted.append(value)
            index += 1
        return " | ".join(compacted)

    rows = []
    for line in element.content.strip().splitlines():
        cells = parse_cells(line)
        if cells and not is_separator(cells):
            rows.append(cells)

    if not rows:
        raise ValueError("Table does not contain a non-empty row")

    if len(rows) == 1:
        parts = [f"Table row:\n{compact(rows[0])}"]
    else:
        context_rows = [rows[0]]
        data_start = 1
        while data_start < len(rows) and not rows[data_start][0]:
            context_rows.append(rows[data_start])
            data_start += 1

        data_rows = rows[data_start:]
        if not data_rows:
            raise ValueError("Table does not contain a recognizable data row")

        context = "\n".join(
            compact(row) for row in context_rows if compact(row)
        )
        parts = []
        for row in data_rows:
            data_row = compact(row)
            if not data_row:
                continue
            parts.append(
                f"Table context:\n{context}\n\nData row:\n{data_row}"
            )

    if not parts:
        raise ValueError("Table does not contain a non-empty data row")

    filing_context = (
        f"Company: {filing.company_name} ({filing.ticker})\n"
        f"Form: {filing.form}"
    )
    parts = [f"{filing_context}\n\n{part}" for part in parts]

    if any(count_tokens(part, tokenizer) > chunk_tokens for part in parts):
        raise ValueError("Table row with context exceeds the chunk token limit")

    chunks = []
    for part in parts:
        sequence = starting_sequence + len(chunks)
        chunks.append(
            DocumentChunk(
                chunk_id=f"{filing.local_path.stem}-{sequence:05d}",
                sequence=sequence,
                chunk_type="table",
                content=part,
                token_count=count_tokens(part, tokenizer),
                source_sequences=(element.sequence,),
                filing=filing,
            )
        )

    return chunks


def chunk_document_elements(
    elements: list[DocumentElement],
    filing: FilingMetadata,
    tokenizer: Tokenizer,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[DocumentChunk]:
    """Chunk text and tables while preserving their original document order."""
    chunks = []
    text_run = []

    def flush_text_run() -> None:
        if not text_run:
            return

        chunks.extend(
            chunk_text_elements(
                elements=text_run,
                filing=filing,
                tokenizer=tokenizer,
                chunk_tokens=chunk_tokens,
                overlap_tokens=overlap_tokens,
                starting_sequence=len(chunks),
            )
        )
        text_run.clear()

    for element in elements:
        if element.element_type == "text":
            text_run.append(element)
            continue

        if element.element_type == "table":
            flush_text_run()
            chunks.extend(
                chunk_table_element(
                    element=element,
                    filing=filing,
                    tokenizer=tokenizer,
                    chunk_tokens=chunk_tokens,
                    starting_sequence=len(chunks),
                )
            )
            continue

        raise ValueError(f"Unsupported element type: {element.element_type}")

    flush_text_run()
    return chunks


def write_chunks_jsonl(
    chunks: list[DocumentChunk],
    output_path: Path,
) -> Path:
    """Write chunks and their filing provenance as JSON Lines."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        for chunk in chunks:
            record = {
                "chunk_id": chunk.chunk_id,
                "sequence": chunk.sequence,
                "chunk_type": chunk.chunk_type,
                "content": chunk.content,
                "token_count": chunk.token_count,
                "source_sequences": list(chunk.source_sequences),
                "filing": {
                    "ticker": chunk.filing.ticker,
                    "company_name": chunk.filing.company_name,
                    "cik": chunk.filing.cik,
                    "form": chunk.filing.form,
                    "filing_date": chunk.filing.filing_date,
                    "accession_number": chunk.filing.accession_number,
                    "source_url": chunk.filing.source_url,
                    "local_path": str(chunk.filing.local_path),
                },
            }
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    return output_path


def load_chunks_jsonl(input_path: Path) -> list[DocumentChunk]:
    """Load saved JSON Lines records back into document chunks."""
    chunks = []

    with input_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
                filing_record = record["filing"]
                filing = FilingMetadata(
                    ticker=filing_record["ticker"],
                    company_name=filing_record.get(
                        "company_name", filing_record["ticker"]
                    ),
                    cik=filing_record["cik"],
                    form=filing_record["form"],
                    filing_date=filing_record["filing_date"],
                    accession_number=filing_record["accession_number"],
                    source_url=filing_record["source_url"],
                    local_path=Path(filing_record["local_path"]),
                )
                chunks.append(
                    DocumentChunk(
                        chunk_id=record["chunk_id"],
                        sequence=record["sequence"],
                        chunk_type=record["chunk_type"],
                        content=record["content"],
                        token_count=record["token_count"],
                        source_sequences=tuple(record["source_sequences"]),
                        filing=filing,
                    )
                )
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(
                    f"Invalid chunk record on line {line_number}"
                ) from error

    chunk_ids = [chunk.chunk_id for chunk in chunks]
    sequences = [chunk.sequence for chunk in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Chunk IDs must be unique")
    if len(sequences) != len(set(sequences)):
        raise ValueError("Chunk sequence numbers must be unique")

    return chunks
