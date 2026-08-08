"""Tools for parsing and cleaning SEC filing HTML."""

import re
from pathlib import Path

from bs4 import BeautifulSoup, Tag

from financial_rag.models import DocumentElement


TEXT_BLOCK_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "div"]
HIDDEN_STYLE_PATTERN = re.compile(
    r"(?:display\s*:\s*none|visibility\s*:\s*hidden)",
    re.IGNORECASE,
)


def clean_html(html: str) -> BeautifulSoup:
    """Parse HTML and remove elements that are not document content."""
    soup = BeautifulSoup(html, "html.parser")

    for element in soup.find_all(["script", "style", "noscript"]):
        element.decompose()

    for element in soup.find_all(True):
        if element.attrs is None:
            continue

        style = str(element.get("style", ""))
        is_hidden = element.has_attr("hidden")
        is_aria_hidden = str(element.get("aria-hidden", "")).lower() == "true"

        if is_hidden or is_aria_hidden or HIDDEN_STYLE_PATTERN.search(style):
            element.decompose()

    return soup


def extract_text_blocks(
    soup: BeautifulSoup,
    minimum_characters: int = 20,
) -> list[str]:
    """Extract non-table text from the smallest meaningful HTML blocks."""
    blocks = []

    for element in soup.find_all(TEXT_BLOCK_TAGS):
        if element.find_parent("table") is not None:
            continue

        if element.find(TEXT_BLOCK_TAGS) is not None:
            continue

        text = re.sub(r"\s+", " ", element.get_text(" ", strip=True)).strip()
        if len(text) >= minimum_characters:
            blocks.append(text)

    return blocks


def table_to_markdown(table: Tag) -> str:
    """Convert one HTML table into a rectangular Markdown table."""
    rows = []

    for row in table.find_all("tr"):
        values = []

        for cell in row.find_all(["th", "td"], recursive=False):
            text = re.sub(r"\s+", " ", cell.get_text(" ", strip=True)).strip()
            text = text.replace("|", "\\|")

            try:
                colspan = max(int(cell.get("colspan", 1)), 1)
            except (TypeError, ValueError):
                colspan = 1

            values.append(text)
            values.extend([""] * (colspan - 1))

        if values and any(values):
            rows.append(values)

    if not rows:
        return ""

    column_count = max(len(row) for row in rows)
    rectangular_rows = [
        row + [""] * (column_count - len(row))
        for row in rows
    ]

    header = rectangular_rows[0]
    body = rectangular_rows[1:]
    markdown_lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * column_count) + " |",
    ]
    markdown_lines.extend(
        "| " + " | ".join(row) + " |"
        for row in body
    )
    return "\n".join(markdown_lines)


def extract_document_elements(
    soup: BeautifulSoup,
    minimum_characters: int = 20,
) -> list[DocumentElement]:
    """Extract text and tables in their original document order."""
    elements = []
    candidate_tags = [*TEXT_BLOCK_TAGS, "table"]

    for element in soup.find_all(candidate_tags):
        if element.name == "table":
            if element.find_parent("table") is not None:
                continue
            content = table_to_markdown(element)
            element_type = "table"
        else:
            if element.find_parent("table") is not None:
                continue
            if element.find(candidate_tags) is not None:
                continue
            content = re.sub(
                r"\s+",
                " ",
                element.get_text(" ", strip=True),
            ).strip()
            element_type = "text"

        if len(content) < minimum_characters:
            continue

        elements.append(
            DocumentElement(
                sequence=len(elements),
                element_type=element_type,
                content=content,
            )
        )

    return elements


def write_parsed_elements(
    elements: list[DocumentElement],
    output_directory: Path,
) -> list[Path]:
    """Write ordered text and table elements into separate directories."""
    text_directory = output_directory / "text"
    table_directory = output_directory / "tables"
    text_directory.mkdir(parents=True, exist_ok=True)
    table_directory.mkdir(parents=True, exist_ok=True)

    paths = []
    for element in elements:
        destination = (
            table_directory
            if element.element_type == "table"
            else text_directory
        )
        path = destination / f"{element.sequence:05d}.md"
        path.write_text(element.content + "\n", encoding="utf-8")
        paths.append(path)

    return paths


def parse_filing_file(
    input_path: Path,
    output_directory: Path,
    minimum_characters: int = 20,
) -> list[Path]:
    """Parse one SEC filing HTML file and save its document elements."""
    html = input_path.read_text(encoding="utf-8")
    soup = clean_html(html)
    elements = extract_document_elements(
        soup,
        minimum_characters=minimum_characters,
    )
    return write_parsed_elements(elements, output_directory)
