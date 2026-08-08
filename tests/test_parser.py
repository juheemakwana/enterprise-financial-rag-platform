from financial_rag.models import DocumentElement
from financial_rag.parser import (
    clean_html,
    extract_document_elements,
    extract_text_blocks,
    parse_filing_file,
    table_to_markdown,
    write_parsed_elements,
)


def test_clean_html_removes_non_content_and_preserves_financial_values() -> None:
    html = """
    <html>
      <head>
        <style>.hidden { display: none; }</style>
        <script>console.log("not document content")</script>
      </head>
      <body>
        <div style="display: none">
          <ix:header>
            <ix:hidden>
              Machine-only XBRL context 0001045810 2026 FY false
            </ix:hidden>
          </ix:header>
        </div>
        <div hidden>Hidden HTML metadata</div>
        <div aria-hidden="true">Hidden accessibility metadata</div>
        <h1>Annual Report</h1>
        <p>
          Revenue was
          <ix:nonfraction name="us-gaap:Revenue">130500</ix:nonfraction>.
        </p>
        <table>
          <tr><th>Year</th><th>Revenue</th></tr>
          <tr><td>2026</td><td>$130,500</td></tr>
        </table>
      </body>
    </html>
    """

    soup = clean_html(html)

    assert soup.find("script") is None
    assert soup.find("style") is None
    assert "Machine-only XBRL context" not in soup.get_text(" ", strip=True)
    assert "Hidden HTML metadata" not in soup.get_text(" ", strip=True)
    assert "Hidden accessibility metadata" not in soup.get_text(" ", strip=True)
    assert "Revenue was 130500" in soup.get_text(" ", strip=True)
    assert soup.find("table") is not None


def test_extract_text_blocks_avoids_parent_duplicates_and_table_text() -> None:
    html = """
    <html><body>
      <div class="page-wrapper">
        <h1>Annual Report</h1>
        <div>Management discusses business performance here.</div>
        <div>
          Revenue was
          <ix:nonfraction name="us-gaap:Revenue">130500</ix:nonfraction>.
        </div>
      </div>
      <table>
        <tr><th>Year</th><th>Revenue</th></tr>
        <tr><td>2026</td><td>$130,500</td></tr>
      </table>
    </body></html>
    """

    blocks = extract_text_blocks(clean_html(html), minimum_characters=1)

    assert blocks == [
        "Annual Report",
        "Management discusses business performance here.",
        "Revenue was 130500 .",
    ]


def test_table_to_markdown_preserves_rows_columns_and_xbrl_values() -> None:
    html = """
    <table>
      <tr><th>Year</th><th>Revenue | USD</th><th>Change</th></tr>
      <tr>
        <td>2026</td>
        <td><ix:nonfraction name="us-gaap:Revenue">130,500</ix:nonfraction></td>
        <td>114%</td>
      </tr>
      <tr><td colspan="2">Total revenue</td><td>130,500</td></tr>
    </table>
    """
    table = clean_html(html).find("table")

    markdown = table_to_markdown(table)

    assert markdown == (
        "| Year | Revenue \\| USD | Change |\n"
        "| --- | --- | --- |\n"
        "| 2026 | 130,500 | 114% |\n"
        "| Total revenue |  | 130,500 |"
    )


def test_extract_document_elements_preserves_text_and_table_order() -> None:
    html = """
    <html><body>
      <h1>Financial Results</h1>
      <div>Revenue increased during the year.</div>
      <table>
        <tr><th>Year</th><th>Revenue</th></tr>
        <tr><td>2026</td><td>130,500</td></tr>
      </table>
      <div>Management expects continued growth.</div>
    </body></html>
    """

    elements = extract_document_elements(
        clean_html(html),
        minimum_characters=1,
    )

    assert elements == [
        DocumentElement(0, "text", "Financial Results"),
        DocumentElement(1, "text", "Revenue increased during the year."),
        DocumentElement(
            2,
            "table",
            "| Year | Revenue |\n"
            "| --- | --- |\n"
            "| 2026 | 130,500 |",
        ),
        DocumentElement(3, "text", "Management expects continued growth."),
    ]


def test_write_parsed_elements_separates_text_and_tables(tmp_path) -> None:
    elements = [
        DocumentElement(0, "text", "Financial Results"),
        DocumentElement(
            1,
            "table",
            "| Year | Revenue |\n| --- | --- |\n| 2026 | 130,500 |",
        ),
        DocumentElement(2, "text", "Management discussion"),
    ]

    paths = write_parsed_elements(elements, tmp_path / "parsed")

    assert paths == [
        tmp_path / "parsed" / "text" / "00000.md",
        tmp_path / "parsed" / "tables" / "00001.md",
        tmp_path / "parsed" / "text" / "00002.md",
    ]
    assert paths[0].read_text(encoding="utf-8") == "Financial Results\n"
    assert paths[1].read_text(encoding="utf-8").startswith("| Year | Revenue |")


def test_parse_filing_file_reads_extracts_and_writes_html(tmp_path) -> None:
    input_path = tmp_path / "NVDA_10-K.html"
    input_path.write_text(
        """
        <html><body>
          <div style="display:none">Machine-only XBRL metadata</div>
          <h1>Financial Results</h1>
          <table>
            <tr><th>Year</th><th>Revenue</th></tr>
            <tr><td>2026</td><td>130,500</td></tr>
          </table>
        </body></html>
        """,
        encoding="utf-8",
    )
    output_directory = tmp_path / "parsed"

    paths = parse_filing_file(
        input_path,
        output_directory,
        minimum_characters=1,
    )

    assert paths == [
        output_directory / "text" / "00000.md",
        output_directory / "tables" / "00001.md",
    ]
    assert paths[0].read_text(encoding="utf-8") == "Financial Results\n"
    assert "| 2026 | 130,500 |" in paths[1].read_text(encoding="utf-8")
