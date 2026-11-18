"""Generate the 10-K-like PDF fixtures deterministically (reportlab, no downloads).

Run: uv run python tests/fixtures/make_10k_extract.py
Writes `example_10k_extract.pdf` (born-digital, 3 pages: Part/Item headings, two financial
tables, running footer with page numbers) and `corrupt.pdf`
(garbage bytes behind a %PDF header). The company is fictional; figures are invented.
"""

from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

HERE = Path(__file__).parent
BODY = ParagraphStyle("body", fontName="Times-Roman", fontSize=10.5, leading=14, spaceAfter=8)
PART = ParagraphStyle("part", fontName="Helvetica-Bold", fontSize=18, leading=22, spaceAfter=10)
ITEM = ParagraphStyle("item", fontName="Helvetica-Bold", fontSize=14, leading=18, spaceAfter=8)
SUB = ParagraphStyle("sub", fontName="Helvetica-Bold", fontSize=11.5, leading=15, spaceAfter=6)
CAPTION = ParagraphStyle("cap", fontName="Helvetica-Oblique", fontSize=10, leading=13, spaceAfter=4)

FILLER = (
    "Our results of operations are affected by many factors, including consumer demand, "
    "competition, supply chain conditions and general economic trends. We believe our "
    "diversified product portfolio positions us well, but there can be no assurance that "
    "historical trends will continue. Management monitors these factors on an ongoing basis."
)


def footer(canvas: Any, doc: Any) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.drawString(72, 40, "Example Corp. | 2025 Form 10-K")
    canvas.drawRightString(540, 40, f"Page {doc.page}")
    canvas.restoreState()


def grid(rows: list[list[str]]) -> Table:
    table = Table(rows, colWidths=[200, 100, 100], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 9),
                ("FONT", (0, 1), (-1, -1), "Helvetica", 9),
                ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ]
        )
    )
    return table


def story() -> list[Any]:
    s: list[Any] = []
    s += [Paragraph("PART I", PART), Paragraph("Item 1. Business", ITEM)]
    s += [Paragraph("Overview", SUB)]
    s += [Paragraph(FILLER, BODY) for _ in range(3)]
    s += [Paragraph("Item 1A. Risk Factors", ITEM)]
    s += [Paragraph("Risks Related to Our Business", SUB)]
    s += [Paragraph(FILLER, BODY) for _ in range(4)]
    s += [PageBreak()]
    s += [Paragraph("PART II", PART)]
    s += [Paragraph("Item 7. Management's Discussion and Analysis", ITEM)]
    s += [Paragraph("Results of Operations", SUB)]
    s += [Paragraph(FILLER, BODY) for _ in range(2)]
    s += [Paragraph("Table 1: Net sales by segment (in millions of dollars)", CAPTION)]
    s += [
        grid(
            [
                ["Segment", "FY2025", "FY2024"],
                ["Americas", "12,450", "11,320"],
                ["Europe", "8,210", "7,980"],
                ["Asia Pacific", "6,940", "6,100"],
                ["Total net sales", "27,600", "25,400"],
            ]
        ),
        Spacer(1, 10),
    ]
    s += [Paragraph(FILLER, BODY) for _ in range(5)]
    s += [PageBreak()]
    s += [Paragraph("Item 8. Financial Statements and Supplementary Data", ITEM)]
    s += [Paragraph("Consolidated Balance Sheets", SUB)]
    s += [Paragraph(FILLER, BODY)]
    s += [Paragraph("Table 2: Selected balance sheet data (in millions of dollars)", CAPTION)]
    rows = [["Line item", "FY2025", "FY2024"]]
    rows += [[f"Line item {n}", f"{1000 + n * 37:,}", f"{900 + n * 31:,}"] for n in range(1, 16)]
    s += [grid(rows), Spacer(1, 10)]
    s += [Paragraph(FILLER, BODY) for _ in range(2)]
    return s


def build_pdf(path: Path) -> None:
    doc = BaseDocTemplate(
        str(path),
        pagesize=LETTER,
        invariant=1,  # fixed timestamps/ids so the output is byte-reproducible
        title="Example Corp. 2025 Form 10-K (extract)",
    )
    frame = Frame(72, 72, 468, 648, id="f")
    doc.addPageTemplates([PageTemplate(id="p", frames=[frame], onPage=footer)])
    doc.build(story())


if __name__ == "__main__":
    build_pdf(HERE / "example_10k_extract.pdf")
    (HERE / "corrupt.pdf").write_bytes(b"%PDF-1.7\n" + bytes(range(256)) * 8 + b"\x00garbage")
