"""PDF parser backed by Docling: layout analysis + table structure, OCR off by default.

SEC filings are born-digital, so OCR is unnecessary and slow (`Settings.pdf_ocr_enabled`).
Docling is imported lazily, on first parse, so app start and unit tests do not pay for it.

Mapping notes:
- Docling reports every section header as level 1, so heading levels come from a small SEC
  heuristic: "PART I" -> 1, "Item 1A. ..." -> 2, any other heading -> one below the last
  PART/ITEM heading (3 under an Item, 2 under a bare Part). Before any PART/ITEM heading, and
  for non-SEC documents, Docling's own level is kept. Skipped levels are fine for the chunker.
- Page headers/footers/page numbers are dropped (Docling classifies them as furniture; the
  label check is a second line of defence). The chunker does no cleanup of its own.
- Table captions Docling attaches to a table are emitted on the TableBlock, not as paragraphs.
- Bounding boxes are in PDF points with a bottom-left origin: y0 is the bottom edge.
"""

import io
import logging
import re
from typing import TYPE_CHECKING, Any

from finrag.core.domain import (
    Block,
    BoundingBox,
    HeadingBlock,
    ParagraphBlock,
    ParsedDocument,
    TableBlock,
)
from finrag.core.errors import ParseError

if TYPE_CHECKING:
    from docling.document_converter import DocumentConverter

log = logging.getLogger(__name__)

_PART = re.compile(r"^\s*part\s+[ivx]+\b", re.IGNORECASE)
_ITEM = re.compile(r"^\s*item\s+\d+[a-c]?\s*[.:\u2013\u2014-]", re.IGNORECASE)
_SKIPPED_LABELS = {"page_header", "page_footer"}
_TEXT_LABELS = {"text", "paragraph", "list_item", "footnote", "formula", "code", "caption"}


class DoclingParser:
    def __init__(self, ocr_enabled: bool = False) -> None:
        self._ocr_enabled = ocr_enabled
        self._converter: DocumentConverter | None = None

    def parse(self, data: bytes) -> ParsedDocument:
        from docling.datamodel.base_models import ConversionStatus, DocumentStream
        from docling.exceptions import ConversionError

        converter = self._get_converter()
        try:
            result = converter.convert(
                DocumentStream(name="document.pdf", stream=io.BytesIO(data)),
                raises_on_error=False,
            )
        except ConversionError as exc:
            raise ParseError(f"The PDF could not be read: {exc}") from exc
        if result.status not in (ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS):
            errors = "; ".join(e.error_message for e in result.errors) or "unknown error"
            raise ParseError(
                f"The PDF could not be read (corrupt or unsupported document): {errors}"
            )
        blocks = _map_document(result.document)
        if not blocks:
            raise ParseError("The PDF contains no extractable text.")
        return ParsedDocument(blocks=blocks)

    def _get_converter(self) -> "DocumentConverter":
        if self._converter is None:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption

            options = PdfPipelineOptions(do_ocr=self._ocr_enabled)
            self._converter = DocumentConverter(
                allowed_formats=[InputFormat.PDF],
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)},
            )
        return self._converter


def _map_document(doc: Any) -> tuple[Block, ...]:
    blocks: list[Block] = []
    caption_refs: set[str] = set()
    for item, _ in doc.iterate_items():
        if type(item).__name__ == "TableItem":
            caption_refs.update(ref.cref for ref in item.captions)
    structural_level = 0  # level of the last PART/ITEM heading; 0 = none seen yet
    for item, _ in doc.iterate_items():
        label = str(getattr(item.label, "value", item.label))
        if label in _SKIPPED_LABELS or not item.prov:
            continue
        page = item.prov[0].page_no
        bbox = _bbox(item.prov[0], doc, page)
        kind = type(item).__name__
        if kind == "TableItem":
            table = _table(item, doc, page, bbox)
            if table is not None:
                blocks.append(table)
        elif kind in ("SectionHeaderItem", "TitleItem"):
            text = item.text.strip()
            if not text:
                continue
            level, structural_level = _heading_level(
                text, 1 if kind == "TitleItem" else item.level, structural_level
            )
            blocks.append(HeadingBlock(text=text, level=level, page=page, bbox=bbox))
        elif label in _TEXT_LABELS and item.self_ref not in caption_refs:
            text = item.text.strip()
            if text:
                blocks.append(ParagraphBlock(text=text, page=page, bbox=bbox))
    return tuple(blocks)


def _heading_level(text: str, docling_level: int, structural_level: int) -> tuple[int, int]:
    """Return (level for this heading, updated structural level)."""
    if _PART.match(text):
        return 1, 1
    if _ITEM.match(text):
        return 2, 2
    if structural_level:
        return structural_level + 1, structural_level
    return max(docling_level, 1), structural_level


def _table(item: Any, doc: Any, page: int, bbox: BoundingBox | None) -> TableBlock | None:
    rows = [
        line for line in item.export_to_markdown(doc).splitlines() if line.lstrip().startswith("|")
    ]
    if len(rows) < 3:  # header + separator + at least one body row
        return None
    caption = item.caption_text(doc).strip() or None
    return TableBlock(
        markdown="\n".join(rows[2:]),
        page=page,
        header="\n".join(rows[:2]),
        caption=caption,
        bbox=bbox,
    )


def _bbox(prov: Any, doc: Any, page: int) -> BoundingBox | None:
    box = prov.bbox
    if box is None:
        return None
    if str(getattr(box.coord_origin, "value", box.coord_origin)) == "TOPLEFT":
        height = doc.pages[page].size.height
        top, bottom = height - box.t, height - box.b
    else:
        top, bottom = box.t, box.b
    return BoundingBox(
        x0=min(box.l, box.r),
        y0=min(top, bottom),
        x1=max(box.l, box.r),
        y1=max(top, bottom),
    )
