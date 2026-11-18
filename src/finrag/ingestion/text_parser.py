"""Plain-text parser: paragraphs separated by blank lines, all on page 1."""

import re

from finrag.core.errors import ParseError
from finrag.ingestion.parsed import Block, ParagraphBlock, ParsedDocument

_BLANK_LINES = re.compile(r"\n\s*\n")


class TextParser:
    def parse(self, data: bytes) -> ParsedDocument:
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ParseError("The file is not valid UTF-8 text.") from exc
        if "\x00" in text:
            raise ParseError("The file contains binary data, not plain text.")
        paragraphs = [p.strip() for p in _BLANK_LINES.split(text.replace("\r\n", "\n"))]
        blocks: tuple[Block, ...] = tuple(ParagraphBlock(text=p, page=1) for p in paragraphs if p)
        if not blocks:
            raise ParseError("The file contains no text.")
        return ParsedDocument(blocks=blocks)
