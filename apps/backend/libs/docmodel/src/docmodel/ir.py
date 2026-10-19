"""Document intermediate representation shared by every stage.

A document is an ordered list of blocks (paragraph, heading, list item, ...).
Each block carries its plain text, its sentences, and the protected spans that a
rewrite must reproduce exactly. `anchor` points back into the source file so an
export can replace exactly that block and leave everything else untouched.
"""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class DocFormat(StrEnum):
    TXT = "txt"
    MD = "md"
    DOCX = "docx"


class BlockKind(StrEnum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE_CELL = "table_cell"
    CAPTION = "caption"
    QUOTE = "quote"
    CODE = "code"
    REFERENCE_ENTRY = "reference_entry"


class SpanKind(StrEnum):
    CITATION = "citation"
    URL = "url"
    DOI = "doi"
    EMAIL = "email"
    QUOTATION = "quotation"
    CODE = "code"
    MATH = "math"
    LINK = "link"
    # Source-format inline element the rewrite must not touch: a differently
    # formatted run, hyperlink, footnote reference, field, image, ...
    INLINE = "inline"


class ProtectedSpan(BaseModel):
    kind: SpanKind
    start: int
    end: int
    text: str
    # Parser-specific handle, e.g. the index of the DOCX inline element group.
    ref: int | None = None


class Sentence(BaseModel):
    index: int
    start: int
    end: int
    text: str


class Block(BaseModel):
    id: str
    kind: BlockKind
    order: int
    text: str
    sentences: list[Sentence] = Field(default_factory=list)
    spans: list[ProtectedSpan] = Field(default_factory=list)
    rewritable: bool = False
    skip_reason: str | None = None
    style: str | None = None
    heading_level: int | None = None
    # Section heading this block sits under (used for section-aware policies).
    section: str | None = None
    anchor: dict[str, Any] = Field(default_factory=dict)

    @property
    def word_count(self) -> int:
        return len(self.text.split())


class Document(BaseModel):
    format: DocFormat
    blocks: list[Block]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def word_count(self) -> int:
        return sum(b.word_count for b in self.blocks)

    def block(self, block_id: str) -> Block:
        for b in self.blocks:
            if b.id == block_id:
                return b
        raise KeyError(block_id)
