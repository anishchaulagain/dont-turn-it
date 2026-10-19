"""Shared block construction: spans, sentences, rewritable decision, sections."""

import re
from typing import Any

from docmodel.ir import Block, BlockKind, ProtectedSpan
from docmodel.protect import detect_spans
from docmodel.segment import split_sentences

MIN_REWRITE_WORDS = 8
_REWRITABLE_KINDS = {BlockKind.PARAGRAPH, BlockKind.LIST_ITEM}
_REFERENCES_HEADING = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\.?\s+)?(references|bibliography|works cited|literature cited|"
    r"sources|reference list)\s*:?\s*$",
    re.IGNORECASE,
)


_FRONT_MATTER_HEADING = re.compile(
    r"^\s*(contents|table of contents|list of (figures|tables|abbreviations)|glossary)\s*:?\s*$",
    re.IGNORECASE,
)
# Captions typed as ordinary paragraphs: "Figure 3. ...", "Table 2: ...", "Fig. 1 ..."
_CAPTION = re.compile(r"^\s*(figure|fig\.|table|chart|diagram|image|plate)\s+[A-Z]?\d+[.:\s]", re.I)
# Manual table-of-contents lines end in a tab (or dot leader) and a page number.
_TOC_ENTRY = re.compile(r"(\t|\.{3,}\s*)\s*[ivxlc\d]+\s*$", re.IGNORECASE)
# A closing bracket alone does not end a sentence: "Word count: 3,119 (excl. references)".
_SENTENCE_END = re.compile(r"[.!?…:;][\"”’)\]]*\s*$|[\"”’]\s*$")


def is_references_heading(text: str) -> bool:
    return bool(_REFERENCES_HEADING.match(text))


def _prose_problem(text: str, sentences: int) -> str | None:
    """Why a paragraph-styled block is not really prose (labels, TOC lines, captions)."""
    if _TOC_ENTRY.search(text):
        return "toc_entry"
    if sentences <= 1 and not _SENTENCE_END.search(text):
        # Title-page lines and labels: "Module: 517HSC - An Introduction to ..."
        return "not_a_sentence"
    return None


class BlockBuilder:
    """Assigns ids/order and tracks the current section while blocks are added."""

    def __init__(self, *, protect_quotations: bool = True) -> None:
        self.blocks: list[Block] = []
        self._section: str | None = None
        self._in_references = False
        self._in_front_matter = False
        self._protect_quotations = protect_quotations

    def add(
        self,
        kind: BlockKind,
        text: str,
        *,
        anchor: dict[str, Any],
        inline_spans: list[ProtectedSpan] | None = None,
        style: str | None = None,
        heading_level: int | None = None,
        skip_reason: str | None = None,
    ) -> Block | None:
        if not text.strip():
            return None
        if kind is BlockKind.HEADING:
            self._section = text.strip()
            self._in_references = is_references_heading(text)
            self._in_front_matter = bool(_FRONT_MATTER_HEADING.match(self._section))
        elif self._in_references and kind in _REWRITABLE_KINDS:
            kind = BlockKind.REFERENCE_ENTRY
        elif kind is BlockKind.PARAGRAPH and _CAPTION.match(text):
            kind = BlockKind.CAPTION

        spans = detect_spans(
            text, protect_quotations=self._protect_quotations, existing=inline_spans or []
        )
        block = Block(
            id=f"b{len(self.blocks) + 1:05d}",
            kind=kind,
            order=len(self.blocks),
            text=text,
            sentences=split_sentences(text, spans) if kind is not BlockKind.CODE else [],
            spans=spans,
            style=style,
            heading_level=heading_level,
            section=self._section if kind is not BlockKind.HEADING else None,
            anchor=anchor,
        )
        if skip_reason is None:
            if kind not in _REWRITABLE_KINDS:
                skip_reason = f"kind:{kind.value}"
            elif self._in_front_matter:
                skip_reason = "front_matter"
            elif block.word_count < MIN_REWRITE_WORDS:
                skip_reason = "too_short"
            else:
                skip_reason = _prose_problem(text, len(block.sentences))
        block.rewritable = skip_reason is None
        block.skip_reason = skip_reason
        self.blocks.append(block)
        return block
