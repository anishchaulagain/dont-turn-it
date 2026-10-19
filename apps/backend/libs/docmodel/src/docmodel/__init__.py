"""Document IR, parsers, protected spans and lossless export."""

from docmodel.docx_format import apply_docx_rewrites, parse_docx
from docmodel.ir import (
    Block,
    BlockKind,
    DocFormat,
    Document,
    ProtectedSpan,
    Sentence,
    SpanKind,
)
from docmodel.parsers.text import apply_text_rewrites, parse_markdown, parse_txt
from docmodel.protect import detect_spans, mask, split_tokens, tokens_in, unmask
from docmodel.segment import split_sentences


def parse(data: bytes, fmt: DocFormat) -> Document:
    if fmt is DocFormat.DOCX:
        return parse_docx(data)
    text = data.decode("utf-8-sig")
    return parse_markdown(text) if fmt is DocFormat.MD else parse_txt(text)


def export(data: bytes, doc: Document, rewrites: dict[str, str]) -> bytes:
    """Apply rewrites (block id -> masked rewritten text) to the original file."""
    if doc.format is DocFormat.DOCX:
        return apply_docx_rewrites(data, doc, rewrites)
    text = data.decode("utf-8-sig")
    plain = {
        block_id: unmask(masked, mask(doc.block(block_id).text, doc.block(block_id).spans)[1])
        for block_id, masked in rewrites.items()
    }
    return apply_text_rewrites(text, doc, plain).encode("utf-8")


__all__ = [
    "Block",
    "BlockKind",
    "DocFormat",
    "Document",
    "ProtectedSpan",
    "Sentence",
    "SpanKind",
    "apply_docx_rewrites",
    "apply_text_rewrites",
    "detect_spans",
    "export",
    "mask",
    "parse",
    "parse_docx",
    "parse_markdown",
    "parse_txt",
    "split_sentences",
    "split_tokens",
    "tokens_in",
    "unmask",
]
