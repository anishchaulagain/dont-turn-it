"""Plain-text and Markdown parsers.

Every block's anchor is a character range in the source string, so export can
splice rewritten text back in and leave all other bytes untouched.
"""

import re

from docmodel.build import BlockBuilder
from docmodel.ir import BlockKind, DocFormat, Document, ProtectedSpan, SpanKind

_BLANK = re.compile(r"\r?\n(?:[ \t]*\r?\n)+")
_MD_HEADING = re.compile(r"^(#{1,6})[ \t]+(.*?)[ \t#]*$")
_MD_FENCE = re.compile(r"^[ \t]*(```|~~~)")
_MD_LIST = re.compile(r"^[ \t]*(?:[-*+]|\d+[.)])[ \t]+")
_MD_EMPHASIS = re.compile(
    r"\*\*[^*\n]+\*\*|(?<![*\w])\*[^*\n]+\*(?![*\w])|__[^_\n]+__|(?<![_\w])_[^_\n]+_(?![_\w])"
)


def _looks_like_heading(text: str) -> bool:
    line = text.strip()
    return (
        "\n" not in line
        and "\t" not in line  # "2. Methods\t5" is a contents entry, not a heading
        and 0 < len(line.split()) <= 10
        and not line.endswith((".", ",", ";", ":", "?", "!"))
    )


def parse_txt(source: str) -> Document:
    builder = BlockBuilder()
    pos = 0
    for sep in [*_BLANK.finditer(source), None]:
        end = sep.start() if sep else len(source)
        chunk = source[pos:end]
        if chunk.strip():
            s = pos + (len(chunk) - len(chunk.lstrip()))
            e = pos + len(chunk.rstrip())
            text = source[s:e]
            kind = BlockKind.HEADING if _looks_like_heading(text) else BlockKind.PARAGRAPH
            builder.add(kind, text, anchor={"start": s, "end": e})
        pos = sep.end() if sep else len(source)
    return Document(format=DocFormat.TXT, blocks=builder.blocks)


def _emphasis_spans(text: str) -> list[ProtectedSpan]:
    return [
        ProtectedSpan(kind=SpanKind.INLINE, start=m.start(), end=m.end(), text=m.group(0))
        for m in _MD_EMPHASIS.finditer(text)
    ]


def parse_markdown(source: str) -> Document:
    builder = BlockBuilder()
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))

    def add_range(
        kind: BlockKind,
        s: int,
        e: int,
        *,
        heading_level: int | None = None,
        skip_reason: str | None = None,
    ) -> None:
        raw = source[s:e]
        lead = len(raw) - len(raw.lstrip())
        s, e = s + lead, s + len(raw.rstrip())
        text = source[s:e]
        prose = kind in (BlockKind.PARAGRAPH, BlockKind.LIST_ITEM)
        builder.add(
            kind,
            text,
            anchor={"start": s, "end": e},
            inline_spans=_emphasis_spans(text) if prose else [],
            heading_level=heading_level,
            skip_reason=skip_reason,
        )

    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if _MD_FENCE.match(line):
            j = i + 1
            while j < len(lines) and not _MD_FENCE.match(lines[j]):
                j += 1
            end_line = min(j + 1, len(lines))
            add_range(BlockKind.CODE, offsets[i], offsets[end_line])
            i = end_line
            continue
        heading = _MD_HEADING.match(line.rstrip("\r\n"))
        if heading:
            s = offsets[i] + heading.start(2)
            add_range(
                BlockKind.HEADING,
                s,
                s + len(heading.group(2)),
                heading_level=len(heading.group(1)),
            )
            i += 1
            continue
        if line.lstrip().startswith("|"):
            j = i
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                j += 1
            add_range(BlockKind.TABLE_CELL, offsets[i], offsets[j], skip_reason="table")
            i = j
            continue
        if line.lstrip().startswith(">"):
            j = i
            while j < len(lines) and lines[j].lstrip().startswith(">"):
                j += 1
            add_range(BlockKind.QUOTE, offsets[i], offsets[j])
            i = j
            continue
        marker = _MD_LIST.match(line)
        if marker:
            # One block per item; indented continuation lines belong to the item.
            j = i + 1
            while (
                j < len(lines)
                and lines[j].strip()
                and lines[j][:1] in (" ", "\t")
                and not _MD_LIST.match(lines[j])
            ):
                j += 1
            add_range(BlockKind.LIST_ITEM, offsets[i] + marker.end(), offsets[j])
            i = j
            continue
        j = i + 1
        while (
            j < len(lines)
            and lines[j].strip()
            and not _MD_HEADING.match(lines[j].rstrip("\r\n"))
            and not _MD_FENCE.match(lines[j])
            and not _MD_LIST.match(lines[j])
            and not lines[j].lstrip().startswith((">", "|"))
        ):
            j += 1
        add_range(BlockKind.PARAGRAPH, offsets[i], offsets[j])
        i = j
    return Document(format=DocFormat.MD, blocks=builder.blocks)


def apply_text_rewrites(source: str, doc: Document, replacements: dict[str, str]) -> str:
    """Splice replacement text (already unmasked) into the source by block anchor."""
    out = source
    targets = sorted(
        (doc.block(block_id) for block_id in replacements),
        key=lambda b: b.anchor["start"],
        reverse=True,
    )
    for block in targets:
        s, e = block.anchor["start"], block.anchor["end"]
        if out[s:e] != block.text:
            raise ValueError(f"source changed under block {block.id}")
        out = out[:s] + replacements[block.id] + out[e:]
    return out
