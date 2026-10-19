"""DOCX parsing and in-place export.

Parsing walks every body paragraph (tables included, text boxes excluded) and
splits it into segments:

* text in the paragraph's dominant run formatting: rewritable prose;
* everything else (differently formatted runs, hyperlinks, fields, footnote
  references, images, math, content controls): opaque *inline groups*, exposed
  to the rewriter as INLINE protected spans (⟦F1⟧ ...) and copied back verbatim;
* zero-width markers (bookmarks): kept, re-emitted around the new content.

Export re-opens the original file, re-runs the same analysis on the anchored
paragraph, and rebuilds only that paragraph's runs. Styles, numbering, headers,
footers, footnotes, comments elsewhere and every untouched paragraph stay
byte-identical in the XML.
"""

import copy
import io
import re
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

from docx import Document as open_docx
from docx.opc.oxml import serialize_part_xml
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

from docmodel.build import BlockBuilder
from docmodel.ir import BlockKind, DocFormat, Document, ProtectedSpan, SpanKind
from docmodel.protect import mask, split_tokens

Element = Any  # lxml element (python-docx's oxml classes); lxml ships no type stubs.

_W_P, _W_R, _W_T = qn("w:p"), qn("w:r"), qn("w:t")
_W_PPR, _W_RPR = qn("w:pPr"), qn("w:rPr")
_TEXTBOX, _TABLE_CELL = qn("w:txbxContent"), qn("w:tc")

# Run children that make a run opaque (not plain text).
_OPAQUE_RUN_CHILDREN = {
    qn(f"w:{t}")
    for t in (
        "footnoteReference", "endnoteReference", "commentReference", "drawing", "pict",
        "object", "sym", "fldChar", "instrText", "ptab", "ruby", "annotationRef",
    )
}  # fmt: skip
_MARKERS = {qn("w:bookmarkStart"), qn("w:bookmarkEnd")}
_DROP = {qn("w:proofErr"), qn("w:permStart"), qn("w:permEnd")}
_TRACKED = {qn(f"w:{t}") for t in ("ins", "del", "moveFrom", "moveTo")}
_COMMENT_RANGE = {qn("w:commentRangeStart"), qn("w:commentRangeEnd")}
# rPr children that do not change how text looks.
_RPR_IGNORE = {qn(f"w:{t}") for t in ("lang", "noProof", "rFonts", "spellStart", "spellEnd")}


@dataclass
class _Segment:
    kind: Literal["text", "inline", "marker"]
    text: str = ""
    elements: list[Element] = field(default_factory=list)
    signature: str | None = None


@dataclass
class ParagraphAnalysis:
    text: str
    spans: list[ProtectedSpan]
    groups: list[list[Element]]
    dominant_rpr: Element | None
    markers: list[Element]
    problems: list[str]


def _has_ancestor(el: Element, tags: set[str]) -> bool:
    parent = el.getparent()
    while parent is not None:
        if parent.tag in tags:
            return True
        parent = parent.getparent()
    return False


def iter_paragraphs(body: Element) -> Iterator[tuple[int, Element, bool]]:
    """Yield (index, w:p, in_table) in document order; index is the export anchor."""
    index = 0
    for p in body.iter(_W_P):
        if _has_ancestor(p, {_TEXTBOX}):
            continue
        yield index, p, _has_ancestor(p, {_TABLE_CELL})
        index += 1


def _rpr_signature(run: Element) -> str:
    rpr = run.find(_W_RPR)
    if rpr is None:
        return ""
    parts = []
    for child in rpr:
        if child.tag in _RPR_IGNORE:
            continue
        attrs = sorted((k, v) for k, v in child.attrib.items() if "rsid" not in k)
        parts.append(f"{child.tag}{attrs}")
    return "|".join(sorted(parts))


def _run_text(run: Element) -> str:
    out = []
    for child in run:
        tag = child.tag
        if tag == _W_T:
            out.append(child.text or "")
        elif tag == qn("w:tab"):
            out.append("\t")
        elif tag in (qn("w:br"), qn("w:cr")):
            out.append("\n")
        elif tag == qn("w:noBreakHyphen"):
            out.append("-")
    return "".join(out)


def _display_text(elements: list[Element]) -> str:
    return "".join(t.text or "" for el in elements for t in el.iter(_W_T))


def _segments(p: Element) -> tuple[list[_Segment], list[str]]:
    segments: list[_Segment] = []
    problems: list[str] = []
    field_group: list[Element] | None = None
    field_depth = 0

    for child in p:
        tag = child.tag
        if tag == _W_PPR or tag in _DROP:
            continue
        if field_group is not None:
            field_group.append(child)
            for fc in child.iter(qn("w:fldChar")):
                kind = fc.get(qn("w:fldCharType"))
                field_depth += 1 if kind == "begin" else -1 if kind == "end" else 0
            if field_depth == 0:
                segments.append(
                    _Segment("inline", _display_text(field_group), elements=field_group)
                )
                field_group = None
            continue
        if tag in _MARKERS:
            segments.append(_Segment("marker", elements=[child]))
            continue
        if tag in _TRACKED:
            problems.append("tracked_changes")
        if tag in _COMMENT_RANGE:
            problems.append("comments")
            segments.append(_Segment("marker", elements=[child]))
            continue
        if tag == _W_R:
            begins = [
                fc for fc in child.iter(qn("w:fldChar")) if fc.get(qn("w:fldCharType")) == "begin"
            ]
            if begins:
                field_group, field_depth = [child], 1
                continue
            if any(c.tag in _OPAQUE_RUN_CHILDREN for c in child):
                segments.append(_Segment("inline", _display_text([child]), elements=[child]))
            else:
                segments.append(
                    _Segment(
                        "text", _run_text(child), elements=[child], signature=_rpr_signature(child)
                    )
                )
            continue
        # hyperlink, fldSimple, smartTag, sdt, oMath, ins/del, ...: opaque.
        segments.append(_Segment("inline", _display_text([child]), elements=[child]))

    if field_group is not None:  # unterminated field: keep it opaque
        segments.append(_Segment("inline", _display_text(field_group), elements=field_group))
        problems.append("unterminated_field")
    return segments, problems


def analyze_paragraph(p: Element) -> ParagraphAnalysis:
    segments, problems = _segments(p)

    weight: dict[str, int] = {}
    for s in segments:
        if s.kind == "text" and s.text.strip():
            weight[s.signature or ""] = weight.get(s.signature or "", 0) + len(s.text)
    dominant = max(weight, key=lambda k: weight[k]) if weight else ""

    # Text runs in another format become inline groups; adjacent runs with the same
    # format merge into one group so Word's rsid splits don't multiply tokens.
    merged: list[_Segment] = []
    for s in segments:
        if s.kind == "text" and (s.signature == dominant or not s.text.strip()):
            s = _Segment("text", s.text, s.elements, dominant)
        elif s.kind == "text":
            s = _Segment("inline", s.text, s.elements, s.signature)
        prev = merged[-1] if merged else None
        if (
            prev is not None
            and prev.kind == s.kind
            and s.kind in ("text", "inline")
            and prev.signature is not None
            and prev.signature == s.signature
        ):
            prev.text += s.text
            prev.elements.extend(s.elements)
        else:
            merged.append(s)

    text_parts: list[str] = []
    spans: list[ProtectedSpan] = []
    groups: list[list[Element]] = []
    markers: list[Element] = []
    dominant_rpr: Element | None = None
    pos = 0
    for s in merged:
        if s.kind == "marker":
            markers.extend(s.elements)
            continue
        if s.kind == "inline":
            spans.append(
                ProtectedSpan(
                    kind=SpanKind.INLINE,
                    start=pos,
                    end=pos + len(s.text),
                    text=s.text,
                    ref=len(groups),
                )  # fmt: skip
            )
            groups.append(s.elements)
        elif dominant_rpr is None and s.elements:
            dominant_rpr = s.elements[0].find(_W_RPR)
        text_parts.append(s.text)
        pos += len(s.text)
    return ParagraphAnalysis(
        text="".join(text_parts),
        spans=spans,
        groups=groups,
        dominant_rpr=dominant_rpr,
        markers=markers,
        problems=sorted(set(problems)),
    )


_HEADING_STYLE = re.compile(r"^heading\s*(\d)$", re.IGNORECASE)


def _classify(style_name: str, has_numbering: bool, in_table: bool) -> tuple[BlockKind, int | None]:
    name = style_name.lower()
    if in_table:
        return BlockKind.TABLE_CELL, None
    heading = _HEADING_STYLE.match(style_name)
    if heading:
        return BlockKind.HEADING, int(heading.group(1))
    if name in ("title", "subtitle"):
        return BlockKind.HEADING, 0
    if name == "caption":
        return BlockKind.CAPTION, None
    if "quote" in name:
        return BlockKind.QUOTE, None
    if name in ("bibliography",):
        return BlockKind.REFERENCE_ENTRY, None
    if "code" in name or "source" in name or "macro" in name:
        return BlockKind.CODE, None
    if has_numbering or name.startswith("list"):
        return BlockKind.LIST_ITEM, None
    return BlockKind.PARAGRAPH, None


def parse_docx(data: bytes) -> Document:
    doc = open_docx(io.BytesIO(data))
    style_names = {s.style_id: s.name for s in doc.styles if s.style_id}
    builder = BlockBuilder()
    for index, p, in_table in iter_paragraphs(doc.element.body):
        analysis = analyze_paragraph(p)
        ppr = p.find(_W_PPR)
        style_id = ""
        has_numbering = False
        if ppr is not None:
            pstyle = ppr.find(qn("w:pStyle"))
            style_id = pstyle.get(qn("w:val"), "") if pstyle is not None else ""
            has_numbering = ppr.find(qn("w:numPr")) is not None
        style = style_names.get(style_id, style_id) or "Normal"
        kind, level = _classify(style, has_numbering, in_table)
        builder.add(
            kind,
            analysis.text,
            anchor={"p_index": index},
            inline_spans=analysis.spans,
            style=style,
            heading_level=level,
            skip_reason=analysis.problems[0] if analysis.problems else None,
        )
    return Document(format=DocFormat.DOCX, blocks=builder.blocks)


def _new_run(text: str, rpr: Element | None) -> Element:
    run = OxmlElement("w:r")
    if rpr is not None:
        run.append(copy.deepcopy(rpr))
    for i, line in enumerate(text.split("\n")):
        if i:
            run.append(OxmlElement("w:br"))
        for j, piece in enumerate(line.split("\t")):
            if j:
                run.append(OxmlElement("w:tab"))
            if piece:
                t = OxmlElement("w:t")
                t.text = piece
                t.set(qn("xml:space"), "preserve")
                run.append(t)
    return run


def apply_docx_rewrites(data: bytes, doc_ir: Document, replacements: dict[str, str]) -> bytes:
    """Rebuild the anchored paragraphs from masked replacement text.

    `replacements` maps block id -> rewritten text that still contains the block's
    placeholder tokens (⟦F1⟧, ⟦C2⟧, ...), exactly as verified by gate G1.
    """
    doc = open_docx(io.BytesIO(data))
    paragraphs = {i: p for i, p, _ in iter_paragraphs(doc.element.body)}
    for block_id, masked_text in replacements.items():
        block = doc_ir.block(block_id)
        p = paragraphs[block.anchor["p_index"]]
        analysis = analyze_paragraph(p)
        if analysis.text != block.text:
            raise ValueError(f"document changed under block {block.id}")
        _, legend = mask(block.text, block.spans)

        content: list[Element] = []
        for piece in split_tokens(masked_text):
            span = legend.get(piece)
            if span is None:
                if piece:
                    content.append(_new_run(piece, analysis.dominant_rpr))
            elif span.kind is SpanKind.INLINE and span.ref is not None:
                content.extend(copy.deepcopy(el) for el in analysis.groups[span.ref])
            else:
                content.append(_new_run(span.text, analysis.dominant_rpr))

        starts = [m for m in analysis.markers if m.tag == qn("w:bookmarkStart")]
        ends = [m for m in analysis.markers if m.tag != qn("w:bookmarkStart")]
        for child in list(p):
            if child.tag != _W_PPR:
                p.remove(child)
        for el in [*starts, *content, *ends]:
            p.append(el)

    return _replace_part(data, doc.part.partname.lstrip("/"), serialize_part_xml(doc.element))


def _replace_part(original: bytes, part_name: str, new_xml: bytes) -> bytes:
    """Copy the original package entry by entry and swap only the main document part.

    `Document.save()` would re-serialise every XML part (styles, numbering, footers,
    settings, core properties). Copying the other entries' raw bytes guarantees the
    export changes nothing but the rewritten paragraphs.
    """
    out = io.BytesIO()
    with (
        zipfile.ZipFile(io.BytesIO(original)) as src,
        zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as dst,
    ):
        for info in src.infolist():
            payload = new_xml if info.filename == part_name else src.read(info.filename)
            dst.writestr(info, payload, compress_type=info.compress_type)
    return out.getvalue()
