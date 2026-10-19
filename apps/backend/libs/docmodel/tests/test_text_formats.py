from docmodel import BlockKind, DocFormat, export, mask, parse

TXT = """Introduction

Artificial intelligence plays a crucial role in modern research (Smith, 2020). Moreover, \
it is important to note that results vary.

Short line.

References

Smith, J. (2020). A study of things. Journal of Stuff, 4(2), 1-10.
"""


def test_txt_blocks_and_reference_section() -> None:
    doc = parse(TXT.encode(), DocFormat.TXT)
    assert [b.kind for b in doc.blocks] == [
        BlockKind.HEADING,
        BlockKind.PARAGRAPH,
        BlockKind.PARAGRAPH,
        BlockKind.HEADING,
        BlockKind.REFERENCE_ENTRY,
    ]
    body = doc.blocks[1]
    assert body.rewritable and body.section == "Introduction"
    assert len(body.sentences) == 2
    assert doc.blocks[2].skip_reason == "too_short"
    assert not doc.blocks[4].rewritable


def test_txt_export_only_touches_rewritten_block() -> None:
    src = TXT.replace("\n", "\r\n")
    doc = parse(src.encode(), DocFormat.TXT)
    body = doc.blocks[1]
    masked, _ = mask(body.text, body.spans)
    new = masked.replace("plays a crucial role in", "shapes").replace(
        "Moreover, it is important to note that results", "Results"
    )
    out = export(src.encode(), doc, {body.id: new}).decode()
    rewritten = "Artificial intelligence shapes modern research (Smith, 2020). Results vary."
    assert rewritten in out
    assert out.replace(rewritten, body.text) == src


MD = """# Title

Some *emphasised* intro paragraph that is long enough to be rewritten by the system.

- First list item that is long enough to be considered for rewriting here.
- Short item

```python
print("code stays")
```

> A quoted block stays as it is.

| a | b |
|---|---|
"""


def test_markdown_structure() -> None:
    doc = parse(MD.encode(), DocFormat.MD)
    assert [b.kind for b in doc.blocks] == [
        BlockKind.HEADING,
        BlockKind.PARAGRAPH,
        BlockKind.LIST_ITEM,
        BlockKind.LIST_ITEM,
        BlockKind.CODE,
        BlockKind.QUOTE,
        BlockKind.TABLE_CELL,
    ]
    para = doc.blocks[1]
    masked, _ = mask(para.text, para.spans)
    assert "⟦F1⟧" in masked and "*emphasised*" not in masked
    assert doc.blocks[2].text.startswith("First list item")
    assert [b.rewritable for b in doc.blocks] == [False, True, True, False, False, False, False]


def test_markdown_export_keeps_markers_and_emphasis() -> None:
    doc = parse(MD.encode(), DocFormat.MD)
    item = doc.blocks[2]
    out = export(MD.encode(), doc, {item.id: "A rewritten first item."}).decode()
    assert "- A rewritten first item.\n- Short item" in out
    para = doc.blocks[1]
    masked, _ = mask(para.text, para.spans)
    out = export(MD.encode(), doc, {para.id: masked.replace("intro", "opening")}).decode()
    assert "Some *emphasised* opening paragraph" in out


def test_non_prose_paragraphs_are_not_rewritten() -> None:
    src = (
        # > 10 words, so the TXT heading heuristic does not claim it.
        "Module: 517HSC – An Introduction to Healthy Development and Wellbeing Across Life\n\n"
        "Word count: 3,119 words (excluding title page, contents, references and appendices)\n\n"
        "Contents\n\n"
        "2. Physiological Development Across Life Stages (LO1)\t5\n\n"
        "Introduction\n\n"
        "Figure 1. Liam's ecological system (after Bronfenbrenner, 1979) shown as rings.\n\n"
        "Chapter one covers the methods used in this report in some detail.... 4\n\n"
        "Liam is nine and attends a local primary school with his two younger siblings.\n"
    )
    doc = parse(src.encode(), DocFormat.TXT)
    reasons = {b.text[:12]: (b.kind.value, b.skip_reason) for b in doc.blocks}
    assert reasons["Module: 517H"] == ("paragraph", "not_a_sentence")
    assert reasons["Word count: "] == ("paragraph", "not_a_sentence")
    assert reasons["2. Physiolog"] == ("paragraph", "front_matter")
    assert reasons["Figure 1. Li"] == ("caption", "kind:caption")
    assert reasons["Chapter one "] == ("paragraph", "toc_entry")
    assert reasons["Liam is nine"] == ("paragraph", None)
