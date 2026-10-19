"""Overlapping sentence windows over the prose of a document.

Windows never cross a section heading or include non-prose blocks (tables, code,
references); every eligible sentence is covered by up to `size` windows, whose
scores are pooled back onto it.
"""

from detection.types import SentenceRef, Window
from docmodel.ir import BlockKind, Document

PROSE_KINDS = {BlockKind.PARAGRAPH, BlockKind.LIST_ITEM}
# Paragraph-styled blocks the parser recognised as not being prose at all.
NOT_PROSE = {"front_matter", "toc_entry", "not_a_sentence"}
MIN_SENTENCE_WORDS = 3


def prose_runs(doc: Document) -> list[list[SentenceRef]]:
    """Group eligible sentences into runs split at headings and non-prose blocks."""
    runs: list[list[SentenceRef]] = [[]]
    for block in doc.blocks:
        if block.kind not in PROSE_KINDS or block.skip_reason in NOT_PROSE:
            if runs[-1]:
                runs.append([])
            continue
        for s in block.sentences:
            words = len(s.text.split())
            if words >= MIN_SENTENCE_WORDS:
                runs[-1].append(
                    SentenceRef(block_id=block.id, sentence_index=s.index, text=s.text, words=words)
                )
    return [r for r in runs if r]


def build_windows(doc: Document, size: int = 5, stride: int = 1) -> list[Window]:
    windows: list[Window] = []
    for run in prose_runs(doc):
        if len(run) <= size:
            windows.append(Window(index=len(windows), sentences=run))
            continue
        for start in range(0, len(run) - size + 1, stride):
            windows.append(Window(index=len(windows), sentences=run[start : start + size]))
        if (len(run) - size) % stride:
            windows.append(Window(index=len(windows), sentences=run[-size:]))
    return windows


def eligible_words(doc: Document) -> int:
    return sum(s.words for run in prose_runs(doc) for s in run)
