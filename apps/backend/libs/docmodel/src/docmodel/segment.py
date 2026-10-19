"""Rule-based sentence segmentation tuned for academic and technical prose.

Handles abbreviations (e.g., et al., Fig.), initials, decimals, citations that
trail a full stop ("... growth. [12] Next"), and never splits inside a protected
span (citation, quotation, URL, code).
"""

import re

from docmodel.ir import ProtectedSpan, Sentence

_ABBREVIATIONS = {
    "e.g", "i.e", "etc", "et al", "al", "cf", "vs", "viz", "approx", "ca", "resp",
    "fig", "figs", "eq", "eqs", "tab", "sec", "secs", "ch", "chap", "vol", "vols",
    "no", "nos", "pp", "p", "ed", "eds", "rev", "dept", "univ", "inc", "ltd", "co",
    "corp", "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "jan", "feb", "mar",
    "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "u.s", "u.k",
    "ph.d", "m.sc", "b.sc", "a.m", "p.m", "min", "max", "est",
}  # fmt: skip

# Candidate boundary: terminal punctuation, optional closing quotes/brackets, whitespace.
_BOUNDARY = re.compile(r"[.!?…]+[\"'”’)\]]*(?=\s+)")
# What may start a new sentence.
_STARTER = re.compile(r"\s+[\"'“‘(\[]?[A-Z0-9⟦]")
# A numeric citation placed after the full stop belongs to the previous sentence.
_TRAILING_CITATION = re.compile(r"\s+\[\d+(?:\s*[-–,]\s*\d+)*\](?=\s|$)")


def _word_before(text: str, end: int) -> str:
    m = re.search(r"([A-Za-z][A-Za-z.]*)\.?$", text[:end])
    return m.group(1).lower().rstrip(".") if m else ""


def _is_abbreviation(text: str, dot: int) -> bool:
    word = _word_before(text, dot)
    if not word:
        return False
    if word in _ABBREVIATIONS:
        return True
    # Single initial: "J. Smith"
    return len(word) == 1 and word.isalpha() and text[dot - 1].isupper()


def split_sentences(text: str, protected: list[ProtectedSpan] | None = None) -> list[Sentence]:
    spans = protected or []
    cuts: list[int] = []
    for m in _BOUNDARY.finditer(text):
        end = m.end()
        if any(s.start < end < s.end for s in spans):
            continue
        punct = text[m.start()]
        if punct == "." and m.end() - m.start() == 1 and _is_abbreviation(text, m.start()):
            continue
        if not _STARTER.match(text, end):
            continue
        trailing = _TRAILING_CITATION.match(text, end)
        if trailing:
            end = trailing.end()
        cuts.append(end)

    sentences: list[Sentence] = []
    start = 0
    for cut in [*cuts, len(text)]:
        chunk = text[start:cut]
        lead = len(chunk) - len(chunk.lstrip())
        s, e = start + lead, start + len(chunk.rstrip())
        if e > s:
            sentences.append(Sentence(index=len(sentences), start=s, end=e, text=text[s:e]))
        start = cut
    return sentences
