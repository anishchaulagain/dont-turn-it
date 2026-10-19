"""Protected spans: text a rewrite must reproduce character for character.

Before a block goes to the LLM, every protected span is replaced by a placeholder
token such as ⟦C1⟧ (citation 1). The verification gate G1 checks that each token
comes back exactly once, and `unmask` restores the original text.
"""

import re
from collections.abc import Iterable

from docmodel.ir import ProtectedSpan, SpanKind

TOKEN_OPEN, TOKEN_CLOSE = "⟦", "⟧"
TOKEN_RE = re.compile(r"⟦([A-Z])(\d+)⟧")

_LETTER = {
    SpanKind.CITATION: "C",
    SpanKind.URL: "U",
    SpanKind.DOI: "D",
    SpanKind.EMAIL: "E",
    SpanKind.QUOTATION: "Q",
    SpanKind.CODE: "K",
    SpanKind.MATH: "M",
    SpanKind.LINK: "L",
    SpanKind.INLINE: "F",
}

_YEAR = r"(?:1[6-9]|20)\d{2}[a-z]?"
_NAME = r"[A-Z][A-Za-z'’\-]+"

_PATTERNS: list[tuple[SpanKind, re.Pattern[str]]] = [
    # Markdown / inline code and math first so their contents are not re-matched.
    (SpanKind.CODE, re.compile(r"`[^`\n]+`")),
    (SpanKind.MATH, re.compile(r"\$[^$\n]+\$|\\\([^\n]+?\\\)")),
    (SpanKind.LINK, re.compile(r"!?\[[^\]\n]+\]\([^)\s]+\)")),
    (SpanKind.DOI, re.compile(r"\b(?:https?://(?:dx\.)?doi\.org/)?10\.\d{4,9}/[^\s\"<>]+")),
    (SpanKind.URL, re.compile(r"\bhttps?://[^\s<>\"]+|\bwww\.[^\s<>\"]+")),
    (SpanKind.EMAIL, re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")),
    # Parenthetical author-year, possibly several: (see Smith, 2020; Lee & Wu, 2019a, p. 4)
    (
        SpanKind.CITATION,
        re.compile(
            r"\((?:(?:see|e\.g\.,?|cf\.|i\.e\.,?)\s+)?[A-Z][^()\n]{0,200}?\b"
            + _YEAR
            + r"(?:[^()\n]{0,60})?\)"
        ),
    ),
    # Narrative: Smith (2020), Smith and Lee (2019), Smith et al. (2021, p. 3)
    (
        SpanKind.CITATION,
        re.compile(
            _NAME
            + r"(?:\s+(?:and|&)\s+"
            + _NAME
            + r"|\s+et\s+al\.)?,?\s+\("
            + _YEAR
            + r"(?:,\s*pp?\.\s*\d+(?:[-–]\d+)?)?\)"
        ),
    ),
    # MLA: (Smith 45), (Smith and Lee 45-47)
    (
        SpanKind.CITATION,
        re.compile(
            r"\(" + _NAME + r"(?:\s+(?:and|&)\s+" + _NAME + r"|\s+et\s+al\.)?\s+\d+(?:[-–]\d+)?\)"
        ),
    ),
    # Numeric: [3], [3, 5], [3-7], [3–5, 9]
    (SpanKind.CITATION, re.compile(r"\[\d+(?:\s*[-–,]\s*\d+)*\]")),
    # Quotations of 3+ words (straight or curly double quotes).
    (
        SpanKind.QUOTATION,
        re.compile(r"“[^”\n]*?\S+\s+\S+\s+\S+[^”\n]*?”|\"[^\"\n]*?\S+\s+\S+\s+\S+[^\"\n]*?\""),
    ),
]

_TRAILING_PUNCT = ".,;:!?)]}'\""


def _trim_trailing(kind: SpanKind, start: int, end: int, text: str) -> int:
    """URLs/DOIs swallow sentence punctuation; give it back."""
    if kind not in (SpanKind.URL, SpanKind.DOI):
        return end
    while end > start and text[end - 1] in _TRAILING_PUNCT:
        # Keep a closing paren that belongs to the URL itself.
        if text[end - 1] == ")" and text[start:end].count("(") >= text[start:end].count(")"):
            break
        end -= 1
    return end


def detect_spans(
    text: str,
    *,
    protect_quotations: bool = True,
    existing: Iterable[ProtectedSpan] = (),
) -> list[ProtectedSpan]:
    """Find semantic protected spans. Spans overlapping `existing` ones are dropped."""
    taken = sorted(existing, key=lambda s: s.start)
    found: list[ProtectedSpan] = list(taken)
    for kind, pattern in _PATTERNS:
        if kind is SpanKind.QUOTATION and not protect_quotations:
            continue
        for m in pattern.finditer(text):
            start, end = m.start(), _trim_trailing(kind, m.start(), m.end(), text)
            if end <= start or any(s.start < end and start < s.end for s in found):
                continue
            found.append(ProtectedSpan(kind=kind, start=start, end=end, text=text[start:end]))
    return sorted(found, key=lambda s: s.start)


def token_for(kind: SpanKind, n: int) -> str:
    return f"{TOKEN_OPEN}{_LETTER[kind]}{n}{TOKEN_CLOSE}"


def mask(text: str, spans: list[ProtectedSpan]) -> tuple[str, dict[str, ProtectedSpan]]:
    """Replace spans by numbered tokens. Returns masked text and token -> span."""
    counters: dict[SpanKind, int] = {}
    legend: dict[str, ProtectedSpan] = {}
    out: list[str] = []
    pos = 0
    for span in sorted(spans, key=lambda s: s.start):
        counters[span.kind] = counters.get(span.kind, 0) + 1
        token = token_for(span.kind, counters[span.kind])
        legend[token] = span
        out.append(text[pos : span.start])
        out.append(token)
        pos = span.end
    out.append(text[pos:])
    return "".join(out), legend


def tokens_in(text: str) -> list[str]:
    return [m.group(0) for m in TOKEN_RE.finditer(text)]


def split_tokens(text: str) -> list[str]:
    """Split masked text into plain-text pieces and tokens, in order."""
    parts: list[str] = []
    pos = 0
    for m in TOKEN_RE.finditer(text):
        if m.start() > pos:
            parts.append(text[pos : m.start()])
        parts.append(m.group(0))
        pos = m.end()
    if pos < len(text):
        parts.append(text[pos:])
    return parts


def unmask(text: str, legend: dict[str, ProtectedSpan]) -> str:
    """Restore original span text. Unknown tokens raise: G1 should have caught them."""

    def restore(m: re.Match[str]) -> str:
        span = legend.get(m.group(0))
        if span is None:
            raise KeyError(f"unknown placeholder {m.group(0)}")
        return span.text

    return TOKEN_RE.sub(restore, text)
