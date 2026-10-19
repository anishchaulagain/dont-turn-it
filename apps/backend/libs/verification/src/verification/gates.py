"""Deterministic knowledge-preservation gates (docs/PLAN.md §4.1).

All gates compare the *masked* original with the *masked* rewrite, so citations,
URLs and formatted inline elements are already reduced to tokens (⟦C1⟧, ⟦F2⟧)
and their years/numbers are not double-counted.

G1 placeholders · G2 numbers · G3 entities/glossary · G4 polarity & modality ·
S3 length ratio. G5 (bidirectional NLI) lives in `verification.nli` because it
needs a model.
"""

import re
from collections import Counter
from typing import Any

from pydantic import BaseModel, Field

from docmodel.protect import TOKEN_RE, tokens_in


class GateResult(BaseModel):
    gate: str
    passed: bool
    hard: bool = True
    score: float | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class GatePolicy(BaseModel):
    """Per writing-mode strictness (docs/PLAN.md §3.0)."""

    # Research papers: adding hedges (weakening a claim) also fails.
    strict_hedges: bool = False
    # Business / process docs: must/shall/should/may must match exactly (RFC 2119).
    strict_deontic: bool = False
    glossary: list[str] = Field(default_factory=list)
    # Removing stock filler ("Moreover, it is important to note that") legitimately
    # shortens text by 20-45%; real information loss is caught by G2-G5, not here.
    min_length_ratio: float = 0.5
    max_length_ratio: float = 1.25


# ---------- G1: placeholders ----------
def check_placeholders(original: str, rewrite: str) -> GateResult:
    want, got = Counter(tokens_in(original)), Counter(tokens_in(rewrite))
    missing = sorted((want - got).elements())
    extra = sorted((got - want).elements())
    return GateResult(
        gate="G1_placeholders",
        passed=not missing and not extra,
        details={"missing": missing, "unexpected_or_duplicated": extra},
    )


# ---------- G2: numbers ----------
_UNITS_WORDS = {
    "zero": 0, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}  # fmt: skip
# "one" is left out on purpose: "one of the reasons" is not a quantity.
_MULTIPLIERS = {"hundred": 100, "thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}
_NUM_WORD = "|".join(sorted(_UNITS_WORDS, key=len, reverse=True))
_NUMBER_RE = re.compile(
    r"(?P<cur>[$€£¥])?\s?"
    # Digits not glued to letters ("Qwen3", "H2O" are identifiers, not quantities).
    r"(?:(?<![A-Za-z_\d.,])(?P<digits>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"|(?i:\b(?P<words>(?:" + _NUM_WORD + r")(?:[-\s](?:" + _NUM_WORD + r"))?)\b))"
    r"(?:(?i:\s?(?P<mult>hundred|thousand|million|billion)\b)|(?P<smult>[kKMB])(?![A-Za-z]))?"
    r"(?i:\s?(?P<pct>%|percent\b|per cent\b|percentage points?\b))?",
)
_SHORT_MULT = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}


def _words_value(words: str) -> float:
    return float(sum(_UNITS_WORDS[w] for w in re.split(r"[-\s]+", words.lower()) if w))


def extract_numbers(text: str) -> Counter[str]:
    plain = TOKEN_RE.sub(" ", text)
    found: Counter[str] = Counter()
    for m in _NUMBER_RE.finditer(plain):
        if not (m.group("digits") or m.group("words")):
            continue
        if m.group("digits"):
            value = float(m.group("digits").replace(",", ""))
        else:
            value = _words_value(m.group("words"))
        if m.group("mult"):
            value *= _MULTIPLIERS[m.group("mult").lower()]
        elif m.group("smult"):
            value *= _SHORT_MULT[m.group("smult").lower()]
        unit = "%" if m.group("pct") else (m.group("cur") or "")
        if "point" in (m.group("pct") or "").lower():
            unit = "pp"
        found[f"{unit}{value:g}"] += 1
    return found


def check_numbers(original: str, rewrite: str) -> GateResult:
    want, got = extract_numbers(original), extract_numbers(rewrite)
    missing = sorted((want - got).elements())
    added = sorted((got - want).elements())
    return GateResult(
        gate="G2_numbers",
        passed=not missing and not added,
        details={"missing": missing, "added": added},
    )


# ---------- G3: entities & glossary ----------
_SENTENCE_START = re.compile(r"(?:^|[.!?]\s+|[\n:;]\s*|[\"“(]\s*)$")
_CAP_WORD = r"[A-Z][a-z]+(?:[-'’][A-Z]?[a-z]+)?"
_ACRONYM = r"[A-Z]{2,}[A-Za-z0-9]*"
_CAMEL = r"[A-Z][a-z]+[A-Z][A-Za-z]*"
_CAP_SEQ = re.compile(
    rf"\b(?:{_CAMEL}|{_CAP_WORD}|{_ACRONYM})(?:\s+(?:{_CAMEL}|{_CAP_WORD}|{_ACRONYM}))*"
)
_NOT_ENTITIES = {"I", "A", "An", "The", "This", "That", "These", "Those", "It", "We", "They"}


def extract_entities(text: str) -> set[str]:
    """Heuristic proper-noun finder (capitalised words not at sentence start,
    acronyms, CamelCase). spaCy NER in ml-inference supplements it."""
    plain = TOKEN_RE.sub(" ", text)
    found: set[str] = set()
    for m in _CAP_SEQ.finditer(plain):
        words = m.group(0).split()
        at_start = bool(_SENTENCE_START.search(plain[: m.start()]))
        if at_start:
            # Drop the sentence-initial word unless it is an acronym/CamelCase.
            head = words[0]
            if not (head.isupper() and len(head) > 1) and not re.search(r"[a-z][A-Z]", head):
                words = words[1:]
        words = [w for w in words if w not in _NOT_ENTITIES]
        if words:
            found.add(" ".join(words))
    return found


def _contains(text: str, phrase: str, *, case_sensitive: bool = False) -> bool:
    flags = 0 if case_sensitive else re.IGNORECASE
    return re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text, flags) is not None


def check_entities(original: str, rewrite: str, glossary: list[str] | None = None) -> GateResult:
    orig_entities, new_entities = extract_entities(original), extract_entities(rewrite)
    missing = sorted(e for e in orig_entities if not _contains(rewrite, e))
    added = sorted(e for e in new_entities if not _contains(original, e))
    glossary_missing = sorted(
        term
        for term in glossary or []
        if _contains(original, term, case_sensitive=True)
        and not _contains(rewrite, term, case_sensitive=True)
    )
    return GateResult(
        gate="G3_entities",
        passed=not missing and not added and not glossary_missing,
        details={"missing": missing, "added": added, "glossary_missing": glossary_missing},
    )


# ---------- G4: polarity, hedging, certainty, quantifiers, obligations ----------
def _lexicon(*words: str) -> re.Pattern[str]:
    return re.compile(r"\b(?:" + "|".join(words) + r")\b", re.IGNORECASE)


_NEGATION = re.compile(
    r"\b(?:not|no|never|none|nothing|nobody|neither|nor|cannot|without)\b|n['’]t\b", re.IGNORECASE
)
_HEDGES = _lexicon(
    "may", "might", "could", "possibly", "perhaps", "potentially", "likely", "unlikely",
    "probably", "suggests?", "suggested", "indicates?", "indicated", "appears?", "appeared",
    "seems?", "seemed", "plausibly", "presumably", "tends? to", "partly", "partially",
    "somewhat", "approximately", "roughly", "nearly", "almost",
)  # fmt: skip
# Causal verbs ("causes") are left to NLI: as nouns they are too ambiguous here.
_CERTAINTY = _lexicon(
    "certainly", "definitely", "clearly", "undoubtedly", "obviously", "proves?", "proved",
    "proven", "demonstrates?", "demonstrated", "confirms?", "confirmed", "establishes",
    "always", "invariably", "conclusively", "guarantees?",
)  # fmt: skip
_UNIVERSAL = _lexicon("all", "every", "each", "always", "never", "none", "entire(?:ly)?")
_PARTIAL = _lexicon(
    "some", "many", "most", "few", "several", "often", "sometimes", "rarely", "usually",
    "frequently", "occasionally", "generally", "typically", "majority", "minority",
)  # fmt: skip
_DEONTIC = re.compile(
    r"\b(must not|must|shall not|shall|should not|should|may not|may|required|"
    r"is not required|optional|recommended|prohibited)\b",
    re.IGNORECASE,
)


# Obligation strength, checked in every mode: "must" -> "should" weakens a claim
# even in an essay. Synonyms within a class ("must" / "need to") are fine.
# "may"/"can" are left to the hedge check because they are mostly epistemic.
_OBLIGATION_STRONG = _lexicon(
    "must", "shall", "required to", "is required", "are required", "needs? to", "have to",
    "has to", "mandatory", "obliged to",
)  # fmt: skip
_OBLIGATION_MEDIUM = _lexicon("should", "ought to", "recommended", "advisable")


def _count(pattern: re.Pattern[str], text: str) -> int:
    return len(pattern.findall(TOKEN_RE.sub(" ", text)))


def check_polarity(original: str, rewrite: str, policy: GatePolicy | None = None) -> GateResult:
    policy = policy or GatePolicy()
    problems: list[str] = []
    counts: dict[str, tuple[int, int]] = {}
    for name, pattern in (
        ("negation", _NEGATION),
        ("hedges", _HEDGES),
        ("certainty", _CERTAINTY),
        ("universal", _UNIVERSAL),
        ("partial", _PARTIAL),
        ("obligation_strong", _OBLIGATION_STRONG),
        ("obligation_medium", _OBLIGATION_MEDIUM),
    ):
        counts[name] = (_count(pattern, original), _count(pattern, rewrite))

    if counts["obligation_strong"][0] != counts["obligation_strong"][1]:
        problems.append("obligation_strength_changed")
    elif counts["obligation_medium"][0] != counts["obligation_medium"][1]:
        problems.append("obligation_strength_changed")

    if counts["negation"][0] != counts["negation"][1]:
        problems.append("negation_changed")
    hedge_o, hedge_r = counts["hedges"]
    if hedge_r < hedge_o:
        problems.append("hedge_removed")  # claim strengthened
    if policy.strict_hedges and hedge_r > hedge_o:
        problems.append("hedge_added")  # claim weakened
    cert_o, cert_r = counts["certainty"]
    if cert_r > cert_o:
        problems.append("certainty_added")
    if counts["universal"][0] != counts["universal"][1]:
        problems.append("universal_quantifier_changed")
    if counts["partial"][0] > 0 and counts["partial"][1] == 0:
        problems.append("partial_quantifier_removed")
    if policy.strict_deontic:
        want = Counter(m.lower() for m in _DEONTIC.findall(original))
        got = Counter(m.lower() for m in _DEONTIC.findall(rewrite))
        if want != got:
            problems.append("obligation_changed")
            counts["deontic"] = (sum(want.values()), sum(got.values()))
    return GateResult(
        gate="G4_polarity_modality",
        passed=not problems,
        details={"problems": problems, "counts": counts},
    )


# ---------- S3: length ----------
def check_length(original: str, rewrite: str, policy: GatePolicy | None = None) -> GateResult:
    policy = policy or GatePolicy()
    n_o = max(1, len(TOKEN_RE.sub(" x ", original).split()))
    ratio = len(TOKEN_RE.sub(" x ", rewrite).split()) / n_o
    return GateResult(
        gate="S3_length",
        passed=policy.min_length_ratio <= ratio <= policy.max_length_ratio,
        hard=False,
        score=round(ratio, 3),
    )


class Verdict(BaseModel):
    passed: bool
    results: list[GateResult]

    @property
    def failures(self) -> list[GateResult]:
        return [r for r in self.results if not r.passed]

    def feedback(self) -> str:
        """Human-readable failure summary, fed back to the rewriter on retry."""
        lines = []
        for r in self.failures:
            facts = {k: v for k, v in r.details.items() if v and k != "counts"}
            lines.append(f"- {r.gate}: {facts or r.score}")
        return "\n".join(lines)


def run_deterministic_gates(
    original_masked: str, rewrite_masked: str, policy: GatePolicy | None = None
) -> Verdict:
    policy = policy or GatePolicy()
    results = [
        check_placeholders(original_masked, rewrite_masked),
        check_numbers(original_masked, rewrite_masked),
        check_entities(original_masked, rewrite_masked, policy.glossary),
        check_polarity(original_masked, rewrite_masked, policy),
        check_length(original_masked, rewrite_masked, policy),
    ]
    return Verdict(passed=all(r.passed for r in results), results=results)
