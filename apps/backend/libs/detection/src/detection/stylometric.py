"""Explainable stylometric indicators (D3 in docs/PLAN.md §8A.2).

Weak on their own, but they are cheap, need no model, and every point of score
comes with a human-readable reason, which is what makes a highlight actionable.
The lexicon drifts with each LLM generation; keep it in sync with the labelled set.
"""

import math
import re
import statistics
from dataclasses import dataclass
from typing import Literal

from detection.types import SentenceScore, Window

# (phrase regex, weight, label). Weight ~ how strongly the phrase signals stock LLM prose.
_STOCK_PHRASES: list[tuple[str, float, str]] = [
    (r"it(?: is|'s) (?:important|worth|crucial) (?:to note|noting|to remember)", 1.0, "stock phrase"),
    (r"plays? an? (?:crucial|pivotal|vital|key|significant) role", 1.0, "stock phrase"),
    (r"in today'?s (?:fast-paced|ever-changing|digital|modern) (?:world|landscape|age)", 1.2, "stock opener"),
    (r"\bdelv(?:e|es|ing) (?:into|deeper)", 1.0, "LLM-favoured verb"),
    (r"\b(?:rich |intricate )?tapestry\b", 1.0, "LLM-favoured word"),
    (r"\ba testament to\b", 1.0, "stock phrase"),
    (r"\bnavigat(?:e|ing) the (?:complexities|intricacies|challenges)", 1.0, "stock phrase"),
    (r"\b(?:ever-evolving|ever-changing) (?:landscape|world|field)", 1.0, "stock phrase"),
    (r"\bunlock(?:s|ing)? the (?:full )?potential\b", 1.0, "stock phrase"),
    (r"\b(?:a|the) (?:realm|landscape) of\b", 0.7, "LLM-favoured word"),
    (r"\bnot only\b[^.]{3,80}\bbut also\b", 0.6, "'not only ... but also'"),
    (r"\bin (?:conclusion|summary)\b,?", 0.6, "formulaic wrap-up"),
    (r"\bunderscor(?:e|es|ed|ing)\b", 0.6, "LLM-favoured verb"),
    (r"\b(?:(?:showcas|leverag|streamlin|elevat)(?:e|es|ed|ing)|(?:foster|empower)(?:s|ed|ing)?|harness(?:es|ed|ing)?)\b", 0.4, "LLM-favoured verb"),
    (r"\b(?:pivotal|multifaceted|meticulous(?:ly)?|paramount|commendable|intricate|nuanced)\b", 0.5, "LLM-favoured word"),
    (r"\b(?:seamless(?:ly)?|robust|comprehensive|cutting-edge|transformative|holistic)\b", 0.35, "generic intensifier"),
    (r"\b(?:crucial|vital|essential|significant(?:ly)?|notably)\b", 0.2, "generic emphasis"),
]  # fmt: skip
_TECH_DOC_FILLER = [
    (r"\b(?:simply|just|easily|effortlessly)\b", 0.5, "filler in instructions"),
]
_OPENERS = re.compile(
    r"^(?:Moreover|Furthermore|Additionally|In addition|Overall|Ultimately|Consequently|"
    r"Notably|Importantly|Interestingly|Indeed|Thus|Hence|Therefore|In essence)\b,?"
)
_TRICOLON = re.compile(r"\b\w+(?: \w+){0,2}, \w+(?: \w+){0,2},? and \w+(?: \w+){0,2}\b")

Mode = Literal["essay", "story", "research_paper", "technical_docs", "business_docs", "generic"]


@dataclass(frozen=True)
class _Lexicon:
    phrases: list[tuple[re.Pattern[str], float, str]]
    burstiness_weight: float


def _lexicon(mode: Mode) -> _Lexicon:
    raw = list(_STOCK_PHRASES)
    burst = 1.0
    if mode == "technical_docs":
        # Good docs are deliberately uniform; rhythm is not evidence there.
        raw += _TECH_DOC_FILLER
        burst = 0.0
    elif mode == "business_docs":
        burst = 0.4
    return _Lexicon([(re.compile(p, re.IGNORECASE), w, label) for p, w, label in raw], burst)


def _burstiness(lengths: list[int]) -> float:
    """0..1: how uniform sentence lengths are (1 = suspiciously uniform)."""
    if len(lengths) < 3:
        return 0.0
    mean = statistics.fmean(lengths)
    cv = statistics.pstdev(lengths) / mean if mean else 0.0
    # Human prose typically has CV around 0.5-0.7; LLM prose often 0.2-0.35.
    return max(0.0, min(1.0, (0.55 - cv) / 0.3))


class StylometricProvider:
    name = "stylometric"
    kind: Literal["local"] = "local"

    def __init__(self, mode: Mode = "generic") -> None:
        self._lex = _lexicon(mode)

    def _sentence_signal(self, text: str) -> tuple[float, list[str]]:
        load, reasons = 0.0, []
        for pattern, weight, label in self._lex.phrases:
            for m in pattern.finditer(text):
                load += weight
                reasons.append(f"{label}: “{m.group(0)}”")
        opener = _OPENERS.match(text.strip())
        if opener:
            load += 0.5
            reasons.append(f"generic transition: “{opener.group(0)}”")
        if _TRICOLON.search(text):
            load += 0.1  # weak: lists of three are ordinary in human prose too
            reasons.append("list of three (tricolon)")
        if text.count("—") >= 2:
            load += 0.3
            reasons.append("heavy em-dash use")
        return load, reasons

    async def score(self, windows: list[Window]) -> list[SentenceScore]:
        per_sentence: dict[tuple[str, int], tuple[float, list[str], list[float]]] = {}
        for window in windows:
            uniform = _burstiness([s.words for s in window.sentences])
            for s in window.sentences:
                key = (s.block_id, s.sentence_index)
                if key not in per_sentence:
                    load, reasons = self._sentence_signal(s.text)
                    per_sentence[key] = (load, reasons, [])
                per_sentence[key][2].append(uniform)

        out: list[SentenceScore] = []
        for (block_id, idx), (load, reasons, uniforms) in per_sentence.items():
            uniform = statistics.fmean(uniforms) * self._lex.burstiness_weight
            reasons = list(reasons)
            if uniform >= 0.5:
                reasons.append("uniform sentence lengths around it (low burstiness)")
            # Saturating combination: phrases dominate, rhythm nudges.
            lexical = 1.0 - math.exp(-load)
            score = min(1.0, 0.75 * lexical + 0.25 * uniform)
            out.append(
                SentenceScore(
                    block_id=block_id,
                    sentence_index=idx,
                    provider=self.name,
                    score=round(score, 4),
                    reasons=reasons,
                )
            )
        return out
