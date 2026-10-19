"""G5: bidirectional entailment between original and rewrite (docs/PLAN.md §4.1).

forward  (original sentence  ⇐ whole rewrite):  catches information loss
backward (rewrite sentence   ⇐ whole original): catches added / hallucinated content

Checking each sentence against the whole other paragraph (SummaC-style) tolerates
merges and splits without needing an explicit sentence alignment.
"""

from typing import Protocol

from docmodel.protect import TOKEN_RE
from docmodel.segment import split_sentences
from verification.gates import GateResult


class EntailmentScorer(Protocol):
    async def entailment(self, pairs: list[tuple[str, str]]) -> list[float]:
        """P(premise entails hypothesis) for each (premise, hypothesis) pair."""
        ...


def _readable(masked: str) -> str:
    # NLI models never saw ⟦C1⟧ tokens; a neutral stand-in keeps grammar intact.
    return TOKEN_RE.sub("[ref]", masked)


async def check_entailment(
    original_masked: str,
    rewrite_masked: str,
    scorer: EntailmentScorer,
    *,
    forward_threshold: float = 0.8,
    backward_threshold: float = 0.8,
) -> GateResult:
    original, rewrite = _readable(original_masked), _readable(rewrite_masked)
    orig_sents = [s.text for s in split_sentences(original)] or [original]
    new_sents = [s.text for s in split_sentences(rewrite)] or [rewrite]

    pairs = [(rewrite, s) for s in orig_sents] + [(original, s) for s in new_sents]
    scores = await scorer.entailment(pairs)
    forward, backward = scores[: len(orig_sents)], scores[len(orig_sents) :]

    lost = [s for s, p in zip(orig_sents, forward, strict=True) if p < forward_threshold]
    added = [s for s, p in zip(new_sents, backward, strict=True) if p < backward_threshold]
    return GateResult(
        gate="G5_entailment",
        passed=not lost and not added,
        score=round(min(min(forward), min(backward)), 4),
        details={
            "not_preserved": lost,
            "not_supported": added,
            "forward": [round(p, 4) for p in forward],
            "backward": [round(p, 4) for p in backward],
        },
    )
