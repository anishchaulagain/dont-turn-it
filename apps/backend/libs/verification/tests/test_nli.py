import asyncio

from verification import check_entailment


class WordOverlapScorer:
    """Stand-in for the NLI model: entailed if the hypothesis' words appear in the premise."""

    def __init__(self) -> None:
        self.calls: list[list[tuple[str, str]]] = []

    async def entailment(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.calls.append(pairs)
        out = []
        for premise, hypothesis in pairs:
            words = {w.strip(".,").lower() for w in hypothesis.split()}
            seen = {w.strip(".,").lower() for w in premise.split()}
            out.append(len(words & seen) / max(1, len(words)))
        return out


def test_entailment_both_directions() -> None:
    scorer = WordOverlapScorer()
    original = "The trial ended early ⟦C1⟧. Costs rose sharply."
    same = "Costs rose sharply. The trial ended early ⟦C1⟧."
    result = asyncio.run(check_entailment(original, same, scorer))
    assert result.passed
    premise, hypothesis = scorer.calls[0][0]
    assert "⟦" not in premise and "[ref]" in premise

    lost = asyncio.run(check_entailment(original, "The trial ended early ⟦C1⟧.", scorer))
    assert not lost.passed and lost.details["not_preserved"] == ["Costs rose sharply."]

    added = asyncio.run(check_entailment(original, same + " Investors panicked overnight.", scorer))
    assert not added.passed and added.details["not_supported"] == ["Investors panicked overnight."]
