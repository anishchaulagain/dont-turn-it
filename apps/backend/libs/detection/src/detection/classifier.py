"""Window classifiers (D1/D2 in docs/PLAN.md §8A.2) and their HTTP client.

The models themselves run in the ml-inference service; this module only sends
window texts and pools the window probabilities back onto sentences.
"""

import statistics
from collections.abc import Awaitable, Callable
from typing import Literal

import httpx

from detection.types import SentenceScore, Window

ClassifyFn = Callable[[list[str]], Awaitable[list[float]]]


def pool_windows(
    windows: list[Window], probs: list[float], provider: str, how: Literal["mean", "max"] = "mean"
) -> list[SentenceScore]:
    collected: dict[tuple[str, int], list[float]] = {}
    for window, p in zip(windows, probs, strict=True):
        for s in window.sentences:
            collected.setdefault((s.block_id, s.sentence_index), []).append(p)
    pool = statistics.fmean if how == "mean" else max
    return [
        SentenceScore(
            block_id=block_id,
            sentence_index=idx,
            provider=provider,
            score=round(min(1.0, max(0.0, pool(values))), 4),
        )
        for (block_id, idx), values in collected.items()
    ]


class WindowClassifierProvider:
    kind: Literal["local"] = "local"

    def __init__(self, name: str, classify: ClassifyFn, batch_size: int = 16) -> None:
        self.name = name
        self._classify = classify
        self._batch = batch_size

    async def score(self, windows: list[Window]) -> list[SentenceScore]:
        probs: list[float] = []
        for i in range(0, len(windows), self._batch):
            probs.extend(await self._classify([w.text for w in windows[i : i + self._batch]]))
        return pool_windows(windows, probs, self.name)


def ml_inference_classifier(
    base_url: str, model: str, client: httpx.AsyncClient | None = None, timeout: float = 60.0
) -> ClassifyFn:
    """POST {base_url}/v1/detect {"model": ..., "texts": [...]} -> {"probs": [...]}."""

    async def classify(texts: list[str]) -> list[float]:
        http = client or httpx.AsyncClient(timeout=timeout)
        try:
            response = await http.post(
                f"{base_url.rstrip('/')}/v1/detect", json={"model": model, "texts": texts}
            )
            response.raise_for_status()
            probs = response.json()["probs"]
            return [float(p) for p in probs]
        finally:
            if client is None:
                await http.aclose()

    return classify
