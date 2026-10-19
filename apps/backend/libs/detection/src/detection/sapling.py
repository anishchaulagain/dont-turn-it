"""Optional external reference detector: Sapling AI Detector (D5, docs/PLAN.md §8A.1).

Development and benchmarking only until Sapling grants written permission for
in-product use. Off unless explicitly constructed with a key. Never send user
documents without their consent, and never train on these outputs.

Free key quota (checked 2026-10-19): 50k characters/day, 250k/month.
"""

import re
from typing import Literal

import httpx

from detection.types import SentenceScore, Window

SAPLING_URL = "https://api.sapling.ai/api/v1/aidetect"


def _norm(text: str) -> str:
    return re.sub(r"\W+", " ", text).strip().lower()


class QuotaExceededError(RuntimeError):
    pass


class SaplingProvider:
    name = "sapling"
    kind: Literal["external"] = "external"

    def __init__(
        self,
        api_key: str,
        *,
        daily_char_budget: int = 50_000,
        client: httpx.AsyncClient | None = None,
        url: str = SAPLING_URL,
    ) -> None:
        if not api_key:
            raise ValueError("SaplingProvider needs an API key")
        self._key = api_key
        self._budget = daily_char_budget
        self._used = 0
        self._client = client
        self._url = url

    async def _detect(self, text: str) -> dict[str, object]:
        if self._used + len(text) > self._budget:
            raise QuotaExceededError("Sapling daily character budget exhausted")
        http = self._client or httpx.AsyncClient(timeout=60.0)
        try:
            response = await http.post(
                self._url, json={"key": self._key, "text": text, "sent_scores": True}
            )
            response.raise_for_status()
            self._used += len(text)
            data: dict[str, object] = response.json()
            return data
        finally:
            if self._client is None:
                await http.aclose()

    async def score(self, windows: list[Window]) -> list[SentenceScore]:
        # One request per distinct sentence run (windows overlap; send the union once).
        sentences = {(s.block_id, s.sentence_index): s for w in windows for s in w.sentences}
        if not sentences:
            return []
        ordered = list(sentences.values())
        data = await self._detect(" ".join(s.text for s in ordered))
        items = data.get("sentence_scores")
        remote = [
            (_norm(str(item.get("sentence", ""))), float(item.get("score", 0.0)))
            for item in (items if isinstance(items, list) else [])
            if isinstance(item, dict)
        ]
        out: list[SentenceScore] = []
        cursor = 0
        for s in ordered:
            target = _norm(s.text)
            for j in range(cursor, len(remote)):
                if remote[j][0] and (remote[j][0] in target or target in remote[j][0]):
                    out.append(
                        SentenceScore(
                            block_id=s.block_id,
                            sentence_index=s.sentence_index,
                            provider=self.name,
                            score=round(min(1.0, max(0.0, remote[j][1])), 4),
                        )
                    )
                    cursor = j + 1
                    break
        return out
