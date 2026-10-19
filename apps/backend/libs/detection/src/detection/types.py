from enum import StrEnum
from typing import Literal, Protocol

from pydantic import BaseModel, Field


class SentenceRef(BaseModel):
    block_id: str
    sentence_index: int
    text: str
    words: int


class Window(BaseModel):
    """Consecutive prose sentences inside one section (Turnitin-style overlap)."""

    index: int
    sentences: list[SentenceRef]

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.sentences)


class SentenceScore(BaseModel):
    block_id: str
    sentence_index: int
    provider: str
    score: float = Field(ge=0.0, le=1.0)  # 1 = AI-like
    reasons: list[str] = Field(default_factory=list)


class Band(StrEnum):
    LOW = "low"
    UNCERTAIN = "uncertain"
    ELEVATED = "elevated"
    ABSTAIN = "abstain"


class DetectorProvider(Protocol):
    @property
    def name(self) -> str: ...

    # "external" providers send text to a third party (docs/PLAN.md §8A.1).
    @property
    def kind(self) -> Literal["local", "external"]: ...

    async def score(self, windows: list[Window]) -> list[SentenceScore]: ...
