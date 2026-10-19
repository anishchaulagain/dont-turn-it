"""Run providers over windows, fuse their scores, and band them (docs/PLAN.md §3.4).

Fusion weights below are *uncalibrated defaults*. Phase 0 fits a logistic-regression
fusion + isotonic calibration per writing mode on the labelled set and replaces
them via `FusionConfig`.
"""

import asyncio
from collections import defaultdict

import structlog
from pydantic import BaseModel, Field

from detection.types import Band, DetectorProvider, SentenceScore
from detection.windows import build_windows, eligible_words
from docmodel.ir import Document

logger = structlog.get_logger(__name__)

DOCUMENT_MIN_WORDS = 300


class FusionConfig(BaseModel):
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "desklib": 0.45,
            "fakespot": 0.25,
            "fast_detectgpt": 0.15,
            "stylometric": 0.15,
            # External references are reported, never fused (docs/PLAN.md §8A.1).
            "sapling": 0.0,
        }
    )
    low_below: float = 0.35
    elevated_from: float = 0.65
    # Providers disagreeing by more than this on a sentence -> abstain.
    max_disagreement: float = 0.6
    # Document-level share of elevated words below this is shown as "*".
    low_share_threshold: float = 0.2


class SentenceAnalysis(BaseModel):
    block_id: str
    sentence_index: int
    score: float
    band: Band
    provider_scores: dict[str, float]
    reasons: list[str]


class DocumentAnalysis(BaseModel):
    sentences: list[SentenceAnalysis]
    eligible_words: int
    elevated_share: float | None
    document_band: str  # "insufficient_text" | "*" | "low" | "uncertain" | "elevated"
    providers: list[str]
    failed_providers: dict[str, str] = Field(default_factory=dict)


def _band(score: float, spread: float, cfg: FusionConfig) -> Band:
    if spread > cfg.max_disagreement:
        return Band.ABSTAIN
    if score < cfg.low_below:
        return Band.LOW
    if score >= cfg.elevated_from:
        return Band.ELEVATED
    return Band.UNCERTAIN


async def analyze(
    doc: Document,
    providers: list[DetectorProvider],
    cfg: FusionConfig | None = None,
    *,
    window_size: int = 5,
) -> DocumentAnalysis:
    cfg = cfg or FusionConfig()
    windows = build_windows(doc, size=window_size)
    results = await asyncio.gather(*(p.score(windows) for p in providers), return_exceptions=True)

    failed: dict[str, str] = {}
    per_sentence: dict[tuple[str, int], list[SentenceScore]] = defaultdict(list)
    for provider, result in zip(providers, results, strict=True):
        if isinstance(result, BaseException):
            failed[provider.name] = repr(result)
            logger.warning("detector_failed", provider=provider.name, error=repr(result))
            continue
        for score in result:
            per_sentence[(score.block_id, score.sentence_index)].append(score)

    words = {(s.block_id, s.sentence_index): s.words for w in windows for s in w.sentences}
    sentences: list[SentenceAnalysis] = []
    for key, scores in per_sentence.items():
        by_provider = {s.provider: s.score for s in scores}
        fused_inputs = [(cfg.weights.get(p, 0.0), v) for p, v in by_provider.items()]
        total = sum(w for w, _ in fused_inputs)
        if total == 0:
            continue
        fused = sum(w * v for w, v in fused_inputs) / total
        weighted = [v for p, v in by_provider.items() if cfg.weights.get(p, 0.0) > 0]
        spread = max(weighted) - min(weighted) if len(weighted) > 1 else 0.0
        sentences.append(
            SentenceAnalysis(
                block_id=key[0],
                sentence_index=key[1],
                score=round(fused, 4),
                band=_band(fused, spread, cfg),
                provider_scores=by_provider,
                reasons=[r for s in scores for r in s.reasons],
            )
        )
    order = {(b.id, s.index): i for i, b in enumerate(doc.blocks) for s in b.sentences}
    sentences.sort(key=lambda s: order.get((s.block_id, s.sentence_index), 0))

    total_words = eligible_words(doc)
    if total_words < DOCUMENT_MIN_WORDS:
        share, doc_band = None, "insufficient_text"
    else:
        elevated = sum(
            words.get((s.block_id, s.sentence_index), 0)
            for s in sentences
            if s.band is Band.ELEVATED
        )
        share = round(elevated / total_words, 4)
        if share < cfg.low_share_threshold:
            doc_band = "*"
        else:
            doc_band = Band.ELEVATED.value if share >= 0.5 else Band.UNCERTAIN.value

    return DocumentAnalysis(
        sentences=sentences,
        eligible_words=total_words,
        elevated_share=share,
        document_band=doc_band,
        providers=[p.name for p in providers if p.name not in failed],
        failed_providers=failed,
    )
