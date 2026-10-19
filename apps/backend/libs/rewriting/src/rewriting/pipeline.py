"""End-to-end run on one file: analyse -> rewrite -> export -> re-analyse.

Shared by the `dtt` CLI and the document service. The output file is the original
with only the rewritten paragraphs replaced (DOCX: only word/document.xml changes).
Scores are AI-likeness *indicators* from the configured detectors, before and after.
"""

import statistics
from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, Field

from detection import Band, DetectorProvider, DocumentAnalysis, analyze
from docmodel import DocFormat, Document, export, parse
from rewriting.engine import BlockOutcome, EngineConfig, RewriteEngine
from rewriting.llm import ChatLLM
from rewriting.modes import Mode
from rewriting.voices import Voice
from verification import EntailmentScorer

Scope = Literal["flagged", "flagged_uncertain", "all"]
ProgressFn = Callable[[str, int, int], None]


class ScoreSummary(BaseModel):
    # Word-weighted mean AI-likeness over eligible sentences, 0 (human-like) .. 1.
    score: float | None
    document_band: str
    elevated_share: float | None
    eligible_words: int
    bands: dict[str, int] = Field(default_factory=dict)
    providers: list[str] = Field(default_factory=list)


class ParagraphChange(BaseModel):
    block_id: str
    section: str | None
    status: str
    reason: str | None
    original: str
    rewrite: str | None
    score_before: float | None
    score_after: float | None
    flagged_sentences: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class PipelineResult(BaseModel):
    mode: str
    scope: Scope
    before: ScoreSummary
    after: ScoreSummary
    paragraphs: list[ParagraphChange]
    output: bytes = Field(repr=False, exclude=True)


def _words(doc: Document) -> dict[tuple[str, int], int]:
    return {(b.id, s.index): len(s.text.split()) for b in doc.blocks for s in b.sentences}


def summarize(doc: Document, analysis: DocumentAnalysis) -> ScoreSummary:
    words = _words(doc)
    weighted = [(s.score, words.get((s.block_id, s.sentence_index), 0)) for s in analysis.sentences]
    total = sum(w for _, w in weighted)
    bands: dict[str, int] = {}
    for s in analysis.sentences:
        bands[s.band.value] = bands.get(s.band.value, 0) + 1
    return ScoreSummary(
        score=round(sum(v * w for v, w in weighted) / total, 4) if total else None,
        document_band=analysis.document_band,
        elevated_share=analysis.elevated_share,
        eligible_words=analysis.eligible_words,
        bands=bands,
        providers=analysis.providers,
    )


def _block_scores(analysis: DocumentAnalysis) -> dict[str, float]:
    per: dict[str, list[float]] = {}
    for s in analysis.sentences:
        per.setdefault(s.block_id, []).append(s.score)
    return {k: round(statistics.fmean(v), 4) for k, v in per.items()}


def _config_for(scope: Scope, base: EngineConfig | None) -> EngineConfig:
    config = (base or EngineConfig()).model_copy()
    config.rewrite_bands = (
        {Band.ELEVATED, Band.UNCERTAIN} if scope == "flagged_uncertain" else {Band.ELEVATED}
    )
    return config


async def run_pipeline(
    data: bytes,
    fmt: DocFormat,
    mode: Mode,
    *,
    llm: ChatLLM,
    providers: list[DetectorProvider],
    scorer: EntailmentScorer | None = None,
    scope: Scope = "flagged",
    config: EngineConfig | None = None,
    voice: Voice | None = None,
    on_progress: ProgressFn | None = None,
) -> PipelineResult:
    def progress(stage: str, done: int = 0, total: int = 0) -> None:
        if on_progress:
            on_progress(stage, done, total)

    progress("analyzing")
    doc = parse(data, fmt)
    before = await analyze(doc, providers)

    engine = RewriteEngine(llm, mode, scorer=scorer, config=_config_for(scope, config), voice=voice)
    outcomes: list[BlockOutcome] = await engine.rewrite_document(
        doc,
        analysis=None if scope == "all" else before,
        on_progress=lambda done, total: progress("rewriting", done, total),
    )

    progress("exporting")
    rewrites = {o.block_id: o.rewrite_masked for o in outcomes if o.rewrite_masked}
    output = export(data, doc, rewrites) if rewrites else data

    progress("rescoring")
    after_doc = parse(output, fmt)
    after = await analyze(after_doc, providers)

    scores_before, scores_after = _block_scores(before), _block_scores(after)
    flagged: dict[str, list[str]] = {}
    reasons: dict[str, list[str]] = {}
    for s in before.sentences:
        if s.band in (Band.ELEVATED, Band.UNCERTAIN):
            text = doc.block(s.block_id).sentences[s.sentence_index].text
            flagged.setdefault(s.block_id, []).append(text)
            bucket = reasons.setdefault(s.block_id, [])
            bucket.extend(r for r in s.reasons if r not in bucket)

    paragraphs = [
        ParagraphChange(
            block_id=o.block_id,
            section=doc.block(o.block_id).section,
            status=o.status,
            reason=o.reason,
            original=o.original,
            rewrite=o.rewrite,
            score_before=scores_before.get(o.block_id),
            score_after=(scores_after if o.rewrite else scores_before).get(o.block_id),
            flagged_sentences=flagged.get(o.block_id, []),
            reasons=reasons.get(o.block_id, []),
        )
        for o in outcomes
    ]
    progress("done")
    return PipelineResult(
        mode=mode.id,
        scope=scope,
        before=summarize(doc, before),
        after=summarize(after_doc, after),
        paragraphs=paragraphs,
        output=output,
    )
