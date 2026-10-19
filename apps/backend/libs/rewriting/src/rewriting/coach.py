"""Coach view (docs/PLAN.md §3.0): the writer stays the author.

Instead of rewriting paragraphs automatically, the coach shows each flagged sentence
with the reasons it reads as generic, offers three verified alternatives on request,
checks the writer's own edits with the same meaning gates, and exports exactly the
sentences the writer accepted.
"""

import statistics
from typing import Any, Literal

from pydantic import BaseModel, Field

from detection import Band, DetectorProvider, DocumentAnalysis, StylometricProvider, analyze
from detection.types import SentenceRef, Window
from docmodel import DocFormat, Document, export, parse
from docmodel.ir import Block, BlockKind, ProtectedSpan
from docmodel.protect import mask, remask, unmask
from docmodel.segment import split_sentences
from rewriting.llm import ChatLLM
from rewriting.modes import Mode
from rewriting.pipeline import ScoreSummary, summarize
from rewriting.prompts import clean_output, suggestion_messages
from rewriting.voices import Voice
from verification import EntailmentScorer, GateResult, check_entailment, run_deterministic_gates

_VISIBLE_KINDS = {
    BlockKind.HEADING,
    BlockKind.PARAGRAPH,
    BlockKind.LIST_ITEM,
    BlockKind.QUOTE,
    BlockKind.CAPTION,
}


# ---------- document view ----------
class SentenceView(BaseModel):
    index: int
    text: str
    band: Band | None = None
    score: float | None = None
    reasons: list[str] = Field(default_factory=list)


class BlockView(BaseModel):
    id: str
    kind: BlockKind
    section: str | None
    heading_level: int | None
    editable: bool
    text: str
    sentences: list[SentenceView]


class CoachDocument(BaseModel):
    blocks: list[BlockView]
    summary: ScoreSummary
    flagged: int


def build_view(doc: Document, analysis: DocumentAnalysis) -> CoachDocument:
    scores = {(s.block_id, s.sentence_index): s for s in analysis.sentences}
    blocks: list[BlockView] = []
    flagged = 0
    for block in doc.blocks:
        if block.kind not in _VISIBLE_KINDS:
            continue
        sentences = []
        for s in block.sentences:
            a = scores.get((block.id, s.index))
            if a is not None and a.band in (Band.ELEVATED, Band.UNCERTAIN):
                flagged += 1
            sentences.append(
                SentenceView(
                    index=s.index,
                    text=s.text,
                    band=a.band if a else None,
                    score=a.score if a else None,
                    reasons=a.reasons if a else [],
                )
            )
        blocks.append(
            BlockView(
                id=block.id,
                kind=block.kind,
                section=block.section,
                heading_level=block.heading_level,
                editable=block.rewritable,
                text=block.text,
                sentences=sentences,
            )
        )
    return CoachDocument(blocks=blocks, summary=summarize(doc, analysis), flagged=flagged)


# ---------- checks on a single sentence ----------
class Check(BaseModel):
    gate: str
    passed: bool
    message: str
    # Advisory checks are shown but do not block (the writer decides): length only.
    blocking: bool = True


class SentenceCheck(BaseModel):
    passed: bool
    score: float
    checks: list[Check]


_POLARITY_MESSAGES = {
    "negation_changed": "A 'not', 'no' or 'never' was added or removed.",
    "hedge_removed": "A hedge (may, might, suggests…) was removed, so the claim got stronger.",
    "hedge_added": "A hedge was added, so the claim got weaker.",
    "certainty_added": "Certainty words (clearly, proves…) were added.",
    "universal_quantifier_changed": "'All', 'every', 'always' or 'never' changed.",
    "partial_quantifier_removed": "'Some', 'most', 'common' or a similar qualifier was removed.",
    "obligation_strength_changed": "The strength of 'must' / 'should' changed.",
    "obligation_changed": "An obligation word (must, shall, should, may) changed.",
}


def _explain(result: GateResult, legend: dict[str, ProtectedSpan]) -> str:
    d: dict[str, Any] = result.details
    if result.passed:
        return "OK"
    if result.gate == "G1_placeholders":
        lost = [
            legend[t].text or "(formatted element)" for t in d.get("missing", []) if t in legend
        ]
        return "Protected text missing or changed: " + "; ".join(f"“{t}”" for t in lost)
    if result.gate == "G2_numbers":
        parts = []
        if d.get("missing"):
            parts.append("missing " + ", ".join(d["missing"]))
        if d.get("added"):
            parts.append("added " + ", ".join(d["added"]))
        return "Numbers changed: " + "; ".join(parts)
    if result.gate == "G3_entities":
        parts = []
        if d.get("missing") or d.get("glossary_missing"):
            parts.append(
                "missing " + ", ".join([*d.get("missing", []), *d.get("glossary_missing", [])])
            )
        if d.get("added"):
            parts.append("new " + ", ".join(d["added"]))
        return "Names or terms changed: " + "; ".join(parts)
    if result.gate == "G4_polarity_modality":
        return " ".join(_POLARITY_MESSAGES.get(p, p) for p in d.get("problems", []))
    if result.gate == "S3_length":
        return "Much shorter or longer than the original: check nothing important was dropped."
    if result.gate == "G5_entailment":
        return "The meaning checker thinks this says something different from the original."
    return result.gate


def _sentence_parts(block: Block, index: int) -> tuple[str, dict[str, ProtectedSpan]]:
    """Masked sentence text plus the legend of the protected spans inside it."""
    sentence = block.sentences[index]
    inside = [
        span.model_copy(
            update={"start": span.start - sentence.start, "end": span.end - sentence.start}
        )
        for span in block.spans
        if span.start >= sentence.start and span.end <= sentence.end
    ]
    return mask(sentence.text, inside)


async def style_score(text: str, mode: Mode) -> float:
    refs = [
        SentenceRef(block_id="x", sentence_index=s.index, text=s.text, words=len(s.text.split()))
        for s in split_sentences(text)
    ]
    if not refs:
        return 0.0
    scores = await StylometricProvider(mode.detector_mode).score([Window(index=0, sentences=refs)])
    return round(statistics.fmean(s.score for s in scores), 4)


async def check_sentence(
    mode: Mode,
    block: Block,
    index: int,
    new_text: str,
    *,
    scorer: EntailmentScorer | None = None,
) -> SentenceCheck:
    masked, legend = _sentence_parts(block, index)
    new_masked, _ = remask(new_text, legend)
    policy = mode.gate_policy()
    # A single sentence may legitimately grow a little more than a paragraph would.
    policy.max_length_ratio = 1.6
    verdict = run_deterministic_gates(masked, new_masked, policy)
    results = list(verdict.results)
    if verdict.passed and scorer is not None:
        results.append(await check_entailment(masked, new_masked, scorer))
    checks = [
        Check(gate=r.gate, passed=r.passed, message=_explain(r, legend), blocking=r.hard)
        for r in results
    ]
    return SentenceCheck(
        passed=all(c.passed or not c.blocking for c in checks),
        score=await style_score(new_text, mode),
        checks=checks,
    )


# ---------- suggestions ----------
class Suggestion(BaseModel):
    approach: str
    text: str
    check: SentenceCheck


class _Alternative(BaseModel):
    approach: str = "alternative"
    text: str


class _Alternatives(BaseModel):
    alternatives: list[_Alternative]


async def suggest(
    llm: ChatLLM,
    mode: Mode,
    doc: Document,
    block_id: str,
    index: int,
    *,
    reasons: list[str] | None = None,
    voice: Voice | None = None,
    scorer: EntailmentScorer | None = None,
    model: str = "rewrite",
) -> list[Suggestion]:
    block = doc.block(block_id)
    masked_sentence, legend = _sentence_parts(block, index)
    paragraph, _ = mask(block.text, block.spans)
    result = await llm.judge(
        suggestion_messages(
            mode,
            sentence=masked_sentence,
            paragraph=paragraph,
            legend=legend,
            section=block.section,
            reasons=reasons or [],
            voice=voice,
        ),
        _Alternatives,
        model=model,
    )
    out: list[Suggestion] = []
    seen: set[str] = set()
    for alt in result.alternatives[:3]:
        cleaned = clean_output(alt.text, masked_sentence)
        try:
            text = unmask(cleaned, legend)
        except KeyError:  # invented a placeholder: show it as failing rather than crash
            text = cleaned
        if not text or text in seen or text == block.sentences[index].text:
            continue
        seen.add(text)
        check = await check_sentence(mode, block, index, text, scorer=scorer)
        out.append(Suggestion(approach=alt.approach, text=text, check=check))
    # Verified suggestions first, then the most natural.
    out.sort(key=lambda s: (not s.check.passed, s.check.score))
    return out


# ---------- applying the writer's edits ----------
class Edit(BaseModel):
    block_id: str
    sentence_index: int
    text: str
    source: Literal["suggestion", "manual"] = "manual"


class ExportResult(BaseModel):
    output: bytes = Field(repr=False, exclude=True)
    after: ScoreSummary
    warnings: list[str]
    applied: int


def apply_edits(doc: Document, edits: list[Edit]) -> tuple[dict[str, str], list[str]]:
    """Block id -> masked new text, plus warnings for protected text the writer removed."""
    by_block: dict[str, dict[int, str]] = {}
    for e in edits:
        by_block.setdefault(e.block_id, {})[e.sentence_index] = e.text.strip()
    rewrites: dict[str, str] = {}
    warnings: list[str] = []
    for block_id, sentence_edits in by_block.items():
        block = doc.block(block_id)
        if not block.rewritable:
            warnings.append(f"{block_id}: this block cannot be edited; edits ignored")
            continue
        text = block.text
        for s in sorted(block.sentences, key=lambda s: s.start, reverse=True):
            if s.index in sentence_edits:
                text = text[: s.start] + sentence_edits[s.index] + text[s.end :]
        _, legend = mask(block.text, block.spans)
        masked, missing = remask(text, legend)
        for token in missing:
            span = legend[token]
            label = f"“{span.text}”" if span.text else "a formatted element (e.g. footnote mark)"
            warnings.append(f"{block_id}: {label} was removed or changed by an edit")
        rewrites[block_id] = masked
    return rewrites, warnings


async def export_with_edits(
    data: bytes,
    fmt: DocFormat,
    doc: Document,
    edits: list[Edit],
    providers: list[DetectorProvider],
) -> ExportResult:
    rewrites, warnings = apply_edits(doc, edits)
    output = export(data, doc, rewrites) if rewrites else data
    after_doc = parse(output, fmt)
    after = summarize(after_doc, await analyze(after_doc, providers))
    return ExportResult(output=output, after=after, warnings=warnings, applied=len(edits))
