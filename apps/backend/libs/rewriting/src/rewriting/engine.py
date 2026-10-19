"""Analyze -> Refactor -> Verify -> Select, per block (docs/PLAN.md §3.5).

For each selected block: generate N candidates, run the deterministic gates
(G1-G4, S3) and then G5 entailment on survivors, pick the most natural passing
candidate, retry once with the gate failures as feedback, and otherwise keep the
original. A failing rewrite is never returned.
"""

import asyncio
import statistics
from collections.abc import Callable
from typing import Literal

import structlog
from pydantic import BaseModel, Field

from detection import Band, DocumentAnalysis, StylometricProvider
from detection.types import SentenceRef, Window
from docmodel.ir import Block, BlockKind, Document
from docmodel.protect import mask, unmask
from docmodel.segment import split_sentences
from llm_client import GatewayError
from rewriting.llm import ChatLLM
from rewriting.modes import Freedom, Mode
from rewriting.prompts import PROMPT_VERSION, build_messages, clean_output, user_prompt
from rewriting.voices import Voice
from verification import EntailmentScorer, GateResult, check_entailment, run_deterministic_gates

logger = structlog.get_logger(__name__)

_CONTEXT_CHARS = 600
# Weak rhythm signals stay in the analysis view but are not sent to the rewriter:
# live runs showed the model "fixing" them by breaking up normal lists of three
# and chopping prose into fragments, which reads worse, not more human.
_NOT_FOR_PROMPT = ("list of three", "uniform sentence lengths")


class CandidateResult(BaseModel):
    round: int
    temperature: float
    text: str  # masked
    passed: bool
    gates: list[GateResult] = Field(default_factory=list)
    ai_score: float | None = None
    error: str | None = None


class BlockOutcome(BaseModel):
    block_id: str
    status: Literal["rewritten", "kept_original", "skipped"]
    reason: str | None = None
    freedom: Freedom
    original: str
    original_masked: str
    rewrite: str | None = None
    rewrite_masked: str | None = None
    ai_score_before: float | None = None
    ai_score_after: float | None = None
    prompt_version: str = PROMPT_VERSION
    candidates: list[CandidateResult] = Field(default_factory=list)


class EngineConfig(BaseModel):
    model: str = "rewrite"
    temperatures: list[float] = Field(default_factory=lambda: [0.7, 0.95])
    max_rounds: int = 2
    # Reasoning models (gpt-oss, qwen3) spend part of this on hidden reasoning.
    max_tokens: int = 3000
    concurrency: int = 2
    # Which analysis bands make a block eligible when no explicit selection is given.
    rewrite_bands: set[Band] = Field(default_factory=lambda: {Band.ELEVATED})


class RewriteEngine:
    def __init__(
        self,
        llm: ChatLLM,
        mode: Mode,
        *,
        scorer: EntailmentScorer | None = None,
        config: EngineConfig | None = None,
        glossary: list[str] | None = None,
        voice: Voice | None = None,
    ) -> None:
        self._llm = llm
        self._mode = mode
        self._scorer = scorer
        self._cfg = config or EngineConfig()
        self._policy = mode.gate_policy(glossary)
        self._style = StylometricProvider(mode.detector_mode)
        self._voice = voice

    async def style_score(self, text: str) -> float:
        """Mean stylometric AI-likeness of a text (diagnostic, used to rank candidates)."""
        refs = [
            SentenceRef(
                block_id="x", sentence_index=s.index, text=s.text, words=len(s.text.split())
            )
            for s in split_sentences(text)
        ]
        if not refs:
            return 0.0
        scores = await self._style.score([Window(index=0, sentences=refs)])
        return round(statistics.fmean(s.score for s in scores), 4)

    async def _candidate(
        self, messages_user: str, freedom: Freedom, masked: str, temperature: float, round_: int
    ) -> CandidateResult:
        try:
            raw = await self._llm.complete(
                build_messages(self._mode, freedom, messages_user, self._voice),
                model=self._cfg.model,
                temperature=temperature,
                max_tokens=self._cfg.max_tokens,
            )
        except GatewayError as exc:
            return CandidateResult(
                round=round_, temperature=temperature, text="", passed=False, error=exc.message
            )
        text = clean_output(raw, masked)
        verdict = run_deterministic_gates(masked, text, self._policy)
        gates = list(verdict.results)
        passed = verdict.passed
        if passed and self._scorer is not None:
            try:
                g5 = await check_entailment(masked, text, self._scorer)
            except (GatewayError, ValueError) as exc:
                g5 = GateResult(gate="G5_entailment", passed=False, details={"error": str(exc)})
            gates.append(g5)
            passed = g5.passed
        return CandidateResult(
            round=round_,
            temperature=temperature,
            text=text,
            passed=passed,
            gates=gates,
            ai_score=await self.style_score(text) if text else None,
        )

    async def rewrite_block(
        self,
        block: Block,
        *,
        before: str | None = None,
        after: str | None = None,
        reasons: list[str] | None = None,
    ) -> BlockOutcome:
        freedom = self._mode.freedom_for(block.section)
        masked, legend = mask(block.text, block.spans)
        outcome = BlockOutcome(
            block_id=block.id,
            status="skipped",
            freedom=freedom,
            original=block.text,
            original_masked=masked,
        )
        if not block.rewritable:
            outcome.reason = block.skip_reason or "not_rewritable"
            return outcome
        if freedom == "locked":
            outcome.reason = "section_locked"
            return outcome

        outcome.ai_score_before = await self.style_score(masked)
        feedback: str | None = None
        for round_ in range(1, self._cfg.max_rounds + 1):
            prompt = user_prompt(
                masked=masked,
                legend=legend,
                section=block.section,
                before=before,
                after=after,
                reasons=reasons or [],
                feedback=feedback,
            )
            batch = await asyncio.gather(
                *(
                    self._candidate(prompt, freedom, masked, t, round_)
                    for t in self._cfg.temperatures
                )
            )
            outcome.candidates.extend(batch)
            changed = [c for c in batch if c.passed and c.text.strip() != masked.strip()]
            if changed:
                best = min(changed, key=lambda c: (c.ai_score or 0.0, c.temperature))
                outcome.status = "rewritten"
                outcome.rewrite_masked = best.text
                outcome.rewrite = unmask(best.text, legend)
                outcome.ai_score_after = best.ai_score
                return outcome
            if any(c.passed for c in batch):
                outcome.status, outcome.reason = "kept_original", "no_change_needed"
                return outcome
            failing = [c for c in batch if c.error is None]
            if not failing:
                outcome.status, outcome.reason = "kept_original", "llm_unavailable"
                return outcome
            # Feed back the failures of the candidate that failed the fewest gates.
            worst_first = sorted(failing, key=lambda c: sum(not g.passed for g in c.gates))
            feedback = "\n".join(
                f"- {g.gate}: { {k: v for k, v in g.details.items() if v and k != 'counts'} }"
                for g in worst_first[0].gates
                if not g.passed
            )
        outcome.status, outcome.reason = "kept_original", "failed_verification"
        return outcome

    def select_blocks(
        self, doc: Document, analysis: DocumentAnalysis | None, block_ids: set[str] | None
    ) -> list[Block]:
        if block_ids is not None:
            return [b for b in doc.blocks if b.id in block_ids]
        if analysis is None:
            return [b for b in doc.blocks if b.rewritable]
        flagged = {s.block_id for s in analysis.sentences if s.band in self._cfg.rewrite_bands}
        return [b for b in doc.blocks if b.rewritable and b.id in flagged]

    async def rewrite_document(
        self,
        doc: Document,
        *,
        analysis: DocumentAnalysis | None = None,
        block_ids: set[str] | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[BlockOutcome]:
        targets = self.select_blocks(doc, analysis, block_ids)
        done = 0
        if on_progress:
            on_progress(0, len(targets))
        reasons: dict[str, list[str]] = {}
        for s in analysis.sentences if analysis else []:
            bucket = reasons.setdefault(s.block_id, [])
            bucket.extend(
                r for r in s.reasons if r not in bucket and not r.startswith(_NOT_FOR_PROMPT)
            )

        prose = [b for b in doc.blocks if b.kind in (BlockKind.PARAGRAPH, BlockKind.LIST_ITEM)]
        position = {b.id: i for i, b in enumerate(prose)}

        def neighbour(block: Block, offset: int) -> str | None:
            i = position.get(block.id)
            if i is None or not 0 <= i + offset < len(prose):
                return None
            other = prose[i + offset]
            if other.section != block.section:
                return None
            return mask(other.text, other.spans)[0][:_CONTEXT_CHARS]

        gate = asyncio.Semaphore(self._cfg.concurrency)

        async def run(block: Block) -> BlockOutcome:
            async with gate:
                result = await self.rewrite_block(
                    block,
                    before=neighbour(block, -1),
                    after=neighbour(block, 1),
                    reasons=reasons.get(block.id, [])[:8],
                )
                logger.info(
                    "block_rewritten",
                    block=block.id,
                    status=result.status,
                    reason=result.reason,
                    candidates=len(result.candidates),
                )
                nonlocal done
                done += 1
                if on_progress:
                    on_progress(done, len(targets))
                return result

        return list(await asyncio.gather(*(run(b) for b in targets)))
