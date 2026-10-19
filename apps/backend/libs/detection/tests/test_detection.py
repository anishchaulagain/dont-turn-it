import json

import httpx
import pytest

from detection import (
    Band,
    FusionConfig,
    SaplingProvider,
    StylometricProvider,
    WindowClassifierProvider,
    analyze,
    build_windows,
    ml_inference_classifier,
)
from docmodel import DocFormat, parse

AI_LIKE = (
    "In today's fast-paced world, technology plays a crucial role in education. "
    "Moreover, it is important to note that digital tools foster collaboration. "
    "Furthermore, they unlock the full potential of students. "
    "Additionally, teachers can leverage robust platforms to streamline grading. "
    "In conclusion, technology is a testament to human ingenuity."
)
HUMAN_LIKE = (
    "My grandmother never trusted the microwave. She'd stand two feet back, arms crossed, "
    "as if it might bite. Once, after a power cut, she declared it haunted and unplugged "
    "it for a week. We ate cold soup. Nobody argued."
)


def doc_of(*paragraphs: str, heading: str | None = None) -> str:
    parts = ([heading] if heading else []) + list(paragraphs)
    return "\n\n".join(parts) + "\n"


def test_windows_do_not_cross_sections() -> None:
    text = "Intro\n\n" + AI_LIKE + "\n\nMethods\n\n" + HUMAN_LIKE + "\n"
    doc = parse(text.encode(), DocFormat.TXT)
    windows = build_windows(doc, size=3)
    for w in windows:
        assert len({doc.block(s.block_id).section for s in w.sentences}) == 1
    covered = {(s.block_id, s.sentence_index) for w in windows for s in w.sentences}
    scorable = [
        s
        for b in doc.blocks
        if b.kind == "paragraph"
        for s in b.sentences
        if len(s.text.split()) >= 3
    ]
    assert len(covered) == len(scorable)  # "Nobody argued." is too short to score


async def test_stylometric_separates_stock_prose_and_explains() -> None:
    doc = parse(doc_of(AI_LIKE, HUMAN_LIKE).encode(), DocFormat.TXT)
    scores = await StylometricProvider().score(build_windows(doc))
    ai_block, human_block = doc.blocks[0].id, doc.blocks[1].id
    ai = [s for s in scores if s.block_id == ai_block]
    human = [s for s in scores if s.block_id == human_block]
    assert min(s.score for s in ai) > max(s.score for s in human)
    assert any("stock" in r for s in ai for r in s.reasons)


async def test_tech_docs_mode_ignores_rhythm_but_flags_filler() -> None:
    text = "Simply run the installer. Then just restart the service and it works easily."
    doc = parse(doc_of(text).encode(), DocFormat.TXT)
    scores = await StylometricProvider("technical_docs").score(build_windows(doc))
    assert any("filler" in r for s in scores for r in s.reasons)
    assert not any("burstiness" in r for s in scores for r in s.reasons)


async def test_classifier_pools_overlapping_windows() -> None:
    seen: list[list[str]] = []

    async def classify(texts: list[str]) -> list[float]:
        seen.append(texts)
        return [0.9 if "crucial" in t else 0.1 for t in texts]

    doc = parse(doc_of(AI_LIKE).encode(), DocFormat.TXT)
    provider = WindowClassifierProvider("desklib", classify, batch_size=2)
    scores = await provider.score(build_windows(doc, size=2))
    assert all(len(batch) <= 2 for batch in seen)
    first = next(s for s in scores if s.sentence_index == 0)
    last = next(s for s in scores if s.sentence_index == 4)
    assert first.score == 0.9 and last.score == 0.1


async def test_ml_inference_client_contract() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.url.path == "/v1/detect" and body["model"] == "desklib"
        return httpx.Response(200, json={"probs": [0.25] * len(body["texts"])})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        classify = ml_inference_classifier("http://ml:8200", "desklib", client)
        assert await classify(["a", "b"]) == [0.25, 0.25]


async def test_analyze_fuses_bands_and_tolerates_failing_provider() -> None:
    long_ai = " ".join([AI_LIKE] * 7)  # > 300 eligible words

    async def classify(texts: list[str]) -> list[float]:
        return [0.95] * len(texts)

    class Broken:
        name = "fakespot"
        kind = "local"

        async def score(self, windows: object) -> list[object]:
            raise RuntimeError("model not loaded")

    doc = parse(doc_of(long_ai, heading="Essay").encode(), DocFormat.TXT)
    result = await analyze(
        doc,
        [StylometricProvider(), WindowClassifierProvider("desklib", classify), Broken()],  # type: ignore[list-item]
    )
    assert result.failed_providers.keys() == {"fakespot"}
    assert result.eligible_words >= 300
    assert result.document_band == "elevated"
    assert all(s.band in (Band.ELEVATED, Band.UNCERTAIN, Band.ABSTAIN) for s in result.sentences)
    assert result.sentences[0].provider_scores.keys() == {"stylometric", "desklib"}


async def test_short_documents_get_no_document_score() -> None:
    doc = parse(doc_of(AI_LIKE).encode(), DocFormat.TXT)
    result = await analyze(doc, [StylometricProvider()])
    assert result.document_band == "insufficient_text" and result.elevated_share is None


async def test_disagreement_abstains() -> None:
    async def high(texts: list[str]) -> list[float]:
        return [1.0] * len(texts)

    async def low(texts: list[str]) -> list[float]:
        return [0.0] * len(texts)

    doc = parse(doc_of(HUMAN_LIKE).encode(), DocFormat.TXT)
    result = await analyze(
        doc,
        [WindowClassifierProvider("desklib", high), WindowClassifierProvider("fakespot", low)],
        FusionConfig(),
    )
    assert {s.band for s in result.sentences} == {Band.ABSTAIN}


async def test_sapling_maps_sentence_scores_and_tracks_budget() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["key"] == "k" and body["sent_scores"] is True
        return httpx.Response(
            200,
            json={
                "score": 0.8,
                "sentence_scores": [
                    {"sentence": "My grandmother never trusted the microwave.", "score": 0.1},
                    {
                        "sentence": "She'd stand two feet back, arms crossed, as if it might bite.",
                        "score": 0.2,
                    },
                ],
            },
        )

    doc = parse(doc_of(HUMAN_LIKE).encode(), DocFormat.TXT)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = SaplingProvider("k", client=client, daily_char_budget=10_000)
        scores = await provider.score(build_windows(doc))
        assert [(s.sentence_index, s.score) for s in scores] == [(0, 0.1), (1, 0.2)]
        tiny = SaplingProvider("k", client=client, daily_char_budget=10)
        with pytest.raises(RuntimeError, match="budget"):
            await tiny.score(build_windows(doc))
