import pytest

from docmodel import SpanKind, detect_spans, mask, split_sentences, split_tokens, tokens_in, unmask


def kinds(text: str) -> list[tuple[SpanKind, str]]:
    return [(s.kind, s.text) for s in detect_spans(text)]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Growth rose (Smith, 2020).", "(Smith, 2020)"),
        (
            "Growth rose (see Smith & Lee, 2019a; Wang et al., 2021, p. 4).",
            "(see Smith & Lee, 2019a; Wang et al., 2021, p. 4)",
        ),
        ("Smith et al. (2021) found otherwise.", "Smith et al. (2021)"),
        ("Smith and Lee (2019, p. 12) disagree.", "Smith and Lee (2019, p. 12)"),
        ("As shown in prior work [3, 5-7], it holds.", "[3, 5-7]"),
        ("The poem uses repetition (Smith 45).", "(Smith 45)"),
    ],
)
def test_citations(text: str, expected: str) -> None:
    assert (SpanKind.CITATION, expected) in kinds(text)


def test_urls_and_dois_give_back_trailing_punctuation() -> None:
    found = kinds("See https://example.org/a_(b) and doi 10.1000/xyz123. Done.")
    assert (SpanKind.URL, "https://example.org/a_(b)") in found
    assert (SpanKind.DOI, "10.1000/xyz123") in found


def test_quotations_optional() -> None:
    text = 'She said "this is not over" and left.'
    assert (SpanKind.QUOTATION, '"this is not over"') in kinds(text)
    assert detect_spans(text, protect_quotations=False) == []


def test_numbers_are_not_masked() -> None:
    assert detect_spans("Revenue grew 30% to $4.5 million in 2021.") == []


def test_mask_roundtrip() -> None:
    text = "Rates rose (Smith, 2020) per https://x.org [4]."
    masked, legend = mask(text, detect_spans(text))
    assert masked == "Rates rose ⟦C1⟧ per ⟦U1⟧ ⟦C2⟧."
    assert tokens_in(masked) == ["⟦C1⟧", "⟦U1⟧", "⟦C2⟧"]
    assert split_tokens(masked)[:3] == ["Rates rose ", "⟦C1⟧", " per "]
    assert unmask(masked, legend) == text
    with pytest.raises(KeyError):
        unmask("⟦C9⟧", legend)


def sents(text: str) -> list[str]:
    return [s.text for s in split_sentences(text, detect_spans(text))]


def test_sentence_split_basics() -> None:
    assert sents("First one. Second one! Third?") == ["First one.", "Second one!", "Third?"]


@pytest.mark.parametrize(
    "text",
    [
        "Smith et al. (2020) showed this in Fig. 3 of the paper.",
        "Values rose e.g. in Spain and i.e. in winter.",
        "The mean was 3.5 units according to J. Smith today.",
        'She wrote "Stop. Now. Please." and left the room.',
    ],
)
def test_no_false_splits(text: str) -> None:
    assert sents(text) == [text]


def test_trailing_numeric_citation_stays_with_sentence() -> None:
    assert sents("Growth slowed. [12] Then it recovered.") == [
        "Growth slowed. [12]",
        "Then it recovered.",
    ]


def test_sentence_offsets_match_text() -> None:
    text = "  One sentence here.   Another follows.  "
    for s in split_sentences(text):
        assert text[s.start : s.end] == s.text
