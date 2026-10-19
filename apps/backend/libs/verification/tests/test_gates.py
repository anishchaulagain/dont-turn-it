from collections import Counter

import pytest

from verification import (
    GatePolicy,
    check_entities,
    check_numbers,
    check_placeholders,
    check_polarity,
    extract_entities,
    extract_numbers,
    run_deterministic_gates,
)


def test_placeholders() -> None:
    assert check_placeholders("A ⟦C1⟧ b ⟦F1⟧.", "B ⟦F1⟧ a ⟦C1⟧.").passed
    r = check_placeholders("A ⟦C1⟧ b ⟦F1⟧.", "B ⟦F1⟧ ⟦F1⟧ ⟦C2⟧.")
    assert not r.passed
    assert r.details == {"missing": ["⟦C1⟧"], "unexpected_or_duplicated": ["⟦C2⟧", "⟦F1⟧"]}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Revenue grew 30% to $4.5 million in 2021.", ["%30", "$4.5e+06", "2021"]),
        ("thirty percent of twenty-five samples", ["%30", "25"]),
        ("1,200 users and 3.5 units", ["1200", "3.5"]),
        ("We used Qwen3 and H2O often, not ten times.", ["10"]),
        ("A 2 percentage point rise", ["pp2"]),
        ("One of the reasons is cost.", []),
        ("Cites ⟦C1⟧ only.", []),
    ],
)
def test_extract_numbers(text: str, expected: list[str]) -> None:
    assert extract_numbers(text) == Counter(expected)


def test_number_equivalent_forms_pass() -> None:
    assert check_numbers("Thirty percent failed.", "30% failed.").passed
    assert check_numbers("It cost $4.5 million.", "It cost $4,500,000.").passed


def test_entities() -> None:
    text = "In 2020, researchers at MIT and the World Health Organization studied GPT models."
    assert {"MIT", "World Health Organization", "GPT"} <= extract_entities(text)
    assert "In" not in extract_entities(text)
    ok = "Researchers at MIT and the World Health Organization studied GPT models in 2020."
    assert check_entities(text, ok).passed
    swapped = "In 2020, researchers at Stanford and the World Health Organization studied GPT."
    r = check_entities(text, swapped)
    assert not r.passed and r.details["missing"] == ["MIT"] and r.details["added"] == ["Stanford"]


def test_glossary_is_case_sensitive() -> None:
    r = check_entities(
        "The in vitro assay uses Western blot.", "The western blot assay.", ["Western blot"]
    )
    assert r.details["glossary_missing"] == ["Western blot"]


def test_obligation_strength_checked_in_every_mode() -> None:
    a = "Schools must ensure every student can benefit."
    # Seen live: Nemotron turned "must ensure" into "should make sure".
    assert not check_polarity(a, "Schools should make sure every student can benefit.").passed
    synonym = "Schools need to make sure every student can benefit."
    assert check_polarity(a, synonym).passed
    # Business/process docs additionally require the exact wording (RFC 2119).
    assert not check_polarity(a, synonym, GatePolicy(strict_deontic=True)).passed


def test_hedge_added_only_fails_when_strict() -> None:
    a, b = "The drug reduces symptoms.", "The drug may reduce symptoms."
    assert check_polarity(a, b).passed
    assert not check_polarity(a, b, GatePolicy(strict_hedges=True)).passed


# ---------- perturbation suite (docs/PLAN.md §4.1) ----------
ORIGINAL = (
    "Moreover, it is important to note that the treatment may reduce symptoms in some "
    "patients ⟦C1⟧. Researchers at Oxford did not observe side effects in 30% of the "
    "120 participants."
)
GOOD_PARAPHRASES = [
    "The treatment may reduce symptoms in some patients ⟦C1⟧. Researchers at Oxford did not "
    "see side effects in 30% of the 120 participants.",
    "In some patients, the treatment may ease symptoms ⟦C1⟧, and Oxford researchers did not "
    "observe side effects in 30% of the 120 people who took part.",
]
HARMFUL = {
    "number_changed": ORIGINAL.replace("30%", "35%"),
    "count_changed": ORIGINAL.replace("120", "200"),
    "number_dropped": ORIGINAL.replace(" in 30% of the 120 participants", ""),
    "citation_dropped": ORIGINAL.replace(" ⟦C1⟧", ""),
    "citation_duplicated": ORIGINAL.replace("participants.", "participants ⟦C1⟧."),
    "negation_flipped": ORIGINAL.replace("did not observe", "observed"),
    "hedge_removed": ORIGINAL.replace("may reduce", "reduces"),
    "certainty_added": ORIGINAL.replace("may reduce", "clearly may reduce"),
    "quantifier_dropped": ORIGINAL.replace("in some patients", "in patients"),
    "quantifier_universalised": ORIGINAL.replace("in some patients", "in all patients"),
    "entity_swapped": ORIGINAL.replace("Oxford", "Cambridge"),
    "entity_added": ORIGINAL.replace("Researchers at Oxford", "Researchers at Oxford and Pfizer"),
    "half_deleted": ORIGINAL.split(" Researchers")[0],
    "obligation_added": ORIGINAL.replace("may reduce", "may reduce, and should reduce,"),
}


@pytest.mark.parametrize("rewrite", GOOD_PARAPHRASES)
def test_good_paraphrases_pass(rewrite: str) -> None:
    verdict = run_deterministic_gates(ORIGINAL, rewrite)
    assert verdict.passed, verdict.feedback()


@pytest.mark.parametrize("name", sorted(HARMFUL))
def test_harmful_perturbations_are_caught(name: str) -> None:
    verdict = run_deterministic_gates(ORIGINAL, HARMFUL[name])
    assert not verdict.passed, f"{name} slipped through"
    assert verdict.feedback()


def test_frequency_qualifier_dropped_is_caught() -> None:
    # Live miss on the safeguarding report (prompt v3, Nemotron rewriter).
    a = "Lateness and absence are common when no adult supports a child's morning."
    b = "Lateness and absence appear when no adult helps the child get ready in the morning."
    r = check_polarity(a, b)
    assert not r.passed and "partial_quantifier_removed" in r.details["problems"]
