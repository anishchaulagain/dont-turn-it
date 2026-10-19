"""Knowledge-preservation gates for rewrites (docs/PLAN.md §4.1)."""

from verification.gates import (
    GatePolicy,
    GateResult,
    Verdict,
    check_entities,
    check_length,
    check_numbers,
    check_placeholders,
    check_polarity,
    extract_entities,
    extract_numbers,
    run_deterministic_gates,
)
from verification.nli import EntailmentScorer, check_entailment

__all__ = [
    "EntailmentScorer",
    "GatePolicy",
    "GateResult",
    "Verdict",
    "check_entailment",
    "check_entities",
    "check_length",
    "check_numbers",
    "check_placeholders",
    "check_polarity",
    "extract_entities",
    "extract_numbers",
    "run_deterministic_gates",
]
