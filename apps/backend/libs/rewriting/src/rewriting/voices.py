"""Voice presets: style profiles measured on human-written corpora (docs/PLAN.md §5.4).

A voice gives the rewriter concrete, measured targets (sentence rhythm, contractions,
first person, transition use, word length) and a few short exemplar paragraphs to imitate
for *style only*. Presets are built offline by `scripts/build_voices.py` from licensed
human corpora and stored as JSON in infra/voices/. Every preset records its source and
license so attribution travels with it.
"""

import json
import re
import statistics
from pathlib import Path

from pydantic import BaseModel, Field

from docmodel.segment import split_sentences

_WORD = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")
_CONTRACTION = re.compile(r"\b\w+['’](?:s|t|re|ve|ll|d|m)\b", re.IGNORECASE)
_FIRST_PERSON = re.compile(r"\b(?:I|me|my|mine|myself|we|us|our|ours)\b")
_TRANSITION = re.compile(
    r"^(?:Moreover|Furthermore|Additionally|However|Therefore|Thus|Hence|Overall|"
    r"In addition|In conclusion|Consequently|Also|Finally|First|Second|Firstly|Secondly)\b"
)


class StyleTargets(BaseModel):
    sentence_words_mean: float
    sentence_words_sd: float
    word_chars_mean: float
    contractions_per_100_words: float
    first_person_per_100_words: float
    transition_opener_share: float  # share of sentences that open with a transition word
    question_share: float


class Voice(BaseModel):
    id: str
    label: str
    description: str
    source: str
    license: str
    targets: StyleTargets
    exemplars: list[str] = Field(default_factory=list)


def measure(texts: list[str]) -> StyleTargets:
    """Measure style targets over a set of human-written paragraphs."""
    lengths: list[int] = []
    words = chars = contractions = first_person = openers = questions = 0
    for text in texts:
        tokens = _WORD.findall(text)
        words += len(tokens)
        chars += sum(len(t) for t in tokens)
        contractions += len(_CONTRACTION.findall(text))
        first_person += len(_FIRST_PERSON.findall(text))
        for s in split_sentences(text):
            n = len(_WORD.findall(s.text))
            if n:
                lengths.append(n)
                openers += bool(_TRANSITION.match(s.text))
                questions += s.text.rstrip().endswith("?")
    n_sent = max(1, len(lengths))
    per100 = 100.0 / max(1, words)
    return StyleTargets(
        sentence_words_mean=round(statistics.fmean(lengths), 1) if lengths else 0.0,
        sentence_words_sd=round(statistics.pstdev(lengths), 1) if len(lengths) > 1 else 0.0,
        word_chars_mean=round(chars / max(1, words), 2),
        contractions_per_100_words=round(contractions * per100, 2),
        first_person_per_100_words=round(first_person * per100, 2),
        transition_opener_share=round(openers / n_sent, 3),
        question_share=round(questions / n_sent, 3),
    )


def describe(voice: Voice) -> str:
    """Targets as plain instructions for the prompt."""
    t = voice.targets
    contractions = (
        "common" if t.contractions_per_100_words >= 1.0
        else "occasional" if t.contractions_per_100_words >= 0.3
        else "rare"
    )  # fmt: skip
    first = (
        "often" if t.first_person_per_100_words >= 2.0
        else "sometimes" if t.first_person_per_100_words >= 0.5
        else "rarely"
    )  # fmt: skip
    vocabulary = (
        "Plain, everyday" if t.word_chars_mean < 4.6
        else "Moderately formal" if t.word_chars_mean < 5.2
        else "Formal, subject-specific"
    )  # fmt: skip
    return (
        f"Voice: {voice.label}. {voice.description}\n"
        f"- Sentences average about {t.sentence_words_mean:.0f} words, but vary a lot "
        f"(typical spread ±{t.sentence_words_sd:.0f} words): mix short and long ones.\n"
        f"- Contractions are {contractions}; first person is used {first}.\n"
        f"- Only about {t.transition_opener_share:.0%} of sentences open with a transition "
        "word; most simply start with the subject.\n"
        f"- {vocabulary} vocabulary (average word length {t.word_chars_mean:.1f} letters)."
    )


def default_voices_dir() -> Path:
    from rewriting.modes import default_modes_dir

    return default_modes_dir().parent / "voices"


def load_voices(directory: Path | None = None) -> dict[str, Voice]:
    directory = directory or default_voices_dir()
    voices: dict[str, Voice] = {}
    if not directory.is_dir():
        return voices
    for path in sorted(directory.glob("*.json")):
        voice = Voice.model_validate(json.loads(path.read_text(encoding="utf-8")))
        voices[voice.id] = voice
    return voices
