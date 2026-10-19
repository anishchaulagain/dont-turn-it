"""Writing modes (docs/PLAN.md §3.0), loaded from infra/modes/*.yaml."""

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from verification import GatePolicy

Freedom = Literal["locked", "minimal", "conservative", "moderate", "expressive"]
DetectorMode = Literal[
    "essay", "story", "research_paper", "technical_docs", "business_docs", "generic"
]


class SectionRule(BaseModel):
    match: str
    freedom: Freedom


class ModeGates(BaseModel):
    strict_hedges: bool = False
    strict_deontic: bool = False


class Mode(BaseModel):
    id: str
    audience: Literal["student", "professional"]
    label: str
    phase: int = 1
    default_view: Literal["coach", "rewrite"] = "rewrite"
    freedom: Freedom = "moderate"
    detector_mode: DetectorMode = "generic"
    protect_quotations: bool = True
    gates: ModeGates = Field(default_factory=ModeGates)
    options: dict[str, list[str]] = Field(default_factory=dict)
    sections: list[SectionRule] = Field(default_factory=list)
    style_rules: list[str] = Field(default_factory=list)
    banned: list[str] = Field(default_factory=list)

    def freedom_for(self, section: str | None) -> Freedom:
        if section:
            for rule in self.sections:
                if re.search(rule.match, section, re.IGNORECASE):
                    return rule.freedom
        return self.freedom

    def gate_policy(self, glossary: list[str] | None = None) -> GatePolicy:
        return GatePolicy(
            strict_hedges=self.gates.strict_hedges,
            strict_deontic=self.gates.strict_deontic,
            glossary=glossary or [],
        )


def default_modes_dir() -> Path:
    """MODES_DIR, else the nearest infra/modes walking up from this file or the cwd."""
    env = os.environ.get("MODES_DIR")
    if env:
        return Path(env)
    for base in (Path(__file__).resolve(), Path.cwd().resolve()):
        for parent in [base, *base.parents]:
            candidate = parent / "infra" / "modes"
            if candidate.is_dir():
                return candidate
    raise FileNotFoundError("infra/modes not found; set MODES_DIR")


def load_modes(directory: Path | None = None) -> dict[str, Mode]:
    directory = directory or default_modes_dir()
    modes: dict[str, Mode] = {}
    for path in sorted(directory.glob("*.yaml")):
        raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        mode = Mode.model_validate(raw)
        modes[mode.id] = mode
    return modes
