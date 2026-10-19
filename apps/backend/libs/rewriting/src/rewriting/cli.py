"""`dtt`: analyse and rewrite a local file.

    uv run dtt modes
    uv run dtt analyze essay.docx --mode student.essay
    uv run dtt rewrite essay.docx --mode student.essay --direct

`--direct` calls providers with keys from docker/.env (no Docker). Without it, the AI
Gateway at GATEWAY_URL is used (`make up && make migrate && make dev-gateway`).
SAPLING_API_KEY is only used with --sapling.
"""

import argparse
import asyncio
import sys
from pathlib import Path

from detection import Band, DocumentAnalysis, analyze
from docmodel import DocFormat, Document, parse
from rewriting.llm import LlmEntailmentScorer
from rewriting.modes import Mode, load_modes
from rewriting.pipeline import PipelineResult, Scope, ScoreSummary, run_pipeline
from rewriting.runtime import detector_providers, open_llm

FORMATS = {".txt": DocFormat.TXT, ".md": DocFormat.MD, ".docx": DocFormat.DOCX}
_BAND_MARK = {Band.ELEVATED: "!!", Band.UNCERTAIN: "? ", Band.LOW: "  ", Band.ABSTAIN: "~ "}


def _format_of(path: Path) -> DocFormat:
    fmt = FORMATS.get(path.suffix.lower())
    if fmt is None:
        sys.exit(f"unsupported file type {path.suffix}; use .txt, .md or .docx")
    return fmt


def _print_analysis(doc: Document, result: DocumentAnalysis) -> None:
    by_block: dict[str, list[str]] = {}
    for s in result.sentences:
        block = doc.block(s.block_id)
        text = block.sentences[s.sentence_index].text
        line = f"  {_BAND_MARK[s.band]} {s.score:.2f}  {text[:110]}"
        if s.reasons:
            line += "\n" + "\n".join(f"             · {r}" for r in s.reasons[:4])
        by_block.setdefault(s.block_id, []).append(line)
    for block_id, lines in by_block.items():
        block = doc.block(block_id)
        print(f"\n[{block_id}] {block.kind.value} · section: {block.section or '-'}")
        print("\n".join(lines))
    share = "-" if result.elevated_share is None else f"{result.elevated_share:.0%}"
    print(
        f"\nDocument: {result.document_band} (elevated share {share}, "
        f"{result.eligible_words} eligible words; providers: {', '.join(result.providers)})"
    )
    print("These are style indicators, not evidence of who wrote the text.")


def _fmt_score(s: ScoreSummary) -> str:
    score = "-" if s.score is None else f"{s.score:.2f}"
    return f"{score} ({s.document_band}; bands {s.bands})"


def _report(result: PipelineResult, mode: Mode) -> str:
    lines = [
        f"# Rewrite report ({mode.label})",
        "",
        f"AI-likeness indicator: before {_fmt_score(result.before)} → "
        f"after {_fmt_score(result.after)}",
    ]
    for p in result.paragraphs:
        lines += ["", f"## {p.block_id}: {p.status}" + (f" ({p.reason})" if p.reason else "")]
        lines += ["", "**Original**", "", p.original]
        if p.rewrite:
            lines += ["", "**Rewrite**", "", p.rewrite]
        lines.append(f"\nScore {p.score_before} → {p.score_after} (indicator only)")
    return "\n".join(lines) + "\n"


async def _run(args: argparse.Namespace) -> None:
    modes = load_modes()
    if args.command == "modes":
        for m in modes.values():
            print(f"{m.id:32} {m.label:28} phase {m.phase}  view={m.default_view}")
        return
    if args.mode not in modes:
        sys.exit(f"unknown mode {args.mode!r}; run `dtt modes`")
    mode = modes[args.mode]
    path = Path(args.file)
    fmt = _format_of(path)
    data = path.read_bytes()
    providers = detector_providers(mode, sapling=args.sapling)

    if args.command == "analyze":
        doc = parse(data, fmt)
        _print_analysis(doc, await analyze(doc, providers))
        return

    scope: Scope = (
        "all" if args.all else "flagged_uncertain" if args.include_uncertain else "flagged"
    )
    async with open_llm("direct" if args.direct else "gateway") as llm:
        result = await run_pipeline(
            data,
            fmt,
            mode,
            llm=llm,
            providers=providers,
            scorer=None if args.no_nli else LlmEntailmentScorer(llm),
            scope=scope,
        )

    out_path = Path(args.output) if args.output else path.with_stem(path.stem + ".rewritten")
    out_path.write_bytes(result.output)
    report_path = out_path.with_suffix(".report.md")
    report_path.write_text(_report(result, mode), encoding="utf-8")
    done = sum(p.status == "rewritten" for p in result.paragraphs)
    print(
        f"{done}/{len(result.paragraphs)} selected paragraphs rewritten → {out_path}\n"
        f"AI-likeness indicator: {_fmt_score(result.before)} → {_fmt_score(result.after)}\n"
        f"report → {report_path}"
    )


def main() -> None:
    # Windows consoles default to cp1252, which cannot print ⟦ ⟧ · →.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(prog="dtt", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("modes", help="list writing modes")
    for name in ("analyze", "rewrite"):
        p = sub.add_parser(name)
        p.add_argument("file")
        p.add_argument("--mode", required=True)
        p.add_argument(
            "--sapling", action="store_true", help="also send the text to Sapling (dev only)"
        )
        if name == "rewrite":
            p.add_argument("-o", "--output")
            p.add_argument("--include-uncertain", action="store_true")
            p.add_argument("--all", action="store_true", help="rewrite every rewritable block")
            p.add_argument("--no-nli", action="store_true", help="skip the G5 entailment check")
            p.add_argument(
                "--direct",
                action="store_true",
                help="call providers directly with keys from docker/.env (no gateway/Docker)",
            )
    asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    main()
