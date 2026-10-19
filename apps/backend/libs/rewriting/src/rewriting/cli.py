"""`dtt`: run analysis and rewriting on a local file through the AI Gateway.

    uv run dtt modes
    uv run dtt analyze essay.docx --mode student.essay
    uv run dtt rewrite essay.docx --mode student.essay -o essay.rewritten.docx

Needs the gateway running (`make up && make migrate && make dev-gateway`) with
MISTRAL_API_KEY / GROQ_API_KEY set in docker/.env. Environment:
GATEWAY_URL (default http://localhost:8100), GATEWAY_SERVICE_TOKEN, DTT_TENANT_ID,
SAPLING_API_KEY (only used with --sapling).
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from detection import (
    Band,
    DetectorProvider,
    DocumentAnalysis,
    SaplingProvider,
    StylometricProvider,
    analyze,
)
from docmodel import DocFormat, Document, export, parse
from llm_client import Caller, GatewayClient
from rewriting.direct import DirectLLM, load_env_file, load_routes
from rewriting.engine import BlockOutcome, EngineConfig, RewriteEngine
from rewriting.llm import GatewayLLM, LlmEntailmentScorer
from rewriting.modes import Mode, default_modes_dir, load_modes

_LOCAL_TENANT = "00000000-0000-4000-8000-000000000001"
_FORMATS = {".txt": DocFormat.TXT, ".md": DocFormat.MD, ".docx": DocFormat.DOCX}
_BAND_MARK = {Band.ELEVATED: "!!", Band.UNCERTAIN: "? ", Band.LOW: "  ", Band.ABSTAIN: "~ "}


def _load(path: Path) -> tuple[bytes, Document]:
    fmt = _FORMATS.get(path.suffix.lower())
    if fmt is None:
        sys.exit(f"unsupported file type {path.suffix}; use .txt, .md or .docx")
    data = path.read_bytes()
    return data, parse(data, fmt)


async def _analyze(doc: Document, mode: Mode, sapling: bool) -> DocumentAnalysis:
    providers: list[DetectorProvider] = [StylometricProvider(mode.detector_mode)]
    if sapling:
        key = os.environ.get("SAPLING_API_KEY", "")
        if not key:
            sys.exit("--sapling needs SAPLING_API_KEY")
        providers.append(SaplingProvider(key))
    return await analyze(doc, providers)


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


def _report(outcomes: list[BlockOutcome], mode: Mode) -> str:
    lines = [f"# Rewrite report ({mode.label})", ""]
    counts: dict[str, int] = {}
    for o in outcomes:
        counts[o.status] = counts.get(o.status, 0) + 1
    lines.append(", ".join(f"{k}: {v}" for k, v in sorted(counts.items())) or "nothing selected")
    for o in outcomes:
        lines += ["", f"## {o.block_id}: {o.status}" + (f" ({o.reason})" if o.reason else "")]
        lines += ["", "**Original**", "", o.original]
        if o.rewrite:
            lines += ["", "**Rewrite**", "", o.rewrite]
            lines.append(
                f"\nStyle score {o.ai_score_before:.2f} → {o.ai_score_after or 0:.2f} "
                "(diagnostic only)"
            )
        for c in o.candidates:
            failed = [g.gate for g in c.gates if not g.passed]
            status = "pass" if c.passed else f"fail {failed or c.error}"
            lines.append(f"- round {c.round}, t={c.temperature}: {status}")
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
    data, doc = _load(path)
    result = await _analyze(doc, mode, args.sapling)
    if args.command == "analyze":
        _print_analysis(doc, result)
        return

    config = EngineConfig()
    if args.include_uncertain:
        config.rewrite_bands = {Band.ELEVATED, Band.UNCERTAIN}
    analysis = None if args.all else result
    if args.direct:
        infra = default_modes_dir().parent
        load_env_file(infra.parent / "docker" / ".env")
        direct = DirectLLM(load_routes(infra))
        try:
            engine = RewriteEngine(
                direct,
                mode,
                scorer=None if args.no_nli else LlmEntailmentScorer(direct),
                config=config,
            )
            outcomes = await engine.rewrite_document(doc, analysis=analysis)
        finally:
            await direct.aclose()
        print(f"served by: {direct.served_by}")
    else:
        async with GatewayClient(
            os.environ.get("GATEWAY_URL", "http://localhost:8100"),
            os.environ.get("GATEWAY_SERVICE_TOKEN", "dev-gateway-token-change-me"),
        ) as client:
            tenant = os.environ.get("DTT_TENANT_ID", _LOCAL_TENANT)
            llm = GatewayLLM(client, Caller(tenant_id=tenant))
            engine = RewriteEngine(
                llm,
                mode,
                scorer=None if args.no_nli else LlmEntailmentScorer(llm),
                config=config,
            )
            outcomes = await engine.rewrite_document(doc, analysis=analysis)

    rewrites = {o.block_id: o.rewrite_masked for o in outcomes if o.rewrite_masked}
    out_path = Path(args.output) if args.output else path.with_stem(path.stem + ".rewritten")
    out_path.write_bytes(export(data, doc, rewrites))
    report_path = out_path.with_suffix(".report.md")
    report_path.write_text(_report(outcomes, mode), encoding="utf-8")
    done = sum(o.status == "rewritten" for o in outcomes)
    print(f"{done}/{len(outcomes)} selected blocks rewritten → {out_path}\nreport → {report_path}")


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
