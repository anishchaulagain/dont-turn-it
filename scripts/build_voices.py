"""Build voice presets (infra/voices/*.json) from licensed, human-written corpora.

    cd apps/backend && uv run python ../../scripts/build_voices.py

Every source is human-written, published before Nov 2022 (no LLM contamination), and
licensed for public use; the license and attribution are stored in each preset.
Raw downloads are cached in .data/datasets/ (gitignored).

Not used: PERSUADE 2.0 (CC BY-NC-SA 4.0, non-commercial), BAWE/MICUSP (non-commercial),
Reddit-derived corpora (terms bar commercial use and training).
"""

import html
import json
import os
import random
import re
import statistics
import sys
import time
from pathlib import Path
from xml.etree import ElementTree

import httpx

from rewriting.voices import StyleTargets, Voice, measure

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".data" / "datasets"
OUT = ROOT / "infra" / "voices"
# Wikimedia robot policy: identify the tool and keep volume low (~1 request/s).
# Their policy requires contact details (URL or email), so Wikipedia is only fetched when
# VOICE_BUILDER_CONTACT is set, e.g. VOICE_BUILDER_CONTACT="https://github.com/you/repo".
CONTACT = os.environ.get("VOICE_BUILDER_CONTACT", "").strip()
UA = {"User-Agent": f"DontTurnItVoiceBuilder/0.1 ({CONTACT or 'no contact'}; low volume) httpx"}
RNG = random.Random(17)

_WS = re.compile(r"\s+")


def http() -> httpx.Client:
    return httpx.Client(headers=UA, timeout=60, follow_redirects=True)


def cached(name: str, fetch: "callable[[], str]") -> str:  # type: ignore[valid-type]
    path = CACHE / name
    if path.is_file():
        return path.read_text(encoding="utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = fetch()
    # Normalise CRLF and write without newline translation: on Windows a text-mode write
    # of "\r\n" becomes "\r\r\n", which reads back as a blank line after every line.
    text = text.replace("\r\n", "\n")
    path.write_text(text, encoding="utf-8", newline="\n")
    return text


def good_paragraph(text: str, lo: int = 45, hi: int = 170) -> bool:
    words = text.split()
    if not lo <= len(words) <= hi:
        return False
    if not text.rstrip().endswith((".", "!", "?", '"', "”")):
        return False
    if sum(c.isdigit() for c in text) > 12 or text.count("[") > 2 or text.count("(") > 4:
        return False
    if text.count('"') + text.count("“") > 4:  # dialogue-heavy
        return False
    if text.count("et al") > 1 or "�" in text:  # citation lists, mis-decoded text
        return False
    return not re.search(r"\b(CHAPTER|Fig\.|Table \d|http)", text)


def is_verse(block: str) -> bool:
    """Poems and songs: several short hard-wrapped lines."""
    lines = [line for line in block.splitlines() if line.strip()]
    return len(lines) >= 3 and statistics.median(len(line) for line in lines) < 48


# ---------- Project Gutenberg: classic essays (US public domain) ----------
# Ebook id -> expected title fragment (checked against the file header).
GUTENBERG_BOOKS = {
    386: "Virginibus Puerisque",  # R. L. Stevenson, 1881
    8092: "Tremendous Trifles",  # G. K. Chesterton, 1909
    10343: "Charles and Mary Lamb",  # Lamb, Elia essays (Works vol. 2)
    2944: "Essays",  # R. W. Emerson, Essays: First Series, 1841
}


def _get(client: httpx.Client, url: str, **params: object) -> str:
    """GET with polite retries on 429/5xx; raises instead of returning error pages."""
    for attempt in range(5):
        response = client.get(url, params=params or None)  # type: ignore[arg-type]
        if response.status_code not in (429, 500, 502, 503, 504):
            break
        time.sleep(2**attempt)
    response.raise_for_status()  # never cache error pages
    # All sources here serve UTF-8; httpx's charset guess garbles accented names otherwise.
    return response.content.decode("utf-8", errors="replace")


def gutenberg_paragraphs(client: httpx.Client) -> list[str]:
    paragraphs: list[str] = []
    for book_id, expected in GUTENBERG_BOOKS.items():
        url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
        raw = cached(f"gutenberg/{book_id}.txt", lambda u=url: _get(client, u))
        header = raw[:3000]
        if expected not in header:
            print(
                f"  gutenberg: #{book_id} does not look like {expected!r}; skipped", file=sys.stderr
            )
            continue
        start = raw.find("*** START OF")
        end = raw.find("*** END OF")
        body = raw[raw.find("\n", start) + 1 : end] if start >= 0 and end > start else raw
        before = len(paragraphs)
        for block in re.split(r"\r?\n\s*\r?\n", body):
            if is_verse(block):
                continue
            text = _WS.sub(" ", block).strip()
            if good_paragraph(text) and text.upper() != text:
                paragraphs.append(text)
        print(f"  gutenberg: #{book_id} {expected} -> +{len(paragraphs) - before}")
    return paragraphs


# ---------- Europe PMC: open-access articles, CC BY / CC0, 2015-2021 ----------
PMC_QUERY = (
    '(OPEN_ACCESS:y) AND (LICENSE:"cc by" OR LICENSE:"cc0") AND HAS_FT:y AND '
    "(FIRST_PDATE:[2015-01-01 TO 2021-12-31]) AND "
    '(TITLE:"education" OR TITLE:"social work" OR TITLE:"child development" OR '
    'TITLE:"wellbeing" OR TITLE:"safeguarding")'
)
_SECTIONS = re.compile(r"introduction|background|discussion|conclusion", re.IGNORECASE)


def pmc_paragraphs(client: httpx.Client, articles: int = 40) -> tuple[list[str], list[str]]:
    search = json.loads(
        cached(
            "europepmc/search.json",
            lambda: _get(
                client,
                "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                query=PMC_QUERY,
                format="json",
                pageSize=100,
                resultType="core",
            ),
        )
    )
    paragraphs: list[str] = []
    cited: list[str] = []
    for hit in search.get("resultList", {}).get("result", []):
        license_ = (hit.get("license") or "").lower()
        pmcid = hit.get("pmcid")
        if not pmcid or license_ not in ("cc by", "cc0"):
            continue
        xml = cached(
            f"europepmc/{pmcid}.xml",
            lambda p=pmcid: _get(
                client, f"https://www.ebi.ac.uk/europepmc/webservices/rest/{p}/fullTextXML"
            ),
        )
        try:
            root = ElementTree.fromstring(xml)
        except ElementTree.ParseError:
            continue
        before = len(paragraphs)
        for sec in root.iter("sec"):
            title = "".join(sec.findtext("title") or "")
            if not _SECTIONS.search(title):
                continue
            for p in sec.findall("p"):
                text = _WS.sub(" ", "".join(p.itertext())).strip()
                text = re.sub(r"\s*\[[\d,\s–-]+\]", "", text)  # numeric citation markers
                if good_paragraph(text, hi=190):
                    paragraphs.append(text)
        if len(paragraphs) > before:
            cited.append(f"{hit.get('authorString', '')[:60]} ({hit.get('pubYear')}). "
                         f"{hit.get('title', '')[:90]} {pmcid}, {license_.upper()}")  # fmt: skip
        if len(cited) >= articles:
            break
        time.sleep(0.2)
    print(f"  europepmc: {len(cited)} articles -> {len(paragraphs)} paragraphs")
    return paragraphs, cited


# ---------- Simple English Wikipedia: revisions as of end of 2021 ----------
_P = re.compile(r"<p>(.*?)</p>", re.DOTALL)
_TAG = re.compile(r"<[^>]+>")
_REF = re.compile(r"\[\d+\]|\[citation needed\]")


def wiki_paragraphs(client: httpx.Client, pages: int = 120) -> list[str]:
    api = "https://simple.wikipedia.org/w/api.php"

    def random_titles() -> str:
        titles: list[str] = []
        for _ in range(3):
            data = json.loads(
                _get(
                    client,
                    api,
                    action="query",
                    list="random",
                    rnnamespace=0,
                    rnlimit=50,
                    format="json",
                )  # fmt: skip
            )
            titles += [r["title"] for r in data["query"]["random"]]
            time.sleep(0.5)
        return json.dumps(titles)

    titles = json.loads(cached("simplewiki/random.json", random_titles))
    paragraphs: list[str] = []
    for title in titles[:pages]:
        safe = re.sub(r"[^\w-]+", "_", title)[:80]

        def fetch(t: str = title) -> str:
            revs = json.loads(
                _get(
                    client,
                    api,
                    action="query",
                    prop="revisions",
                    titles=t,
                    rvlimit=1,
                    rvstart="2021-12-31T00:00:00Z",
                    rvdir="older",
                    rvprop="ids",
                    format="json",
                    formatversion=2,
                )  # fmt: skip
            )
            page = revs["query"]["pages"][0]
            if "revisions" not in page:
                return ""  # article created after 2021: skip (contamination guard)
            parsed = json.loads(
                _get(
                    client,
                    api,
                    action="parse",
                    oldid=page["revisions"][0]["revid"],
                    prop="text",
                    format="json",
                    formatversion=2,
                )  # fmt: skip
            )
            return str(parsed.get("parse", {}).get("text", ""))

        page_html = cached(f"simplewiki/{safe}.html", fetch)
        for m in _P.finditer(page_html):
            text = _WS.sub(" ", html.unescape(_REF.sub("", _TAG.sub("", m.group(1))))).strip()
            if good_paragraph(text, lo=35):
                paragraphs.append(text)
        time.sleep(1.0)
    print(f"  simplewiki: {len(titles[:pages])} pages -> {len(paragraphs)} paragraphs")
    return paragraphs


def exemplars(paragraphs: list[str], targets: StyleTargets, n: int = 6) -> list[str]:
    """Short paragraphs whose own style is closest to the corpus profile."""
    pool = [p for p in paragraphs if 50 <= len(p.split()) <= 130]
    RNG.shuffle(pool)

    def distance(p: str) -> float:
        t = measure([p])
        return abs(t.sentence_words_mean - targets.sentence_words_mean) / max(
            1, targets.sentence_words_mean
        ) + abs(t.transition_opener_share - targets.transition_opener_share)

    return sorted(pool[:300], key=distance)[:n]


def write_voice(voice: Voice) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{voice.id}.json"
    path.write_text(voice.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(f"-> {path.relative_to(ROOT)}: {voice.targets.model_dump()}")


def main() -> None:
    with http() as client:
        print("Project Gutenberg essays")
        essays = gutenberg_paragraphs(client)
        if essays:
            t = measure(essays)
            write_voice(
                Voice(
                    id="classic-essay",
                    label="Thoughtful personal essay",
                    description=(
                        "Reflective, first-person essay prose from classic human essayists: "
                        "varied sentences, concrete images, an individual point of view. Use "
                        "its rhythm and directness, not its old-fashioned vocabulary."
                    ),
                    source=(
                        "Project Gutenberg: Stevenson, Virginibus Puerisque (1881); Chesterton, "
                        "Tremendous Trifles (1909); Lamb, Essays of Elia (1823); Emerson, Essays: "
                        f"First Series (1841). {len(essays)} paragraphs."
                    ),
                    license="Public domain in the USA (Project Gutenberg; trademark removed)",
                    targets=t,
                    exemplars=exemplars(essays, t),
                )
            )

        print("Europe PMC open-access articles")
        papers, cited = pmc_paragraphs(client)
        if papers:
            t = measure(papers)
            write_voice(
                Voice(
                    id="academic-article",
                    label="Academic article (education & social care)",
                    description=(
                        "Published research prose in education, psychology and social work: "
                        "precise, plainly worded, claims hedged as in the evidence."
                    ),
                    source=f"Europe PMC open-access articles, 2015-2021 ({len(papers)} "
                    "paragraphs): " + "; ".join(cited[:12]) + ("; ..." if len(cited) > 12 else ""),
                    license="CC BY 4.0 / CC0 (per article; attribution listed in source)",
                    targets=t,
                    exemplars=exemplars(papers, t),
                )
            )

        print("Simple English Wikipedia (2021 revisions)")
        if not CONTACT:
            print("  skipped: set VOICE_BUILDER_CONTACT (Wikimedia robot policy)")
            return
        plain = wiki_paragraphs(client)
        if plain:
            t = measure(plain)
            write_voice(
                Voice(
                    id="plain-explainer",
                    label="Plain explainer",
                    description=(
                        "Clear, everyday explanatory prose for a general reader: short "
                        "sentences, common words, one idea at a time."
                    ),
                    source=f"Simple English Wikipedia articles, revisions as of 2021-12-31 "
                    f"({len(plain)} paragraphs); contributors per page history",
                    license="CC BY-SA 4.0 (exemplars in this file are shared alike)",
                    targets=t,
                    exemplars=exemplars(plain, t),
                )
            )


if __name__ == "__main__":
    main()
