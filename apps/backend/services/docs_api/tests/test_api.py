import io
import time
import zipfile
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import docx
import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from docs_api.config import DocsSettings
from docs_api.main import create_app
from llm_client.types import ChatMessage
from rewriting.llm import ChatLLM

REWRITE = (
    "Technology shapes education ⟦C1⟧, and digital tools help learners reach their full potential."
)


ALTERNATIVES = [
    {"approach": "minimal", "text": "Technology shapes education ⟦C1⟧."},
    {"approach": "plain", "text": "Technology matters in education."},
]


class FakeLLM:
    async def complete(
        self, messages: list[ChatMessage], *, model: str, temperature: float, max_tokens: int
    ) -> str:
        return REWRITE

    async def judge[M: BaseModel](
        self, messages: list[ChatMessage], schema: type[M], *, model: str
    ) -> M:
        if "alternatives" in schema.model_fields:
            return schema.model_validate({"alternatives": ALTERNATIVES})
        return schema.model_validate({"results": []})


@asynccontextmanager
async def fake_llm() -> AsyncIterator[ChatLLM]:
    yield FakeLLM()


def make_docx() -> bytes:
    d = docx.Document()
    d.add_heading("Introduction", level=1)
    d.add_paragraph(
        "Moreover, it is important to note that technology plays a crucial role in education "
        "(Smith, 2020). Furthermore, digital tools unlock the full potential of learners."
    )
    d.add_paragraph("My grandmother never trusted the microwave, and she said so every day.")
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    settings = DocsSettings(
        _env_file=None, environment="test", data_dir=str(tmp_path), nli_enabled=False
    )
    with TestClient(create_app(settings, llm_factory=fake_llm)) as c:
        yield c


def wait_done(client: TestClient, job_id: str) -> dict[str, Any]:
    for _ in range(100):
        job: dict[str, Any] = client.get(f"/v1/jobs/{job_id}").json()
        if job["status"] in ("done", "failed"):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_modes(client: TestClient) -> None:
    modes = {m["id"]: m for m in client.get("/v1/modes").json()}
    assert modes["student.essay"]["default_view"] == "coach"


def test_rewrite_job_end_to_end(client: TestClient) -> None:
    data = make_docx()
    response = client.post(
        "/v1/jobs",
        files={"file": ("essay.docx", data)},
        data={"mode": "student.essay", "scope": "flagged_uncertain"},
    )
    assert response.status_code == 202, response.text
    job = wait_done(client, response.json()["id"])
    assert job["status"] == "done", job["error"]

    result = job["result"]
    assert result["after"]["score"] < result["before"]["score"]
    [para] = result["paragraphs"]
    assert para["status"] == "rewritten" and "(Smith, 2020)" in para["rewrite"]
    assert para["score_after"] < para["score_before"]
    assert para["flagged_sentences"] and para["reasons"]

    download = client.get(f"/v1/jobs/{job['id']}/download")
    assert download.status_code == 200
    assert 'filename="essay.rewritten.docx"' in download.headers["content-disposition"]
    before, after = zipfile.ZipFile(io.BytesIO(data)), zipfile.ZipFile(io.BytesIO(download.content))
    assert [n for n in before.namelist() if before.read(n) != after.read(n)] == [
        "word/document.xml"
    ]


@pytest.mark.parametrize(
    ("files", "form", "code"),
    [
        ({"file": ("notes.pdf", b"%PDF")}, {"mode": "student.essay"}, "invalid_upload"),
        ({"file": ("notes.txt", b"hello")}, {"mode": "nope"}, "invalid_upload"),
        ({"file": ("notes.txt", b"")}, {"mode": "student.essay"}, "invalid_upload"),
    ],
)
def test_rejects_bad_uploads(
    client: TestClient, files: dict[str, Any], form: dict[str, str], code: str
) -> None:
    response = client.post("/v1/jobs", files=files, data=form)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == code


def test_unknown_job_and_early_download(client: TestClient) -> None:
    assert client.get("/v1/jobs/" + "0" * 32).status_code == 404
    assert client.get("/v1/jobs/..%2F..%2Fsecrets").status_code == 404


def test_coach_flow(client: TestClient) -> None:
    created = client.post(
        "/v1/documents", files={"file": ("essay.docx", make_docx())}, data={"mode": "student.essay"}
    )
    assert created.status_code == 201, created.text
    session = created.json()
    doc_id = session["id"]
    para = next(b for b in session["document"]["blocks"] if b["editable"])
    first = para["sentences"][0]
    assert first["band"] in ("elevated", "uncertain") and first["reasons"]
    assert session["document"]["flagged"] >= 1

    target = {"block_id": para["id"], "sentence_index": 0}
    suggestions = client.post(f"/v1/documents/{doc_id}/suggestions", json=target).json()
    assert suggestions[0]["check"]["passed"]
    assert suggestions[0]["text"] == "Technology shapes education (Smith, 2020)."
    assert not suggestions[1]["check"]["passed"]  # dropped the citation

    check = client.post(
        f"/v1/documents/{doc_id}/check", json={**target, "text": "Tech matters to schools."}
    ).json()
    assert not check["passed"]
    assert any("(Smith, 2020)" in c["message"] for c in check["checks"] if not c["passed"])

    assert client.get(f"/v1/documents/{doc_id}/download").status_code == 409
    exported = client.post(
        f"/v1/documents/{doc_id}/export",
        json={"edits": [{**target, "text": suggestions[0]["text"], "source": "suggestion"}]},
    )
    assert exported.status_code == 200, exported.text
    body = exported.json()
    assert body["applied"] == 1 and body["warnings"] == []
    assert body["after"]["score"] < session["document"]["summary"]["score"]

    download = client.get(body["download_url"])
    assert 'filename="essay.edited.docx"' in download.headers["content-disposition"]
    text = [p.text for p in docx.Document(io.BytesIO(download.content)).paragraphs]
    assert text[1].startswith("Technology shapes education (Smith, 2020). Furthermore")


def test_coach_rejects_bad_targets(client: TestClient) -> None:
    session = client.post(
        "/v1/documents", files={"file": ("essay.docx", make_docx())}, data={"mode": "student.essay"}
    ).json()
    heading = session["document"]["blocks"][0]
    bad = client.post(
        f"/v1/documents/{session['id']}/check",
        json={"block_id": heading["id"], "sentence_index": 0, "text": "x"},
    )
    assert bad.status_code == 400 and bad.json()["error"]["code"] == "invalid_request"
    assert client.get("/v1/documents/" + "f" * 32).status_code == 404
