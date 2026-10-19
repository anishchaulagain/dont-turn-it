from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, File, Form, Request, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from detection import analyze
from docs_api.config import DocsSettings
from docs_api.documents import DocumentMeta, DocumentStore
from docs_api.jobs import FORMATS, Job, JobRunner, LlmFactory
from platform_core.errors import AppError, NotFoundError
from rewriting import LlmEntailmentScorer, Mode
from rewriting.coach import (
    CoachDocument,
    Edit,
    SentenceCheck,
    Suggestion,
    build_view,
    check_sentence,
    export_with_edits,
    suggest,
)
from rewriting.pipeline import Scope, ScoreSummary
from rewriting.runtime import detector_providers
from rewriting.voices import Voice

router = APIRouter(prefix="/v1")
health_router = APIRouter()

_MIME = {
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".txt": "text/plain; charset=utf-8",
    ".md": "text/markdown; charset=utf-8",
}


class InvalidUploadError(AppError):
    code = "invalid_upload"


class InvalidRequestError(AppError):
    code = "invalid_request"


class NotReadyError(AppError):
    status_code = status.HTTP_409_CONFLICT
    code = "not_ready"


class ModeInfo(BaseModel):
    id: str
    label: str
    audience: Literal["student", "professional"]
    phase: int
    default_view: Literal["coach", "rewrite"]


class VoiceInfo(BaseModel):
    id: str
    label: str
    description: str
    source: str
    license: str


class CoachSession(BaseModel):
    id: str
    filename: str
    mode: str
    voice: str | None
    document: CoachDocument


class SentenceRequest(BaseModel):
    block_id: str
    sentence_index: int = Field(ge=0)


class CheckRequest(SentenceRequest):
    text: str = Field(min_length=1, max_length=2000)


class ExportRequest(BaseModel):
    edits: list[Edit] = Field(max_length=2000)


class ExportResponse(BaseModel):
    after: ScoreSummary
    warnings: list[str]
    applied: int
    download_url: str


# ---------- state helpers ----------
def _settings(request: Request) -> DocsSettings:
    settings: DocsSettings = request.app.state.settings
    return settings


def _runner(request: Request) -> JobRunner:
    runner: JobRunner = request.app.state.runner
    return runner


def _store(request: Request) -> DocumentStore:
    store: DocumentStore = request.app.state.store
    return store


def _modes(request: Request) -> dict[str, Mode]:
    modes: dict[str, Mode] = request.app.state.modes
    return modes


def _voices(request: Request) -> dict[str, Voice]:
    voices: dict[str, Voice] = request.app.state.voices
    return voices


def _llm_factory(request: Request) -> LlmFactory:
    factory: LlmFactory = request.app.state.llm_factory
    return factory


async def _read_upload(
    request: Request, file: UploadFile, mode: str, voice: str | None
) -> tuple[bytes, str]:
    settings = _settings(request)
    filename = Path(file.filename or "document").name
    if Path(filename).suffix.lower() not in FORMATS:
        raise InvalidUploadError("Upload a .docx, .txt or .md file")
    if mode not in _modes(request):
        raise InvalidUploadError(f"Unknown mode '{mode}'", details={"modes": list(_modes(request))})
    if voice and voice not in _voices(request):
        raise InvalidUploadError(
            f"Unknown voice '{voice}'", details={"voices": list(_voices(request))}
        )
    limit = settings.max_upload_mb * 1024 * 1024
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise InvalidUploadError(f"File is larger than {settings.max_upload_mb} MB")
    if not data:
        raise InvalidUploadError("File is empty")
    return data, filename


def _job(request: Request, job_id: str) -> Job:
    job = _runner(request).get(job_id)
    if job is None:
        raise NotFoundError(f"Job {job_id} not found")
    return job


def _meta(request: Request, doc_id: str) -> DocumentMeta:
    meta = _store(request).meta(doc_id)
    if meta is None:
        raise NotFoundError(f"Document {doc_id} not found")
    return meta


def _sentence(request: Request, meta: DocumentMeta, block_id: str, index: int) -> None:
    doc = _store(request).parsed(meta)
    try:
        block = doc.block(block_id)
    except KeyError as exc:
        raise InvalidRequestError(f"Unknown block {block_id}") from exc
    if not block.rewritable:
        raise InvalidRequestError("This block cannot be edited")
    if index >= len(block.sentences):
        raise InvalidRequestError("Unknown sentence")


# ---------- catalogue ----------
@health_router.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/modes")
async def list_modes(request: Request) -> list[ModeInfo]:
    return [
        ModeInfo(
            id=m.id, label=m.label, audience=m.audience, phase=m.phase, default_view=m.default_view
        )
        for m in _modes(request).values()
    ]


@router.get("/voices")
async def list_voices(request: Request) -> list[VoiceInfo]:
    return [
        VoiceInfo(
            id=v.id, label=v.label, description=v.description, source=v.source, license=v.license
        )
        for v in _voices(request).values()
    ]


# ---------- automatic rewrite jobs ----------
@router.post("/jobs", status_code=status.HTTP_202_ACCEPTED)
async def create_job(
    request: Request,
    file: Annotated[UploadFile, File()],
    mode: Annotated[str, Form()],
    scope: Annotated[Scope, Form()] = "flagged_uncertain",
    voice: Annotated[str | None, Form()] = None,
) -> Job:
    data, filename = await _read_upload(request, file, mode, voice)
    return _runner(request).submit(data, filename, mode, scope, voice or None)


@router.get("/jobs/{job_id}")
async def get_job(request: Request, job_id: str) -> Job:
    return _job(request, job_id)


@router.get("/jobs/{job_id}/download")
async def download_job(request: Request, job_id: str) -> FileResponse:
    job = _job(request, job_id)
    path = _runner(request).output_path(job)
    if job.status != "done" or not path.is_file():
        raise NotReadyError(f"Job is {job.status}")
    return FileResponse(path, media_type=_MIME[job.suffix], filename=job.download_name)


# ---------- coach ----------
async def _session(request: Request, meta: DocumentMeta) -> CoachSession:
    doc = _store(request).parsed(meta)
    mode = _modes(request)[meta.mode]
    view = build_view(doc, await analyze(doc, detector_providers(mode)))
    return CoachSession(
        id=meta.id, filename=meta.filename, mode=meta.mode, voice=meta.voice, document=view
    )


@router.post("/documents", status_code=status.HTTP_201_CREATED)
async def create_document(
    request: Request,
    file: Annotated[UploadFile, File()],
    mode: Annotated[str, Form()],
    voice: Annotated[str | None, Form()] = None,
) -> CoachSession:
    data, filename = await _read_upload(request, file, mode, voice)
    meta = _store(request).create(data, filename, mode, voice or None)
    try:
        return await _session(request, meta)
    except (ValueError, KeyError) as exc:
        raise InvalidUploadError(f"Could not read this file: {exc}") from exc


@router.get("/documents/{doc_id}")
async def get_document(request: Request, doc_id: str) -> CoachSession:
    return await _session(request, _meta(request, doc_id))


@router.post("/documents/{doc_id}/suggestions")
async def sentence_suggestions(
    request: Request, doc_id: str, body: SentenceRequest
) -> list[Suggestion]:
    meta = _meta(request, doc_id)
    _sentence(request, meta, body.block_id, body.sentence_index)
    doc = _store(request).parsed(meta)
    mode = _modes(request)[meta.mode]
    analysis = await analyze(doc, detector_providers(mode))
    reasons = next(
        (
            s.reasons
            for s in analysis.sentences
            if s.block_id == body.block_id and s.sentence_index == body.sentence_index
        ),
        [],
    )
    async with _llm_factory(request)() as llm:
        return await suggest(
            llm,
            mode,
            doc,
            body.block_id,
            body.sentence_index,
            reasons=reasons,
            voice=_voices(request).get(meta.voice or ""),
            scorer=LlmEntailmentScorer(llm) if _settings(request).nli_enabled else None,
        )


@router.post("/documents/{doc_id}/check")
async def check_edit(request: Request, doc_id: str, body: CheckRequest) -> SentenceCheck:
    meta = _meta(request, doc_id)
    _sentence(request, meta, body.block_id, body.sentence_index)
    block = _store(request).parsed(meta).block(body.block_id)
    return await check_sentence(_modes(request)[meta.mode], block, body.sentence_index, body.text)


@router.post("/documents/{doc_id}/export")
async def export_document(request: Request, doc_id: str, body: ExportRequest) -> ExportResponse:
    meta = _meta(request, doc_id)
    store = _store(request)
    doc = store.parsed(meta)
    for edit in body.edits:
        _sentence(request, meta, edit.block_id, edit.sentence_index)
    mode = _modes(request)[meta.mode]
    result = await export_with_edits(
        store.original(meta), meta.fmt, doc, body.edits, detector_providers(mode)
    )
    store.output_path(meta).write_bytes(result.output)
    return ExportResponse(
        after=result.after,
        warnings=result.warnings,
        applied=result.applied,
        download_url=f"/v1/documents/{meta.id}/download",
    )


@router.get("/documents/{doc_id}/download")
async def download_document(request: Request, doc_id: str) -> FileResponse:
    meta = _meta(request, doc_id)
    path = _store(request).output_path(meta)
    if not path.is_file():
        raise NotReadyError("Export the document first")
    return FileResponse(path, media_type=_MIME[meta.suffix], filename=meta.download_name)
