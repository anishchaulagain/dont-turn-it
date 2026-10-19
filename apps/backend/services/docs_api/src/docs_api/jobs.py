"""Background rewrite jobs: in-process queue, state persisted as JSON on disk.

Single-process by design (local / $0 phase). The RabbitMQ worker from the plan
replaces `JobRunner` later without changing the API contract.
"""

import asyncio
import uuid
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import structlog
from pydantic import BaseModel, Field

from docmodel import DocFormat
from rewriting import LlmEntailmentScorer, Mode
from rewriting.llm import ChatLLM
from rewriting.pipeline import ParagraphChange, Scope, ScoreSummary, run_pipeline
from rewriting.runtime import detector_providers
from rewriting.voices import Voice

logger = structlog.get_logger(__name__)

Status = Literal["queued", "analyzing", "rewriting", "exporting", "rescoring", "done", "failed"]
LlmFactory = Callable[[], AbstractAsyncContextManager[ChatLLM]]

FORMATS = {".docx": DocFormat.DOCX, ".txt": DocFormat.TXT, ".md": DocFormat.MD}
# Pipeline stages reported while running; "done" is set only once the output is saved.
_STAGES: dict[str, Status] = {
    "analyzing": "analyzing",
    "rewriting": "rewriting",
    "exporting": "exporting",
    "rescoring": "rescoring",
}


class JobResult(BaseModel):
    before: ScoreSummary
    after: ScoreSummary
    paragraphs: list[ParagraphChange]


class Job(BaseModel):
    id: str
    filename: str
    mode: str
    scope: Scope
    voice: str | None = None
    status: Status = "queued"
    progress_done: int = 0
    progress_total: int = 0
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    result: JobResult | None = None

    @property
    def suffix(self) -> str:
        return Path(self.filename).suffix.lower()

    @property
    def download_name(self) -> str:
        stem = Path(self.filename).stem
        return f"{stem}.rewritten{self.suffix}"


class JobRunner:
    def __init__(
        self,
        data_dir: Path,
        modes: dict[str, Mode],
        llm_factory: LlmFactory,
        *,
        max_concurrent: int = 1,
        nli_enabled: bool = True,
        voices: dict[str, Voice] | None = None,
    ) -> None:
        self._dir = data_dir / "jobs"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._modes = modes
        self._llm_factory = llm_factory
        self._gate = asyncio.Semaphore(max_concurrent)
        self._nli = nli_enabled
        self._voices = voices or {}
        self._jobs: dict[str, Job] = {}
        self._tasks: set[asyncio.Task[None]] = set()

    def _path(self, job_id: str, name: str) -> Path:
        return self._dir / job_id / name

    def _save(self, job: Job) -> None:
        self._path(job.id, "job.json").write_text(job.model_dump_json(indent=2), encoding="utf-8")

    def get(self, job_id: str) -> Job | None:
        job = self._jobs.get(job_id)
        if job is None and len(job_id) == 32 and job_id.isalnum():
            path = self._path(job_id, "job.json")
            if path.is_file():
                job = Job.model_validate_json(path.read_text(encoding="utf-8"))
                self._jobs[job_id] = job
        return job

    def output_path(self, job: Job) -> Path:
        return self._path(job.id, f"output{job.suffix}")

    def submit(
        self, data: bytes, filename: str, mode: str, scope: Scope, voice: str | None = None
    ) -> Job:
        job = Job(id=uuid.uuid4().hex, filename=filename, mode=mode, scope=scope, voice=voice)
        self._path(job.id, "").mkdir(parents=True)
        self._path(job.id, f"original{job.suffix}").write_bytes(data)
        self._jobs[job.id] = job
        self._save(job)
        task = asyncio.create_task(self._run(job, data))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return job

    async def _run(self, job: Job, data: bytes) -> None:
        async with self._gate:
            try:
                await self._execute(job, data)
            except Exception as exc:  # noqa: BLE001 - surface any failure on the job
                logger.exception("job_failed", job=job.id)
                job.status, job.error = "failed", f"{type(exc).__name__}: {exc}"
            job.finished_at = datetime.now(UTC)
            self._save(job)

    async def _execute(self, job: Job, data: bytes) -> None:
        mode = self._modes[job.mode]

        def progress(stage: str, done: int, total: int) -> None:
            job.status = _STAGES.get(stage, job.status)
            if stage == "rewriting":
                job.progress_done, job.progress_total = done, total

        async with self._llm_factory() as llm:
            result = await run_pipeline(
                data,
                FORMATS[job.suffix],
                mode,
                llm=llm,
                providers=detector_providers(mode),
                scorer=LlmEntailmentScorer(llm) if self._nli else None,
                scope=job.scope,
                voice=self._voices.get(job.voice or ""),
                on_progress=progress,
            )
        self.output_path(job).write_bytes(result.output)
        job.result = JobResult(
            before=result.before, after=result.after, paragraphs=result.paragraphs
        )
        job.status = "done"

    async def drain(self) -> None:
        """Wait for running jobs (tests and graceful shutdown)."""
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
