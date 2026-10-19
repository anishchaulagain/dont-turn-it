"""Coach sessions: an uploaded original plus its parsed form, stored on disk."""

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from docmodel import DocFormat, Document, parse
from docs_api.jobs import FORMATS


class DocumentMeta(BaseModel):
    id: str
    filename: str
    mode: str
    voice: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def suffix(self) -> str:
        return Path(self.filename).suffix.lower()

    @property
    def fmt(self) -> DocFormat:
        return FORMATS[self.suffix]

    @property
    def download_name(self) -> str:
        return f"{Path(self.filename).stem}.edited{self.suffix}"


class DocumentStore:
    def __init__(self, data_dir: Path) -> None:
        self._dir = data_dir / "documents"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._parsed: dict[str, Document] = {}

    def _path(self, doc_id: str, name: str) -> Path:
        return self._dir / doc_id / name

    def create(self, data: bytes, filename: str, mode: str, voice: str | None) -> DocumentMeta:
        meta = DocumentMeta(id=uuid.uuid4().hex, filename=filename, mode=mode, voice=voice)
        self._path(meta.id, "").mkdir(parents=True)
        self._path(meta.id, f"original{meta.suffix}").write_bytes(data)
        self._path(meta.id, "meta.json").write_text(meta.model_dump_json(), encoding="utf-8")
        return meta

    def meta(self, doc_id: str) -> DocumentMeta | None:
        if len(doc_id) != 32 or not doc_id.isalnum():
            return None
        path = self._path(doc_id, "meta.json")
        if not path.is_file():
            return None
        return DocumentMeta.model_validate(json.loads(path.read_text(encoding="utf-8")))

    def original(self, meta: DocumentMeta) -> bytes:
        return self._path(meta.id, f"original{meta.suffix}").read_bytes()

    def parsed(self, meta: DocumentMeta) -> Document:
        if meta.id not in self._parsed:
            self._parsed[meta.id] = parse(self.original(meta), meta.fmt)
        return self._parsed[meta.id]

    def output_path(self, meta: DocumentMeta) -> Path:
        return self._path(meta.id, f"edited{meta.suffix}")
