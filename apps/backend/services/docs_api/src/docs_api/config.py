from functools import lru_cache
from typing import Literal

from platform_core.config import BaseServiceSettings


class DocsSettings(BaseServiceSettings):
    app_name: str = "Document Service"

    # Uploaded originals, rewritten outputs and job records live here.
    data_dir: str = ".data/docs_api"
    max_upload_mb: int = 10
    # Free-tier LLM limits make parallel documents pointless; queue them instead.
    max_concurrent_jobs: int = 1

    # "direct": providers with keys from docker/.env (no Docker needed);
    # "gateway": the AI Gateway (quotas, cache, usage log).
    llm_backend: Literal["direct", "gateway"] = "direct"
    nli_enabled: bool = True

    cors_origins: list[str] = ["http://localhost:3000"]
    # Local dev: any localhost port (the frontend may not get :3000). Set to "" when deployed.
    cors_origin_regex: str = r"^http://(localhost|127\.0\.0\.1):\d+$"


@lru_cache
def get_settings() -> DocsSettings:
    return DocsSettings()
