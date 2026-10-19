from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from docs_api.api import health_router, router
from docs_api.config import DocsSettings, get_settings
from docs_api.documents import DocumentStore
from docs_api.jobs import JobRunner, LlmFactory
from llm_client import GatewayError
from platform_core.errors import error_response, register_exception_handlers
from platform_core.logging import configure_logging
from platform_core.middleware import RequestContextMiddleware
from rewriting import load_modes
from rewriting.runtime import open_llm
from rewriting.voices import load_voices

logger = structlog.get_logger(__name__)


def create_app(
    settings: DocsSettings | None = None, llm_factory: LlmFactory | None = None
) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        modes, voices = load_modes(), load_voices()
        factory = llm_factory or partial(open_llm, settings.llm_backend)
        runner = JobRunner(
            Path(settings.data_dir),
            modes,
            factory,
            max_concurrent=settings.max_concurrent_jobs,
            nli_enabled=settings.nli_enabled,
            voices=voices,
        )
        app.state.settings, app.state.modes, app.state.voices = settings, modes, voices
        app.state.runner, app.state.llm_factory = runner, factory
        app.state.store = DocumentStore(Path(settings.data_dir))
        logger.info(
            "startup", backend=settings.llm_backend, modes=sorted(modes), voices=sorted(voices)
        )
        yield
        await runner.drain()

    app = FastAPI(
        title=settings.app_name,
        debug=settings.debug,
        lifespan=lifespan,
        description="Upload a document, flag AI-like writing, rewrite it with meaning "
        "checks, and download the result. Scores are style indicators, not authorship.",
    )
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_origin_regex=settings.cors_origin_regex or None,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
        expose_headers=["Content-Disposition"],
    )
    register_exception_handlers(app)

    @app.exception_handler(GatewayError)
    async def _llm_unavailable(request: Request, exc: GatewayError) -> JSONResponse:
        logger.warning("llm_unavailable", status=exc.status_code, error=exc.message[:300])
        return error_response(
            request,
            status_code=503,
            code="llm_unavailable",
            message="The free language models are busy or rate-limited right now. "
            "Please try again in a minute.",
            headers={"Retry-After": "60"},
        )

    app.include_router(health_router)
    app.include_router(router)
    return app


app = create_app()
