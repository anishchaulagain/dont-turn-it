"""Wiring shared by the CLI and the document service: LLM backend and detectors."""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal

from detection import DetectorProvider, SaplingProvider, StylometricProvider
from llm_client import Caller, GatewayClient
from rewriting.direct import DirectLLM, load_env_file, load_routes
from rewriting.llm import ChatLLM, GatewayLLM
from rewriting.modes import Mode, default_modes_dir

Backend = Literal["direct", "gateway"]
LOCAL_TENANT = "00000000-0000-4000-8000-000000000001"


@asynccontextmanager
async def open_llm(backend: Backend) -> AsyncIterator[ChatLLM]:
    """`direct`: providers called with keys from docker/.env (no Docker needed).
    `gateway`: the AI Gateway at GATEWAY_URL (quotas, cache, usage log)."""
    if backend == "direct":
        infra = default_modes_dir().parent
        load_env_file(infra.parent / "docker" / ".env")
        direct = DirectLLM(load_routes(infra))
        try:
            yield direct
        finally:
            await direct.aclose()
        return
    async with GatewayClient(
        os.environ.get("GATEWAY_URL", "http://localhost:8100"),
        os.environ.get("GATEWAY_SERVICE_TOKEN", "dev-gateway-token-change-me"),
    ) as client:
        tenant = os.environ.get("DTT_TENANT_ID", LOCAL_TENANT)
        yield GatewayLLM(client, Caller(tenant_id=tenant))


def detector_providers(mode: Mode, *, sapling: bool = False) -> list[DetectorProvider]:
    providers: list[DetectorProvider] = [StylometricProvider(mode.detector_mode)]
    if sapling:
        key = os.environ.get("SAPLING_API_KEY", "")
        if not key:
            raise ValueError("Sapling needs SAPLING_API_KEY")
        providers.append(SaplingProvider(key))
    return providers
