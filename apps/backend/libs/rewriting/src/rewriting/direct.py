"""Gateway-free LLM access for local development (`dtt --direct`).

Reads the same routing table as the AI Gateway (infra/gateway/routes.yaml) and the
LiteLLM model map (infra/litellm/config.yaml), then calls providers' OpenAI-compatible
endpoints directly, falling back along the alias chain on 429/5xx/timeouts. No quotas,
breaker, cache or usage log; use the gateway for anything beyond a laptop.

Gemini is never used here for document text (free-tier prompts may be used for
training and human review; docs/PLAN.md §8A.3).
"""

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import structlog
import yaml
from pydantic import BaseModel

from llm_client import GatewayError
from llm_client.types import ChatMessage

logger = structlog.get_logger(__name__)

# litellm provider prefix -> (OpenAI-compatible base URL, API-key env var)
_PROVIDERS: dict[str, tuple[str, str | None]] = {
    "mistral": ("https://api.mistral.ai/v1", "MISTRAL_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "ollama_chat": ("http://localhost:11434/v1", None),
}
_RETRYABLE = {408, 409, 425, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class Route:
    deployment: str
    provider: str
    model: str
    base_url: str
    api_key: str | None
    rpm: int | None = None


def load_env_file(path: Path) -> None:
    """Load KEY=VALUE lines into os.environ without overriding existing values."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_routes(infra: Path) -> dict[str, list[Route]]:
    routes = yaml.safe_load((infra / "gateway" / "routes.yaml").read_text(encoding="utf-8"))
    litellm = yaml.safe_load((infra / "litellm" / "config.yaml").read_text(encoding="utf-8"))
    models = {m["model_name"]: m["litellm_params"]["model"] for m in litellm["model_list"]}

    deployments = routes.get("deployments") or {}
    out: dict[str, list[Route]] = {}
    for alias, spec in routes["aliases"].items():
        chain: list[Route] = []
        for dep in spec["deployments"]:
            target = models.get(dep, "")
            prefix, _, model = target.partition("/")
            if prefix not in _PROVIDERS:
                continue  # gemini (privacy) or unsupported provider
            base_url, key_env = _PROVIDERS[prefix]
            key = os.environ.get(key_env, "").strip() if key_env else None
            if key_env and not key:
                continue
            rpm = (deployments.get(dep) or {}).get("rpm")
            chain.append(Route(dep, prefix, model, base_url, key, rpm))
        out[alias] = chain
    return out


def _strip_fences(text: str) -> str:
    # Reasoning models (qwen3) may prepend <think>...</think> even in JSON mode.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return text.strip()


class DirectLLM:
    """ChatLLM that talks to providers directly using the gateway's alias chains."""

    def __init__(
        self,
        routes: dict[str, list[Route]],
        *,
        client: httpx.AsyncClient | None = None,
        timeout: float = 90.0,
        max_429_retries: int = 3,
    ) -> None:
        self._routes = routes
        self._http = client or httpx.AsyncClient(timeout=timeout)
        self._max_429_retries = max_429_retries
        self._locks: dict[str, asyncio.Lock] = {}
        self._next_slot: dict[str, float] = {}
        self.served_by: dict[str, int] = {}

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _pace(self, route: Route) -> None:
        """Space requests to one deployment by 60/rpm seconds (free tiers burst-limit)."""
        if not route.rpm:
            return
        lock = self._locks.setdefault(route.deployment, asyncio.Lock())
        async with lock:
            now = time.monotonic()
            wait = self._next_slot.get(route.deployment, now) - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._next_slot[route.deployment] = max(now, now + wait) + 60.0 / route.rpm

    async def _post(self, route: Route, body: dict[str, Any]) -> httpx.Response:
        headers = {"Authorization": f"Bearer {route.api_key}"} if route.api_key else {}
        for attempt in range(self._max_429_retries + 1):
            await self._pace(route)
            response = await self._http.post(
                f"{route.base_url}/chat/completions",
                json={**body, "model": route.model},
                headers=headers,
            )
            if response.status_code != 429 or attempt == self._max_429_retries:
                return response
            retry_after = response.headers.get("retry-after", "")
            delay = float(retry_after) if retry_after.replace(".", "", 1).isdigit() else 2**attempt
            logger.info("direct_rate_limited", deployment=route.deployment, retry_in=delay)
            await asyncio.sleep(min(delay, 20.0))
        return response

    async def _chat(self, alias: str, body: dict[str, Any]) -> str:
        chain = self._routes.get(alias) or []
        if not chain:
            raise GatewayError(503, "no_deployment", f"no configured provider for '{alias}'")
        errors: list[str] = []
        for route in chain:
            try:
                response = await self._post(route, body)
            except httpx.HTTPError as exc:
                errors.append(f"{route.deployment}: {type(exc).__name__}")
                continue
            if response.status_code in _RETRYABLE or response.status_code in (401, 403, 404):
                errors.append(f"{route.deployment}: HTTP {response.status_code}")
                logger.warning(
                    "direct_fallback", deployment=route.deployment, status=response.status_code
                )
                continue
            if response.is_error:
                raise GatewayError(response.status_code, "upstream_error", response.text[:300])
            self.served_by[route.deployment] = self.served_by.get(route.deployment, 0) + 1
            content = response.json()["choices"][0]["message"].get("content") or ""
            return str(content)
        raise GatewayError(503, "all_deployments_failed", "; ".join(errors))

    async def complete(
        self, messages: list[ChatMessage], *, model: str, temperature: float, max_tokens: int
    ) -> str:
        return await self._chat(
            model,
            {
                "messages": [m.model_dump(exclude_none=True) for m in messages],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )

    async def judge[M: BaseModel](
        self, messages: list[ChatMessage], schema: type[M], *, model: str
    ) -> M:
        instruction = ChatMessage(
            role="system",
            content=(
                "Respond with a single JSON object matching this JSON schema, and nothing else:\n"
                + json.dumps(schema.model_json_schema())
            ),
        )
        text = await self._chat(
            model,
            {
                "messages": [m.model_dump(exclude_none=True) for m in [instruction, *messages]],
                "temperature": 0,
                "response_format": {"type": "json_object"},
            },
        )
        return schema.model_validate_json(_strip_fences(text))
