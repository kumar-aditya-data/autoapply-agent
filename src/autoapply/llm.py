"""Claude client used by every agent.

Each agent call returns a validated Pydantic object via the SDK's structured
outputs helper (``messages.parse(..., output_format=Model)``), and logs its
token usage. Verified against anthropic 1.5:

- ``client.messages.parse`` / ``client.beta.messages.parse`` -> ``.parsed_output``
- ``response.usage``: ``input_tokens``, ``output_tokens``,
  ``cache_read_input_tokens``, ``cache_creation_input_tokens``
- refusal fallback: ``betas=["server-side-fallback-2026-07-01"], fallbacks="default"``

Prompt caching: the agent instructions and the numbered CV are identical
across every call in a run, so they form a cached ``system`` prefix; the
volatile per-job content goes in the user turn after the breakpoint.
"""

from __future__ import annotations

import logging
import time
from typing import Protocol, TypeVar

import anthropic
from pydantic import BaseModel

from autoapply.config import Settings
from autoapply.usage import UsageLog, UsageRecord

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMError(RuntimeError):
    """A model call failed or returned unusable output."""


class LLMClient(Protocol):
    """What the graph needs from a model: one structured call per agent step."""

    def structured(self, *, run_id: str, agent: str, output_model: type[T], system: str,
                   cached_context: str | None, user: str) -> T: ...


class AnthropicClient:
    """``LLMClient`` backed by the Anthropic API."""

    def __init__(self, settings: Settings, usage_log: UsageLog, max_tokens: int = 16000) -> None:
        self.settings = settings
        self.usage_log = usage_log
        self.max_tokens = max_tokens
        # Zero-arg constructor resolves ANTHROPIC_API_KEY or an `ant auth login` profile.
        self._client = anthropic.Anthropic()

    def structured(self, *, run_id: str, agent: str, output_model: type[T], system: str,
                   cached_context: str | None, user: str) -> T:
        """Make one call and return a validated ``output_model`` instance."""
        system_blocks: list[dict] = [{"type": "text", "text": system}]
        if cached_context:
            system_blocks.append({"type": "text", "text": cached_context})
        system_blocks[-1]["cache_control"] = {"type": "ephemeral"}

        kwargs = dict(
            model=self.settings.model,
            max_tokens=self.max_tokens,
            system=system_blocks,
            thinking={"type": "adaptive"},
            output_config={"effort": self.settings.effort},
            messages=[{"role": "user", "content": user}],
            output_format=output_model,
        )
        started = time.perf_counter()
        try:
            if self.settings.refusal_fallback:
                response = self._client.beta.messages.parse(
                    betas=[FALLBACK_BETA], fallbacks="default", **kwargs
                )
            else:
                response = self._client.messages.parse(**kwargs)
        except anthropic.AuthenticationError as exc:
            raise LLMError("authentication failed: set ANTHROPIC_API_KEY in .env or run `ant auth login`") from exc
        except anthropic.BadRequestError as exc:
            hint = (" (if this model does not support refusal fallbacks, set AUTOAPPLY_REFUSAL_FALLBACK=false)"
                    if self.settings.refusal_fallback else "")
            raise LLMError(f"bad request for model {self.settings.model}: {exc.message}{hint}") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError("rate limited by the API; wait and retry") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"could not reach the Anthropic API: {exc}") from exc
        latency = time.perf_counter() - started

        usage = response.usage
        self.usage_log.append(UsageRecord(
            run_id=run_id,
            agent=agent,
            model=response.model,
            input_tokens=usage.input_tokens or 0,
            output_tokens=usage.output_tokens or 0,
            cache_read_input_tokens=usage.cache_read_input_tokens or 0,
            cache_creation_input_tokens=usage.cache_creation_input_tokens or 0,
            latency_s=round(latency, 2),
        ))
        log.info("%s on %s: in=%s cache_read=%s out=%s (%.1fs)", agent, response.model,
                 usage.input_tokens, usage.cache_read_input_tokens, usage.output_tokens, latency)

        if response.stop_reason == "refusal":
            category = getattr(response.stop_details, "category", None) if response.stop_details else None
            raise LLMError(f"{agent}: model declined the request (category={category})")
        if response.stop_reason == "max_tokens":
            raise LLMError(f"{agent}: output truncated at max_tokens={self.max_tokens}")
        parsed = response.parsed_output
        if parsed is None:
            raise LLMError(f"{agent}: no parseable structured output")
        return parsed
