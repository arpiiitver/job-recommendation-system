"""Native tool-calling transport for the Career Copilot agent.

Kept separate from ``provider.py`` on purpose: the JSON routes (resume
parsing, NL search, explanations, roadmap) keep their existing transport
untouched, while the agent talks to any OpenAI-compatible Chat Completions
endpoint that supports function calling:

* OpenAI itself (``LLM_PROVIDER=openai``),
* Gemini through Google's OpenAI-compatible endpoint (``LLM_PROVIDER=gemini``),
* local models via Ollama (``LLM_PROVIDER=ollama``).

The LLM only decides *which* tool to call and how to phrase the reply; every
number it can quote comes from the trained ranking pipeline via the tools.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from app.core.config import get_settings
from app.services.llm.provider import LlmUsage, _log_usage, strip_fences
from app.services.llm.strict import strict_bail

log = logging.getLogger(__name__)

GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
OLLAMA_DEFAULT_BASE_URL = "http://host.docker.internal:11434/v1"


@dataclass
class ToolCall:
    """One native function call requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str = ""
    parse_error: str | None = None
    # Provider-specific fields that must be echoed back verbatim — e.g.
    # Gemini's ``extra_content.google.thought_signature`` (omitting it on the
    # follow-up request is a 400 INVALID_ARGUMENT).
    extra_content: dict[str, Any] | None = None


@dataclass
class ToolTurn:
    """One assistant turn from ``chat_tools``: text and/or tool calls."""

    content: str
    tool_calls: list[ToolCall]
    usage: LlmUsage
    finish_reason: str = ""


def assistant_tool_message(turn: ToolTurn) -> dict[str, Any]:
    """The assistant message to append before the tool results, echoing each
    call exactly as the provider issued it (ids, raw arguments, signatures)."""
    calls = []
    for c in turn.tool_calls:
        entry: dict[str, Any] = {
            "id": c.id,
            "type": "function",
            "function": {"name": c.name, "arguments": c.raw_arguments or json.dumps(c.arguments)},
        }
        if c.extra_content:
            entry["extra_content"] = c.extra_content
        calls.append(entry)
    return {"role": "assistant", "content": turn.content or None, "tool_calls": calls}


def _extra_content(tool_call: Any) -> dict[str, Any] | None:
    extra = getattr(tool_call, "model_extra", None) or {}
    value = extra.get("extra_content") if isinstance(extra, dict) else None
    return value if isinstance(value, dict) else None


class ToolChatProvider:
    """OpenAI-compatible Chat Completions client with native tool calling."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "",
        timeout: int = 30,
        name: str = "openai",
        max_retries: int = 2,
    ) -> None:
        from openai import OpenAI  # already a backend dependency

        # max_retries: the SDK retries 408/409/429/5xx with exponential
        # backoff + jitter and honours Retry-After (free tiers 429 often).
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url or None,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.model = model
        self.name = name

    def chat_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        max_tokens: int = 900,
        route: str = "copilot",
        temperature: float = 0.2,
    ) -> ToolTurn:
        """One tool-calling turn (never cached: conversations are stateful)."""
        started = time.perf_counter()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        message = choice.message
        calls: list[ToolCall] = []
        for i, tc in enumerate(getattr(message, "tool_calls", None) or []):
            fn = getattr(tc, "function", None)
            if fn is None:
                continue
            raw_args = fn.arguments or "{}"
            err: str | None = None
            try:
                args = json.loads(strip_fences(raw_args) or "{}")
                if not isinstance(args, dict):
                    args, err = {}, "arguments must be a JSON object"
            except (json.JSONDecodeError, ValueError) as exc:
                args, err = {}, f"invalid JSON arguments: {exc}"
            calls.append(
                ToolCall(
                    id=getattr(tc, "id", None) or f"call_{i}",
                    name=fn.name,
                    arguments=args,
                    raw_arguments=raw_args,
                    parse_error=err,
                    extra_content=_extra_content(tc),
                )
            )
        usage_obj = getattr(response, "usage", None)
        usage = LlmUsage(
            provider=self.name,
            model=self.model,
            route=route,
            prompt_tokens=int(getattr(usage_obj, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage_obj, "completion_tokens", 0) or 0),
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
        _log_usage(usage)
        return ToolTurn(
            content=message.content or "",
            tool_calls=calls,
            usage=usage,
            finish_reason=str(getattr(choice, "finish_reason", "") or ""),
        )


def _gemini_model(model: str) -> str:
    # Config may still carry an OpenAI model name — use a current Gemini model.
    if not model or model.startswith("gpt-"):
        return "gemini-2.5-flash"
    return model


def get_tool_provider() -> ToolChatProvider | None:
    """A tool-calling provider for the configured LLM, or None.

    None means the Copilot uses its deterministic intent router (same tools,
    templated replies) — the chat keeps working without a key.
    """
    settings = get_settings()
    if not settings.llm_enabled or not settings.agent_enabled:
        strict_bail("copilot", "LLM/agent disabled via settings")
        return None
    provider = (settings.llm_provider or "openai").lower().strip()
    timeout = settings.llm_timeout_seconds
    try:
        if provider == "ollama":
            model = settings.llm_model if not settings.llm_model.startswith("gpt-") else "llama3.1"
            return ToolChatProvider(
                api_key="ollama",  # Ollama ignores the key; the SDK requires one
                model=model,
                base_url=settings.openai_base_url or OLLAMA_DEFAULT_BASE_URL,
                timeout=timeout,
                name="ollama",
            )
        use_gemini = provider == "gemini" or (not settings.openai_api_key and settings.gemini_api_key)
        if use_gemini:
            if not settings.gemini_api_key:
                strict_bail("copilot", "GEMINI_API_KEY not configured")
                return None
            return ToolChatProvider(
                api_key=settings.gemini_api_key,
                model=_gemini_model(settings.llm_model),
                base_url=GEMINI_OPENAI_BASE_URL,
                timeout=timeout,
                name="gemini",
            )
        if not settings.openai_api_key:
            strict_bail("copilot", "no LLM API key configured")
            return None
        return ToolChatProvider(
            api_key=settings.openai_api_key,
            model=settings.llm_model,
            base_url=settings.openai_base_url,
            timeout=timeout,
            name="openai",
        )
    except ImportError:  # pragma: no cover - openai is a hard dependency
        strict_bail("copilot", "openai package is not installed")
        return None
