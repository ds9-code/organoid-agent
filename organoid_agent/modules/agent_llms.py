"""
Pluggable LLM client.

Wraps the OpenAI Python SDK against any OpenAI-compatible endpoint. The default
backbone is Hermes-4 via Nous Portal, but Azure OpenAI, OpenAI proper, Dartmouth,
LiteLLM, or vLLM proxies all work — just point ``OPENAI_BASE_URL`` at the right
host.

Mirrors the structure of ``medea/modules/agent_llms.py`` so anything you'd write
against Medea's ``AgentLLM`` will read familiarly here.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

DEFAULT_TEMPERATURE = 0.4
# Defaults: local Ollama serving Hermes-4-14B (the project's recommended
# free/local backbone). Override via OPENAI_BASE_URL + AGENT_MODEL for any
# other OpenAI-compatible endpoint (OpenAI, Nous Portal, OpenRouter, Azure, ...).
DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "hf.co/bartowski/NousResearch_Hermes-4-14B-GGUF:Q4_K_M"


@dataclass
class LLMConfig:
    """LLM call hyperparameters."""
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    reasoning_effort: Optional[str] = None  # "low" | "medium" | "high" — for reasoning models


class AgentLLM:
    """Thin OpenAI-compatible chat client.

    Resolves API key / base URL / model from env vars at construction. The same
    instance can be reused by multiple modules.

    Env vars (read at __init__ time, can be overridden by constructor kwargs):
        OPENAI_API_KEY       — required
        OPENAI_BASE_URL      — default: https://api.portal.nousresearch.com/v1
        AGENT_MODEL          — default: Hermes-4-405B
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        llm_name: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        verbose: bool = False,
    ):
        self.config = config or LLMConfig()
        self.model = llm_name or os.environ.get("AGENT_MODEL", DEFAULT_MODEL)
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL
        # Local backends (Ollama, vLLM, LM Studio) don't require an API key, but
        # the OpenAI SDK still wants a non-empty string. Default to a placeholder
        # so the smallest-config "just install Ollama" path works out of the box.
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or "ollama-local"
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("pip install openai") from e
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        if verbose:
            print(f"[AgentLLM] model={self.model!r} via {self.base_url}")

    def chat(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        **kwargs,
    ) -> Any:
        """Single chat-completion call. Returns the raw OpenAI response object."""
        call_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.config.temperature,
        }
        if self.config.max_tokens is not None:
            call_kwargs["max_tokens"] = self.config.max_tokens
        if self.config.top_p is not None:
            call_kwargs["top_p"] = self.config.top_p
        if tools:
            call_kwargs["tools"] = tools
        call_kwargs.update(kwargs)
        return self.client.chat.completions.create(**call_kwargs)

    def __repr__(self) -> str:
        return f"AgentLLM(model={self.model!r}, base_url={self.base_url!r})"
