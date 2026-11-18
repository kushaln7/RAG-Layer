"""LLM Model Factory and Client Management.

Provides centralized instantiation, configuration validation, and caching
for LLM providers including DeepSeek, OpenAI, and offline test mocks.
"""

import logging
import os
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_openai import ChatOpenAI

from config.settings import settings

logger = logging.getLogger(__name__)


class LLMConfigurationError(Exception):
    """Raised when LLM configuration or credentials are missing or invalid."""


class LLMGenerationError(Exception):
    """Raised when an error occurs during LLM response generation."""


# Module-level model instance cache for reuse
_LLM_CACHE: dict[str, BaseChatModel] = {}


def get_llm(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    use_cache: bool = True,
) -> BaseChatModel:
    """Instantiate and return a configured LangChain ChatModel instance.

    Args:
        provider: Provider name (e.g. 'DeepSeek', 'OpenAI', 'fake'). Defaults to settings.LLM_PROVIDER.
        model: Model name/identifier (e.g. 'deepseek-flash', 'gpt-4o'). Defaults to settings.LLM_MODEL.
        api_key: Secret API key. Defaults to settings.LLM_API_KEY.
        base_url: Base API endpoint URL. Defaults to settings.LLM_BASE_URL.
        temperature: Sampling temperature (0.0 to 1.0). Defaults to settings.LLM_TEMPERATURE.
        max_tokens: Maximum tokens in completion. Defaults to settings.LLM_MAX_TOKENS.
        use_cache: Whether to return a cached instance with matching configuration.

    Returns:
        Configured BaseChatModel instance.

    Raises:
        LLMConfigurationError: If provider is unrecognized or required credentials are missing.
    """
    resolved_provider = (provider or getattr(settings, "LLM_PROVIDER", "DeepSeek")).strip().lower()
    resolved_model = model or getattr(settings, "LLM_MODEL", "deepseek-flash")
    if api_key is not None:
        resolved_api_key = api_key
    else:
        resolved_api_key = getattr(settings, "LLM_API_KEY", "") or os.getenv("DEEPSEEK_API_KEY", "")
    resolved_base_url = base_url or getattr(settings, "LLM_BASE_URL", "https://api.deepseek.com/")
    resolved_temperature = temperature if temperature is not None else float(getattr(settings, "LLM_TEMPERATURE", 0.0))
    resolved_max_tokens = max_tokens if max_tokens is not None else int(getattr(settings, "LLM_MAX_TOKENS", 1024))

    cache_key = f"{resolved_provider}:{resolved_model}:{resolved_base_url}:{resolved_temperature}:{resolved_max_tokens}"

    if use_cache and cache_key in _LLM_CACHE:
        return _LLM_CACHE[cache_key]

    logger.info("Initializing LLM client: provider=%s, model=%s", resolved_provider, resolved_model)

    if resolved_provider in ("fake", "mock", "test"):
        llm = FakeListChatModel(
            responses=[
                "I couldn't find this information in the available knowledge base.",
                "Based on the provided context, the required security policy is enforced.",
                "Canonix enterprise knowledge answer generated successfully.",
            ]
        )
        if use_cache:
            _LLM_CACHE[cache_key] = llm
        return llm

    if resolved_provider in ("deepseek", "deepseek-ai"):
        if not resolved_api_key or not resolved_api_key.strip():
            raise LLMConfigurationError(
                "DeepSeek API key is missing. Please set LLM_API_KEY or DEEPSEEK_API_KEY in your .env file."
            )

        # Normalize base_url
        clean_base_url = resolved_base_url.rstrip("/") + "/"
        if not clean_base_url.endswith("/v1/") and "deepseek.com" in clean_base_url:
            # LangChain ChatOpenAI works with base_url="https://api.deepseek.com/" or ".../v1/"
            clean_base_url = clean_base_url

        llm = ChatOpenAI(
            api_key=resolved_api_key,
            base_url=clean_base_url,
            model=resolved_model,
            temperature=resolved_temperature,
            max_tokens=resolved_max_tokens,
            timeout=30.0,
            max_retries=2,
        )

        if use_cache:
            _LLM_CACHE[cache_key] = llm
        return llm

    if resolved_provider in ("openai",):
        if not resolved_api_key or not resolved_api_key.strip():
            raise LLMConfigurationError("OpenAI API key is missing. Please set OPENAI_API_KEY or LLM_API_KEY.")

        llm = ChatOpenAI(
            api_key=resolved_api_key,
            model=resolved_model,
            temperature=resolved_temperature,
            max_tokens=resolved_max_tokens,
            timeout=30.0,
            max_retries=2,
        )
        if use_cache:
            _LLM_CACHE[cache_key] = llm
        return llm

    raise LLMConfigurationError(
        f"Unsupported LLM provider '{resolved_provider}'. Supported providers: 'DeepSeek', 'OpenAI', 'fake'."
    )


def clear_llm_cache() -> None:
    """Clear cached LLM instances."""
    _LLM_CACHE.clear()
