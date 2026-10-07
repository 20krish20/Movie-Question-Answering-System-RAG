"""
LLM backend for the factual pipeline.

Uses any OpenAI-compatible chat API. Defaults to Groq's free tier
(no credit card) with openai/gpt-oss-120b. Groq retires models regularly:
list current ones with  curl https://api.groq.com/openai/v1/models -H "Authorization: Bearer $LLM_API_KEY"

Environment variables:
    LLM_API_KEY    API key (falls back to GROQ_API_KEY / OPENAI_API_KEY)
    LLM_BASE_URL   default: https://api.groq.com/openai/v1
    LLM_MODEL      default: openai/gpt-oss-120b
    LLM_REASONING_EFFORT  low|medium|high for reasoning models (default: low for gpt-oss, unset otherwise)

Other free / cheap providers, just change the env vars:
    Gemini   LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
             LLM_MODEL=gemini-2.5-flash
    OpenAI   LLM_BASE_URL=https://api.openai.com/v1   LLM_MODEL=gpt-4o-mini
"""
import os
from functools import lru_cache

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"


class LLMNotConfigured(RuntimeError):
    pass


def _api_key():
    return os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")


def is_configured() -> bool:
    return bool(_api_key())


@lru_cache(maxsize=1)
def _client():
    from openai import OpenAI

    key = _api_key()
    if not key:
        raise LLMNotConfigured(
            "No LLM API key set. Set LLM_API_KEY (e.g. a free Groq key from https://console.groq.com/keys)."
        )
    return OpenAI(api_key=key, base_url=os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL), timeout=30, max_retries=2)


def call_llm(prompt: str) -> str:
    """Return the model's raw text. The factual pipeline expects one ```python``` block in it."""
    model = os.getenv("LLM_MODEL", DEFAULT_MODEL)
    extra = {}
    effort = os.getenv("LLM_REASONING_EFFORT") or ("low" if "gpt-oss" in model else None)
    if effort:
        extra["reasoning_effort"] = effort
    response = _client().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=2000,  # reasoning models spend tokens thinking before the answer
        **extra,
    )
    return response.choices[0].message.content or ""
