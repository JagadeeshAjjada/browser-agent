"""
Shared LLM factory
==================
Single source of truth for creating a browser-use compatible LLM.
Both personal_assistant.py (CLI) and web_ui.py import get_llm() from here.

Supported providers (set LLM_PROVIDER in .env):
  gemini   — Google Gemini via browser-use's native ChatGoogle wrapper
  groq     — Groq (OpenAI-compatible endpoint)
  ollama   — Local Ollama (OpenAI-compatible endpoint)
"""

import os

from dotenv import load_dotenv

load_dotenv()


# ── Groq / Ollama helper ─────────────────────────────────────────────────────
# browser-use reads llm.provider and llm.model; ChatOpenAI uses model_name
# and has no provider field — subclass fixes both.

def _make_openai_compatible(model: str, base_url: str, api_key: str):
    from langchain_openai import ChatOpenAI as _ChatOpenAI
    from pydantic import ConfigDict

    class _BrowserChatOpenAI(_ChatOpenAI):
        model_config = ConfigDict(
            arbitrary_types_allowed=True,
            extra="allow",
            populate_by_name=True,
        )
        provider: str = "openai"  # required by browser-use

        def model_post_init(self, __context):
            super().model_post_init(__context)
            # browser-use reads .model; ChatOpenAI stores it as .model_name
            self.model = self.model_name

    return _BrowserChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=0.0,
    )


# ── Public API ────────────────────────────────────────────────────────────────

def get_llm():
    """
    Returns a browser-use-compatible LLM instance.
    Reads LLM_PROVIDER, *_API_KEY, and *_MODEL from .env.
    Raises ValueError on bad config so callers can catch and report cleanly.
    """
    llm_provider = os.getenv("LLM_PROVIDER", "gemini").lower()

    # ── Gemini (recommended) ─────────────────────────────────────────────────
    if llm_provider == "gemini":
        from browser_use.llm.google.chat import ChatGoogle

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set in .env. "
                "Get a free key at https://aistudio.google.com/app/apikey"
            )
        return ChatGoogle(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            api_key=api_key,
            temperature=0.0,
        )

    # ── Groq ─────────────────────────────────────────────────────────────────
    elif llm_provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set in .env. "
                "Get a free key at https://console.groq.com"
            )
        return _make_openai_compatible(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key,
        )

    # ── Ollama (local) ────────────────────────────────────────────────────────
    elif llm_provider == "ollama":
        return _make_openai_compatible(
            model=os.getenv("OLLAMA_MODEL", "llama3.2"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            api_key="ollama",
        )

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER: '{llm_provider}'. "
            "Valid options: gemini, groq, ollama"
        )
