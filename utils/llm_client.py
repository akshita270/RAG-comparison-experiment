"""
utils/llm_client.py — Unified LLM interface for OpenAI and Anthropic.

All 4 architectures and the evaluation pipeline use this single class.
This decouples the rest of the codebase from provider-specific SDK calls
and centralises retry/backoff logic in one place.
"""
import time
import logging
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Provider-agnostic wrapper around OpenAI and Anthropic SDKs.

    Usage:
        llm = LLMClient()                          # uses config defaults
        llm = LLMClient(model="gpt-4o-mini")       # override model
        answer = llm.complete("What is RAG?")
    """

    def __init__(
        self,
        model: Optional[str] = None,
        provider: Optional[str] = None,
    ):
        self.provider = provider or config.LLM_PROVIDER
        self.model    = model    or config.LLM_MODEL
        self._init_client()

    def _init_client(self):
        """Instantiate the underlying SDK client once at construction time."""
        if self.provider == "openai":
            from openai import OpenAI
            self._client = OpenAI(api_key=config.OPENAI_API_KEY)
        elif self.provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        else:
            raise ValueError(
                f"Unknown LLM_PROVIDER '{self.provider}'. "
                "Set LLM_PROVIDER=openai or LLM_PROVIDER=anthropic."
            )

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        retries: int = 3,
        retry_delay: float = 2.0,
    ) -> str:
        """
        Send a prompt and return the response string.

        Retries on any exception with exponential backoff — this handles
        transient rate-limit errors without crashing a long experiment run.
        """
        for attempt in range(retries):
            try:
                if self.provider == "openai":
                    return self._openai_complete(prompt, system, temperature, max_tokens)
                else:
                    return self._anthropic_complete(prompt, system, temperature, max_tokens)
            except Exception as exc:
                if attempt < retries - 1:
                    wait = retry_delay * (2 ** attempt)
                    logger.warning(
                        f"LLM call failed (attempt {attempt+1}/{retries}): {exc}. "
                        f"Retrying in {wait:.1f}s …"
                    )
                    time.sleep(wait)
                else:
                    logger.error(f"LLM call failed after {retries} attempts: {exc}")
                    raise

    def _openai_complete(self, prompt, system, temperature, max_tokens) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content

    def _anthropic_complete(self, prompt, system, temperature, max_tokens) -> str:
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        resp = self._client.messages.create(**kwargs)
        return resp.content[0].text
