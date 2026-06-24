from __future__ import annotations

import logging
from typing import Any

import requests

from app.core.config import get_settings


LOGGER = logging.getLogger(__name__)


class LLMServiceError(RuntimeError):
    """Raised when chat model generation fails."""


class LLMService:
    """Provider-aware chat model facade.

    Gemini is the default chat LLM. Ollama remains available for fully local
    chat generation by setting LLM_PROVIDER=ollama.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self.provider = settings.llm_provider
        self.timeout = settings.llm_timeout_seconds
        self.ollama_base_url = settings.ollama_base_url
        self.ollama_model = settings.ollama_chat_model
        self.gemini_api_key = settings.gemini_api_key.strip()
        self.gemini_model = settings.gemini_model.strip()
        self.gemini_api_base_url = settings.gemini_api_base_url

    @property
    def model_name(self) -> str:
        return self.gemini_model if self.provider == "gemini" else self.ollama_model

    def status(self) -> str:
        if self.provider == "gemini":
            return "configured" if self._has_gemini_api_key() else "missing_api_key"
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=3)
            return "healthy" if response.ok else "unhealthy"
        except requests.RequestException:
            return "unhealthy"

    def generate(self, prompt: str, temperature: float = 0.1) -> str:
        if self.provider == "gemini":
            return self._generate_gemini(prompt, temperature=temperature)
        return self._generate_ollama(prompt, temperature=temperature)

    def _generate_gemini(self, prompt: str, *, temperature: float) -> str:
        if not self._has_gemini_api_key():
            raise LLMServiceError("GEMINI_API_KEY or GOOGLE_API_KEY is required when LLM_PROVIDER=gemini")

        url = f"{self.gemini_api_base_url}/v1beta/models/{self.gemini_model}:generateContent"
        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": 2048,
            },
        }
        try:
            response = requests.post(
                url,
                headers={"x-goog-api-key": self.gemini_api_key, "Content-Type": "application/json"},
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            text = self._extract_gemini_text(response.json())
            if not text.strip():
                raise LLMServiceError("Gemini returned an empty response")
            return text.strip()
        except requests.RequestException as exc:
            LOGGER.exception("Gemini generation failed")
            raise LLMServiceError(
                f"Unable to generate with Gemini model '{self.gemini_model}'. "
                "Check GEMINI_API_KEY and network access."
            ) from exc

    def _generate_ollama(self, prompt: str, *, temperature: float) -> str:
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        try:
            response = requests.post(f"{self.ollama_base_url}/api/generate", json=payload, timeout=self.timeout)
            response.raise_for_status()
            text = response.json().get("response", "")
            if not text.strip():
                raise LLMServiceError("Ollama returned an empty response")
            return text.strip()
        except requests.RequestException as exc:
            raise LLMServiceError(
                f"Unable to reach Ollama chat model at {self.ollama_base_url}. "
                f"Confirm 'ollama serve' is running and '{self.ollama_model}' is pulled."
            ) from exc

    @staticmethod
    def _extract_gemini_text(payload: dict[str, Any]) -> str:
        candidates = payload.get("candidates") or []
        if not candidates:
            prompt_feedback = payload.get("promptFeedback") or {}
            raise LLMServiceError(f"Gemini returned no candidates: {prompt_feedback}")
        parts = ((candidates[0].get("content") or {}).get("parts") or [])
        text_parts = [str(part.get("text", "")) for part in parts if part.get("text")]
        return "\n".join(text_parts)

    def _has_gemini_api_key(self) -> bool:
        key = self.gemini_api_key.strip()
        return bool(key and not key.lower().startswith(("replace-", "your-")))
