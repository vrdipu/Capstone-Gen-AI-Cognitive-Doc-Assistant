from __future__ import annotations

from litellm import completion

from app.core.config import get_settings


class LLMService:
    """LiteLLM facade configured for local Ollama by default."""

    def __init__(self) -> None:
        settings = get_settings()
        self.model = f"ollama/{settings.ollama_chat_model}"
        self.api_base = settings.ollama_base_url

    def generate(self, prompt: str, temperature: float = 0.1) -> str:
        response = completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            api_base=self.api_base,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""
