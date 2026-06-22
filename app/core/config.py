from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-backed application settings."""

    ollama_base_url: str = Field(
        default="http://localhost:11434",
        alias="OLLAMA_BASE_URL",
        description="Base URL for the local Ollama HTTP API.",
    )
    chroma_db_dir: Path = Field(
        default=Path("./chroma_db"),
        alias="CHROMA_DB_DIR",
        description="Directory used by persistent ChromaDB storage.",
    )
    ollama_chat_model: str = Field(default="llama3.2:3b", alias="OLLAMA_CHAT_MODEL")
    ollama_embedding_model: str = Field(default="nomic-embed-text", alias="OLLAMA_EMBEDDING_MODEL")
    max_graph_retries: int = Field(default=3, alias="MAX_GRAPH_RETRIES", ge=1, le=10)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    @field_validator("ollama_base_url")
    @classmethod
    def normalize_ollama_url(cls, value: str) -> str:
        normalized = value.strip().rstrip("/")
        if not normalized:
            raise ValueError("OLLAMA_BASE_URL cannot be empty")
        return normalized

    @field_validator("chroma_db_dir")
    @classmethod
    def expand_chroma_dir(cls, value: Path) -> Path:
        return value.expanduser()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
