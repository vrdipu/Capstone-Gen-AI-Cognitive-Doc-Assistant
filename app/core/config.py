from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-backed application settings."""

    app_name: str = Field(default="GenAI Document Assistant", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    debug: bool = Field(default=True, alias="DEBUG")
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT", ge=1, le=65535)
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        alias="OLLAMA_BASE_URL",
        description="Base URL for the local Ollama HTTP API.",
    )
    chroma_db_dir: Path = Field(
        default=Path("./data/vectorstore"),
        validation_alias=AliasChoices("CHROMA_DB_DIR", "CHROMA_PERSIST_DIR"),
        description="Directory used by persistent ChromaDB storage.",
    )
    chroma_collection_name: str = Field(default="documents", alias="CHROMA_COLLECTION_NAME")
    upload_dir: Path = Field(default=Path("./data/uploads"), alias="UPLOAD_DIR")
    max_file_size_mb: int = Field(default=10, alias="MAX_FILE_SIZE_MB", ge=1, le=200)
    allowed_extensions: str = Field(default="pdf,txt,csv,xlsx,xls,docx,json,yaml,yml", alias="ALLOWED_EXTENSIONS")
    chunk_size: int = Field(default=1000, alias="CHUNK_SIZE", ge=100, le=4000)
    chunk_overlap: int = Field(default=150, alias="CHUNK_OVERLAP", ge=0, le=1000)
    top_k_results: int = Field(default=3, alias="TOP_K_RESULTS", ge=1, le=20)
    llm_provider: str = Field(default="gemini", alias="LLM_PROVIDER")
    gemini_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        description="Google Gemini API key used when LLM_PROVIDER=gemini.",
    )
    gemini_model: str = Field(default="gemini-2.5-flash", alias="GEMINI_MODEL")
    gemini_api_base_url: str = Field(
        default="https://generativelanguage.googleapis.com",
        alias="GEMINI_API_BASE_URL",
    )
    gemini_api_version: str = Field(default="v1", alias="GEMINI_API_VERSION")
    gemini_generation_max_retries: int = Field(default=3, alias="GEMINI_GENERATION_MAX_RETRIES", ge=1, le=10)
    gemini_embedding_api_version: str = Field(default="v1beta", alias="GEMINI_EMBEDDING_API_VERSION")
    embedding_provider: str = Field(default="gemini", alias="EMBEDDING_PROVIDER")
    gemini_embedding_model: str = Field(default="gemini-embedding-001", alias="GEMINI_EMBEDDING_MODEL")
    gemini_embedding_batch_size: int = Field(default=8, alias="GEMINI_EMBEDDING_BATCH_SIZE", ge=1, le=100)
    gemini_embedding_batch_delay_seconds: float = Field(
        default=2.0,
        alias="GEMINI_EMBEDDING_BATCH_DELAY_SECONDS",
        ge=0.0,
        le=60.0,
    )
    gemini_embedding_max_retries: int = Field(default=6, alias="GEMINI_EMBEDDING_MAX_RETRIES", ge=1, le=10)
    ollama_chat_model: str = Field(
        default="llama3.2:3b",
        validation_alias=AliasChoices("OLLAMA_CHAT_MODEL", "OLLAMA_MODEL"),
    )
    ollama_embedding_model: str = Field(
        default="nomic-embed-text",
        validation_alias=AliasChoices("OLLAMA_EMBEDDING_MODEL", "EMBEDDING_MODEL"),
    )
    llm_timeout_seconds: int = Field(default=180, alias="LLM_TIMEOUT_SECONDS", ge=10, le=600)
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

    @field_validator("llm_provider")
    @classmethod
    def normalize_llm_provider(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"gemini", "ollama"}:
            raise ValueError("LLM_PROVIDER must be 'gemini' or 'ollama'")
        return normalized

    @field_validator("embedding_provider")
    @classmethod
    def normalize_embedding_provider(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"gemini", "ollama"}:
            raise ValueError("EMBEDDING_PROVIDER must be 'gemini' or 'ollama'")
        return normalized

    @field_validator("gemini_api_base_url")
    @classmethod
    def normalize_gemini_url(cls, value: str) -> str:
        normalized = value.strip().rstrip("/")
        if not normalized:
            raise ValueError("GEMINI_API_BASE_URL cannot be empty")
        return normalized

    @field_validator("gemini_api_version", "gemini_embedding_api_version")
    @classmethod
    def normalize_gemini_api_version(cls, value: str) -> str:
        normalized = value.strip().strip("/")
        if normalized not in {"v1", "v1beta"}:
            raise ValueError("Gemini API version must be 'v1' or 'v1beta'")
        return normalized

    @field_validator("chroma_db_dir")
    @classmethod
    def expand_chroma_dir(cls, value: Path) -> Path:
        return value.expanduser()

    @field_validator("upload_dir")
    @classmethod
    def expand_upload_dir(cls, value: Path) -> Path:
        return value.expanduser()

    @property
    def allowed_extensions_list(self) -> List[str]:
        return [extension.strip().lower().lstrip(".") for extension in self.allowed_extensions.split(",")]

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    @property
    def upload_path(self) -> Path:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        return self.upload_dir

    @property
    def vectorstore_path(self) -> Path:
        self.chroma_db_dir.mkdir(parents=True, exist_ok=True)
        return self.chroma_db_dir


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
