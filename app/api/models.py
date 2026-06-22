from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    success: bool
    message: str
    filename: str
    document_id: str
    chunks_created: int = 0


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    document_ids: list[str] | None = None
    top_k: int = Field(default=3, ge=1, le=20)


class AgentQuestionRequest(QuestionRequest):
    enable_validation: bool = True


class SourceDocument(BaseModel):
    document_id: str
    filename: str
    chunk_text: str
    relevance_score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class AnswerResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]
    confidence: float | None = None
    processing_time_seconds: float


class AgentAnswerResponse(AnswerResponse):
    is_validated: bool
    validation_notes: str | None = None
    agent_steps: list[dict[str, Any]] = Field(default_factory=list)


class SearchResponse(BaseModel):
    results: list[SourceDocument]


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    environment: str
    ollama_status: str = "unknown"
    vectorstore_status: str = "unknown"
