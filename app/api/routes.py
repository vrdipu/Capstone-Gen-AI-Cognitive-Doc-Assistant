from __future__ import annotations

import time
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import requests
from fastapi import APIRouter, File, UploadFile, status

from app.agents.graph import run_document_assistant
from app.api.models import (
    AgentAnswerResponse,
    AgentQuestionRequest,
    AnswerResponse,
    DocumentUploadResponse,
    HealthResponse,
    QuestionRequest,
    SearchResponse,
    SourceDocument,
)
from app.core.config import get_settings
from app.services.ingestion import parse_uploaded_file
from app.services.vector_store import VectorStoreService
from app.utils.exceptions import AppException
from app.utils.validation import FileValidator, QuestionValidator


health_router = APIRouter(tags=["Health"])
document_router = APIRouter(prefix="/documents", tags=["Documents"])
question_router = APIRouter(prefix="/questions", tags=["Questions"])
agent_router = APIRouter(prefix="/agents", tags=["Agents"])
search_router = APIRouter(prefix="/search", tags=["Search"])


def _source_from_result(result: dict, index: int) -> SourceDocument:
    metadata = result.get("metadata") or {}
    return SourceDocument(
        document_id=str(metadata.get("document_id") or metadata.get("source") or result.get("source") or f"chunk-{index}"),
        filename=str(metadata.get("source") or result.get("source") or "unknown"),
        chunk_text=str(result.get("content") or ""),
        relevance_score=float(1.0 - min(float(result.get("distance") or 0.0), 1.0)),
        metadata=metadata,
    )


@health_router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    settings = get_settings()
    ollama_status = "unknown"
    vectorstore_status = "unknown"
    try:
        response = requests.get(f"{settings.ollama_base_url}/api/tags", timeout=3)
        ollama_status = "healthy" if response.ok else "unhealthy"
    except requests.RequestException:
        ollama_status = "unhealthy"
    try:
        VectorStoreService().collection.count()
        vectorstore_status = "healthy"
    except Exception:
        vectorstore_status = "unhealthy"
    status_value = "healthy" if ollama_status == "healthy" and vectorstore_status == "healthy" else "degraded"
    return HealthResponse(
        status=status_value,
        app_name=settings.app_name,
        version="1.0.0",
        environment=settings.app_env,
        ollama_status=ollama_status,
        vectorstore_status=vectorstore_status,
    )


@document_router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
    valid, message = FileValidator.validate_filename(file.filename or "")
    if not valid:
        raise AppException(message, status.HTTP_400_BAD_REQUEST)
    data = await file.read()
    valid, message = FileValidator.validate_size(len(data))
    if not valid:
        raise AppException(message, status.HTTP_400_BAD_REQUEST)

    upload_path = get_settings().upload_path / Path(file.filename or f"upload-{uuid4()}").name
    upload_path.write_bytes(data)
    uploaded = BytesIO(data)
    uploaded.name = file.filename or upload_path.name
    document = parse_uploaded_file(uploaded)
    chunks_created = VectorStoreService().add_document(document)
    return DocumentUploadResponse(
        success=True,
        message="Document uploaded and indexed",
        filename=document.source_name,
        document_id=document.metadata["source"],
        chunks_created=chunks_created,
    )


@question_router.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest) -> AnswerResponse:
    valid, message = QuestionValidator.validate(request.question)
    if not valid:
        raise AppException(message, status.HTTP_400_BAD_REQUEST)
    started = time.perf_counter()
    vector_store = VectorStoreService()
    results = vector_store.query(request.question, k=request.top_k)
    state = run_document_assistant(request.question)
    sources = [_source_from_result(result, index) for index, result in enumerate(state.get("retrieved_context") or results)]
    return AnswerResponse(
        answer=state.get("answer") or "No grounded answer was produced.",
        sources=sources,
        confidence=0.85 if state.get("is_validated") else 0.5,
        processing_time_seconds=round(time.perf_counter() - started, 3),
    )


@agent_router.post("/ask", response_model=AgentAnswerResponse)
async def ask_agent(request: AgentQuestionRequest) -> AgentAnswerResponse:
    valid, message = QuestionValidator.validate(request.question)
    if not valid:
        raise AppException(message, status.HTTP_400_BAD_REQUEST)
    started = time.perf_counter()
    state = run_document_assistant(request.question)
    sources = [_source_from_result(result, index) for index, result in enumerate(state.get("retrieved_context", []))]
    steps = [
        {"agent": "planner", "success": bool(state.get("plan"))},
        {"agent": "retriever", "success": bool(state.get("retrieved_context"))},
        {"agent": "reasoner", "success": bool(state.get("answer"))},
        {"agent": "validator", "success": bool(state.get("is_validated"))},
    ]
    return AgentAnswerResponse(
        answer=state.get("answer") or "No grounded answer was produced.",
        sources=sources,
        confidence=0.9 if state.get("is_validated") else 0.45,
        processing_time_seconds=round(time.perf_counter() - started, 3),
        is_validated=bool(state.get("is_validated")),
        validation_notes=state.get("validation_notes"),
        agent_steps=steps,
    )


@search_router.post("/", response_model=SearchResponse)
async def search(request: QuestionRequest) -> SearchResponse:
    valid, message = QuestionValidator.validate(request.question)
    if not valid:
        raise AppException(message, status.HTTP_400_BAD_REQUEST)
    results = VectorStoreService().query(request.question, k=request.top_k)
    return SearchResponse(results=[_source_from_result(result, index) for index, result in enumerate(results)])
