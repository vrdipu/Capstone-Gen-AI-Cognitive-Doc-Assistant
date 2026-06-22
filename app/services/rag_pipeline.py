from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.agents.graph import run_document_assistant


@dataclass(frozen=True)
class RAGResponse:
    answer: str
    sources: list[dict[str, Any]]
    is_validated: bool


class RAGPipeline:
    def answer(self, question: str) -> RAGResponse:
        state = run_document_assistant(question)
        return RAGResponse(
            answer=state.get("answer", ""),
            sources=state.get("retrieved_context", []),
            is_validated=bool(state.get("is_validated")),
        )
