from __future__ import annotations

from typing import Any

from app.agents.graph import run_document_assistant


class AgentOrchestrator:
    def ask(self, question: str) -> dict[str, Any]:
        state = run_document_assistant(question)
        return {
            "answer": state.get("answer", ""),
            "sources": state.get("retrieved_context", []),
            "validation": {
                "is_validated": bool(state.get("is_validated")),
                "notes": state.get("validation_notes"),
            },
            "agent_steps": state.get("agent_steps", []),
        }
