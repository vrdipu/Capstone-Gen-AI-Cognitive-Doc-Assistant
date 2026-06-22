from __future__ import annotations

from typing import Any

from app.agents.base_agent import AgentResult
from app.agents.graph import planner_node, reasoning_node, responder_node, retriever_node, validator_node


class PlannerAgent:
    name = "planner"

    def process(self, payload: dict[str, Any]) -> AgentResult:
        state = planner_node({"question": str(payload.get("question", ""))})
        return AgentResult(True, state)


class RetrieverAgent:
    name = "retriever"

    def process(self, payload: dict[str, Any]) -> AgentResult:
        state = retriever_node({"question": str(payload.get("question", "")), "search_query": str(payload.get("query", ""))})
        return AgentResult(True, state.get("retrieved_context", []))


class ReasonerAgent:
    name = "reasoner"

    def process(self, payload: dict[str, Any]) -> AgentResult:
        state = reasoning_node(
            {
                "question": str(payload.get("question", "")),
                "retrieved_context": payload.get("retrieved_context", []),
            }
        )
        return AgentResult(True, state.get("reasoning_output", ""))


class ResponderAgent:
    name = "responder"

    def process(self, payload: dict[str, Any]) -> AgentResult:
        state = responder_node(
            {
                "reasoning_output": payload.get("answer") or payload.get("analysis") or "",
                "retrieved_context": payload.get("retrieved_context", []),
            }
        )
        return AgentResult(True, state.get("answer", ""))


class ValidatorAgent:
    name = "validator"

    def process(self, payload: dict[str, Any]) -> AgentResult:
        state: dict[str, Any] = {
            "question": payload.get("question", ""),
            "answer": payload.get("answer") or payload.get("response", ""),
            "retrieved_context": payload.get("retrieved_context", []),
        }
        validated = validator_node(state)
        return AgentResult(True, validated)
