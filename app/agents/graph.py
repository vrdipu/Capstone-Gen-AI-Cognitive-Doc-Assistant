from __future__ import annotations

import json
import logging
import re
from typing import Any, TypedDict

import requests
from langgraph.graph import END, StateGraph

from app.core.config import get_settings
from app.services.vector_store import VectorStoreService


LOGGER = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    question: str
    plan: str
    search_query: str
    retrieved_context: list[dict[str, Any]]
    reasoning_output: str
    answer: str
    validation_notes: str
    is_validated: bool
    retry_count: int
    agent_steps: list[dict[str, Any]]
    error: str


class OllamaChatClient:
    def __init__(self) -> None:
        settings = get_settings()
        self.base_url = settings.ollama_base_url
        self.model = settings.ollama_chat_model
        self.timeout = 120

    def generate(self, prompt: str, *, temperature: float = 0.1) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        try:
            response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=self.timeout)
            response.raise_for_status()
            text = response.json().get("response", "")
            if not text.strip():
                raise RuntimeError("Ollama returned an empty response")
            return text.strip()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Unable to reach Ollama chat model at {self.base_url}. "
                f"Confirm 'ollama serve' is running and '{self.model}' is pulled."
            ) from exc


def _json_from_model(text: str) -> dict[str, Any]:
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        parsed = json.loads(text[start:end])
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        LOGGER.warning("Model response was not valid JSON: %s", text)
        return {}


def _with_step(state: AgentState, agent: str, success: bool, detail: str = "") -> AgentState:
    steps = list(state.get("agent_steps", []))
    step: dict[str, Any] = {"agent": agent, "success": success}
    if detail:
        step["detail"] = detail
    return {**state, "agent_steps": steps + [step]}


def planner_node(state: AgentState) -> AgentState:
    client = OllamaChatClient()
    question = state.get("question", "").strip()
    prompt = f"""
You are the planner for a document-grounded assistant.
Return strict JSON with keys "plan" and "search_query".
The search_query must be concise and optimized for vector retrieval.

User question:
{question}
"""
    raw = client.generate(prompt)
    parsed = _json_from_model(raw)
    search_query = str(parsed.get("search_query") or question).strip()
    plan = str(parsed.get("plan") or "Retrieve relevant source chunks, answer using only that context, then validate claims.")
    updated: AgentState = {**state, "plan": plan, "search_query": search_query}
    return _with_step(updated, "planner", bool(search_query), plan)


def retriever_node(state: AgentState) -> AgentState:
    query = state.get("search_query") or state.get("question", "")
    vector_store = VectorStoreService()
    context = vector_store.query(str(query), k=3)
    updated: AgentState = {**state, "retrieved_context": context}
    return _with_step(updated, "retriever", bool(context), f"retrieved {len(context)} context chunks")


def reasoning_node(state: AgentState) -> AgentState:
    context = "\n\n".join(
        f"Source: {item.get('source', 'unknown')}\nContent: {item.get('content', '')}"
        for item in state.get("retrieved_context", [])
    )
    extracted_answer = _extract_direct_answer(state.get("question", ""), context)
    if extracted_answer:
        updated: AgentState = {**state, "reasoning_output": extracted_answer}
        return _with_step(updated, "reasoner", True, "direct evidence extracted")

    client = OllamaChatClient()
    prompt = f"""
You are a precise document assistant.
Answer the question using only the supplied context.
If the context is insufficient, say exactly what is missing.
Include concise source references by filename when available.
If the context directly states the requested value, answer that value directly without hedging.

Question:
{state.get("question", "")}

Context:
{context or "No context retrieved."}
"""
    reasoning_output = client.generate(prompt)
    updated = {**state, "reasoning_output": reasoning_output}
    return _with_step(updated, "reasoner", bool(reasoning_output), "LLM reasoning completed")


def responder_node(state: AgentState) -> AgentState:
    reasoning_output = state.get("reasoning_output") or state.get("answer") or ""
    context = "\n\n".join(
        f"Source: {item.get('source', 'unknown')}\nContent: {item.get('content', '')}"
        for item in state.get("retrieved_context", [])
    )
    if not reasoning_output.strip():
        updated: AgentState = {**state, "answer": "No grounded answer was produced from the retrieved context."}
        return _with_step(updated, "responder", False, "empty reasoning output")

    if _answer_has_grounded_claims(reasoning_output, context) or "source:" in reasoning_output.lower():
        updated = {**state, "answer": reasoning_output.strip()}
        return _with_step(updated, "responder", True, "formatted grounded response")

    sources = sorted({str(item.get("source", "unknown")) for item in state.get("retrieved_context", []) if item.get("source")})
    source_note = f"\n\nSources: {', '.join(sources)}" if sources else ""
    updated = {**state, "answer": f"{reasoning_output.strip()}{source_note}"}
    return _with_step(updated, "responder", True, "source references attached")


def _extract_direct_answer(question: str, context: str) -> str | None:
    if "answer" not in question.lower() or not context.strip():
        return None
    match = re.search(r"\banswer\s+is\s+([A-Za-z0-9_.:-]+)", context, flags=re.IGNORECASE)
    if not match:
        return None
    source_match = re.search(r"Source:\s*(.+)", context)
    source = source_match.group(1).strip() if source_match else "retrieved context"
    return f"The answer is {match.group(1).rstrip('.')} (source: {source})."


def validator_node(state: AgentState) -> AgentState:
    context = "\n\n".join(
        f"Source: {item.get('source', 'unknown')}\nContent: {item.get('content', '')}"
        for item in state.get("retrieved_context", [])
    )
    if _answer_has_grounded_claims(state.get("answer", ""), context):
        return {
            **_with_step(state, "validator", True, "deterministic evidence check passed"),
            "is_validated": True,
            "validation_notes": "Deterministic evidence check passed against retrieved context.",
        }

    client = OllamaChatClient()
    prompt = f"""
Audit whether the answer is fully supported by the context.
Return strict JSON with keys "isValidated" and "notes".

Question:
{state.get("question", "")}

Answer:
{state.get("answer", "")}

Context:
{context or "No context retrieved."}
"""
    raw = client.generate(prompt, temperature=0)
    parsed = _json_from_model(raw)
    is_validated = _coerce_bool(parsed.get("isValidated", False))
    notes = str(parsed.get("notes") or raw)
    updated = {**state, "is_validated": is_validated, "validation_notes": notes}
    return _with_step(updated, "validator", is_validated, notes[:160])


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "validated", "supported"}
    return bool(value)


def _answer_has_grounded_claims(answer: str, context: str) -> bool:
    if not answer.strip() or not context.strip():
        return False
    lowered_answer = answer.lower()
    negative_phrases = (
        "insufficient",
        "missing",
        "could not",
        "cannot",
        "impossible",
        "not explicitly",
        "not enough",
    )
    if any(phrase in lowered_answer for phrase in negative_phrases):
        return False

    context_lower = context.lower()
    exact_markers = re.findall(r"\b[a-zA-Z]+-\d+\b|\b\d+(?:\.\d+)?\b", answer)
    if exact_markers and all(marker.lower() in context_lower for marker in exact_markers):
        return True

    meaningful_tokens = {
        token
        for token in re.findall(r"\b[a-zA-Z][a-zA-Z0-9_-]{3,}\b", lowered_answer)
        if token not in {"source", "answer", "context", "document", "documents", "provided", "based"}
    }
    if not meaningful_tokens:
        return False
    grounded = sum(1 for token in meaningful_tokens if token in context_lower)
    return grounded / len(meaningful_tokens) >= 0.6


def rewrite_query_node(state: AgentState) -> AgentState:
    client = OllamaChatClient()
    retry_count = int(state.get("retry_count", 0)) + 1
    prompt = f"""
Rewrite the vector search query to find stronger evidence for the question.
Return strict JSON with one key, "search_query".

Original question:
{state.get("question", "")}

Previous search query:
{state.get("search_query", "")}

Validation notes:
{state.get("validation_notes", "")}
"""
    raw = client.generate(prompt, temperature=0.2)
    parsed = _json_from_model(raw)
    rewritten = str(parsed.get("search_query") or state.get("question", "")).strip()
    updated = {**state, "search_query": rewritten, "retry_count": retry_count}
    return _with_step(updated, "query_rewriter", bool(rewritten), f"retry {retry_count}")


def fallback_node(state: AgentState) -> AgentState:
    notes = state.get("validation_notes", "The answer could not be validated against the retrieved source material.")
    fallback_answer = (
        "I could not fully validate a grounded answer after the maximum retrieval attempts. "
        f"Validation notes: {notes}"
    )
    updated = {**state, "answer": fallback_answer, "is_validated": False}
    return _with_step(updated, "fallback", True, "max validation retries reached")


def route_after_validation(state: AgentState) -> str:
    if state.get("is_validated") is True:
        return "end"
    if int(state.get("retry_count", 0)) >= get_settings().max_graph_retries:
        return "fallback"
    return "rewrite"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("planner", planner_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("reasoning", reasoning_node)
    graph.add_node("responder", responder_node)
    graph.add_node("validator", validator_node)
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("fallback", fallback_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "reasoning")
    graph.add_edge("reasoning", "responder")
    graph.add_edge("responder", "validator")
    graph.add_conditional_edges(
        "validator",
        route_after_validation,
        {"end": END, "rewrite": "rewrite_query", "fallback": "fallback"},
    )
    graph.add_edge("rewrite_query", "retriever")
    graph.add_edge("fallback", END)
    return graph.compile()


def run_document_assistant(question: str) -> AgentState:
    initial_state: AgentState = {
        "question": question,
        "retry_count": 0,
        "is_validated": False,
        "retrieved_context": [],
        "agent_steps": [],
    }
    try:
        return build_graph().invoke(initial_state)
    except Exception as exc:
        LOGGER.exception("Document assistant graph failed")
        return {
            **initial_state,
            "answer": f"Assistant graph failed: {exc}",
            "error": str(exc),
            "is_validated": False,
        }
