from __future__ import annotations

import json
import logging
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
    answer: str
    validation_notes: str
    is_validated: bool
    retry_count: int
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
    return {**state, "plan": plan, "search_query": search_query}


def retriever_node(state: AgentState) -> AgentState:
    query = state.get("search_query") or state.get("question", "")
    vector_store = VectorStoreService()
    context = vector_store.query(str(query), k=3)
    return {**state, "retrieved_context": context}


def reasoning_node(state: AgentState) -> AgentState:
    client = OllamaChatClient()
    context = "\n\n".join(
        f"Source: {item.get('source', 'unknown')}\nContent: {item.get('content', '')}"
        for item in state.get("retrieved_context", [])
    )
    prompt = f"""
You are a precise document assistant.
Answer the question using only the supplied context.
If the context is insufficient, say exactly what is missing.
Include concise source references by filename when available.

Question:
{state.get("question", "")}

Context:
{context or "No context retrieved."}
"""
    answer = client.generate(prompt)
    return {**state, "answer": answer}


def validator_node(state: AgentState) -> AgentState:
    client = OllamaChatClient()
    context = "\n\n".join(
        f"Source: {item.get('source', 'unknown')}\nContent: {item.get('content', '')}"
        for item in state.get("retrieved_context", [])
    )
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
    is_validated = bool(parsed.get("isValidated", False))
    notes = str(parsed.get("notes") or raw)
    return {**state, "is_validated": is_validated, "validation_notes": notes}


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
    return {**state, "search_query": rewritten, "retry_count": retry_count}


def fallback_node(state: AgentState) -> AgentState:
    notes = state.get("validation_notes", "The answer could not be validated against the retrieved source material.")
    fallback_answer = (
        "I could not fully validate a grounded answer after the maximum retrieval attempts. "
        f"Validation notes: {notes}"
    )
    return {**state, "answer": fallback_answer, "is_validated": False}


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
    graph.add_node("validator", validator_node)
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("fallback", fallback_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "reasoning")
    graph.add_edge("reasoning", "validator")
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
