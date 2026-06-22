from __future__ import annotations

import json
import logging
from typing import Any, TypedDict

import requests

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
