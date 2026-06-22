from __future__ import annotations

import logging
from typing import Any

import streamlit as st


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)
MAX_VISIBLE_MESSAGES = 12


def _initialize_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("indexed_files", [])


def _visible_messages() -> list[dict[str, str]]:
    return st.session_state.messages[-MAX_VISIBLE_MESSAGES:]


def _append_message(role: str, content: str) -> None:
    st.session_state.messages.append({"role": role, "content": content})


def _index_uploaded_files(files: list[Any]) -> None:
    from app.services.ingestion import parse_uploaded_file
    from app.services.vector_store import VectorStoreService

    vector_store = VectorStoreService()
    indexed_names: list[str] = []
    failures: list[str] = []

    for uploaded_file in files:
        try:
            document = parse_uploaded_file(uploaded_file)
            chunks_added = vector_store.add_document(document)
            indexed_names.append(f"{document.source_name} ({chunks_added} chunks)")
        except Exception as exc:
            LOGGER.exception("Failed to index uploaded file %s", getattr(uploaded_file, "name", "unknown"))
            failures.append(f"{getattr(uploaded_file, 'name', 'unknown')}: {exc}")

    if indexed_names:
        st.session_state.indexed_files.extend(indexed_names)
        st.success("Indexed " + ", ".join(indexed_names))
    if failures:
        st.error("Some files could not be indexed: " + " | ".join(failures))


def _answer_prompt(prompt: str) -> str:
    from app.agents.graph import run_document_assistant

    result = run_document_assistant(prompt)
    answer = result.get("answer") or "I could not produce a grounded answer from the indexed documents."
    if result.get("is_validated") is False:
        answer += "\n\nValidation warning: the final answer could not be fully verified against retrieved context."
    return answer


def main() -> None:
    st.set_page_config(page_title="Agentic RAG Document Assistant", page_icon="AI", layout="wide")
    _initialize_state()

    st.title("Agentic RAG Document Assistant")

    with st.sidebar:
        st.header("Documents")
        uploaded_files = st.file_uploader(
            "Upload source files",
            type=["pdf", "txt", "csv", "xlsx", "json", "yaml", "yml"],
            accept_multiple_files=True,
        )
        if st.button("Index documents", type="primary", disabled=not uploaded_files):
            with st.spinner("Parsing and embedding documents..."):
                _index_uploaded_files(uploaded_files or [])

        if st.session_state.indexed_files:
            st.subheader("Indexed")
            for item in st.session_state.indexed_files[-10:]:
                st.caption(item)

    chat_container = st.container()
    with chat_container:
        for message in _visible_messages():
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    prompt = st.chat_input("Ask a question about your indexed documents")
    if prompt:
        _append_message("user", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Planning, retrieving, reasoning, and validating..."):
                try:
                    answer = _answer_prompt(prompt)
                except Exception as exc:
                    LOGGER.exception("Assistant request failed")
                    answer = f"Request failed: {exc}"
                st.markdown(answer)
        _append_message("assistant", answer)


if __name__ == "__main__":
    main()
