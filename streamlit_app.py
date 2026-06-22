from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st


API_HTTP = requests.Session()
API_HTTP.trust_env = False
CONFIGURED_API_BASE_URL = os.environ.get("API_BASE_URL", "").rstrip("/")
API_BASE_URL_CANDIDATES = tuple(
    dict.fromkeys(
        url.rstrip("/")
        for url in (
            CONFIGURED_API_BASE_URL,
            "http://api:8000",
            "http://host.docker.internal:8000",
            "http://localhost:8000",
        )
        if url
    )
)
MAX_VISIBLE_MESSAGES = 12


def _initialize_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("indexed_files", [])
    st.session_state.setdefault("api_base_url", None)


def _visible_messages() -> list[dict[str, str]]:
    return st.session_state.messages[-MAX_VISIBLE_MESSAGES:]


def _append_message(role: str, content: str) -> None:
    st.session_state.messages.append({"role": role, "content": content})


def _probe_api_base_url() -> str | None:
    cached_url = st.session_state.get("api_base_url")
    candidates = (cached_url, *API_BASE_URL_CANDIDATES) if cached_url else API_BASE_URL_CANDIDATES
    for api_base_url in dict.fromkeys(url for url in candidates if url):
        try:
            response = API_HTTP.get(f"{api_base_url}/health", timeout=5)
            if response.status_code == 200:
                st.session_state.api_base_url = api_base_url
                return api_base_url
        except requests.RequestException:
            continue
    st.session_state.api_base_url = None
    return None


def _api_base_url() -> str:
    api_base_url = st.session_state.get("api_base_url") or _probe_api_base_url()
    if not api_base_url:
        raise RuntimeError("API is not reachable. Tried: " + ", ".join(API_BASE_URL_CANDIDATES))
    return api_base_url


def _upload_documents(files: list[Any]) -> None:
    indexed: list[str] = []
    failures: list[str] = []
    for uploaded_file in files:
        try:
            response = API_HTTP.post(
                f"{_api_base_url()}/documents/upload",
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                timeout=120,
            )
            if response.ok:
                result = response.json()
                indexed.append(f"{result['filename']} ({result['chunks_created']} chunks)")
            else:
                failures.append(f"{uploaded_file.name}: {response.text}")
        except requests.RequestException as exc:
            failures.append(f"{uploaded_file.name}: {exc}")
    if indexed:
        st.session_state.indexed_files.extend(indexed)
        st.success("Indexed " + ", ".join(indexed))
    if failures:
        st.error("Some files could not be indexed: " + " | ".join(failures))


def _ask_agent(question: str) -> str:
    response = API_HTTP.post(f"{_api_base_url()}/agents/ask", json={"question": question, "top_k": 3}, timeout=180)
    response.raise_for_status()
    result = response.json()
    answer = result.get("answer") or "No answer returned."
    if result.get("sources"):
        sources = ", ".join(sorted({source["filename"] for source in result["sources"] if source.get("filename")}))
        if sources:
            answer += f"\n\nSources: {sources}"
    if result.get("is_validated") is False:
        answer += "\n\nValidation warning: answer was not fully validated against source material."
    return answer


def main() -> None:
    st.set_page_config(page_title="GenAI Document Assistant", page_icon=":material/description:", layout="wide")
    _initialize_state()

    st.title("GenAI Document Assistant")
    api_base_url = _probe_api_base_url()
    api_healthy = api_base_url is not None
    if not api_healthy:
        st.warning("API is not reachable. Tried: " + ", ".join(API_BASE_URL_CANDIDATES))
    else:
        st.caption(f"API connected: {api_base_url}")

    with st.sidebar:
        st.header("Documents")
        uploaded_files = st.file_uploader(
            "Upload source files",
            type=["pdf", "txt", "csv", "xlsx", "xls", "docx", "json", "yaml", "yml"],
            accept_multiple_files=True,
        )
        if st.button("Index documents", type="primary", disabled=not uploaded_files or not api_healthy):
            with st.spinner("Uploading and indexing documents..."):
                _upload_documents(uploaded_files or [])

        if st.session_state.indexed_files:
            st.subheader("Indexed")
            for item in st.session_state.indexed_files[-10:]:
                st.caption(item)

    for message in _visible_messages():
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask a question about your indexed documents", disabled=not api_healthy)
    if prompt:
        _append_message("user", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Planning, retrieving, reasoning, and validating..."):
                try:
                    answer = _ask_agent(prompt)
                except Exception as exc:
                    answer = f"Request failed: {exc}"
                st.markdown(answer)
        _append_message("assistant", answer)


if __name__ == "__main__":
    main()
