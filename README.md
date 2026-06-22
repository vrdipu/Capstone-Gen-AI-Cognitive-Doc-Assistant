# Capstone-Gen-AI-Cognitive-Doc-Assistant

Generative AI-powered document assistant using Streamlit, LangGraph, local Ollama models, and persistent ChromaDB vector search.

## Architecture

- `main.py`: Streamlit dashboard with bounded sliding-window chat history.
- `app/core/config.py`: Pydantic settings for environment variables.
- `app/services/ingestion.py`: Robust parsing for PDF, TXT, CSV, XLSX, JSON, YAML, and YML.
- `app/services/vector_store.py`: ChromaDB persistence with Ollama `nomic-embed-text` embeddings and overlapping recursive chunks.
- `app/agents/graph.py`: LangGraph planner, retriever, reasoning, validator, retry router, query rewrite, and fallback nodes.

## Local Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Start Ollama locally and pull the required models:

```powershell
ollama serve
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

If `ollama serve` is already running as a background service, only run the two `ollama pull` commands.

Optional environment configuration:

```powershell
$env:OLLAMA_BASE_URL="http://localhost:11434"
$env:CHROMA_DB_DIR="./chroma_db"
$env:OLLAMA_CHAT_MODEL="llama3.2:3b"
$env:OLLAMA_EMBEDDING_MODEL="nomic-embed-text"
$env:MAX_GRAPH_RETRIES="3"
```

Run the dashboard:

```powershell
streamlit run main.py
```

Open the local Streamlit URL, upload supported documents from the sidebar, index them, and ask questions in the chat box.

## Git Automation Template

Each development step used this terminal automation pattern:

```powershell
git add <changed-files>
git commit -m "Dipu VR : Capstone Project Gen AI Apllication - Step [X]: [Step Title]"
git push
```

## Production Port

The Streamlit app listens on port `8501` by default.
