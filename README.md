# Capstone-Gen-AI-Cognitive-Doc-Assistant

Generative AI-powered document assistant using Streamlit, LangGraph, local Ollama models, and persistent ChromaDB vector search.

## Architecture

- `main.py`: Streamlit dashboard with bounded sliding-window chat history.
- `app/core/config.py`: Pydantic settings for environment variables.
- `app/services/ingestion.py`: Robust parsing for PDF, TXT, CSV, XLSX, JSON, YAML, and YML.
- `app/services/vector_store.py`: ChromaDB persistence with Ollama `nomic-embed-text` embeddings and overlapping recursive chunks.
- `app/agents/graph.py`: LangGraph planner, retriever, reasoning, validator, retry router, query rewrite, and fallback nodes.

## Prerequisites

- Python `3.11` is recommended for native local installs.
- Docker Desktop or Docker Engine is required for container deployment.
- Ollama must run on the host machine with these models:

```powershell
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

## Windows Local Setup

Create and activate a virtual environment:

```powershell
py -3.11 -m venv .venv
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

Note for Windows: Python `3.12` may try to compile `chroma-hnswlib` and fail unless Microsoft C++ Build Tools are installed. Use Python `3.11` for the native local setup, or run Docker.

## macOS Local Setup

Install Python and Ollama:

```bash
brew install python@3.11 ollama
ollama serve
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

Create the app environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run main.py
```

## Linux Local Setup

Install Python, venv support, and Ollama:

```bash
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3-pip curl
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

Create the app environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run main.py
```

## Git Automation Template

Each development step used this terminal automation pattern:

```powershell
git add <changed-files>
git commit -m "Dipu VR : Capstone Project Gen AI Apllication - Step [X]: [Step Title]"
git push
```

## Production Port

The Streamlit app listens on port `8501` by default.

## Docker Deployment

Build the production image:

```powershell
docker build -t capstone-agentic-rag:latest .
```

Run the container while using Ollama from the host machine:

```powershell
docker run --rm -p 8501:8501 -e OLLAMA_BASE_URL=http://host.docker.internal:11434 -v ${PWD}\chroma_db:/app/chroma_db capstone-agentic-rag:latest
```

Linux users can use this host gateway flag when running with plain Docker:

```bash
docker run --rm -p 8501:8501 --add-host=host.docker.internal:host-gateway -e OLLAMA_BASE_URL=http://host.docker.internal:11434 -v "$(pwd)/chroma_db:/app/chroma_db" capstone-agentic-rag:latest
```

## Docker Compose Deployment

Start the application:

```powershell
docker compose up --build -d
```

Check status and logs:

```powershell
docker compose ps
docker compose logs -f document-assistant
```

Stop the application:

```powershell
docker compose down
```

Then open:

```text
http://localhost:8501
```
