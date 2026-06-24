# Capstone-Gen-AI-Cognitive-Doc-Assistant

Generative AI-powered document assistant using FastAPI, Streamlit, LangGraph, runtime-configurable Gemini chat models, Gemini embeddings, and persistent ChromaDB vector search.

## Architecture

- `main.py`: FastAPI backend on port `8000`.
- `streamlit_app.py`: Streamlit frontend on port `8501`.
- `app/api/`: Pydantic API models and route handlers.
- `app/services/ingestion.py`: Robust parsing for PDF, TXT, CSV, XLS, XLSX, DOCX, JSON, YAML, and YML.
- `app/services/vector_store.py`: ChromaDB persistence with Gemini embeddings by default, plus optional Ollama fallback.
- `app/services/llm_service.py`: Gemini chat LLM facade with optional Ollama chat fallback.
- `app/services/rag_pipeline.py`: RAG facade over the agent graph.
- `app/agents/graph.py`: LangGraph planner, retriever, reasoner, validator, retry router, and fallback node.
- `app/utils/`: Input validation, rate limiting, and API exception handling.
- `k8s/`: Kubernetes manifests.

## Prerequisites

- Python `3.11` for native local installs.
- Google Gemini API key for chat generation and embeddings.
- Optional: Ollama only if you set `LLM_PROVIDER=ollama` or `EMBEDDING_PROVIDER=ollama`.

Windows note: Python `3.12` can require Microsoft C++ Build Tools for Chroma native dependencies. Use Python `3.11` or Docker.

## Windows Local Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

$env:LLM_PROVIDER="gemini"
$env:GEMINI_API_KEY="your-gemini-api-key"
$env:GEMINI_MODEL="gemini-2.5-flash-lite"
$env:GEMINI_API_VERSION="v1"
$env:GEMINI_GENERATION_MAX_RETRIES="3"
$env:EMBEDDING_PROVIDER="gemini"
$env:GEMINI_EMBEDDING_MODEL="gemini-embedding-001"
$env:GEMINI_EMBEDDING_API_VERSION="v1beta"
$env:GEMINI_EMBEDDING_BATCH_SIZE="8"
$env:GEMINI_EMBEDDING_BATCH_DELAY_SECONDS="2"
$env:GEMINI_EMBEDDING_MAX_RETRIES="6"
$env:CHROMA_PERSIST_DIR="./data/vectorstore"
$env:UPLOAD_DIR="./data/uploads"

python main.py
```

In another terminal:

```powershell
.\.venv\Scripts\Activate.ps1
$env:API_BASE_URL="http://localhost:8000"
streamlit run streamlit_app.py
```

## macOS Local Setup

```bash
brew install python@3.11

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

export LLM_PROVIDER=gemini
export GEMINI_API_KEY=your-gemini-api-key
export GEMINI_MODEL=gemini-2.5-flash-lite
export GEMINI_API_VERSION=v1
export GEMINI_GENERATION_MAX_RETRIES=3
export EMBEDDING_PROVIDER=gemini
export GEMINI_EMBEDDING_MODEL=gemini-embedding-001
export GEMINI_EMBEDDING_API_VERSION=v1beta
export GEMINI_EMBEDDING_BATCH_SIZE=8
export GEMINI_EMBEDDING_BATCH_DELAY_SECONDS=2
export GEMINI_EMBEDDING_MAX_RETRIES=6
export CHROMA_PERSIST_DIR=./data/vectorstore
export UPLOAD_DIR=./data/uploads
python main.py
```

In another terminal:

```bash
source .venv/bin/activate
export API_BASE_URL=http://localhost:8000
streamlit run streamlit_app.py
```

## Linux Local Setup

```bash
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3-pip curl

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

export LLM_PROVIDER=gemini
export GEMINI_API_KEY=your-gemini-api-key
export GEMINI_MODEL=gemini-2.5-flash-lite
export GEMINI_API_VERSION=v1
export GEMINI_GENERATION_MAX_RETRIES=3
export EMBEDDING_PROVIDER=gemini
export GEMINI_EMBEDDING_MODEL=gemini-embedding-001
export GEMINI_EMBEDDING_API_VERSION=v1beta
export GEMINI_EMBEDDING_BATCH_SIZE=8
export GEMINI_EMBEDDING_BATCH_DELAY_SECONDS=2
export GEMINI_EMBEDDING_MAX_RETRIES=6
export CHROMA_PERSIST_DIR=./data/vectorstore
export UPLOAD_DIR=./data/uploads
python main.py
```

In another terminal:

```bash
source .venv/bin/activate
export API_BASE_URL=http://localhost:8000
streamlit run streamlit_app.py
```

## API Checks

```bash
curl http://localhost:8000/health
curl http://localhost:8000/docs
```

Upload:

```bash
curl -X POST "http://localhost:8000/documents/upload" -F "file=@test.pdf"
```

Ask:

```bash
curl -X POST "http://localhost:8000/agents/ask" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the main topic?","top_k":3,"enable_validation":true}'
```

## Docker Build

```powershell
docker build -t capstone-agentic-rag:latest -f Dockerfile .
docker build -t capstone-agentic-rag-frontend:latest -f Dockerfile.streamlit .
```

The Docker images are generic. API keys, chat models, embedding models, and provider choices are supplied when containers start through environment variables. Do not bake real API keys into the image.

Plain Docker API run on Windows PowerShell:

```powershell
$env:GEMINI_API_KEY="your-gemini-api-key"
docker run --rm -p 8000:8000 `
  -e LLM_PROVIDER=gemini `
  -e GEMINI_API_KEY=$env:GEMINI_API_KEY `
  -e GEMINI_MODEL=gemini-2.5-flash-lite `
  -e GEMINI_API_VERSION=v1 `
  -e EMBEDDING_PROVIDER=gemini `
  -e GEMINI_EMBEDDING_MODEL=gemini-embedding-001 `
  -e GEMINI_EMBEDDING_API_VERSION=v1beta `
  -v ${PWD}\data:/app/data `
  capstone-agentic-rag:latest
```

Plain Docker API run on Linux:

```bash
export GEMINI_API_KEY=your-gemini-api-key
docker run --rm -p 8000:8000 --add-host=host.docker.internal:host-gateway \
  -e LLM_PROVIDER=gemini \
  -e GEMINI_API_KEY="$GEMINI_API_KEY" \
  -e GEMINI_MODEL=gemini-2.5-flash-lite \
  -e GEMINI_API_VERSION=v1 \
  -e EMBEDDING_PROVIDER=gemini \
  -e GEMINI_EMBEDDING_MODEL=gemini-embedding-001 \
  -e GEMINI_EMBEDDING_API_VERSION=v1beta \
  -v "$(pwd)/data:/app/data" \
  capstone-agentic-rag:latest
```

## Docker Compose

```powershell
Copy-Item .env.example .env
# Edit .env and set GEMINI_API_KEY, GEMINI_MODEL, and any provider/model overrides.
docker compose up --build -d
docker compose ps
docker compose logs -f api
docker compose logs -f frontend
```

To reuse already-built images on another machine:

```powershell
Copy-Item .env.example .env
# Edit .env for that machine's API key/model values.
docker compose up -d
```

You can also override values from the shell without editing compose:

```powershell
$env:GEMINI_API_KEY="your-gemini-api-key"
$env:GEMINI_MODEL="gemini-2.5-flash-lite"
$env:GEMINI_EMBEDDING_MODEL="gemini-embedding-001"
docker compose up -d
```

Open:

```text
API: http://localhost:8000/docs
UI:  http://localhost:8501
```

Stop:

```powershell
docker compose down
```

## Embedding Provider Notes

The default vector collection uses Gemini embeddings and is stored separately from the previous Ollama embedding collection. After switching providers, upload/index your documents again so ChromaDB contains vectors from the active embedding model.

To use the previous local embedding stack instead:

```powershell
$env:EMBEDDING_PROVIDER="ollama"
$env:OLLAMA_EMBEDDING_MODEL="nomic-embed-text"
ollama serve
ollama pull nomic-embed-text
docker compose up --build -d
```

## Kubernetes

```bash
sh scripts/docker-build.sh
kubectl apply -k k8s/
kubectl get pods -n genai-assistant
kubectl get svc -n genai-assistant
```

The frontend service uses NodePort `30501`.

Cleanup:

```bash
sh scripts/k8s-cleanup.sh
```
