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

See [Architecture Diagrams](docs/architecture-diagrams.md) for the application flow, LangGraph RAG loop, Docker Compose deployment, Kubernetes deployment, and runtime configuration diagrams.

## Prerequisites

- Docker Desktop or Docker Engine with Docker Compose for the recommended deployment path.
- Python `3.11` only if running the app without Docker.
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

## Docker Deployment From Released Images

The recommended deployment path is Docker Compose with already-published images from Docker Hub.

Released image version:

```text
API:      dirajan/capstone-agentic-rag:v1.0.0
Frontend: dirajan/capstone-agentic-rag-frontend:v1.0.0
```

The images are generic. API keys, chat models, embedding models, and provider choices are supplied at container startup through `.env` or shell environment variables. Do not bake real API keys into the image.

Windows PowerShell:

```powershell
Copy-Item .env.example .env
notepad .env
```

Set at least these values in `.env`:

```env
API_IMAGE_REPOSITORY=dirajan/capstone-agentic-rag
FRONTEND_IMAGE_REPOSITORY=dirajan/capstone-agentic-rag-frontend
IMAGE_TAG=v1.0.0
GEMINI_API_KEY=your-gemini-api-key
GOOGLE_API_KEY=your-gemini-api-key
GEMINI_MODEL=gemini-2.5-flash-lite
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
```

Pull and start:

```powershell
docker compose pull
docker compose up -d
docker compose ps
```

macOS/Linux:

```bash
cp .env.example .env
nano .env
docker compose pull
docker compose up -d
docker compose ps
```

Open:

```text
UI:       http://localhost:8501
API docs: http://localhost:8000/docs
Health:   http://localhost:8000/health
```

Stop:

```bash
docker compose down
```

Clear indexed documents and uploaded files:

```bash
docker compose down -v
docker compose up -d
```

Change deployed version:

```env
IMAGE_TAG=v1.0.0
```

Then run:

```bash
docker compose pull
docker compose up -d --force-recreate
```

## Docker Build

```powershell
docker build -t capstone-agentic-rag:latest -f Dockerfile .
docker build -t capstone-agentic-rag-frontend:latest -f Dockerfile.streamlit .
```

Tag a local build as a release image:

```powershell
docker tag capstone-agentic-rag:latest dirajan/capstone-agentic-rag:v1.0.0
docker tag capstone-agentic-rag-frontend:latest dirajan/capstone-agentic-rag-frontend:v1.0.0
```

Push release images:

```powershell
docker push dirajan/capstone-agentic-rag:v1.0.0
docker push dirajan/capstone-agentic-rag-frontend:v1.0.0
```

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

## Docker Compose Quick Reference

```powershell
Copy-Item .env.example .env
# Edit .env and set GEMINI_API_KEY plus any model/provider overrides.
# IMAGE_TAG selects the pushed Docker image version, for example v1.0.0.
docker compose pull
docker compose up -d
docker compose ps
docker compose logs -f api
docker compose logs -f frontend
```

To reuse already-built images on another machine:

```powershell
Copy-Item .env.example .env
# Edit .env for that machine's API key/model values and IMAGE_TAG.
docker compose up -d
```

You can also override values from the shell without editing compose:

```powershell
$env:GEMINI_API_KEY="your-gemini-api-key"
$env:GEMINI_MODEL="gemini-2.5-flash-lite"
$env:GEMINI_EMBEDDING_MODEL="gemini-embedding-001"
$env:IMAGE_TAG="v1.0.0"
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

The Kubernetes manifests deploy the released Docker images:

```text
dirajan/capstone-agentic-rag:v1.0.0
dirajan/capstone-agentic-rag-frontend:v1.0.0
```

Docker Desktop Kubernetes on Windows:

```powershell
kubectl config use-context docker-desktop

# Optional if local proxy variables interfere with Docker Desktop Kubernetes.
$env:HTTP_PROXY=""
$env:HTTPS_PROXY=""
$env:ALL_PROXY=""
$env:NO_PROXY="localhost,127.0.0.1,kubernetes.docker.internal"

kubectl apply -f k8s\namespace.yaml

$geminiKey = ((Get-Content .env | Where-Object { $_ -match '^GEMINI_API_KEY=' } | Select-Object -First 1) -split '=', 2)[1]
kubectl -n genai-assistant create secret generic genai-secrets `
  --from-literal=GEMINI_API_KEY=$geminiKey `
  --from-literal=GOOGLE_API_KEY=$geminiKey `
  --dry-run=client -o yaml | kubectl apply -f -

kubectl apply -k k8s
kubectl -n genai-assistant rollout status deployment/genai-api
kubectl -n genai-assistant rollout status deployment/genai-frontend
kubectl -n genai-assistant get pods,svc,pvc,hpa,ingress
```

macOS/Linux:

```bash
kubectl config use-context docker-desktop
kubectl apply -f k8s/namespace.yaml

GEMINI_API_KEY_VALUE="$(grep '^GEMINI_API_KEY=' .env | head -n 1 | cut -d= -f2-)"
kubectl -n genai-assistant create secret generic genai-secrets \
  --from-literal=GEMINI_API_KEY="$GEMINI_API_KEY_VALUE" \
  --from-literal=GOOGLE_API_KEY="$GEMINI_API_KEY_VALUE" \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl apply -k k8s
kubectl -n genai-assistant rollout status deployment/genai-api
kubectl -n genai-assistant rollout status deployment/genai-frontend
kubectl -n genai-assistant get pods,svc,pvc,hpa,ingress
```

The frontend service uses NodePort `30501`:

```text
UI: http://localhost:30501
```

Health checks:

```powershell
kubectl -n genai-assistant exec deploy/genai-api -- python -c "import requests; print(requests.get('http://localhost:8000/health', timeout=10).text)"
kubectl -n genai-assistant exec deploy/genai-frontend -- python -c "import requests; print(requests.get('http://api:8000/health', timeout=10).status_code)"
```

Cleanup:

```bash
kubectl delete namespace genai-assistant
```
