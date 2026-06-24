# Architecture Diagrams

This document shows how the GenAI Document Assistant works and how it is deployed with Docker Compose and Kubernetes.

## 1. Application Working Architecture

```mermaid
flowchart LR
    user[User Browser] --> ui[Streamlit UI<br/>streamlit_app.py]
    ui -->|HTTP REST| api[FastAPI Backend<br/>main.py]

    subgraph backend[Backend Application Layer]
        api --> routes[API Routes<br/>app/api/routes.py]
        routes --> ingestion[Document Ingestion<br/>PDF, TXT, CSV, XLSX, DOCX, JSON, YAML]
        routes --> agent[LangGraph Agent<br/>app/agents/graph.py]
        routes --> search[Search Endpoint]
    end

    subgraph rag[RAG Services]
        ingestion --> splitter[Recursive Text Splitter<br/>chunk 1000, overlap 150]
        splitter --> embeddings[Gemini Embeddings<br/>gemini-embedding-001]
        embeddings --> chroma[(Persistent ChromaDB<br/>/app/data/vectorstore)]
        search --> chroma
        agent --> chroma
    end

    subgraph llm[Model Layer]
        agent --> gemini[Gemini Chat Model<br/>gemini-2.5-flash-lite]
        embeddings --> geminiEmbed[Gemini Embedding API]
    end

    chroma --> agent
    gemini --> agent
    agent --> routes
    routes --> ui
```

## 2. Document Upload And Indexing Flow

```mermaid
sequenceDiagram
    actor User
    participant UI as Streamlit UI
    participant API as FastAPI /documents/upload
    participant Parser as Ingestion Parser
    participant Splitter as Recursive Splitter
    participant Emb as Gemini Embeddings
    participant DB as ChromaDB

    User->>UI: Upload PDF or supported file
    UI->>API: POST /documents/upload
    API->>Parser: Parse file bytes into clean text
    Parser-->>API: ParsedDocument
    API->>Splitter: Split text into overlapping chunks
    Splitter-->>API: Chunk list
    API->>Emb: Batch embed chunks
    Emb-->>API: Vector embeddings
    API->>DB: Upsert chunks, metadata, vectors
    DB-->>API: Persisted
    API-->>UI: File indexed with chunk count
    UI-->>User: Show indexing result
```

## 3. LangGraph Agentic RAG Flow

```mermaid
flowchart TD
    q[User Question] --> planner[Planner Node<br/>Create retrieval query]
    planner --> retriever[Retriever Node<br/>Query ChromaDB top-k context]
    retriever --> reasoner[Reasoning Node<br/>Answer from retrieved context]
    reasoner --> responder[Responder Node<br/>Attach sources]
    responder --> validator[Validator Node<br/>Check grounding]

    validator -->|validated true| done[Final Answer]
    validator -->|validated false and retries left| rewrite[Query Rewrite Node]
    rewrite --> retriever
    validator -->|max retries reached| fallback[Fallback Node<br/>Stop loop safely]
    fallback --> done

    subgraph safeguards[Runtime Safeguards]
        retries[Max graph retries: 3]
        quota[Gemini retry and quota handling]
        grounding[Deterministic source grounding check]
    end

    retries -.-> validator
    quota -.-> reasoner
    grounding -.-> validator
```

## 4. Docker Compose Deployment Diagram

```mermaid
flowchart TB
    browser[Browser<br/>http://localhost:8501] --> frontendPort[Host Port 8501]
    apiClient[API Client<br/>http://localhost:8000] --> apiPort[Host Port 8000]

    subgraph host[Docker Host]
        subgraph compose[Docker Compose Project]
            frontendPort --> frontend[frontend service<br/>dirajan/capstone-agentic-rag-frontend:<IMAGE_TAG>]
            apiPort --> api[api service<br/>dirajan/capstone-agentic-rag:<IMAGE_TAG>]
            frontend -->|API_BASE_URL=http://api:8000| api
            api --> volume[(api-data volume<br/>/app/data)]
        end
    end

    api --> gemini[Google Gemini APIs<br/>Chat and Embeddings]

    env[.env file<br/>API key, models, IMAGE_TAG] -.runtime env.-> api
    env -.runtime env.-> frontend
```

Runtime image selection:

```env
API_IMAGE_REPOSITORY=dirajan/capstone-agentic-rag
FRONTEND_IMAGE_REPOSITORY=dirajan/capstone-agentic-rag-frontend
IMAGE_TAG=v1.0.0
```

## 5. Kubernetes Deployment Diagram

```mermaid
flowchart TB
    browser[Browser] --> nodeport[NodePort Service<br/>frontend:30501]

    subgraph cluster[Docker Desktop Kubernetes Cluster]
        ns[Namespace<br/>genai-assistant]

        subgraph frontendLayer[Frontend Layer]
            nodeport --> frontendPod[genai-frontend Pod<br/>Streamlit<br/>dirajan/capstone-agentic-rag-frontend:v1.0.0]
            frontendSvc[frontend Service<br/>8501 -> 30501] --> frontendPod
        end

        subgraph apiLayer[API and Agent Layer]
            frontendPod -->|http://api:8000| apiSvc[api Service<br/>ClusterIP:8000]
            apiSvc --> apiPod[genai-api Pod<br/>FastAPI + LangGraph<br/>dirajan/capstone-agentic-rag:v1.0.0]
            hpa[HorizontalPodAutoscaler<br/>genai-api-hpa] -.scales.-> apiPod
        end

        subgraph configLayer[Runtime Configuration]
            cfg[ConfigMap<br/>models, provider, chunking]
            secret[Secret<br/>GEMINI_API_KEY]
            pvc[(PVC genai-data<br/>ChromaDB and uploads)]
        end

        cfg -.env.-> apiPod
        secret -.env.-> apiPod
        pvc -.mount /app/data.-> apiPod
        ingress[Ingress<br/>genai-assistant.local] -.optional.-> frontendSvc
    end

    apiPod --> gemini[Google Gemini APIs<br/>Chat and Embeddings]
```

Kubernetes access points:

```text
UI:       http://localhost:30501
API:      internal service http://api:8000
Namespace: genai-assistant
```

## 6. Runtime Configuration Summary

```mermaid
flowchart LR
    envFile[.env or Kubernetes Secret/ConfigMap] --> runtime[Container Runtime]
    runtime --> api[API Container]
    runtime --> frontend[Frontend Container]

    envFile --> key[GEMINI_API_KEY]
    envFile --> model[GEMINI_MODEL]
    envFile --> embed[GEMINI_EMBEDDING_MODEL]
    envFile --> tag[IMAGE_TAG]
    envFile --> chunk[Chunking and retry settings]
```

Key point: images are built once and reused across systems. API keys, model names, embedding model names, and image tags are injected at runtime.
