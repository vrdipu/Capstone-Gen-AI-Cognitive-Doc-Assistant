from __future__ import annotations

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import agent_router, document_router, health_router, question_router, search_router
from app.core.config import get_settings
from app.core.logging import app_logger
from app.utils.exceptions import register_exception_handlers
from app.utils.rate_limiter import RateLimitMiddleware


settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    app_logger.info(f"Starting {settings.app_name}")
    yield


app = FastAPI(
    title=settings.app_name,
    description="AI-powered document assistant using RAG and agentic validation.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RateLimitMiddleware, requests_per_minute=60)

app.include_router(health_router)
app.include_router(document_router)
app.include_router(question_router)
app.include_router(agent_router)
app.include_router(search_router)
register_exception_handlers(app)


@app.get("/", tags=["Root"])
async def root() -> dict[str, str]:
    return {"message": f"Welcome to {settings.app_name}"}


if __name__ == "__main__":
    uvicorn.run("main:app", host=settings.api_host, port=settings.api_port, reload=settings.debug)
