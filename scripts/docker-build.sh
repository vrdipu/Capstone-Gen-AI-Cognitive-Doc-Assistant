#!/usr/bin/env sh
set -eu

docker build -t capstone-agentic-rag:latest -f Dockerfile .
docker build -t capstone-agentic-rag-frontend:latest -f Dockerfile.streamlit .
