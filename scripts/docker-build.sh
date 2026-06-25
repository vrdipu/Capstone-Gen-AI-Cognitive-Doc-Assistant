#!/usr/bin/env sh
set -eu

IMAGE_TAG="${IMAGE_TAG:-v1.0.0}"
API_IMAGE_REPOSITORY="${API_IMAGE_REPOSITORY:-dirajan/capstone-agentic-rag}"
FRONTEND_IMAGE_REPOSITORY="${FRONTEND_IMAGE_REPOSITORY:-dirajan/capstone-agentic-rag-frontend}"
PLATFORMS="${PLATFORMS:-linux/amd64,linux/arm64}"

docker buildx create --name capstone-multiarch --use 2>/dev/null || docker buildx use capstone-multiarch
docker buildx inspect --bootstrap

docker buildx build \
  --platform "$PLATFORMS" \
  -t "$API_IMAGE_REPOSITORY:$IMAGE_TAG" \
  -f Dockerfile \
  --push \
  .

docker buildx build \
  --platform "$PLATFORMS" \
  -t "$FRONTEND_IMAGE_REPOSITORY:$IMAGE_TAG" \
  -f Dockerfile.streamlit \
  --push \
  .
