param(
    [string]$ImageTag = $(if ($env:IMAGE_TAG) { $env:IMAGE_TAG } else { "v1.0.0" }),
    [string]$ApiImageRepository = $(if ($env:API_IMAGE_REPOSITORY) { $env:API_IMAGE_REPOSITORY } else { "dirajan/capstone-agentic-rag" }),
    [string]$FrontendImageRepository = $(if ($env:FRONTEND_IMAGE_REPOSITORY) { $env:FRONTEND_IMAGE_REPOSITORY } else { "dirajan/capstone-agentic-rag-frontend" }),
    [string]$Platforms = $(if ($env:PLATFORMS) { $env:PLATFORMS } else { "linux/amd64,linux/arm64" })
)

$ErrorActionPreference = "Stop"

docker buildx create --name capstone-multiarch --use 2>$null
if ($LASTEXITCODE -ne 0) {
    docker buildx use capstone-multiarch
}

docker buildx inspect --bootstrap

docker buildx build `
    --platform $Platforms `
    -t "${ApiImageRepository}:${ImageTag}" `
    -f Dockerfile `
    --push `
    .

docker buildx build `
    --platform $Platforms `
    -t "${FrontendImageRepository}:${ImageTag}" `
    -f Dockerfile.streamlit `
    --push `
    .

docker buildx imagetools inspect "${ApiImageRepository}:${ImageTag}"
docker buildx imagetools inspect "${FrontendImageRepository}:${ImageTag}"
