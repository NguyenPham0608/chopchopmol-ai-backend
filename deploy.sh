#!/bin/bash
set -e

IMAGE="nguyen51304/chopchopmol-backend"
TAG="${1:-latest}"

echo "Building ${IMAGE}:${TAG} ..."
docker build --platform linux/amd64 -t "${IMAGE}:${TAG}" .

echo "Pushing ${IMAGE}:${TAG} ..."
docker push "${IMAGE}:${TAG}"

echo ""
echo "Done! Image: ${IMAGE}:${TAG}"
echo ""
echo "RunPod template settings:"
echo "  Container Image:  ${IMAGE}:${TAG}"
echo "  Container Disk:   20 GB"
echo "  Expose HTTP Port: 10000"
echo "  Docker Command:   (leave empty — CMD in Dockerfile)"
echo ""
echo "  Environment Variables:"
echo "    ANTHROPIC_API_KEY=<your key>"
echo "    OPENAI_API_KEY=<your key>"
