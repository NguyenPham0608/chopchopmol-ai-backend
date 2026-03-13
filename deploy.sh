#!/bin/bash
set -e

export DOCKER_HOST=unix:///Users/buupham/.docker/run/docker.sock

IMAGE="nguyen51304/chopchopmol-backend"
TAG="${1:-latest}"

echo "Building ${IMAGE}:${TAG} ..."
docker build --platform linux/amd64 -t "${IMAGE}:${TAG}" .

echo "Pushing ${IMAGE}:${TAG} ..."
docker push "${IMAGE}:${TAG}"

echo ""
echo "=== Done! Image: ${IMAGE}:${TAG} ==="
echo ""
echo "RunPod Template settings:"
echo "  Container Image:   ${IMAGE}:${TAG}"
echo "  Container Disk:    20 GB"
echo "  Expose HTTP Ports: 10000"
echo "  Expose TCP Ports:  22"
echo "  Docker Command:    (leave empty)"
echo ""
echo "  Environment Variables:"
echo "    ANTHROPIC_API_KEY = <your key>"
echo "    OPENAI_API_KEY   = <your key>"
echo ""
echo "  For SSH: add your public key in RunPod Settings > SSH Keys"
