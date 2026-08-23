#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TAG=${TAG:-cz22edd/pytorch:maniskillv2-100m}

docker build -t "$TAG" "$ROOT_DIR"
