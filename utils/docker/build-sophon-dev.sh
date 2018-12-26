#!/usr/bin/env bash

set -exu
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

./build-sophon.sh
docker build -f Dockerfile.sophon.dev -t pytorch/glow/sophon/dev:0.1 .
