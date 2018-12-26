#!/usr/bin/env bash

set -exu
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

./build.sh
docker build -f Dockerfile.sophon -t pytorch/glow/sophon:0.1 .
