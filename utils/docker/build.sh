#!/usr/bin/env bash

set -exu
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

docker build -f Dockerfile -t pytorch/glow:0.1 .
