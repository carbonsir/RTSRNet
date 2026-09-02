#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

DATA="${1:-/root/autodl-tmp/data}"
shift || true

python -u go.py --data "$DATA" "$@"
