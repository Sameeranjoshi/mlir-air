#!/bin/bash
# air-explorer.sh -- Run air-opt dependency passes then launch Model Explorer
# Usage: air-explorer.sh <input.mlir> [model-explorer options, e.g. --port 8080]
set -e

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <input.mlir> [--port PORT]" >&2
    exit 1
fi

INPUT="$1"; shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AIR_OPT="${AIR_OPT:-$(dirname "$SCRIPT_DIR")/../../install/bin/air-opt}"

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

echo "Running air-opt passes on $INPUT ..."
"$AIR_OPT" "$INPUT" \
    -air-dependency \
    -canonicalize \
    -air-dependency-canonicalize \
    "-air-dependency-parse-graph=output-dir=$TMPDIR" \
    -o /dev/null

echo "DOT files written to $TMPDIR"
ls "$TMPDIR"/*.dot 2>/dev/null

echo "Launching Model Explorer (adapter locked to air_explorer) ..."
python -m air_explorer.launch "$TMPDIR/combined.dot" "$@"
