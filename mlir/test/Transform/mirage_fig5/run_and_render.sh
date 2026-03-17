#!/bin/bash
# Run AIR dependency passes on attention_block.mlir and render the ACDG.
# Usage: ./run_and_render.sh [output_dir]
# Default output_dir: /tmp/mirage_fig5_dot

set -x
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${1:-/tmp/mirage_fig5_dot}"
mkdir -p "$OUT_DIR"

INPUT="$SCRIPT_DIR/attention_block.mlir"
echo "Input: $INPUT"
echo "Output dir: $OUT_DIR"

# Run passes. Use -air-dependency-schedule-opt only if AIE is enabled (pass may not exist on GPU build).
PIPELINE="-air-dependency -canonicalize -air-dependency-canonicalize"
if air-opt --help 2>/dev/null | grep -q "air-dependency-schedule-opt"; then
  PIPELINE="-air-dependency -canonicalize -air-dependency-schedule-opt -air-dependency-canonicalize"
fi

air-opt "$INPUT" $PIPELINE -air-dependency-parse-graph="output-dir=$OUT_DIR" -o /dev/null

echo ""
echo "DOT files written to $OUT_DIR"
ls -la "$OUT_DIR"
