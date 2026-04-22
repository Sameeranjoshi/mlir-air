#!/usr/bin/env bash
# Usage:
#   utils/run_csl_ci.sh             # full: build + lit + simulator sweep
#   utils/run_csl_ci.sh --fast      # skip simulator (lit-only; <1s)
#   utils/run_csl_ci.sh --no-build  # skip ninja install
#   utils/run_csl_ci.sh --only=fadds_e2e    # one wafer only, full pipeline
#   utils/run_csl_ci.sh --verbose   # stream per-wafer output instead of capturing
#   utils/run_csl_ci.sh -j 16       # parallelism (passes through to run_csl_sdk.sh)
#
# Exit 0 iff every selected tier reports green.
#
# Tiers:
#   Tier 1: MLIR lit tests (fast, <1s, always runs)
#   Tier 2: Full simulator sweep (slow, ~5min at -j 8, skipped with --fast)
#
# The same script is run by:
#   - Developers locally, before pushing
#   - .githooks/pre-push (with --fast)
#   - .github/workflows/csl-ci.yml (full, on PR)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

FAST=0
NO_BUILD=0
ONLY=""
VERBOSE=0
JOBS=8

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fast) FAST=1; shift;;
    --no-build) NO_BUILD=1; shift;;
    --only=*) ONLY="${1#--only=}"; shift;;
    --verbose) VERBOSE=1; shift;;
    -j) JOBS="$2"; shift 2;;
    -j*) JOBS="${1#-j}"; shift;;
    -h|--help) sed -n '2,15p' "$0" | sed 's/^# //'; exit 0;;
    *) echo "unknown flag: $1" >&2; exit 2;;
  esac
done

# ---- env ----
module load python/3.10.3 cmake/3.26.0 rocm/6.2.1 singularity 2>/dev/null || true
if [[ -d "$ROOT_DIR/sandbox" ]]; then
  source "$ROOT_DIR/sandbox/bin/activate"
fi
source "$ROOT_DIR/utils/env_setup_gpu.sh" install llvm/install >/dev/null 2>&1

# ---- build ----
if [[ $NO_BUILD -eq 0 ]]; then
  echo "=== Build (ninja install) ==="
  (cd "$ROOT_DIR/build" && ninja install) || { echo "BUILD FAILED" >&2; exit 1; }
fi

# ---- Tier 1: lit ----
echo "=== Tier 1: MLIR lit tests ==="

LIT_BIN="${LIT:-}"
if [[ -z "$LIT_BIN" ]]; then
  LIT_BIN="$(command -v lit 2>/dev/null || true)"
fi
if [[ -z "$LIT_BIN" ]]; then
  for cand in "$HOME/.local/bin/lit" "/usr/local/bin/lit"; do
    [[ -x "$cand" ]] && LIT_BIN="$cand" && break
  done
fi
if [[ -z "$LIT_BIN" ]] || [[ ! -x "$LIT_BIN" ]]; then
  echo "lit not found; set LIT=/path/to/lit or activate the sandbox" >&2
  exit 1
fi

"$LIT_BIN" \
  "$ROOT_DIR/build/mlir/test/Dialect/CSL" \
  "$ROOT_DIR/build/mlir/test/Conversion/AIRToCSL" \
  "$ROOT_DIR/build/mlir/test/Targets/CSLEmit"
# lit exits non-zero on failure; set -e causes script to abort here.

if [[ $FAST -eq 1 ]]; then
  echo "=== --fast: skipping Tier 2 (simulator) ==="
  echo "ALL GREEN (Tier 1 only)"
  exit 0
fi

# ---- Tier 2: emit + simulator ----
echo ""
echo "=== Tier 2: emit + simulator sweep ==="

EMIT_ROOT="$(mktemp -d)"
echo "emit root: $EMIT_ROOT"

# Collect test files.
TESTS_GLOB=(
  "$ROOT_DIR/mlir/test/Targets/CSLEmit/e2e"/*.mlir
  "$ROOT_DIR/mlir/test/Targets/CSLEmit/e2e/scientific"/*.mlir
  "$ROOT_DIR/mlir/test/Targets/CSLEmit/e2e/auto-vectorize"/*.mlir
  "$ROOT_DIR/mlir/test/Targets/CSLEmit/e2e/switch"/*.mlir
)

emit_count=0
for f in "${TESTS_GLOB[@]}"; do
  [[ -f "$f" ]] || continue
  wafer_stem="$(basename "$f" .mlir)"
  if [[ -n "$ONLY" ]] && [[ "$wafer_stem" != "$ONLY" ]]; then
    continue
  fi
  if air-opt "$f" -csl-auto-vectorize -csl-infer-exports 2>/dev/null \
       | air-translate --emit-csl "--output-dir=$EMIT_ROOT" >/dev/null 2>&1; then
    emit_count=$((emit_count + 1))
  fi
done

wafer_count=$(find "$EMIT_ROOT" -maxdepth 1 -mindepth 1 -type d | wc -l)
echo "emitted: $emit_count test file(s), $wafer_count wafer dir(s)"

if [[ -n "$ONLY" ]]; then
  # Remove any wafer dirs that don't match the requested name.
  for dir in "$EMIT_ROOT"/*/; do
    [[ -d "$dir" ]] || continue
    name="$(basename "$dir")"
    [[ "$name" == "$ONLY" ]] || rm -rf "$dir"
  done
  wafer_count=$(find "$EMIT_ROOT" -maxdepth 1 -mindepth 1 -type d | wc -l)
  if [[ $wafer_count -eq 0 ]]; then
    echo "no wafer matched --only=$ONLY" >&2
    exit 2
  fi
fi

SIM_LOG="$(mktemp)"
sim_rc=0
if [[ $VERBOSE -eq 1 ]]; then
  bash "$ROOT_DIR/utils/run_csl_sdk.sh" -j "$JOBS" "$EMIT_ROOT" | tee "$SIM_LOG" || sim_rc=$?
else
  bash "$ROOT_DIR/utils/run_csl_sdk.sh" -j "$JOBS" "$EMIT_ROOT" > "$SIM_LOG" 2>&1 || sim_rc=$?
fi

# Summarize — match "PASS <name>" / "FAIL <name>" lines printed by run_csl_sdk.sh.
pass=$(grep -cE "^  PASS |^PASS " "$SIM_LOG" 2>/dev/null || true)
fail=$(grep -cE "^  FAIL |^FAIL " "$SIM_LOG" 2>/dev/null || true)
echo "simulator: $pass pass / $fail fail"

if [[ $fail -gt 0 ]] || [[ $sim_rc -ne 0 ]]; then
  echo ""
  echo "=== Failures ==="
  grep -E "^  FAIL |^FAIL " "$SIM_LOG" || true
  if [[ $VERBOSE -eq 0 ]]; then
    echo ""
    echo "rerun with --verbose to see full output, or inspect: $SIM_LOG"
  fi
  exit 1
fi

echo ""
echo "ALL GREEN (lit: ok, simulator: $pass/$pass)"
