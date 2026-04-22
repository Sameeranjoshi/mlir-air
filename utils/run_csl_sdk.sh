#!/usr/bin/env bash
# Usage: run_csl_sdk.sh [-j N] <emission-root>
#
# Iterates every subdirectory of <emission-root> that contains an executable
# commands_wse3.sh and runs it. The emission root is the directory passed to
# `air-translate --emit-csl --output-dir=<dir>`, which contains one
# subdirectory per csl.wafer.
#
# Exercises every lit-emittable wafer, including (v5):
#   - e2e/dsds/                       @get_dsd(mem1d_dsd, ...) + @fmacs/@fadds
#   - e2e/control_flow_if/            scf.if + arith.cmpf/cmpi
#   - e2e/imports/                    generic csl.import_module (<math>, etc.)
#   - e2e/scientific/                 saxpy, dot, stencil_1d, norm_sq
#   - e2e/simd_dsd/                   N-PE SIMD DSD kernel on 4x2 subgrid
#
# Requires: cslc, cs_python on PATH.
#
# Exit status: 0 if every wafer ran to "SUCCESS!", 1 otherwise.
#
# CAVEAT: The Cerebras SDK singularity container may cap simultaneous cslc
# invocations.  If high -j produces sporadic failures not seen at -j 1,
# reduce JOBS.  Empirically start with -j 4 or -j 8.
set -uo pipefail

JOBS="${JOBS:-4}"

# Parse flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    -j) JOBS="$2"; shift 2;;
    -j*) JOBS="${1#-j}"; shift;;
    -*) echo "unknown flag: $1" >&2; exit 2;;
    *) break;;
  esac
done

ROOT="${1:?usage: run_csl_sdk.sh [-j N] <emission-root>}"
if [[ ! -d "$ROOT" ]]; then
  echo "error: not a directory: $ROOT" >&2
  exit 2
fi

# run_one_wafer <root> <name>
#   - Runs commands_wse3.sh inside the wafer dir.
#   - Writes full output to <root>/<name>/run.log.
#   - Echoes a single "PASS <name>" or "FAIL <name>" line to stdout.
run_one_wafer() {
  local root="$1"
  local name="$2"
  local dir="$root/$name"
  local cmd="$dir/commands_wse3.sh"

  if [[ ! -x "$cmd" ]]; then
    return 0  # silently skip dirs without commands_wse3.sh
  fi

  if ( cd "$dir" && ./commands_wse3.sh >"$dir/run.log" 2>&1 && grep -q "SUCCESS!" "$dir/run.log" ); then
    echo "PASS $name"
  else
    echo "FAIL $name"
  fi
}

export -f run_one_wafer

# Build a stable ordered list of wafer directory names.
mapfile -t wafer_names < <(
  find "$ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort
)

if [[ ${#wafer_names[@]} -eq 0 ]]; then
  echo "error: no subdirectories found under $ROOT" >&2
  exit 2
fi

any_run=0
for name in "${wafer_names[@]}"; do
  [[ -x "$ROOT/$name/commands_wse3.sh" ]] && any_run=1 && break
done

if [[ $any_run -eq 0 ]]; then
  echo "error: no executable commands_wse3.sh found under $ROOT" >&2
  exit 2
fi

echo "Running ${#wafer_names[@]} wafer(s) with -j $JOBS ..."

# Run wafers in parallel; collect all output lines.
results=$(
  printf '%s\n' "${wafer_names[@]}" \
    | xargs -I{} -P "$JOBS" bash -c 'run_one_wafer "$@"' _ "$ROOT" {}
)

# Print each result line for visibility.
while IFS= read -r line; do
  echo "  $line"
done <<< "$results"

# Summarize.
pass_count=$(grep -cE "^PASS " <<< "$results" || true)
fail_count=$(grep -cE "^FAIL " <<< "$results" || true)

echo ""
echo "=== Summary: $pass_count PASS, $fail_count FAIL ==="

if [[ $fail_count -gt 0 ]]; then
  echo "Failed wafers:"
  grep -E "^FAIL " <<< "$results" | awk '{print "  " $2}'
  exit 1
fi

exit 0
