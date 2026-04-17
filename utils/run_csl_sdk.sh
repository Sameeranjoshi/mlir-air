#!/usr/bin/env bash
# Usage: run_csl_sdk.sh <emission-root>
#
# Iterates every subdirectory of <emission-root> that contains an executable
# commands_wse3.sh and runs it. The emission root is the directory passed to
# `air-translate --emit-csl --output-dir=<dir>`, which contains one
# subdirectory per csl.wafer.
#
# Requires: cslc, cs_python on PATH.
#
# Exit status: 0 if every wafer ran to "SUCCESS!", 1 otherwise.
set -u

ROOT="${1:?usage: run_csl_sdk.sh <emission-root>}"
if [[ ! -d "$ROOT" ]]; then
  echo "error: not a directory: $ROOT" >&2
  exit 2
fi

any_fail=0
any_run=0
for dir in "$ROOT"/*/; do
  [[ -d "$dir" ]] || continue
  name="$(basename "$dir")"
  cmd="$dir/commands_wse3.sh"
  if [[ -x "$cmd" ]]; then
    any_run=1
    echo "=== $name ==="
    if ( cd "$dir" && ./commands_wse3.sh 2>&1 | tee run.log | grep -q "SUCCESS!" ); then
      echo "  PASS"
    else
      echo "  FAIL (see $dir/run.log)"
      any_fail=1
    fi
  fi
done

if [[ $any_run -eq 0 ]]; then
  echo "error: no executable commands_wse3.sh found under $ROOT" >&2
  exit 2
fi

exit $any_fail
