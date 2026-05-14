#!/usr/bin/env bash
# Bootstrap validation: copy the golden files to a temp dir,
# compile with cslc, and run on CS-3.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP="$(mktemp -d)"
cp "$HERE/vecadd_layout.csl.golden" "$TMP/vecadd_layout.csl"
cp "$HERE/vecadd_pe.csl.golden"     "$TMP/vecadd_pe.csl"
cp "$HERE/run.py.golden"            "$TMP/run.py"
cd "$TMP"
cslc --arch=wse3 ./vecadd_layout.csl --fabric-dims=8,3 \
     --fabric-offsets=4,1 -o out --memcpy --channels 1
cs_python run.py --name out --check
