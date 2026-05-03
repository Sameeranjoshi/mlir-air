#!/usr/bin/env bash
# update-ce.sh — pull the latest air-opt/air-translate from the GitHub CI
# nightly release and hot-swap the Compiler Explorer instance.
#
# CE spawns the binary fresh per compilation request, so no CE restart is
# needed — the swap is instant from the user's perspective.
#
# Usage:
#   bash utils/update-ce.sh              # update from latest nightly release
#   bash utils/update-ce.sh --branch foo # update from a specific branch tag
#   bash utils/update-ce.sh --artifact <run-id>  # from a specific workflow run

set -euo pipefail

REPO="Sameeranjoshi/mlir-air"
INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/install/bin"
BRANCH="${2:-air-to-fire}"
TMPDIR_DL=$(mktemp -d)
trap 'rm -rf "$TMPDIR_DL"' EXIT

case "${1:-}" in
  --artifact)
    RUN_ID="$2"
    echo "=== Downloading artifact from run $RUN_ID ==="
    gh run download "$RUN_ID" \
      --repo "$REPO" \
      --name air-tools-linux-x64 \
      --dir "$TMPDIR_DL"
    ;;
  --branch)
    BRANCH="${2:-air-to-fire}"
    TAG="nightly-${BRANCH}"
    echo "=== Downloading latest release for branch: $BRANCH ==="
    gh release download "$TAG" \
      --repo "$REPO" \
      --pattern "air-tools-linux-x64.tar.gz" \
      --dir "$TMPDIR_DL"
    ;;
  *)
    # Default: latest nightly release for air-to-fire (or main when merged)
    TAG="nightly-${BRANCH}"
    echo "=== Downloading latest nightly release ($TAG) ==="
    gh release download "$TAG" \
      --repo "$REPO" \
      --pattern "air-tools-linux-x64.tar.gz" \
      --dir "$TMPDIR_DL" 2>/dev/null || {
        echo "Release tag $TAG not found — falling back to latest release"
        gh release download \
          --repo "$REPO" \
          --pattern "air-tools-linux-x64.tar.gz" \
          --dir "$TMPDIR_DL"
      }
    ;;
esac

TAR=$(find "$TMPDIR_DL" -name "*.tar.gz" | head -1)
[[ -z "$TAR" ]] && { echo "ERROR: no tarball found in download" >&2; exit 1; }

echo "=== Extracting to $INSTALL_DIR ==="
mkdir -p "$INSTALL_DIR"
tar -xzf "$TAR" -C "$INSTALL_DIR"
chmod +x "$INSTALL_DIR"/air-opt "$INSTALL_DIR"/air-translate "$INSTALL_DIR"/air-runner

echo ""
echo "Updated binaries:"
ls -lh "$INSTALL_DIR"/air-opt "$INSTALL_DIR"/air-translate "$INSTALL_DIR"/air-runner

echo ""
echo "CE is hot-swapped — next compilation in the browser uses the new binary."
echo "CE URL (via SSH tunnel): http://localhost:10245"
