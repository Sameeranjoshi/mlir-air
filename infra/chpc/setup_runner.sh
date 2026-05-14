#!/usr/bin/env bash
# setup_runner.sh — one-time setup of GitHub Actions self-hosted runner on CHPC.
#
# Run this ONCE after SSH-ing into the CHPC login node with 2FA:
#   bash infra/chpc/setup_runner.sh
#
# What it does:
#   1. Downloads the GitHub Actions runner binary into ~/actions-runner/
#   2. Configures it with your repo token (prompts once)
#   3. Writes ~/actions-runner/.env  (SDK path, module setup)
#   4. Adds a crontab entry to restart the runner after login node reboots
#   5. Starts the runner inside a new tmux session called "gh-runner"
#
# Prerequisites on CHPC:
#   - tmux available (it is on all CHPC login nodes)
#   - The Cerebras SDK already copied to ~/cerebras-sdk/SDK_1_4/
#   - singularity loadable via  module load singularity

set -euo pipefail

RUNNER_DIR="$HOME/actions-runner"
WORK_DIR="/scratch/$USER/runner-work"
SDK_PATH="$HOME/cerebras-sdk/SDK_1_4"
RUNNER_VERSION="2.316.1"  # update to latest from github.com/actions/runner/releases

# ── 1. download runner ──────────────────────────────────────────────────────────
echo "=== Downloading GitHub Actions runner v${RUNNER_VERSION} ==="
mkdir -p "$RUNNER_DIR"
cd "$RUNNER_DIR"

RUNNER_TAR="actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz"
if [[ ! -f "$RUNNER_TAR" ]]; then
  curl -fsSL \
    "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/${RUNNER_TAR}" \
    -o "$RUNNER_TAR"
  tar xzf "$RUNNER_TAR"
fi
echo "Runner binary ready."

# ── 2. configure ───────────────────────────────────────────────────────────────
echo ""
echo "=== Configuring runner ==="
echo "Go to: https://github.com/Sameeranjoshi/mlir-air/settings/actions/runners/new"
echo "Select: Linux / x64"
echo "Copy the token shown (starts with A...) and paste below."
echo ""
read -rp "Repo runner token: " RUNNER_TOKEN

mkdir -p "$WORK_DIR"

./config.sh \
  --url "https://github.com/Sameeranjoshi/mlir-air" \
  --token "$RUNNER_TOKEN" \
  --name "chpc-$(hostname -s)" \
  --labels "self-hosted,chpc,linux" \
  --work "$WORK_DIR" \
  --unattended \
  --replace

# ── 3. write .env — injected into every job ────────────────────────────────────
echo ""
echo "=== Writing .env ==="
cat > "$RUNNER_DIR/.env" <<EOF
# Cerebras SDK — added to PATH so cslc/cs_python are found on compute nodes
SDK_PATH=${SDK_PATH}
PATH=${SDK_PATH}:\$PATH

# CHPC module system — sourced by run_csl_ci.sh internally, but needed here
# for the runner shell to find 'module' command
MODULESHOME=/uufs/chpc.utah.edu/sys/modulefiles
EOF
echo ".env written."

# ── 4. crontab entry for auto-restart after login node reboot ──────────────────
echo ""
echo "=== Adding crontab restart entry ==="
RESTART_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/restart_runner.sh"
CRON_LINE="@reboot sleep 60 && bash ${RESTART_SCRIPT}"

# add only if not already present
( crontab -l 2>/dev/null | grep -v 'restart_runner'; echo "$CRON_LINE" ) | crontab -
echo "Crontab updated."

# ── 5. start in tmux ───────────────────────────────────────────────────────────
echo ""
echo "=== Starting runner in tmux session 'gh-runner' ==="
tmux new-session -d -s gh-runner \
  "cd ${RUNNER_DIR} && ./run.sh; read -p 'Runner exited. Press enter to restart manually.'"

echo ""
echo "Done! Runner is live in tmux session 'gh-runner'."
echo "  Attach with:  tmux attach -t gh-runner"
echo "  Detach with:  Ctrl-B then D"
echo ""
echo "The runner will auto-restart after login node reboots (via crontab)."
