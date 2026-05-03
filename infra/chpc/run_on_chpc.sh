#!/usr/bin/env bash
# run_on_chpc.sh — discover a free CPU slot and run a command on it via srun.
#
# Usage (called by GitHub Actions workflow):
#   bash infra/chpc/run_on_chpc.sh <command> [args...]
#
# Environment overrides (set in runner .env or exported before calling):
#   CHPC_CLUSTER    force a specific cluster   (e.g. notchpeak)
#   CHPC_PARTITION  force a specific partition
#   CHPC_ACCOUNT    force a specific account
#   CHPC_QOS        force a specific qos
#   CHPC_CPUS       CPUs to request            (default: 8)
#   CHPC_MEM        Memory to request          (default: 16G)
#   CHPC_TIME       Wall-time limit            (default: 00:20:00)
#   SDK_PATH        Path to Cerebras SDK dir   (default: ~/cerebras-sdk/SDK_1_4)

set -euo pipefail

CPUS="${CHPC_CPUS:-8}"
MEM="${CHPC_MEM:-16G}"
TIME="${CHPC_TIME:-00:20:00}"
SDK_PATH="${SDK_PATH:-$HOME/cerebras-sdk/SDK_1_4}"

# ── discover resources unless explicitly overridden ────────────────────────────
if [[ -n "${CHPC_CLUSTER:-}" && -n "${CHPC_PARTITION:-}" && \
      -n "${CHPC_ACCOUNT:-}" ]]; then
  CLUSTER="$CHPC_CLUSTER"
  PARTITION="$CHPC_PARTITION"
  ACCOUNT="$CHPC_ACCOUNT"
  QOS="${CHPC_QOS:-}"
  echo "[run_on_chpc] Using forced allocation: $CLUSTER / $PARTITION / $ACCOUNT / ${QOS:-<no-qos>}"
else
  echo "[run_on_chpc] Discovering free CPU slot (min ${CPUS} cores)..."

  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  BEST=$(bash "$SCRIPT_DIR/find_free_cpu.sh" "$CPUS" 2>/dev/null \
         | grep -v '^$' | grep -v '^-' | grep -v '^CLUSTER' \
         | grep -v '^To ' | grep -v '^  srun' \
         | head -1)

  if [[ -z "$BEST" ]]; then
    echo "[run_on_chpc] No free CPU slot found. Falling back to default allocation." >&2
    # Let SLURM queue — use first available partition from mychpc batch
    BEST=$(mychpc batch 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' \
           | grep -i 'partition' | grep -oP -- '--partition=\S+ --account=\S+( --qos=\S+)?' \
           | head -1)
    [[ -z "$BEST" ]] && { echo "[run_on_chpc] ERROR: cannot determine any allocation." >&2; exit 1; }
  fi

  CLUSTER=$(echo "$BEST" | awk '{print $1}')
  PARTITION=$(echo "$BEST" | awk '{print $2}')
  ACCOUNT=$(echo "$BEST"  | awk '{print $6}')
  QOS=$(echo "$BEST"      | awk '{print $7}')

  echo "[run_on_chpc] Selected: cluster=$CLUSTER partition=$PARTITION account=$ACCOUNT qos=${QOS:-<none>}"
fi

# ── build srun args ────────────────────────────────────────────────────────────
SRUN_ARGS=(
  --cluster="$CLUSTER"
  --partition="$PARTITION"
  --account="$ACCOUNT"
  --ntasks=1
  --cpus-per-task="$CPUS"
  --mem="$MEM"
  --time="$TIME"
)
[[ -n "${QOS:-}" && "$QOS" != "?" ]] && SRUN_ARGS+=(--qos="$QOS")

# ── inject SDK into PATH for the compute node ──────────────────────────────────
export PATH="$SDK_PATH:$PATH"

echo "[run_on_chpc] srun ${SRUN_ARGS[*]} -- $*"
exec srun "${SRUN_ARGS[@]}" -- "$@"
