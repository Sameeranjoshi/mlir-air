#!/usr/bin/env bash
# find_free_cpu.sh — list CPU slots available on CHPC partitions you can access.
# Adapted from free_gpus.sh for CPU-only workloads (no gres needed).
#
# Usage:
#   ./find_free_cpu.sh [min_cores]      # default: 8
#
# Output (one line per candidate, sorted by idle CPUs descending):
#   CLUSTER  PARTITION  NODE  IDLE/TOTAL_CPUS  TIMELIMIT  ACCOUNT  QOS
#
# The first output line is the "best" slot — use it directly in srun.

MIN_CORES="${1:-8}"

# ── parse your allocations: partition → account, qos ──────────────────────────
declare -A PART_ACCOUNT PART_QOS

while IFS= read -r line; do
  part=$(echo "$line" | grep -oP '(?<=--partition=)\S+')
  acct=$(echo "$line" | grep -oP '(?<=--account=)\S+')
  qos=$( echo "$line" | grep -oP '(?<=--qos=)\S+')
  [[ -n "$part" && -n "$acct" ]] && PART_ACCOUNT["$part"]="$acct" && PART_QOS["$part"]="$qos"
# include all partitions (not just GPU ones)
done < <(mychpc batch 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | grep -i 'partition')

[[ ${#PART_ACCOUNT[@]} -eq 0 ]] && { echo "No partitions found in your allocations." >&2; exit 1; }

# ── header ─────────────────────────────────────────────────────────────────────
printf "\n%-12s %-28s %-12s %-16s %-12s %-20s %s\n" \
  "CLUSTER" "PARTITION" "NODE" "IDLE/TOTAL" "TIMELIMIT" "ACCOUNT" "QOS"
printf '%0.s-' {1..115}; echo

# ── query each cluster ──────────────────────────────────────────────────────────
# sinfo -o "%C" gives CPUs in A/I/O/T (allocated/idle/other/total) format per node
RESULTS=""

for cluster in granite notchpeak kingspeak lonepeak; do
  while IFS='|' read -r node cpus state part timelimit; do
    [[ -z "${PART_ACCOUNT[$part]}" ]] && continue

    # skip GPU-only partitions (let the GPU script handle those)
    echo "$part" | grep -qi 'gpu' && continue

    # %C gives "alloc/idle/other/total" — extract idle and total
    idle=$(echo "$cpus"  | awk -F'/' '{print $2}')
    total=$(echo "$cpus" | awk -F'/' '{print $4}')

    [[ -z "$idle" || -z "$total" ]] && continue
    [[ "$idle" -lt "$MIN_CORES" ]] && continue

    acct="${PART_ACCOUNT[$part]:-?}"
    qos="${PART_QOS[$part]:-?}"

    # collect for sorting
    RESULTS+="$cluster|$part|$node|${idle}/${total}|$timelimit|$acct|$qos|$idle\n"

  done < <(sinfo -M "$cluster" -h -N \
              -o "%N|%C|%T|%P|%l" 2>/dev/null \
           | grep -v "^$" \
           | sort -u \
           | awk -F'|' '$3 != "allocated" && $3 != "alloc" && \
                         $3 != "down"      && $3 != "drain" && \
                         $3 != "maint"     && $3 != "comp"')
done

# sort by idle CPUs descending, print
echo -e "$RESULTS" | sort -t'|' -k8 -rn | while IFS='|' read -r cluster part node cpus timelimit acct qos _idle; do
  [[ -z "$cluster" ]] && continue
  printf "%-12s %-28s %-12s %-16s %-12s %-20s %s\n" \
    "$cluster" "$part" "$node" "$cpus" "$timelimit" "$acct" "$qos"
done

echo ""
echo "To allocate (interactive):"
echo "  srun --cluster=<CLUSTER> --partition=<PARTITION> --account=<ACCOUNT> --qos=<QOS> \\"
echo "       --ntasks=1 --cpus-per-task=${MIN_CORES} --time=00:15:00 --mem=16G --pty bash"
