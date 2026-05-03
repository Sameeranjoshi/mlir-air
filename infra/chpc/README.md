# CHPC Self-Hosted CI — Future Plan

## Current approach (simple, working)

Tests run locally via the pre-push hook before every `git push` (~36s).

```bash
# Re-enable the hook (already done):
git config core.hooksPath .githooks

# Run manually anytime:
bash utils/run_csl_ci.sh --no-build -j 8

# Bypass for docs/WIP pushes:
git push --no-verify
```

## When to switch to CHPC CI

When the test suite gets slow enough that a 36s local block is annoying,
or when you want CI to run on PRs from collaborators.

## Setup (one-time, ~10 min)

1. SSH into CHPC login node (2FA approval on phone — one-time per session):
   ```bash
   ssh yournetid@lonepeak.chpc.utah.edu
   ```

2. Copy the Cerebras SDK to CHPC:
   ```bash
   # From this machine:
   rsync -avz /home/bricklib_dataflow/sdk/SDK_1_4/ \
     yournetid@lonepeak.chpc.utah.edu:~/cerebras-sdk/SDK_1_4/
   ```

3. On CHPC, run the one-time setup:
   ```bash
   git clone https://github.com/Sameeranjoshi/mlir-air
   cd mlir-air
   bash infra/chpc/setup_runner.sh
   # Paste the runner token from:
   # github.com/Sameeranjoshi/mlir-air → Settings → Actions → Runners → New
   ```

4. Done. Every push triggers:
   - `csl-lit` job on GitHub cloud (Tier 1, no SDK needed)
   - `csl-simulator` job on CHPC (Tier 2, full SDK simulator)

## How it works

```
git push
  → GitHub webhook
    → runner agent (login node, tmux session "gh-runner")
      → infra/chpc/run_on_chpc.sh
        → find_free_cpu.sh   (picks best free partition/account/qos)
        → srun               (compute node, 8 cores, 20 min limit)
          → utils/run_csl_ci.sh --no-build -j 8
```

- Runner agent is a lightweight HTTP polling loop — no login node abuse.
- `find_free_cpu.sh` adapted from the free_gpus.sh pattern.
- `[skip-sim]` in a commit message skips the simulator job.
- Crontab auto-restarts the runner after login node reboots.

## Files

| File | Purpose |
|------|---------|
| `find_free_cpu.sh` | Finds free CPU slots across CHPC clusters |
| `run_on_chpc.sh` | Discovers slot, fires srun |
| `setup_runner.sh` | One-time runner registration + tmux start |
| `restart_runner.sh` | Crontab target for post-reboot restarts |
| `../../.github/workflows/csl-ci.yml` | Workflow with both jobs |
