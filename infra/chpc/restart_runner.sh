#!/usr/bin/env bash
# restart_runner.sh — called by crontab after login node reboots.
# Starts a new tmux session and runs the runner agent.

RUNNER_DIR="$HOME/actions-runner"

# If session already exists, do nothing
tmux has-session -t gh-runner 2>/dev/null && exit 0

tmux new-session -d -s gh-runner \
  "cd ${RUNNER_DIR} && ./run.sh; read -p 'Runner exited. Press enter.'"
