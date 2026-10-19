#!/bin/bash
# Run several seeds of no-squares-launcher.sh on a machine without OAR.
# Default: one run after the other (safe with a single GPU).
# PARALLEL=true starts them all at once (only if GPU memory and CPUs allow it).
# Arguments are passed through to the launcher; "--seed <s>" is appended.
# Each run writes to logs/local_<timestamp>_seed<s>.out / .err.
# Examples:
#    ./run-seeds-local.sh 30
#    ./run-seeds-local.sh 30 --max_epochs 52
#    PARALLEL=true NUM_SEEDS=4 ./run-seeds-local.sh 30
# Long runs: start it inside tmux/screen, or with
#    nohup ./run-seeds-local.sh 30 > logs/seeds.out 2>&1 &

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p logs

NUM_SEEDS="${NUM_SEEDS:-10}"
FIRST_SEED="${FIRST_SEED:-0}"
PARALLEL="${PARALLEL:-false}"
STAMP="$(date +%Y_%m_%d_%H_%M_%S)"

FAILED=()
PIDS=()
PID_SEEDS=()

for ((s = FIRST_SEED; s < FIRST_SEED + NUM_SEEDS; s++)); do
  LOG="logs/local_${STAMP}_seed${s}"
  echo "$(date +%H:%M:%S) seed $s -> $LOG.out / $LOG.err"
  if [ "$PARALLEL" = true ]; then
    ./no-squares-launcher.sh "$@" --seed "$s" > "$LOG.out" 2> "$LOG.err" &
    PIDS+=("$!")
    PID_SEEDS+=("$s")
  else
    if ! ./no-squares-launcher.sh "$@" --seed "$s" > >(tee "$LOG.out") 2> >(tee "$LOG.err" >&2); then
      echo "  seed $s failed, see $LOG.err" >&2
      FAILED+=("$s")
    fi
  fi
done

if [ "$PARALLEL" = true ]; then
  for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
      echo "  seed ${PID_SEEDS[$i]} failed" >&2
      FAILED+=("${PID_SEEDS[$i]}")
    fi
  done
fi

echo "$(date +%H:%M:%S) done: $NUM_SEEDS runs, ${#FAILED[@]} failed${FAILED[*]:+ (seeds: ${FAILED[*]})}"
[ "${#FAILED[@]}" -eq 0 ]
