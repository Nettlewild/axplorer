#!/bin/bash
# Submit one OAR job per seed (they run in parallel, each with its own GPU,
# walltime and log, as set by the #OAR lines in no-squares-launcher.sh).
# Arguments are passed through to the launcher; "--seed <s>" is appended.
# Examples:
#    ./submit-seeds-oar.sh 30
#    ./submit-seeds-oar.sh 30 --max_epochs 52
#    NUM_SEEDS=5 FIRST_SEED=100 ./submit-seeds-oar.sh 30

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p logs

NUM_SEEDS="${NUM_SEEDS:-10}"
FIRST_SEED="${FIRST_SEED:-0}"

if ! command -v oarsub >/dev/null 2>&1; then
  echo "oarsub not found: use run-seeds-local.sh on machines without OAR." >&2
  exit 1
fi

for ((s = FIRST_SEED; s < FIRST_SEED + NUM_SEEDS; s++)); do
  JOB_CMD=$(printf '%q ' ./no-squares-launcher.sh "$@" --seed "$s")
  echo "Seed $s: oarsub -S \"$JOB_CMD\""
  oarsub -S "$JOB_CMD" || echo "  submission failed for seed $s" >&2
done
