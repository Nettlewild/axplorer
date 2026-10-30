#!/bin/bash

#OAR -q production
#OAR -p musa
#OAR -l /host=1/gpu=1,walltime=96:00:00
#OAR -O logs/%jobid%.out
#OAR -E logs/%jobid%.err

# Launcher for the "no squares" problem (4-cycle-free graphs, env "square").
# Override N when submitting, and optionally pass any additional train.py
# arguments after it. Examples:
#    oarsub -S "./no-squares-launcher.sh 30"
#    oarsub -S "./no-squares-launcher.sh 30 --max_epochs 52"
#    oarsub -S "./no-squares-launcher.sh --max_epochs 52"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p logs

export PYTHONPATH="$(dirname "$SCRIPT_DIR"):${PYTHONPATH:-}"

# The Python code parallelizes scoring/local search with process workers.
# Keep numerical libraries from spawning extra threads inside each worker.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

POSITIONAL_ARGS=()
EXTRA_ARGS=()
while [ "$#" -gt 0 ]; do
  if [ "${#POSITIONAL_ARGS[@]}" -lt 1 ] && [[ "$1" != --* ]]; then
    POSITIONAL_ARGS+=("$1")
    shift
  else
    EXTRA_ARGS=("$@")
    break
  fi
done

# PROBLEM'S PARAMETERS
ENV_NAME="square"
N="${N:-${POSITIONAL_ARGS[0]:-33}}" #OK

#ENCODING_TOKENS="${ENCODING_TOKENS:-single_integer}"
ENCODING_TOKENS="${ENCODING_TOKENS:-adjacency}"
#POW2BASE="${POW2BASE:-1}"
POW2BASE="${POW2BASE:-4}"
MAKE_OBJECT_CANONICAL="${MAKE_OBJECT_CANONICAL:-false}"

#Per essere vicini a patternboost originale,
#mi sembra che possa andar bene tener ei valori di default
#GENSIZE="${GENSIZE:-5000}"
#POP_SIZE="${POP_SIZE:-1000}"
# NTEST="${NTEST:-100}"
# MAX_EPOCHS="${MAX_EPOCHS:-250}"
# MAX_STEPS="${MAX_STEPS:-100}"
# NUM_EVAL_STEPS="${NUM_EVAL_STEPS:-100}"
# NUM_SAMPLES_FROM_MODEL="${NUM_SAMPLES_FROM_MODEL:-10000}"
# GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-1000}"
NUM_WORKERS="${NUM_WORKERS:-16}"
# BATCH_SIZE="${BATCH_SIZE:-32}"
# N_LAYER="${N_LAYER:-4}"
# N_EMBD="${N_EMBD:-256}"
# Il paper originale usa la temeperatura?
# Si, ma forse non consentiva l'aumento dinamico?
# TEMPERATURE="${TEMPERATURE:-0.8}"
# TEMP_SPAN="${TEMP_SPAN:-6}"

# ENVIRONMENTAL PARAMETERS: what do they do?
# --- 13/10/2026 ---
AUGMENT_DATA_REPRESENTATION="${AUGMENT_DATA_REPRESENTATION:-false}"
NO_POSITIONAL="${NO_POSITIONAL:-false}"

# Default max_len = tokens needed to encode a graph + BOS/EOS.
# Sparse encodings use one token (single_integer) or two tokens (sequence_k_tokens)
# per edge, bounded by Reiman's bound ex(N, C4) <= N/4 * (1 + sqrt(4N - 3)).
# The dense encoding uses one token per pow2base entries of the upper triangle.
MAX_EDGES=$(awk -v n="$N" 'BEGIN { b = n / 4 * (1 + sqrt(4 * n - 3)); print (b == int(b)) ? b : int(b) + 1 }')
case "$ENCODING_TOKENS" in
  single_integer)    DEFAULT_MAX_LEN=$(( 2 + MAX_EDGES )) ;;
  sequence_k_tokens) DEFAULT_MAX_LEN=$(( 2 + 2 * MAX_EDGES )) ;;
  adjacency)         DEFAULT_MAX_LEN=$(( 2 + (N * (N - 1) / 2 + POW2BASE - 1) / POW2BASE )) ;;
  *) echo "Unknown ENCODING_TOKENS: $ENCODING_TOKENS" >&2; exit 1 ;;
esac
MAX_LEN="${MAX_LEN:-$DEFAULT_MAX_LEN}"
EXP_NAME="${ENV_NAME}_N${N}"

ENV_PREFIX="$HOME/micromamba/envs/env_axplorer"

# Unique run id, so that parallel runs never share a dump folder
# (checkpoint/<exp_name>/<exp_id>). OAR_JOB_ID is unique per job; outside OAR
# fall back to timestamp + PID + random suffix. The seed, if given via --seed,
# is included for readability. Override with EXP_ID=... or --exp_id ...
SEED_TAG=""
ARG_EXP_ID=""
DUMP_PATH="checkpoint"
for ((i = 0; i < ${#EXTRA_ARGS[@]}; i++)); do
  case "${EXTRA_ARGS[$i]}" in
    --seed)        SEED_TAG="seed${EXTRA_ARGS[$((i + 1))]}_" ;;
    --seed=*)      SEED_TAG="seed${EXTRA_ARGS[$i]#--seed=}_" ;;
    --exp_id)      ARG_EXP_ID="${EXTRA_ARGS[$((i + 1))]}" ;;
    --exp_id=*)    ARG_EXP_ID="${EXTRA_ARGS[$i]#--exp_id=}" ;;
    --dump_path)   DUMP_PATH="${EXTRA_ARGS[$((i + 1))]}" ;;
    --dump_path=*) DUMP_PATH="${EXTRA_ARGS[$i]#--dump_path=}" ;;
  esac
done
# if [ -n "${OAR_JOB_ID:-}" ]; then
#   DEFAULT_EXP_ID="${SEED_TAG}job${OAR_JOB_ID}"
# else
DEFAULT_EXP_ID="${SEED_TAG}$(date +%Y_%m_%d_%H_%M_%S)_$$_${RANDOM}"
# fi
EXP_ID="${EXP_ID:-$DEFAULT_EXP_ID}"


echo "No squares run:"
echo "  env_name=$ENV_NAME N=$N encoding_tokens=$ENCODING_TOKENS pow2base=$POW2BASE max_len=$MAX_LEN"
echo "  exp_name=$EXP_NAME exp_id=$EXP_ID"
echo "Running on $(hostname)"
echo "nproc: $(nproc)"
echo "CPU affinity:"
grep Cpus_allowed_list /proc/self/status
echo "Python CPU count / affinity:"
python -c "import os; print('os.cpu_count =', os.cpu_count()); print('affinity =', len(os.sched_getaffinity(0)))"
if [ -f /sys/fs/cgroup/cpu.max ]; then
  cat /sys/fs/cgroup/cpu.max
fi
# echo "OAR nodefile: ${OAR_NODEFILE:-<unset>}"
# if [ -n "${OAR_NODEFILE:-}" ]; then
#   sort "$OAR_NODEFILE" | uniq -c
# fi
nvidia-smi

# grid5000
#LD_LIBRARY_PATH="$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}" \
#"$HOME/.local/bin/micromamba" run -n "env_axplorer" python -c "import torch; print('cuda:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"

#which python
#python -c "import torch; print('cuda:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"

# clai
conda run -n patternboost python -c "import torch; print('cuda:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"

# sofia
source "/home/angileri/flora/bin/patternboost/bin/activate"
python -c "import torch; print('cuda:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"

CMD=(
  python train.py
  --env_name "$ENV_NAME"
  --exp_name "$EXP_NAME"
  --exp_id "$EXP_ID"
  --N "$N"
  --encoding_tokens "$ENCODING_TOKENS"
  --pow2base "$POW2BASE"
  --make_object_canonical "$MAKE_OBJECT_CANONICAL"
  --max_len "$MAX_LEN"
  # --gensize "$GENSIZE"
  # --pop_size "$POP_SIZE"
  # --ntest "$NTEST"
  # --max_epochs "$MAX_EPOCHS"
  # --max_steps "$MAX_STEPS"
  # --num_eval_steps "$NUM_EVAL_STEPS"
  # --num_samples_from_model "$NUM_SAMPLES_FROM_MODEL"
  # --gen_batch_size "$GEN_BATCH_SIZE"
  # --batch_size "$BATCH_SIZE"
  # --n_layer "$N_LAYER"
  # --n_embd "$N_EMBD"
  --augment_data_representation "$AUGMENT_DATA_REPRESENTATION"
  # --temperature "$TEMPERATURE"
  # --temp_span "$TEMP_SPAN"
  --no_positional "$NO_POSITIONAL"
  --num_workers "$NUM_WORKERS"
)

if [ -n "${TRAIN_EXTRA_ARGS:-}" ]; then
  read -r -a ENV_EXTRA_ARGS <<< "$TRAIN_EXTRA_ARGS"
  CMD+=("${ENV_EXTRA_ARGS[@]}")
fi

CMD+=("${EXTRA_ARGS[@]}")

printf 'Training command:'
printf ' %q' "${CMD[@]}"
printf '\n'


run_training() {
  # grid5000
  #"${CMD[@]}"
  #LD_LIBRARY_PATH="$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}" \
  #"$HOME/.local/bin/micromamba" run -n "env_axplorer" "${CMD[@]}"
  # clai
  #conda run --no-capture-output -n patternboost "${CMD[@]}"
  # sofia
  "${CMD[@]}"
}

# Automatic restart after an out-of-memory crash (CUDA OOM, MemoryError, or
# process killed by the kernel/cgroup OOM killer, exit code 137).
# The command is relaunched unchanged, with the same exp_id: train.py then
# resumes from model.pt / optimizer.pt / train_data.pkl / epoch.txt of the run
# folder. The run is abandoned after MAX_OOM_RESTARTS consecutive OOM crashes
# without any epoch completed in between (the counter resets on progress).
MAX_OOM_RESTARTS="${MAX_OOM_RESTARTS:-5}"
RUN_DIR="$DUMP_PATH/$EXP_NAME/${ARG_EXP_ID:-$EXP_ID}"
EPOCH_FILE="$RUN_DIR/epoch.txt"
OOM_PATTERN='out of memory|OutOfMemoryError|MemoryError|CUBLAS_STATUS_ALLOC_FAILED|Cannot allocate memory'

read_epoch() { cat "$EPOCH_FILE" 2>/dev/null || echo "none"; }

attempt=0
oom_streak=0
while true; do
  attempt=$((attempt + 1))
  ATTEMPT_LOG="logs/${EXP_NAME}_${ARG_EXP_ID:-$EXP_ID}_attempt${attempt}.log"
  epoch_before="$(read_epoch)"
  echo "[launcher] $(date '+%F %T') attempt $attempt (epoch.txt: $epoch_before), output also in $ATTEMPT_LOG"

  run_training 2>&1 | tee "$ATTEMPT_LOG"
  status=${PIPESTATUS[0]}

  if [ "$status" -eq 0 ]; then
    echo "[launcher] $(date '+%F %T') training finished (exit code 0)"
    break
  fi

  if [ "$status" -ne 137 ] && ! tail -n 200 "$ATTEMPT_LOG" | grep -qE "$OOM_PATTERN"; then
    echo "[launcher] $(date '+%F %T') training failed with exit code $status (not an out-of-memory error): no restart"
    exit "$status"
  fi

  epoch_after="$(read_epoch)"
  if [ "$epoch_after" != "$epoch_before" ]; then
    oom_streak=1
  else
    oom_streak=$((oom_streak + 1))
  fi
  echo "[launcher] $(date '+%F %T') OUT OF MEMORY (exit code $status) at epoch.txt=$epoch_after; consecutive OOMs without progress: $oom_streak/$MAX_OOM_RESTARTS"
  if [ "$oom_streak" -ge "$MAX_OOM_RESTARTS" ]; then
    echo "[launcher] too many OOM crashes without progress: giving up"
    exit "$status"
  fi
  if [ ! -f "$RUN_DIR/model.pt" ]; then
    echo "[launcher] no model.pt yet in $RUN_DIR: the run restarts from scratch"
  fi
  nvidia-smi
  sleep 30
done
