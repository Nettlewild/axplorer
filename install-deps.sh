#!/bin/bash
# Installa nel venv le librerie Python usate dalla repo (pip).
# Uso:  ./install-deps.sh            (venv di default: ~/patternboost-env)
#       VENV=/altro/percorso ./install-deps.sh
set -euo pipefail

VENV="${VENV:-$HOME/patternboost-env}"

# crea il venv se non esiste
[ -d "$VENV" ] || python3 -m venv "$VENV"
source "$VENV/bin/activate"

python -m pip install --upgrade pip

# training (train.py, src/)
python -m pip --no-cache-dir install torch numpy numba psutil

# plotting (plot_experiments.py, tools/plot_score_histograms.py)
python -m pip install matplotlib pandas seaborn

# notebook new_envs.ipynb (facoltativo)
# python -m pip install jupyterlab

python -c "import torch, numpy, numba, psutil; print('torch', torch.__version__, '| CUDA build:', torch.version.cuda)"
