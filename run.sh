#!/usr/bin/env bash
# One entry point for the whole project. Everything runs inside the b2txt25 conda env.
#
# Usage (from the repo root):
#   bash run.sh setup        # clone repo + create conda env (Mac / MPS version)
#   bash run.sh download     # unzip Dryad data downloaded in the browser (~11.6 GB) into data/
#   bash run.sh check        # verify env, GPU and data
#   bash run.sh eval-quick   # pretrained baseline PER on 2 sessions (smoke test)
#   bash run.sh eval         # pretrained baseline PER on the full val split
#   bash run.sh time         # time training on MPS and CPU, estimate full run
#   bash run.sh train NAME BATCHES [extra mac_train.py args]
#   bash run.sh all          # setup + download + check + eval-quick + eval
#
# Every run is logged to logs/<command>_<timestamp>.log. Long steps keep the Mac awake.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$ROOT/nejm-brain-to-text"
ENV=b2txt25
mkdir -p "$ROOT/logs" "$ROOT/results"

# ---------- conda ----------
# An active venv (e.g. VS Code's selected interpreter) puts its bin first on PATH and
# wins over `conda run`, so drop it before using conda.
if [ -n "${VIRTUAL_ENV:-}" ]; then
  PATH="$(echo "$PATH" | tr ':' '\n' | grep -vx "$VIRTUAL_ENV/bin" | paste -sd: -)"
  unset VIRTUAL_ENV
  export PATH
fi
if ! command -v conda >/dev/null 2>&1; then
  for c in "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3"; do
    if [ -x "$c/bin/conda" ]; then export PATH="$c/bin:$PATH"; break; fi
  done
fi
command -v conda >/dev/null 2>&1 || { echo "conda not found. Install miniforge first."; exit 1; }
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

in_env() { conda run --no-capture-output -n "$ENV" "$@"; }
awake() { if command -v caffeinate >/dev/null; then caffeinate -i "$@"; else "$@"; fi; }

log_to() {  # tee all output of the rest of the script to a log file
  local f="$ROOT/logs/$1_$(date +%Y%m%d_%H%M%S).log"
  exec > >(tee -a "$f") 2>&1
  echo "Logging to $f"
}

# ---------- steps ----------
do_setup() {
  if [ ! -d "$REPO/.git" ]; then
    git clone https://github.com/Neuroprosthetics-Lab/nejm-brain-to-text.git "$REPO"
  else
    echo "Repo already cloned."
  fi
  if [ -d "$ROOT/_to_delete" ]; then rm -rf "$ROOT/_to_delete"; fi

  if conda env list | awk '{print $1}' | grep -qx "$ENV"; then
    echo "Conda env $ENV already exists."
  else
    conda create -n "$ENV" python=3.10 -y
  fi
  in_env pip install --upgrade pip
  in_env pip install torch torchvision torchaudio
  in_env pip install numpy==2.1.2 pandas==2.3.0 matplotlib==3.10.1 scipy==1.15.2 \
    scikit-learn==1.6.1 tqdm==4.67.1 g2p_en==2.1.0 h5py==3.13.0 omegaconf==2.3.0 \
    editdistance==0.8.1 jupyter==1.1.1 redis==5.2.1
  in_env pip install -e "$REPO"
  echo "Setup done."
}

do_download() {
  if [ -d "$REPO/data/hdf5_data_final" ] && [ -d "$REPO/data/t15_pretrained_rnn_baseline" ]; then
    echo "Data already present, skipping download."
    return
  fi
  # Dryad's API now refuses scripted file downloads (HTTP 401), so download the two zips
  # in a browser and this step unzips them from data/ or ~/Downloads.
  # Safari auto-unzips, leaving folders in ~/Downloads instead: move those in.
  local moved=0
  for dir in hdf5_data_final t15_pretrained_rnn_baseline; do
    if [ -d "$HOME/Downloads/$dir" ] && [ ! -d "$REPO/data/$dir" ]; then
      echo "Moving ~/Downloads/$dir into $REPO/data/"
      mv "$HOME/Downloads/$dir" "$REPO/data/"
      moved=$((moved + 1))
    fi
  done
  if [ -d "$REPO/data/hdf5_data_final" ] && [ -d "$REPO/data/t15_pretrained_rnn_baseline" ]; then
    echo "Data ready ($moved folder(s) moved from ~/Downloads)."
    return
  fi

  local found=0
  for z in t15_copyTask_neuralData.zip t15_pretrained_rnn_baseline.zip; do
    local src=""
    for d in "$REPO/data" "$HOME/Downloads"; do
      if [ -f "$d/$z" ]; then src="$d/$z"; break; fi
    done
    if [ -z "$src" ]; then
      echo "MISSING: $z (not in $REPO/data or ~/Downloads)"
      continue
    fi
    found=$((found + 1))
    echo "Unzipping $src ..."
    awake unzip -q -n "$src" -d "$REPO/data"
  done
  for p in t15_copyTask.pkl t15_personalUse.pkl; do
    if [ -f "$HOME/Downloads/$p" ] && [ ! -f "$REPO/data/$p" ]; then cp "$HOME/Downloads/$p" "$REPO/data/"; fi
  done
  if [ "$found" -lt 2 ]; then
    cat <<EOF

Download these from the Dryad page in your browser (Files section):
  https://datadryad.org/dataset/doi:10.5061/dryad.dncjsxm85
    - t15_copyTask_neuralData.zip      (11.05 GB)  required
    - t15_pretrained_rnn_baseline.zip  (485 MB)    required
Leave them in ~/Downloads, then rerun: bash run.sh download
EOF
    return 1
  fi
  echo "Data ready. You can delete the two .zip files to free ~11.5 GB."
}

do_check() {
  in_env python -c "import torch, h5py, omegaconf, editdistance, torchaudio; \
print('torch', torch.__version__, '| MPS available:', torch.backends.mps.is_available())"
  local n=0
  if [ -d "$REPO/data/hdf5_data_final" ]; then n=$(ls "$REPO/data/hdf5_data_final" | wc -l | tr -d ' '); fi
  echo "Session folders: $n (expect 45)"
  [ -f "$REPO/data/t15_pretrained_rnn_baseline/checkpoint/best_checkpoint" ] \
    && echo "Pretrained baseline: found" || echo "Pretrained baseline: MISSING"
}

do_eval() {  # $1 = extra args
  (cd "$REPO/model_training" && awake conda run --no-capture-output -n "$ENV" \
    python "$ROOT/scripts/mac_greedy_eval.py" \
      --model_path ../data/t15_pretrained_rnn_baseline \
      --data_dir ../data/hdf5_data_final \
      --csv_path ../data/t15_copyTaskData_description.csv \
      --out "$ROOT/results/baseline_val_greedy$2.csv" $1)
}

do_time() {
  (cd "$REPO/model_training" && \
    awake conda run --no-capture-output -n "$ENV" python "$ROOT/scripts/time_training.py" \
      --device mps --batches 300 --out "$ROOT/results/timing.csv" && \
    awake conda run --no-capture-output -n "$ENV" python "$ROOT/scripts/time_training.py" \
      --device cpu --batches 100 --out "$ROOT/results/timing.csv")
}

do_train() {  # NAME BATCHES [extra]
  local name="$1" batches="$2"; shift 2
  (cd "$REPO/model_training" && awake conda run --no-capture-output -n "$ENV" \
    python "$ROOT/scripts/mac_train.py" --name "$name" --batches "$batches" \
      --runs_dir "$ROOT/results/runs" "$@")
}

# ---------- dispatch ----------
cmd="${1:-help}"; shift || true
case "$cmd" in
  setup)      log_to setup;      do_setup ;;
  download)   log_to download;   do_download ;;
  check)      log_to check;      do_check ;;
  eval-quick) log_to eval_quick; do_eval "--max_sessions 2" "_quick" ;;
  eval)       log_to eval;       do_eval "" "" ;;
  time)       log_to time;       do_time ;;
  train)      [ $# -ge 2 ] || { echo "Usage: bash run.sh train NAME BATCHES [args]"; exit 1; }
              log_to "train_$1"; do_train "$@" ;;
  all)        log_to all; do_setup; do_download; do_check; do_eval "--max_sessions 2" "_quick"; do_eval "" "" ;;
  *)          sed -n 2,14p "$0" ;;
esac
