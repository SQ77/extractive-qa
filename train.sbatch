#!/bin/bash

# Generic Slurm job for training (train.py). Arguments after train.sbatch are forwarded to train.py.
# Submit from the repository root, e.g.:
#   sbatch --job-name=train-baseline train.sbatch --mode baseline
#   sbatch --job-name=train-newsqa   train.sbatch --mode newsqa  --base_model_dir qa_model
#   sbatch --job-name=train-distill  train.sbatch --mode distill --base_model_dir qa_model
#   sbatch --job-name=train-joint    train.sbatch --mode joint
#
# newsqa/distill start from the baseline checkpoint, so submit them after train-baseline finishes
# (or chain with: sbatch --dependency=afterok:<jobid> ...).
# Use a distinct --job-name per run: it names the log file in logs/slurm/.

#SBATCH --job-name=qa-train
#SBATCH --partition=gpu-long
# Joint training (~97k questions, 2 epochs) is the longest run; override with --time if needed.
#SBATCH --time=04:00:00
#SBATCH --gpus=h100-47:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=20000

set -euo pipefail

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-qa}"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
LOG_DIR="$REPO_ROOT/logs/slurm"
mkdir -p "$LOG_DIR"
exec > "$LOG_DIR/${SLURM_JOB_NAME:-job}-${SLURM_JOB_ID:-$$}.out" 2>&1

cd "$REPO_ROOT"

set +u
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u

# Keep HF caches in the repo-independent default (~/.cache); unbuffered so logs stream live.
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

python train.py "$@"
