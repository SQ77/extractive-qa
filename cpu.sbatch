#!/bin/bash

# Generic CPU-only Slurm job for the non-training scripts. Arguments after cpu.sbatch are the
# script and its arguments. Submit from the repository root, e.g.:
#   sbatch --job-name=preprocess cpu.sbatch preprocess_newsqa.py
#   sbatch --job-name=bootstrap  cpu.sbatch bootstrap_significance.py
#   sbatch --job-name=bootstrap  cpu.sbatch bootstrap_significance.py --pred_dir outputs_train
#
# preprocess_newsqa.py downloads the raw data, so compute nodes need internet access. If they
# don't, run it on a login/interactive node instead, or copy the .jsonl.gz files in by hand.
# Use a distinct --job-name per run: it names the log file in logs/slurm/.

#SBATCH --job-name=qa-cpu
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8000

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

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

python "$@"
