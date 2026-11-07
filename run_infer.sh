#!/bin/bash
# Pipeline script called by job.sbatch: inference + evaluation for any model on any
# SQuAD-format dev file(s). All arguments are forwarded to infer.py.
#
# Example:
#   sbatch --job-name=infer-base job.sbatch \
#       --model_dir qa_model --dev_files qa/dev-v1.1.json qa/dev-newsqa.json
set -euo pipefail

python infer.py "$@"
