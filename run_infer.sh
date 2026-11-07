#!/bin/bash
# Pipeline script called by job.sbatch: run qa_model on both dev sets.
# Extra args passed to sbatch are forwarded here and on to infer.py.
set -euo pipefail

python infer.py \
    --model_dir qa_model \
    --dev_files qa/dev-v1.1.json qa/dev-newsqa.json \
    --out_dir outputs \
    "$@"
