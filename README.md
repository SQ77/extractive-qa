# Diagnosing and Mitigating Catastrophic Forgetting in Cross-Domain Extractive QA

CS4248 Group Project (Group G05).

We fine-tune BERT-base (`bert-base-uncased`) on SQuAD v1.1, then adapt it to a NewsQA subset, and measure how much SQuAD performance is forgotten. We compare two mitigations against plain sequential fine-tuning:

- **Joint training**: train from scratch on SQuAD + NewsQA combined.
- **Distillation regularization**: a Learning-without-Forgetting-style KL penalty on start/end logits against a frozen SQuAD-only teacher.

Differences are tested with paired bootstrap significance tests.

## Results (BERT-base)

| Model | SQuAD EM | SQuAD F1 | NewsQA EM | NewsQA F1 |
|---|---|---|---|---|
| SQuAD-only baseline | 80.92 | 88.40 | 39.58 | 55.82 |
| Sequential fine-tune (SQuAD, then NewsQA) | 74.84 | 85.44 | 48.55 | 64.24 |
| Joint training | 81.12 | 88.58 | 45.96 | 61.81 |
| Distillation-regularized fine-tune | 80.08 | 88.06 | 45.66 | 61.47 |

SQuAD is evaluated on the full v1.1 dev set (10,570 questions). NewsQA is evaluated on a 1,003-question dev subset. See the `*_results.md` files for details and significance tests.

## Repository layout

| File | Purpose |
|---|---|
| `preprocess_newsqa.ipynb` | Builds the NewsQA subset in SQuAD format from the raw data |
| `train_qa.ipynb` | SQuAD v1.1 baseline fine-tuning and evaluation |
| `finetune_bert_on_newsqa.ipynb` | Sequential fine-tuning of the baseline on NewsQA |
| `finetune_bert_on_newsqa_distill.ipynb` | Sequential fine-tuning with the distillation regularizer |
| `train_joint_squad_newsqa.ipynb` | Joint training on SQuAD + NewsQA |
| `bootstrap_significance.ipynb` | Paired bootstrap significance tests on per-question scores |
| `infer.py` | Runs inference and EM/F1 evaluation of a checkpoint on SQuAD-format dev files |
| `run_infer.sh` | Wrapper that launches `infer.py` |
| `job.sbatch` | Generic Slurm job script for running inference on the SoC compute cluster |
| `qa/evaluate-v2.0.py` | Official SQuAD scoring script |
| `qa/predictions_*.json` | Saved dev-set predictions for every model (used by the bootstrap tests) |
| `bert_baseline_results.md`, `newsqa_finetune_results.md`, `distillation_results.md`, `joint_training_results.md` | Per-experiment write-ups |
| `outputs/` | Predictions written by cluster inference jobs (created when `job.sbatch` runs) |
| `Soc_cluster/README.md` | SoC Compute Cluster unofficial guide |

## Reproducing

There are two ways to reproduce the results.

### Option 1: Google Drive (everything included)

The shared Google Drive folder contains every file in this repo plus all the datasets (SQuAD, NewsQA) and trained model checkpoints, so nothing needs to be downloaded or rebuilt separately.

Drive link: `<add link here>`

### Option 2: This repo

This repo contains the Colab notebooks (for training) and the scripts for running inference on the NUS SoC compute cluster. Data and model checkpoints are **not** included in the repo, so you must get them yourself (or copy them from the Drive).

#### A. Colab notebooks (training and evaluation)

1. **Get the data** and place it under `qa/`:
   - SQuAD v1.1: `train-v1.1.json`, `dev-v1.1.json`
   - NewsQA (MRQA 2019 Shared Task release): `NewsQA_train.jsonl.gz`, `NewsQA_dev.jsonl.gz` in `qa/newsqa_raw/`
2. Run `preprocess_newsqa.ipynb` to produce `qa/train-newsqa.json` and `qa/dev-newsqa.json`.
3. Run the notebooks in this order (checkpoints are written to `qa_model*/`):
   1. `train_qa.ipynb` (produces `qa_model/`)
   2. `finetune_bert_on_newsqa.ipynb` (produces `qa_model_newsqa/`)
   3. `finetune_bert_on_newsqa_distill.ipynb` (produces `qa_model_newsqa_distill/`)
   4. `train_joint_squad_newsqa.ipynb` (produces `qa_model_joint/`)
   5. `bootstrap_significance.ipynb`

#### B. NUS SoC compute cluster (inference and evaluation only)

`job.sbatch`, `run_infer.sh` and `infer.py` evaluate an existing checkpoint on SQuAD-format dev files. Replace `<user>` with your SoC username.

**1. SSH in** with a plain terminal (not VSCode Remote-SSH):

```bash
ssh -J <user>@stujump.comp.nus.edu.sg <user>@xlogin.comp.nus.edu.sg
```

**2. Clone the repo** into your home directory:

```bash
git clone https://github.com/SQ77/extractive-qa.git
cd extractive-qa
mkdir -p qa_model
```

**3. Copy the files git doesn't carry.** From your laptop, in the repo root, copy the model files and the two dev sets:

```bash
scp -J <user>@stujump.comp.nus.edu.sg qa_model/config.json qa_model/model.safetensors qa_model/tokenizer.json qa_model/tokenizer_config.json <user>@xlogin.comp.nus.edu.sg:extractive-qa/qa_model/

scp -J <user>@stujump.comp.nus.edu.sg qa/dev-v1.1.json qa/dev-newsqa.json <user>@xlogin.comp.nus.edu.sg:extractive-qa/qa/
```

**4. One-time environment setup.** Get a compute node; don't install anything on xlogin:

```bash
salloc -t 120
srun --pty bash
```

Install Miniconda (see `Soc_cluster/README.md`), then create the environment:

```bash
conda create -n qa python=3.11
conda activate qa
pip install torch transformers datasets numpy
```

Leave the node with `exit` twice (the shell, then the allocation).

**5. Run the evaluation.** From the repo root, check the layout:

```bash
cd extractive-qa
ls qa_model/model.safetensors qa/dev-v1.1.json qa/dev-newsqa.json qa/evaluate-v2.0.py
```

Submit the jobs:

```bash
sbatch --job-name=infer-base job.sbatch --model_dir qa_model --dev_files qa/dev-v1.1.json qa/dev-newsqa.json

sbatch --job-name=infer-joint job.sbatch --model_dir qa_model_joint --dev_files qa/dev-newsqa.json
```

Monitor them:

```bash
squeue --me
tail -f logs/slurm/qa-infer-<jobid>.out
```

**6. Get the results.** The EM/F1 for each dataset are printed in the log, under `=== qa_model on qa/dev-v1.1.json ===` and `=== qa_model on qa/dev-newsqa.json ===`. Download the predictions from your laptop:

```bash
scp -r <user>@xlogin.comp.nus.edu.sg:extractive-qa/outputs .
```

## Setup

Python 3 (3.11 was used on the cluster). Main dependencies: `torch`, `transformers`, `datasets`, `accelerate`, `evaluate`, `numpy`. The notebooks also use Google Drive mounting when run on Colab.

| Task | Environment |
|---|---|
| Training (notebooks) | Google Colab with a single T4 GPU and fp16. Each model takes roughly 1 to 1.5 hours. The `pip install` cell is already at the top of each notebook. |
| Inference and evaluation (`infer.py`) | NUS SoC compute cluster, one H100 (47 GB) via `job.sbatch`, or any CUDA GPU. Needs only `torch`, `transformers`, `datasets`, `numpy`; see Reproducing, Option 2B. |
