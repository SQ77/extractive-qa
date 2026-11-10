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

SQuAD is evaluated on the full v1.1 dev set (10,570 questions). NewsQA is evaluated on a 1,003-question dev subset.

## Repository layout

| File | Purpose |
|---|---|
| `infer.py` | Runs inference and EM/F1 evaluation of a checkpoint on SQuAD-format dev files |
| `run_infer.sh` | Wrapper that launches `infer.py` |
| `job.sbatch` | Generic Slurm job script for running inference on the SoC compute cluster |
| `train.py` | Training script (all four modes) for all four modes, with evaluation |
| `train.sbatch` | Slurm job script for `train.py` |
| `preprocess_newsqa.py` | Builds the NewsQA subset in SQuAD format from the raw data |
| `bootstrap_significance.py` | Paired bootstrap significance tests on per-question scores (CPU only) |
| `cpu.sbatch` | Slurm job script (no GPU) for `preprocess_newsqa.py` and `bootstrap_significance.py` |
| `qa/evaluate-v2.0.py` | Official SQuAD scoring script |
| `qa/predictions_*.json` | Saved dev-set predictions for every model, produced in Colab (the results reported in the write-up; used by the bootstrap tests) |
| `outputs/` | Predictions written by cluster inference jobs (created when `job.sbatch` runs) |
| `outputs_train/` | Predictions and metrics from retraining the models on the cluster with `train.py`, to reproduce the Colab results in `qa/` |
| `Soc_cluster/README.md` | SoC Compute Cluster unofficial guide |

## Reproducing

This repo contains the scripts for preprocessing, training and inference on the NUS SoC compute cluster. Data and model checkpoints are **not** included in the repo, so you must get them yourself (or copy them from the zipped files submitted).

#### A. NUS SoC compute cluster (inference and evaluation only)

`job.sbatch`, `run_infer.sh` and `infer.py` evaluate an existing checkpoint on SQuAD-format dev files. Replace `<user>` with your SoC username.

**1. SSH in** with a plain terminal (not VSCode Remote-SSH):

```bash
ssh -J <user>@stujump.comp.nus.edu.sg <user>@xlogin.comp.nus.edu.sg
```

**2. Clone the repo** into your home directory:

```bash
git clone https://github.com/SQ77/extractive-qa.git
cd extractive-qa
mkdir -p qa_model qa_model_newsqa qa_model_newsqa_distill qa_model_joint
```

**3. Copy the files git doesn't carry.** From your laptop, in the repo root, copy the model files for each checkpoint you want to evaluate (`qa_model`, `qa_model_newsqa`, `qa_model_newsqa_distill`, `qa_model_joint`) and the two dev sets:

```bash
for m in qa_model qa_model_newsqa qa_model_newsqa_distill qa_model_joint; do
  scp -J <user>@stujump.comp.nus.edu.sg $m/config.json $m/model.safetensors $m/tokenizer.json $m/tokenizer_config.json <user>@xlogin.comp.nus.edu.sg:extractive-qa/$m/
done

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
pip install torch transformers datasets numpy 'accelerate>=1.1.0'
```

Leave the node with `exit` twice (the shell, then the allocation).

**5. Run the evaluation.** From the repo root, check the layout:

```bash
cd extractive-qa
ls qa_model/model.safetensors qa_model_newsqa/model.safetensors qa_model_newsqa_distill/model.safetensors qa_model_joint/model.safetensors qa/dev-v1.1.json qa/dev-newsqa.json qa/evaluate-v2.0.py
```

Submit the jobs:

```bash
sbatch --job-name=infer-base job.sbatch --model_dir qa_model --dev_files qa/dev-v1.1.json qa/dev-newsqa.json

sbatch --job-name=infer-newsqa job.sbatch --model_dir qa_model_newsqa --dev_files qa/dev-v1.1.json qa/dev-newsqa.json

sbatch --job-name=infer-distill job.sbatch --model_dir qa_model_newsqa_distill --dev_files qa/dev-v1.1.json qa/dev-newsqa.json

sbatch --job-name=infer-joint job.sbatch --model_dir qa_model_joint --dev_files qa/dev-newsqa.json
```

Monitor them:

```bash
squeue --me
tail -f logs/slurm/qa-infer-<jobid>.out
```

**6. Get the results.** The EM/F1 for each dataset are printed in the log, under headers of the form `=== <model_dir> on <dev_file> ===`, e.g. `=== qa_model_newsqa on qa/dev-v1.1.json ===`. They are also saved in `outputs/` as `metrics_<model>_<dev>.json`, next to the predictions in `predictions_<model>_<dev>.json`. Download both from your laptop:

```bash
scp -J <user>@stujump.comp.nus.edu.sg -r <user>@xlogin.comp.nus.edu.sg:extractive-qa/outputs .
```

#### B. NUS SoC compute cluster (training)

`train.py` trains the four models, then evaluates on the dev sets with the official script. `train.sbatch` is the Slurm wrapper (`gpu` partition, 3 h limit, one H100). Complete steps 1 to 4 of section A first, and also copy the training data to the cluster (`qa/train-v1.1.json`, `qa/train-newsqa.json`, `qa/dev-v1.1.json`, `qa/dev-newsqa.json`, `qa/evaluate-v2.0.py`). `qa/train-newsqa.json` and `qa/dev-newsqa.json` are produced by `preprocess_newsqa.py` (section C).

| `--mode` | Starts from | Saves to |
|---|---|---|
| `baseline` | `bert-base-uncased` | `qa_model/` |
| `newsqa` | `--base_model_dir` (default `qa_model`) | `qa_model_newsqa/` |
| `distill` | `--base_model_dir` (also the frozen teacher) | `qa_model_newsqa_distill/` |
| `joint` | `bert-base-uncased` | `qa_model_joint/` |

`newsqa` and `distill` need the baseline checkpoint, so chain them after it:

```bash
b=$(sbatch --parsable --job-name=train-baseline train.sbatch --mode baseline)
sbatch --dependency=afterok:$b --job-name=train-newsqa  train.sbatch --mode newsqa
sbatch --dependency=afterok:$b --job-name=train-distill train.sbatch --mode distill
sbatch --job-name=train-joint train.sbatch --mode joint
```

Add `--max_train_samples 2000 --max_eval_samples 500` for a quick smoke test. Predictions and `metrics_*.json` are written to `outputs_train/` (not `qa/`, which holds the Colab results, so they are not overwritten); compare them with the numbers in the Results table. Exact numbers can differ slightly from the reported results (different GPU, no gradient checkpointing, nondeterministic kernels).

**Monitor the jobs.**

```bash
squeue --me                                        
tail -f logs/slurm/train-baseline-<jobid>.out      
```

`<jobid>` is printed by `sbatch`, or stored in `$b` if you used the chained commands above.

**Get the results.** After a job completes, check that it produced:

- the model directory in the repo root (`qa_model/`, `qa_model_newsqa/`, `qa_model_newsqa_distill/` or `qa_model_joint/`), containing `model.safetensors`;
- `outputs_train/predictions_<model_dir>_<dev>.json` and `outputs_train/metrics_<model_dir>_<dev>.json` for each dev set the mode scores (for example `metrics_qa_model_dev-v1.1.json`).

The EM/F1 scores are also printed in the log under `--- <model_dir>_<dev> (<dev_file>) ---` headers. Quick look on the cluster:

```bash
cat outputs_train/metrics_qa_model_dev-v1.1.json
```

Download the predictions and metrics from your laptop:

```bash
scp -J <user>@stujump.comp.nus.edu.sg -r <user>@xlogin.comp.nus.edu.sg:extractive-qa/outputs_train .
```

The trained models are already on the cluster, so you can run inference with them as in section A. To keep a copy locally, download a model directory the same way (each `model.safetensors` is about 1.3 GB).

#### C. NUS SoC compute cluster (preprocessing and significance tests)

These two scripts need no GPU, so they run through `cpu.sbatch`:

```bash
sbatch --job-name=preprocess cpu.sbatch preprocess_newsqa.py      # writes qa/train-newsqa.json, qa/dev-newsqa.json
sbatch --job-name=bootstrap  cpu.sbatch bootstrap_significance.py # reads the predictions in qa/
```

`preprocess_newsqa.py` downloads the raw MRQA files; if compute nodes have no internet, run it on a login node or copy the two `.jsonl.gz` files into `qa/newsqa_raw/` first. `bootstrap_significance.py` takes `--pred_dir outputs_train` or per-model `--squad_pred MODEL=PATH` / `--newsqa_pred MODEL=PATH` overrides to test predictions produced on the cluster.

## Setup

Python 3 (3.11 was used on the cluster). Main dependencies: `torch`, `transformers`, `datasets`, `accelerate`, `evaluate`, `numpy`.

| Task | Environment |
|---|---|
| Training (`train.py`) | NUS SoC compute cluster, one H100 via `train.sbatch` (3 h limit). Section B. |
| Inference and evaluation (`infer.py`) | NUS SoC compute cluster, one H100 (47 GB) via `job.sbatch`, or any CUDA GPU. Needs only `torch`, `transformers`, `datasets`, `numpy`; see section A. |
