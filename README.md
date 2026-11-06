# Diagnosing and Mitigating Catastrophic Forgetting in Cross-Domain Extractive QA

CS4248 Group Project (Group G05).

We fine-tune BERT-base (`bert-base-uncased`) on SQuAD v1.1, then adapt it to a NewsQA subset, and measure how much SQuAD performance is forgotten. We compare two mitigations against plain sequential fine-tuning:

- **Joint training**: train from scratch on SQuAD + NewsQA combined.
- **Distillation regularization**: a Learning-without-Forgetting-style KL penalty on start/end logits against a frozen SQuAD-only teacher.

Differences are tested with paired bootstrap significance tests.

## Results (BERT-base)

| Model | SQuAD EM | SQuAD F1 | NewsQA EM | NewsQA F1 |
|---|---|---|---|---|
| SQuAD-only baseline | 80.92 | 88.40 | not tested | not tested |
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
| `qa/evaluate-v2.0.py` | Official SQuAD scoring script |
| `qa/predictions_*.json` | Saved dev-set predictions for every model (used by the bootstrap tests) |
| `bert_baseline_results.md`, `newsqa_finetune_results.md`, `distillation_results.md`, `joint_training_results.md` | Per-experiment write-ups |

## Reproducing

Data and model checkpoints are **not** included in this repo.

1. **Get the data** and place it under `qa/`:
   - SQuAD v1.1: `train-v1.1.json`, `dev-v1.1.json` from https://rajpurkar.github.io/SQuAD-explorer/
   - NewsQA (MRQA 2019 Shared Task release): `NewsQA_train.jsonl.gz`, `NewsQA_dev.jsonl.gz` in `qa/newsqa_raw/`
   - NewsQA is subject to its own license terms; do not redistribute it.
2. Run `preprocess_newsqa.ipynb` to produce `qa/train-newsqa.json` and `qa/dev-newsqa.json`.
3. Run the notebooks in this order (checkpoints are written to `qa_model*/`):
   1. `train_qa.ipynb` (produces `qa_model/`)
   2. `finetune_bert_on_newsqa.ipynb` (produces `qa_model_newsqa/`)
   3. `finetune_bert_on_newsqa_distill.ipynb` (produces `qa_model_newsqa_distill/`)
   4. `train_joint_squad_newsqa.ipynb` (produces `qa_model_joint/`)
   5. `bootstrap_significance.ipynb`

## Setup

Python 3 with a CUDA GPU (experiments ran on a single T4 with fp16). Main dependencies: `torch`, `transformers`, `datasets`, `accelerate`, `numpy`. Training for each model takes roughly 1 to 1.5 hours on a T4.
