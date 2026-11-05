## Joint Training: BERT on SQuAD v1.1 + NewsQA Combined

**Setup.** A fresh `bert-base-uncased` trained from scratch on SQuAD v1.1 train and the NewsQA subset combined into a single training set from the start (not continuing from the existing SQuAD-only checkpoint), as a comparison point against both the SQuAD-only baseline and the sequential fine-tune.

**Hyperparameters:**

| Hyperparameter | Value |
|---|---|
| Training data | SQuAD v1.1 train (87,599) + NewsQA subset train (up to 10,000), combined and shuffled |
| Max sequence length | 384 |
| Doc stride | 128 |
| Batch size (effective) | 8 x 2 grad. accum. = 16 |
| Learning rate | 3e-5 (same as the SQuAD-only baseline, since this also trains from scratch) |
| Epochs | 2 |
| Weight decay | 0.01 |
| Precision | fp16 |

**Results.** Evaluated separately on the SQuAD v1.1 dev set (10,570 questions) and the NewsQA dev subset (1,003 questions):

| Model | SQuAD v1.1 EM | SQuAD v1.1 F1 | NewsQA EM | NewsQA F1 |
|---|---|---|---|---|
| SQuAD-only baseline | 80.92 | 88.40 | not tested | not tested |
| Sequential fine-tune (SQuAD, then NewsQA) | 74.84 | 85.44 | 48.55 | 64.24 |
| Joint training (SQuAD + NewsQA together) | 81.12 | 88.58 | 45.96 | 61.81 |

**Analysis.** Joint training avoids the forgetting seen in the sequential fine-tune entirely: SQuAD performance is unchanged from the baseline (within noise, if anything marginally better). This confirms the forgetting observed in the sequential run was caused specifically by training on NewsQA after SQuAD was already fine-tuned, not some other confound.

However, joint training adapts to NewsQA less fully than the sequential fine-tune did (EM 45.96 vs. 48.55, F1 61.81 vs. 64.24). This is expected: NewsQA is only about 10% of the combined training set, so gradient updates are dominated by SQuAD and the model never gets to specialize on NewsQA's answer style the way the sequential run did.

This is a clean illustration of the stability-plasticity tradeoff in continual learning: sequential fine-tuning maximizes adaptation to the new domain at the cost of forgetting the old one, joint training preserves the old domain perfectly but under-adapts to the new one.

**What this sets up.** The distillation-regularized fine-tune (`finetune_bert_on_newsqa_distill.ipynb`) is being tested against both reference points above. The target is to land between them: closer to the sequential fine-tune's NewsQA numbers (48.55/64.24) while keeping closer to joint training's SQuAD retention (81.12/88.58), rather than the sequential run's steep SQuAD cost (74.84/85.44).

## Artifact Locations

- **Model (Drive):** `/content/drive/MyDrive/CS4248 Group Project/qa_model_joint`
- **Predictions:** `qa/predictions_squad_after_joint.json`, `qa/predictions_newsqa_after_joint.json`
- **Notebook (source of truth):** `train_joint_squad_newsqa.ipynb`
