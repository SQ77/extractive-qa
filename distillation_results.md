## Distillation-Regularized Fine-tune: NewsQA with Boundary-Distillation Regularization

**Setup.** Starting from the BERT-base model already fine-tuned on SQuAD v1.1 (see `bert_baseline_results.md`), we continued fine-tuning on the NewsQA subset, same as `finetune_bert_on_newsqa.ipynb`, but added a Learning-without-Forgetting style regularizer (Li and Hoiem, 2016). Two copies of the SQuAD-trained checkpoint are loaded: a frozen teacher and a trainable student. A custom `DistillationTrainer` (overrides `compute_loss`) adds a KL-divergence penalty between the student's and teacher's start and end logits on the same NewsQA inputs, alongside the normal NewsQA span cross-entropy loss.

**Hyperparameters:**

| Hyperparameter | Value |
|---|---|
| Base model | `qa_model` (BERT-base fine-tuned on SQuAD v1.1) |
| Training data | NewsQA subset, 10,000 question cap |
| Max sequence length | 384 |
| Doc stride | 128 |
| Batch size (effective) | 8 x 2 grad. accum. = 16 |
| Learning rate | 2e-5 (same as the plain sequential fine-tune, for a fair comparison) |
| Epochs | 2 |
| Weight decay | 0.01 |
| Precision | fp16 |
| Distill weight | 1.0 |
| Distill temperature | 2.0 |
| N-best / max answer length | 20 / 30 tokens |

**Training.** Combined loss (NewsQA task loss plus the distillation KL penalty, logged together rather than separately) decreased from 2.05 (step 500) to 1.94 (step 1000) to 1.85 (step 1500), a normal decelerating curve. This total is not directly comparable to the plain sequential fine-tune's task-only loss, since the distillation term pulls in the opposite direction of the task loss by design (it grows as the student moves away from the frozen teacher to fit NewsQA), so a slower apparent decrease here is expected rather than a sign of a problem.

**Results.** Evaluated with the official scorer on both the NewsQA dev subset (1,003 questions) and the original SQuAD v1.1 dev set (10,570 questions, to check forgetting):

| Model | SQuAD v1.1 EM | SQuAD v1.1 F1 | NewsQA EM | NewsQA F1 |
|---|---|---|---|---|
| SQuAD-only baseline | 80.92 | 88.40 | not tested | not tested |
| Sequential fine-tune (SQuAD, then NewsQA) | 74.84 | 85.44 | 48.55 | 64.24 |
| Joint training (SQuAD + NewsQA together) | 81.12 | 88.58 | 45.96 | 61.81 |
| Distillation-regularized fine-tune | 80.08 | 88.06 | 45.66 | 61.47 |

**Analysis.** The regularizer sharply reduces forgetting relative to plain sequential fine-tuning: SQuAD EM drops only 0.84 points and F1 only 0.34 points from baseline, versus 6.08 and 2.96 points for the unregularized sequential fine-tune, roughly a 7 to 9-fold reduction in forgetting on this run.

However, the result did not land where we expected. Going in, the target (see `joint_training_results.md`) was for distillation to land between the two reference points: closer to the sequential fine-tune's NewsQA numbers (48.55/64.24) while keeping closer to joint training's SQuAD retention (81.12/88.58). Instead, distillation's NewsQA scores (45.66/61.47) are essentially tied with joint training's (45.96/61.81), marginally lower, and its SQuAD retention is slightly worse than joint training's (80.08/88.06 vs. 81.12/88.58).

**Significance testing.** We ran `bootstrap_significance.ipynb` (paired bootstrap, B = 10,000 resamples) over per-question EM/F1 on both dev sets, using the official `get_raw_scores()` function from `evaluate-v2.0.py`. The SQuAD-only baseline's predictions were regenerated from the `qa_model` checkpoint for this test, since the original run never saved `predictions.json` persistently (see the gotcha in `bert_baseline_results.md`); the regenerated run scores 81.08 EM / 88.58 F1, a trivial ~0.16/0.18 point difference from the originally reported 80.92/88.40 consistent with ordinary inference nondeterminism, not a correction:

| Comparison | Metric | Diff | 95% CI | p-value | Significant? |
|---|---|---|---|---|---|
| baseline vs sequential (SQuAD) | EM | +6.23 | [+5.51, +6.95] | 0.0000 | yes |
| baseline vs joint (SQuAD) | EM | -0.04 | [-0.59, +0.52] | 0.9242 | no |
| baseline vs distillation (SQuAD) | EM | +1.00 | [+0.51, +1.49] | 0.0004 | yes |
| sequential vs joint (SQuAD) | EM | -6.27 | [-7.01, -5.54] | 0.0000 | yes |
| sequential vs distillation (SQuAD) | EM | -5.23 | [-5.84, -4.62] | 0.0000 | yes |
| joint vs distillation (SQuAD) | EM | +1.04 | [+0.44, +1.64] | 0.0012 | yes |
| baseline vs sequential (SQuAD) | F1 | +3.14 | [+2.67, +3.60] | 0.0000 | yes |
| baseline vs joint (SQuAD) | F1 | +0.00 | [-0.37, +0.38] | 1.0000 | no |
| baseline vs distillation (SQuAD) | F1 | +0.52 | [+0.20, +0.84] | 0.0022 | yes |
| sequential vs joint (SQuAD) | F1 | -3.14 | [-3.62, -2.67] | 0.0000 | yes |
| sequential vs distillation (SQuAD) | F1 | -2.62 | [-3.02, -2.23] | 0.0000 | yes |
| joint vs distillation (SQuAD) | F1 | +0.52 | [+0.12, +0.91] | 0.0084 | yes |
| sequential vs joint (NewsQA) | EM | +2.59 | [+0.30, +4.99] | 0.0330 | yes |
| sequential vs distillation (NewsQA) | EM | +2.89 | [+0.90, +4.99] | 0.0056 | yes |
| joint vs distillation (NewsQA) | EM | +0.30 | [-1.89, +2.59] | 0.8190 | no |
| sequential vs joint (NewsQA) | F1 | +2.43 | [+0.54, +4.34] | 0.0112 | yes |
| sequential vs distillation (NewsQA) | F1 | +2.76 | [+1.07, +4.51] | 0.0012 | yes |
| joint vs distillation (NewsQA) | F1 | +0.33 | [-1.56, +2.26] | 0.7274 | no |

Adding the baseline comparison resolves the open question more decisively than we expected: **joint training is statistically indistinguishable from the untouched baseline** (EM p = 0.9242, F1 p = 1.0000, diffs of -0.04 and 0.00 respectively), meaning joint training shows zero detectable forgetting, not just "very little." **Distillation, in contrast, has a small but statistically real residual forgetting effect** relative to the untouched baseline (EM +1.00, p = 0.0004; F1 +0.52, p = 0.0022): it is far better than sequential fine-tuning, but it does not fully eliminate forgetting the way joint training does. On NewsQA, joint training and distillation remain statistically indistinguishable from each other (p = 0.82 EM, p = 0.73 F1).

**Conclusion.** Joint training now dominates distillation even more clearly than the joint-vs-distillation comparison alone suggested: it achieves zero detectable forgetting (not merely "less"), while distillation leaves a small but statistically real residual forgetting cost, and the two are tied on NewsQA adaptation. This is the opposite of what motivated building distillation as a presumably stronger, architectural alternative to the simpler data-side joint-training fix, and is worth raising with the team before framing distillation as "the" mitigation in the report.

**Next step.** Discuss with Ewen whether to (a) report this honestly as a negative result for distillation relative to joint training, which is still a legitimate finding, (b) try a `DISTILL_WEIGHT` sweep (e.g. 0.25, 0.5, 1.0, 2.0) to see if a weaker regularization strength trades some SQuAD retention for more NewsQA plasticity and finds a strictly better operating point than joint training, or (c) pivot to EWC or LoRA/adapters as an additional mitigation comparison.

## Artifact Locations

- **Base model used (Drive):** `/content/drive/MyDrive/CS4248 Group Project/qa_model`
- **Fine-tuned model (Drive):** `/content/drive/MyDrive/CS4248 Group Project/qa_model_newsqa_distill`
- **Predictions:** `qa/predictions_newsqa_distill.json`, `qa/predictions_squad_after_newsqa_distill.json`
- **Notebook (source of truth):** `finetune_bert_on_newsqa_distill.ipynb`

## TODO / Next Steps

- [x] Run `bootstrap_significance.ipynb` to get significance tests for all pairwise comparisons on both dev sets.
- [ ] Add significance markers to the results tracker in `bert_baseline_results.md` and `report_draft.md` once the bootstrap test is done.
- [ ] If distillation does not show a significant advantage over joint training, discuss with Ewen whether to run a `DISTILL_WEIGHT` sweep or pivot to EWC/LoRA as an additional mitigation comparison.
