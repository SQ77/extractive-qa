## NewsQA: Continued Fine-tuning of the BERT Baseline

**Setup.** Starting from the BERT-base model already fine-tuned on SQuAD v1.1 (see `bert_baseline_results.md`), we continued fine-tuning on a NewsQA subset (sourced from the MRQA 2019 Shared Task's pre-combined release), to test whether this improves domain robustness across non-Wikipedia prose, as proposed.

**Hyperparameters:**

| Hyperparameter | Value |
|---|---|
| Base model | `qa_model` (BERT-base fine-tuned on SQuAD v1.1) |
| Training data | NewsQA subset, 10,000 question cap |
| Max sequence length | 384 |
| Doc stride | 128 |
| Batch size (effective) | 8 x 2 grad. accum. = 16 |
| Learning rate | 2e-5 (lower than the original 3e-5, since the model was already converged) |
| Epochs | 2 |
| Weight decay | 0.01 |
| Precision | fp16 |
| N-best / max answer length | 20 / 30 tokens |

**Results.** Evaluated with the official scorer on both the NewsQA dev subset (1,003 questions) and the original SQuAD v1.1 dev set (10,570 questions, to check for forgetting):

| Metric | SQuAD only (baseline) | After NewsQA fine-tune | Change |
|---|---|---|---|
| SQuAD v1.1 EM | 80.92 | 74.84 | -6.08 |
| SQuAD v1.1 F1 | 88.40 | 85.44 | -2.96 |
| NewsQA EM | not tested | 48.55 | - |
| NewsQA F1 | not tested | 64.24 | - |

**Analysis.** The NewsQA scores themselves are in line with typical BERT-scale baselines reported for this dataset (NewsQA is intrinsically harder than SQuAD: longer documents, noisier crowdsourced answers, more sliding-window truncation), so the pipeline is working as expected rather than being broken.

However, the drop in SQuAD performance after continued fine-tuning on NewsQA is a clear sign of catastrophic forgetting: the model improved on NewsQA partly by overwriting what it had learned on SQuAD, rather than becoming robust across both domains. EM dropped roughly twice as much as F1 did, which suggests the model is still finding approximately the right answer region but is less precise about exact span boundaries after the second round of fine-tuning.

This means sequential fine-tuning, as done here, does not fully achieve the "domain robustness" goal from the proposal, since it trades SQuAD performance for NewsQA performance rather than improving both.

**Next step.** Try joint training instead: combine SQuAD and NewsQA into a single training set from the start (rather than fine-tuning on NewsQA after SQuAD is already done), so both distributions are seen together in every batch. This is expected to reduce forgetting compared to sequential fine-tuning, and gives a genuine three-way comparison for the report: SQuAD-only baseline, sequential fine-tune (this run), and joint training.

## Artifact Locations

- **Base model used (Drive):** `/content/drive/MyDrive/CS4248 Group Project/qa_model`
- **Fine-tuned model (Drive):** `/content/drive/MyDrive/CS4248 Group Project/qa_model_newsqa`
- **NewsQA data:** `qa/train-newsqa.json`, `qa/dev-newsqa.json` (produced by `preprocess_newsqa.ipynb`)
- **Predictions:** `qa/predictions_newsqa_finetuned.json`, `qa/predictions_squad_after_newsqa_finetune.json`
- **Notebooks (source of truth):** `preprocess_newsqa.ipynb`, `finetune_bert_on_newsqa.ipynb`

## TODO / Next Steps

- [ ] Build a joint-training notebook that combines SQuAD v1.1 and the NewsQA subset into one training set from the start, and compare against both the SQuAD-only baseline and this sequential fine-tune.
- [ ] If forgetting needs to be reduced without full joint training, consider: lower learning rate, fewer epochs, freezing lower transformer layers, or mixing a portion of SQuAD examples into the NewsQA fine-tuning data (rehearsal).
- [ ] Add this three-way comparison (SQuAD-only, sequential fine-tune, joint training) to the main results tracker in `bert_baseline_results.md` once joint training is done.
