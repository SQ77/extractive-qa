## Baseline: BERT-base Fine-tuned on SQuAD v1.1

**Setup.** As our baseline, we fine-tuned `bert-base-uncased` (110M parameters) on the full SQuAD v1.1 training set (87,599 question-context pairs), following the standard extractive QA formulation: question and context are jointly encoded as `[CLS] question [SEP] context [SEP]`, with a sliding window (stride 128) over contexts exceeding the 384-token limit, and separate linear heads predicting start and end token logits.

**Hyperparameters:**

| Hyperparameter | Value |
|---|---|
| Max sequence length | 384 |
| Doc stride | 128 |
| Batch size (effective) | 8 x 2 grad. accum. = 16 |
| Learning rate | 3e-5 |
| Epochs | 2 |
| Weight decay | 0.01 |
| Precision | fp16 |
| N-best / max answer length | 20 / 30 tokens |
| Hardware | 1x NVIDIA T4 GPU |

**Training.** Training loss decreased monotonically from 2.29 (step 500) to 0.70 (final step, 11,062 total steps over 2 epochs), converging smoothly with no signs of divergence. Total training time was approximately 1h20m.

**Results.** We evaluated on the full SQuAD v1.1 dev set (10,570 questions) using the official scorer, obtaining:

| Metric | Score |
|---|---|
| Exact Match (EM) | 80.92 |
| F1 | 88.40 |

These results are consistent with the original BERT paper's reported SQuAD v1.1 dev performance for `bert-base-uncased` (approximately 80.8 EM / 88.5 F1), confirming our fine-tuning pipeline is correctly implemented and the model generalizes well rather than overfitting the training set.

This baseline serves as the reference point against which we compare our proposed DeBERTa-v3-base model with boundary-aware span prediction (Section X).

---

## Run Metadata (internal reference, not for report)

- **Date run:** 2026-10-08
- **Environment:** Google Colab, 1x Tesla T4 GPU
- **Library versions:** `transformers==5.19.0` (installed fresh via `!pip install -U transformers datasets accelerate evaluate wandb`)
- **Seed:** not explicitly set in the notebook. `TrainingArguments` defaults to `seed=42`. If exact reproducibility matters later, set `ta_kwargs["seed"] = 42` explicitly rather than relying on the default.

## Artifact Locations

- **Trained model (Drive):** `/content/drive/MyDrive/CS4248 Group Project/qa_model`
- **Trained model (local backup):** `~/Downloads/qa_model` (config.json, model.safetensors approximately 415MB, tokenizer files, training_args.bin, no optimizer checkpoints, approximately 416MB total)
- **`predictions.json`:** not saved anywhere persistent. It was written to the Colab session's local working directory, not Drive. If we need to re-inspect individual predictions (for error analysis) or re-score without rerunning inference, we'll need to rerun the inference and postprocessing cells first. Consider saving this to Drive directly next time (`PREDICTIONS_FILE = os.path.join(DATA_DIR, "predictions_bert_baseline.json")`).
- **Notebook (source of truth):** `train_qa.ipynb` in this folder.

## Known Gotchas

- The scoring cell (`subprocess.run([...], check=True)`) does not capture the child process's stdout by default. `evaluate-v2.0.py`'s printed EM/F1 JSON got lost on the first run (only `CompletedProcess(returncode=0)` showed up). Fixed by adding `capture_output=True, text=True` and printing `result.stdout`. Apply this fix once and it's done for all future runs of this notebook.
- Downloading the full Drive `qa_model` folder can balloon to 3GB+ if per-epoch checkpoints (`checkpoint-*` subfolders, which include full optimizer state) are included. Only the top-level model files are needed for inference/handoff, so exclude `checkpoint-*` when zipping/downloading.

## Results Tracker (fill in as more models are trained)

| Model | SQuAD v1.1 EM | SQuAD v1.1 F1 | NewsQA F1 | Adversarial SQuAD F1 | Notes |
|---|---|---|---|---|---|
| BiDAF (baseline) | TBD | TBD | TBD | TBD | not yet run |
| BERT-base (baseline) | 80.92 | 88.40 | TBD | TBD | this run, 2026-10-08 |
| BERT-base + boundary-aware (ours, v1) | TBD | TBD | TBD | TBD | not yet implemented, planned first |
| DeBERTa-v3-base + boundary-aware (ours, v2) | TBD | TBD | TBD | TBD | not yet implemented, planned if time allows |

## Design Decisions

- **2026-10-08:** Decided to implement the boundary-aware span-prediction head on top of the already-trained BERT-base checkpoint first, rather than starting with DeBERTa-v3-base. Rationale: DeBERTa training takes about another 1.5 hours, and BERT is already trained and working, so it is faster to debug the new head and auxiliary loss on BERT before committing more GPU time to a slower backbone.
- Prof confirmed in class that the proposal is not hard-set and the team can be flexible with it, so swapping the primary backbone (or the order in which models are tried) is acceptable as long as the final report explains what was actually done and why.
- Plan: validate the boundary-aware method on BERT first. If time allows, also run it on DeBERTa-v3-base for the headline result, since that keeps the comparison the proposal promised (BiDAF vs BERT baseline vs DeBERTa + boundary-aware). If time does not allow, BERT + boundary-aware can serve as the main reported result instead, with this decision explained in the report.

## TODO / Next Steps (mapped to proposal breakdown of work)

- [ ] **Member 1**: NewsQA subset preprocessing and token alignment for the additional dataset.
- [ ] **Member 2**: Swap `MODEL_CHECKPOINT` to `microsoft/deberta-v3-base`, rerun training pipeline.
- [ ] **Member 2/3**: Implement the boundary-aware span-prediction head (the project's main novel contribution, not yet started).
- [ ] **Member 3**: Span decoding and inference testing for the new model once trained.
- [ ] **Member 4**: BiDAF baseline implementation and evaluation.
- [ ] **Member 4**: Adversarial SQuAD (AddSent/AddOneSent) evaluation run.
- [ ] **Member 4**: Layer-wise learning-rate decay ablation.
- [ ] **Member 5**: Error analysis on BERT baseline predictions (needs `predictions.json` saved first, see gotcha above).
- [ ] Save `predictions.json` to Drive going forward so it isn't lost between sessions.
