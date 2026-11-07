"""Training script for the cluster.

Modes:
    baseline  train_qa.ipynb                          bert-base-uncased -> SQuAD          -> qa_model
    newsqa    finetune_bert_on_newsqa.ipynb           qa_model -> NewsQA                  -> qa_model_newsqa
    distill   finetune_bert_on_newsqa_distill.ipynb   qa_model -> NewsQA + KL to teacher  -> qa_model_newsqa_distill
    joint     train_joint_squad_newsqa.ipynb          bert-base-uncased -> SQuAD + NewsQA -> qa_model_joint

After training, the model is evaluated on SQuAD dev and NewsQA dev with the official script.
Predictions and metrics are written to --out_dir (default: outputs_train/), so they never
overwrite the saved predictions in qa/.

Example:
    python train.py --mode baseline
    python train.py --mode newsqa --base_model_dir qa_model
"""
import argparse
import dataclasses
import inspect
import json
import os
import subprocess
import sys

import torch
import torch.nn.functional as F
from datasets import Dataset, concatenate_datasets
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Evaluation helpers are shared with infer.py so training-time and inference-time
# scoring use identical code.
from infer import MAX_LENGTH, DOC_STRIDE, load_squad_json as load_eval_json
from infer import make_feature_fn, postprocess, run_inference

MODEL_CHECKPOINT = "bert-base-uncased"
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2
NUM_EPOCHS = 2
SEED = 42

# mode -> (output dir, learning rate, run name)
MODES = {
    "baseline": ("qa_model", 3e-5, "bert-base-uncased-squad-v1.1"),
    "newsqa": ("qa_model_newsqa", 2e-5, "bert-newsqa-continued-finetune"),
    "distill": ("qa_model_newsqa_distill", 2e-5, "bert-newsqa-distill"),
    "joint": ("qa_model_joint", 3e-5, "bert-squad-newsqa-joint"),
}

# Distillation hyperparameters (finetune_bert_on_newsqa_distill.ipynb)
DISTILL_WEIGHT = 1.0
DISTILL_TEMPERATURE = 2.0


def load_train_json(path):
    """Like infer.load_squad_json but keeps the answers (needed for training)."""
    with open(path, encoding="utf-8") as f:
        squad = json.load(f)
    ids, questions, contexts, answers = [], [], [], []
    for article in squad["data"]:
        for para in article["paragraphs"]:
            for qa in para["qas"]:
                ids.append(qa["id"])
                questions.append(qa["question"])
                contexts.append(para["context"])
                answers.append(
                    {
                        "text": [a["text"] for a in qa["answers"]],
                        "answer_start": [a["answer_start"] for a in qa["answers"]],
                    }
                )
    return Dataset.from_dict(
        {"id": ids, "question": questions, "context": contexts, "answers": answers}
    )


def make_train_feature_fn(tokenizer):
    def prepare_train_features(examples):
        questions = [q.lstrip() for q in examples["question"]]
        tokenized = tokenizer(
            questions,
            examples["context"],
            max_length=MAX_LENGTH,
            truncation="only_second",
            stride=DOC_STRIDE,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        sample_map = tokenized.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized.pop("offset_mapping")

        start_positions, end_positions = [], []
        for i, offsets in enumerate(offset_mapping):
            answer = examples["answers"][sample_map[i]]
            start_char = answer["answer_start"][0]
            end_char = start_char + len(answer["text"][0])
            sequence_ids = tokenized.sequence_ids(i)

            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while idx < len(sequence_ids) and sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            if offsets[context_start][0] > start_char or offsets[context_end][1] < end_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                idx = context_start
                while idx <= context_end and offsets[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)
                idx = context_end
                while idx >= context_start and offsets[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        tokenized["start_positions"] = start_positions
        tokenized["end_positions"] = end_positions
        return tokenized

    return prepare_train_features


class DistillationTrainer(Trainer):
    """Task loss + KL(student || frozen teacher) on start/end logits (LwF-style)."""

    def __init__(self, *args, teacher_model=None, distill_weight=1.0, distill_temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.distill_weight = distill_weight
        self.distill_temperature = distill_temperature

    def _kl(self, student_logits, teacher_logits):
        t = self.distill_temperature
        student_log_probs = F.log_softmax(student_logits / t, dim=-1)
        teacher_probs = F.softmax(teacher_logits / t, dim=-1)
        return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (t ** 2)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        task_loss = outputs.loss

        teacher_inputs = {k: v for k, v in inputs.items() if k not in ("start_positions", "end_positions")}
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**teacher_inputs)

        distill_loss = (
            self._kl(outputs.start_logits, teacher_outputs.start_logits)
            + self._kl(outputs.end_logits, teacher_outputs.end_logits)
        ) / 2
        loss = task_loss + self.distill_weight * distill_loss
        return (loss, outputs) if return_outputs else loss


def build_training_args(output_dir, lr, run_name, gradient_checkpointing):
    # Only pass kwargs the installed transformers version accepts (same approach as the notebooks).
    fields = {f.name for f in dataclasses.fields(TrainingArguments)}

    def pick(*pairs):
        for name, value in pairs:
            if name in fields:
                return {name: value}
        return {}

    kw = {"output_dir": output_dir}
    kw.update(pick(("eval_strategy", "no"), ("evaluation_strategy", "no")))
    # Checkpoints at the end of each epoch would each be ~1.3GB; the final model is
    # saved explicitly with trainer.save_model, so skip intermediate saves.
    kw.update(pick(("save_strategy", "no")))
    kw["learning_rate"] = lr
    kw["per_device_train_batch_size"] = BATCH_SIZE
    kw["per_device_eval_batch_size"] = BATCH_SIZE
    kw["gradient_accumulation_steps"] = GRADIENT_ACCUMULATION_STEPS
    kw["num_train_epochs"] = NUM_EPOCHS
    kw["weight_decay"] = 0.01
    kw.update(pick(("warmup_ratio", 0.1)))
    kw.update(pick(("fp16", torch.cuda.is_available())))
    kw.update(pick(("gradient_checkpointing", gradient_checkpointing)))
    kw.update(pick(("dataloader_num_workers", 0)))
    kw.update(pick(("report_to", "none")))
    kw.update(pick(("run_name", run_name)))
    kw.update(pick(("seed", SEED)))
    return TrainingArguments(**kw)


def make_trainer(trainer_cls, tokenizer, **kwargs):
    params = set(inspect.signature(Trainer.__init__).parameters)
    tok_kw = "processing_class" if "processing_class" in params else "tokenizer"
    return trainer_cls(**kwargs, **{tok_kw: tokenizer})


def evaluate(model, tokenizer, dev_file, out_dir, tag, eval_script, batch_size):
    """Inference + official scoring on one dev file; writes predictions_<tag>.json and metrics_<tag>.json."""
    raw = load_eval_json(dev_file)
    feats = raw.map(make_feature_fn(tokenizer), batched=True, remove_columns=raw.column_names)
    start_logits, end_logits = run_inference(model, feats, batch_size)
    preds = postprocess(raw, feats, start_logits, end_logits)

    pred_path = os.path.join(out_dir, f"predictions_{tag}.json")
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(preds, f)
    result = subprocess.run(
        [sys.executable, eval_script, dev_file, pred_path],
        check=True, capture_output=True, text=True,
    )
    print(f"--- {tag} ({dev_file}) ---\n{result.stdout}", flush=True)
    try:
        metrics = json.loads(result.stdout)
    except json.JSONDecodeError:
        return
    with open(os.path.join(out_dir, f"metrics_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=list(MODES))
    ap.add_argument("--data_dir", default="qa")
    ap.add_argument("--output_dir", default=None, help="Where to save the model (default depends on --mode).")
    ap.add_argument("--base_model_dir", default="qa_model",
                    help="SQuAD-trained checkpoint to start from (newsqa/distill modes only).")
    ap.add_argument("--out_dir", default="outputs_train", help="Where predictions/metrics are written.")
    ap.add_argument("--eval_script", default="qa/evaluate-v2.0.py")
    ap.add_argument("--eval_batch_size", type=int, default=64)
    ap.add_argument("--gradient_checkpointing", action="store_true",
                    help="Notebooks used it for Colab's T4; not needed on an H100 (results are unaffected).")
    ap.add_argument("--max_train_samples", type=int, default=None, help="Smoke-test: cap training questions.")
    ap.add_argument("--max_eval_samples", type=int, default=None, help="Smoke-test: cap dev questions.")
    args = ap.parse_args()

    default_out, lr, run_name = MODES[args.mode]
    model_out = args.output_dir or default_out

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0), flush=True)
    else:
        sys.exit("No GPU detected; refusing to train BERT on CPU.")
    set_seed(SEED)
    os.makedirs(args.out_dir, exist_ok=True)

    d = args.data_dir
    squad_train_f, squad_dev_f = os.path.join(d, "train-v1.1.json"), os.path.join(d, "dev-v1.1.json")
    news_train_f, news_dev_f = os.path.join(d, "train-newsqa.json"), os.path.join(d, "dev-newsqa.json")

    # Which files each mode needs, and which checkpoint it starts from.
    if args.mode == "baseline":
        start_from, train_files, dev_files = MODEL_CHECKPOINT, [squad_train_f], [squad_dev_f]
    elif args.mode == "joint":
        start_from, train_files, dev_files = MODEL_CHECKPOINT, [squad_train_f, news_train_f], [squad_dev_f, news_dev_f]
    else:
        start_from, train_files, dev_files = args.base_model_dir, [news_train_f], [news_dev_f, squad_dev_f]
    for p in train_files + dev_files + [args.eval_script] + ([start_from] if args.mode in ("newsqa", "distill") else []):
        if not os.path.exists(p):
            sys.exit(f"Missing: {p}")

    # Tokenizer comes from the starting checkpoint (the saved qa_model for continued fine-tuning).
    tokenizer = AutoTokenizer.from_pretrained(start_from)
    assert tokenizer.is_fast, "Need a fast tokenizer for offset mapping"

    print("Loading training data...", flush=True)
    parts = [load_train_json(p) for p in train_files]
    train_raw = parts[0] if len(parts) == 1 else concatenate_datasets(parts).shuffle(seed=SEED)
    if args.max_train_samples:
        train_raw = train_raw.select(range(min(args.max_train_samples, len(train_raw))))
    print(f"Training questions: {len(train_raw)}", flush=True)

    train_dataset = train_raw.map(
        make_train_feature_fn(tokenizer), batched=True, remove_columns=train_raw.column_names
    )

    model = AutoModelForQuestionAnswering.from_pretrained(start_from)
    targs = build_training_args(model_out, lr, run_name, args.gradient_checkpointing)

    if args.mode == "distill":
        teacher = AutoModelForQuestionAnswering.from_pretrained(args.base_model_dir).to("cuda")
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        trainer = make_trainer(
            DistillationTrainer, tokenizer, model=model, args=targs, train_dataset=train_dataset,
            teacher_model=teacher, distill_weight=DISTILL_WEIGHT, distill_temperature=DISTILL_TEMPERATURE,
        )
    else:
        trainer = make_trainer(Trainer, tokenizer, model=model, args=targs, train_dataset=train_dataset)

    print(f"Training ({args.mode}), lr={lr}, start from {start_from}...", flush=True)
    trainer.train()
    trainer.save_model(model_out)
    tokenizer.save_pretrained(model_out)
    print(f"Model saved to {model_out}", flush=True)

    # Evaluate with the same code path as infer.py.
    model = trainer.model
    for dev_file in dev_files:
        if args.max_eval_samples:
            # Smoke-test only: scoring a truncated dev file against its full gold file would be wrong,
            # so write a truncated copy of the gold file as well.
            with open(dev_file, encoding="utf-8") as f:
                squad = json.load(f)
            left, keep = args.max_eval_samples, []
            for art in squad["data"]:
                paras = []
                for para in art["paragraphs"]:
                    if left <= 0:
                        break
                    para = dict(para, qas=para["qas"][:left])
                    left -= len(para["qas"])
                    paras.append(para)
                if paras:
                    keep.append(dict(art, paragraphs=paras))
            dev_file = os.path.join(args.out_dir, "smoke_" + os.path.basename(dev_file))
            with open(dev_file, "w", encoding="utf-8") as f:
                json.dump({"data": keep}, f)
        tag = f"{os.path.basename(os.path.normpath(model_out))}_{os.path.splitext(os.path.basename(dev_file))[0]}"
        evaluate(model, tokenizer, dev_file, args.out_dir, tag, args.eval_script, args.eval_batch_size)


if __name__ == "__main__":
    main()
