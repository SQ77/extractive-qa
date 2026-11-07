"""Run extractive-QA inference with a saved checkpoint on one or more SQuAD-format dev files.

Mirrors run_inference / postprocess_qa_predictions from the notebooks.

Example:
    python infer.py --model_dir qa_model \
        --dev_files qa/dev-v1.1.json qa/dev-newsqa.json --out_dir outputs
"""
import argparse
import collections
import json
import os
import subprocess
import sys

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

MAX_LENGTH = 384
DOC_STRIDE = 128
N_BEST = 20
MAX_ANSWER_LENGTH = 30


def load_squad_json(path):
    with open(path, encoding="utf-8") as f:
        squad = json.load(f)
    ids, questions, contexts = [], [], []
    for article in squad["data"]:
        for para in article["paragraphs"]:
            for qa in para["qas"]:
                ids.append(qa["id"])
                questions.append(qa["question"])
                contexts.append(para["context"])
    return Dataset.from_dict({"id": ids, "question": questions, "context": contexts})


def make_feature_fn(tokenizer):
    def prepare_validation_features(examples):
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
        example_ids = []
        for i in range(len(tokenized["input_ids"])):
            example_ids.append(examples["id"][sample_map[i]])
            sequence_ids = tokenized.sequence_ids(i)
            tokenized["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None
                for k, o in enumerate(tokenized["offset_mapping"][i])
            ]
        tokenized["example_id"] = example_ids
        return tokenized

    return prepare_validation_features


def run_inference(model, dataset, batch_size):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    eval_set = dataset.remove_columns(["example_id", "offset_mapping"])
    eval_set.set_format("torch")

    starts, ends = [], []
    with torch.no_grad():
        for i in range(0, len(eval_set), batch_size):
            batch = eval_set[i : i + batch_size]
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            starts.append(out.start_logits.cpu().numpy())
            ends.append(out.end_logits.cpu().numpy())
    return np.concatenate(starts), np.concatenate(ends)


def postprocess(examples, features, all_start_logits, all_end_logits):
    id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature_id in enumerate(features["example_id"]):
        features_per_example[id_to_index[feature_id]].append(i)

    predictions = {}
    for example_index, example in enumerate(examples):
        context = example["context"]
        valid = []
        for fi in features_per_example[example_index]:
            start_logit = all_start_logits[fi]
            end_logit = all_end_logits[fi]
            offsets = features[fi]["offset_mapping"]
            start_idxs = np.argsort(start_logit)[-1 : -N_BEST - 1 : -1].tolist()
            end_idxs = np.argsort(end_logit)[-1 : -N_BEST - 1 : -1].tolist()
            for s in start_idxs:
                for e in end_idxs:
                    if s >= len(offsets) or e >= len(offsets) or offsets[s] is None or offsets[e] is None:
                        continue
                    if e < s or e - s + 1 > MAX_ANSWER_LENGTH:
                        continue
                    valid.append(
                        {
                            "score": start_logit[s] + end_logit[e],
                            "text": context[offsets[s][0] : offsets[e][1]],
                        }
                    )
        predictions[example["id"]] = max(valid, key=lambda x: x["score"])["text"] if valid else ""
    return predictions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--dev_files", nargs="+", required=True)
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--eval_script", default="qa/evaluate-v2.0.py")
    args = ap.parse_args()

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0), flush=True)
    else:
        print("WARNING: no GPU detected, running on CPU", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    assert tokenizer.is_fast, "Need a fast tokenizer for offset mapping"
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_dir)
    model_name = os.path.basename(os.path.normpath(args.model_dir))

    for dev_file in args.dev_files:
        dev_name = os.path.splitext(os.path.basename(dev_file))[0]
        print(f"\n=== {model_name} on {dev_file} ===", flush=True)
        raw = load_squad_json(dev_file)
        feats = raw.map(make_feature_fn(tokenizer), batched=True, remove_columns=raw.column_names)
        start_logits, end_logits = run_inference(model, feats, args.batch_size)
        preds = postprocess(raw, feats, start_logits, end_logits)

        out_path = os.path.join(args.out_dir, f"predictions_{model_name}_{dev_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(preds, f)
        print(f"Predictions written to {out_path}", flush=True)

        result = subprocess.run(
            [sys.executable, args.eval_script, dev_file, out_path],
            check=True, capture_output=True, text=True,
        )
        print(result.stdout, flush=True)


if __name__ == "__main__":
    main()
