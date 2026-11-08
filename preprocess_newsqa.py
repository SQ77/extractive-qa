"""Build the NewsQA subset in SQuAD v1.1 format.

Downloads the MRQA 2019 NewsQA train/dev files (skipped if already present), converts them to the
SQuAD JSON schema, runs character- and token-level alignment checks, and writes
<data_dir>/train-newsqa.json and <data_dir>/dev-newsqa.json.

The compute nodes may have no internet access. If the download fails, download the two .jsonl.gz
files elsewhere and copy them to <data_dir>/newsqa_raw/ as NewsQA_train.jsonl.gz / NewsQA_dev.jsonl.gz.

Example:
    python preprocess_newsqa.py
"""
import argparse
import gzip
import json
import os
import random
import sys
import urllib.request

from transformers import AutoTokenizer

MODEL_CHECKPOINT = "bert-base-uncased"  # only used for the alignment sanity checks
MAX_LENGTH = 384
DOC_STRIDE = 128
RANDOM_SEED = 42

MRQA_TRAIN_URL = "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/NewsQA.jsonl.gz"
MRQA_DEV_URL = "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/NewsQA.jsonl.gz"


def download(url, dest):
    if os.path.exists(dest):
        print(f"Already downloaded: {dest}", flush=True)
        return
    print(f"Downloading {url} ...", flush=True)
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as e:
        if os.path.exists(dest):
            os.remove(dest)
        sys.exit(
            f"Could not download {url} ({e}).\n"
            f"Download it elsewhere and copy it to {dest}, then rerun."
        )
    print(f"Saved to {dest}", flush=True)


def load_mrqa_jsonl(path):
    documents = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "header" in row and "context" not in row:
                continue  # first line is dataset metadata, not a document
            documents.append(row)
    return documents


def mrqa_to_squad_format(documents, max_questions=None, seed=RANDOM_SEED):
    """MRQA-format NewsQA documents -> nested SQuAD v1.1 JSON schema."""
    paragraphs = []
    n_dropped = 0

    for doc in documents:
        context = doc["context"]
        qas_out = []
        for qa in doc["qas"]:
            detected = qa.get("detected_answers", [])
            if not detected:
                n_dropped += 1
                continue

            answers = []
            for det in detected:
                # answer_start is kept as given; any misalignment is flagged by the checks below.
                answers.append({"text": det["text"], "answer_start": det["char_spans"][0][0]})

            qas_out.append(
                {"id": qa.get("qid", qa.get("id")), "question": qa["question"], "answers": answers}
            )

        if qas_out:
            paragraphs.append({"context": context, "qas": qas_out})

    random.Random(seed).shuffle(paragraphs)

    if max_questions is not None:
        selected, total = [], 0
        for p in paragraphs:
            if total >= max_questions:
                break
            selected.append(p)
            total += len(p["qas"])
        paragraphs = selected

    n_questions = sum(len(p["qas"]) for p in paragraphs)
    print(
        f"Converted {n_questions} questions across {len(paragraphs)} paragraphs "
        f"({n_dropped} dropped, no detected answer)",
        flush=True,
    )
    return {"data": [{"title": "NewsQA", "paragraphs": paragraphs}]}


def flatten(squad_data):
    return [
        (p["context"], qa)
        for article in squad_data["data"]
        for p in article["paragraphs"]
        for qa in p["qas"]
    ]


def check_char_alignment(squad_data, n=200, seed=RANDOM_SEED):
    flat = flatten(squad_data)
    sample = random.Random(seed).sample(flat, min(n, len(flat)))
    mismatches = 0
    for context, qa in sample:
        ans = qa["answers"][0]
        start = ans["answer_start"]
        extracted = context[start : start + len(ans["text"])]
        if extracted != ans["text"]:
            mismatches += 1
            if mismatches <= 3:
                print(f"  mismatch -> expected {ans['text']!r}, got {extracted!r}")
    print(f"Character-level check: {len(sample) - mismatches}/{len(sample)} correct", flush=True)


def check_token_alignment(tokenizer, squad_data, n=200, seed=RANDOM_SEED):
    flat = flatten(squad_data)
    sample = random.Random(seed).sample(flat, min(n, len(flat)))
    correct = not_found = 0

    for context, qa in sample:
        ans = qa["answers"][0]
        start_char = ans["answer_start"]
        end_char = start_char + len(ans["text"])
        tokenized = tokenizer(
            qa["question"].lstrip(), context,
            max_length=MAX_LENGTH, truncation="only_second", stride=DOC_STRIDE,
            return_overflowing_tokens=True, return_offsets_mapping=True, padding="max_length",
        )

        found = False
        for i, offsets in enumerate(tokenized["offset_mapping"]):
            sequence_ids = tokenized.sequence_ids(i)
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while idx < len(sequence_ids) and sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            if offsets[context_start][0] > start_char or offsets[context_end][1] < end_char:
                continue  # answer not inside this sliding window

            idx = context_start
            while idx <= context_end and offsets[idx][0] <= start_char:
                idx += 1
            start_pos = idx - 1
            idx = context_end
            while idx >= context_start and offsets[idx][1] >= end_char:
                idx -= 1
            end_pos = idx + 1

            if context[offsets[start_pos][0] : offsets[end_pos][1]].strip() == ans["text"].strip():
                correct += 1
            found = True
            break

        if not found:
            not_found += 1

    print(
        f"Token-level check: {correct}/{len(sample)} spans decode back to the gold answer "
        f"({not_found} examples had no window containing the answer)",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="qa")
    ap.add_argument("--max_train_questions", type=int, default=10000,
                    help="Cap on NewsQA train questions (use 0 for no cap).")
    ap.add_argument("--max_dev_questions", type=int, default=1000,
                    help="Cap on NewsQA dev questions (use 0 for no cap).")
    ap.add_argument("--skip_checks", action="store_true", help="Skip the alignment sanity checks.")
    args = ap.parse_args()

    raw_dir = os.path.join(args.data_dir, "newsqa_raw")
    os.makedirs(raw_dir, exist_ok=True)
    train_gz = os.path.join(raw_dir, "NewsQA_train.jsonl.gz")
    dev_gz = os.path.join(raw_dir, "NewsQA_dev.jsonl.gz")
    download(MRQA_TRAIN_URL, train_gz)
    download(MRQA_DEV_URL, dev_gz)

    print("Parsing raw NewsQA files...", flush=True)
    train_docs, dev_docs = load_mrqa_jsonl(train_gz), load_mrqa_jsonl(dev_gz)
    print(f"Raw documents: {len(train_docs)} train, {len(dev_docs)} dev")
    print(f"Raw questions: {sum(len(d['qas']) for d in train_docs)} train, "
          f"{sum(len(d['qas']) for d in dev_docs)} dev", flush=True)

    print("Converting to SQuAD v1.1 format...", flush=True)
    train_squad = mrqa_to_squad_format(train_docs, args.max_train_questions or None)
    dev_squad = mrqa_to_squad_format(dev_docs, args.max_dev_questions or None)

    if not args.skip_checks:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
        assert tokenizer.is_fast, "Need a fast tokenizer for offset mapping"
        for name, data in (("Train", train_squad), ("Dev", dev_squad)):
            print(f"{name} set:", flush=True)
            check_char_alignment(data)
            check_token_alignment(tokenizer, data)

    for name, data in (("train-newsqa.json", train_squad), ("dev-newsqa.json", dev_squad)):
        path = os.path.join(args.data_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
