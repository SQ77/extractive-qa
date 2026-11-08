"""Paired bootstrap significance tests.

Compares the SQuAD-only baseline, sequential fine-tune, joint training and distillation-regularized
fine-tune, using per-question EM/F1 from the official scorer. CPU only; needs numpy.

By default it reads the saved predictions in qa/. To test predictions produced on the cluster
(train.py -> outputs_train/, infer.py -> outputs/), override individual files, e.g.:
    python bootstrap_significance.py \
        --squad_pred baseline=outputs/predictions_qa_model_dev-v1.1.json \
        --squad_pred sequential=outputs/predictions_qa_model_newsqa_dev-v1.1.json

If predictions_squad_baseline.json is missing, create it with infer.py first:
    python infer.py --model_dir qa_model --dev_files qa/dev-v1.1.json --out_dir outputs
and pass it with --squad_pred baseline=outputs/predictions_qa_model_dev-v1.1.json.
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np

N_RESAMPLES = 10000
SEED = 42

# model -> predictions file name (inside --pred_dir)
SQUAD_PREDICTIONS = {
    "baseline": "predictions_squad_baseline.json",
    "sequential": "predictions_squad_after_newsqa_finetune.json",
    "joint": "predictions_squad_after_joint.json",
    "distillation": "predictions_squad_after_newsqa_distill.json",
}
NEWSQA_PREDICTIONS = {
    "sequential": "predictions_newsqa_finetuned.json",
    "joint": "predictions_newsqa_after_joint.json",
    "distillation": "predictions_newsqa_distill.json",
}


def load_scorer(path):
    # The official script has a hyphen in its name, so load it by path.
    spec = importlib.util.spec_from_file_location("squad_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def per_question_scores(squad_eval, dev_file, predictions_file):
    with open(dev_file, encoding="utf-8") as f:
        dataset = json.load(f)["data"]
    with open(predictions_file, encoding="utf-8") as f:
        preds = json.load(f)
    return squad_eval.get_raw_scores(dataset, preds)  # (exact_raw, f1_raw)


def paired_bootstrap(scores_a, scores_b, qids, n_resamples=N_RESAMPLES, seed=SEED):
    """Berg-Kirkpatrick et al. (2012)-style paired bootstrap.

    Returns (mean_a, mean_b, observed_diff, p_value, ci_low, ci_high): p_value is the two-sided
    probability that the sign of the difference flips under resampling; the CI is a 95% percentile
    interval on the resampled a-minus-b difference.
    """
    rng = np.random.default_rng(seed)
    a = np.array([scores_a[q] for q in qids])
    b = np.array([scores_b[q] for q in qids])
    n = len(qids)

    observed_diff = a.mean() - b.mean()
    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        diffs[i] = a[idx].mean() - b[idx].mean()

    if observed_diff >= 0:
        p_value = 2 * np.mean(diffs <= 0)
    else:
        p_value = 2 * np.mean(diffs >= 0)
    p_value = min(p_value, 1.0)

    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    return a.mean(), b.mean(), observed_diff, p_value, ci_low, ci_high


def compare_all_pairs(squad_eval, dev_file, predictions_by_model, dataset_name, n_resamples):
    exact_by_model, f1_by_model = {}, {}
    for name, pred_file in predictions_by_model.items():
        exact_by_model[name], f1_by_model[name] = per_question_scores(squad_eval, dev_file, pred_file)

    qids = sorted(set.intersection(*(set(d) for d in exact_by_model.values())))
    print(f"\n=== {dataset_name} ({len(qids)} questions) ===", flush=True)

    names = list(predictions_by_model)
    for metric, scores_by_model in (("EM", exact_by_model), ("F1", f1_by_model)):
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                mean_a, mean_b, diff, p, lo, hi = paired_bootstrap(
                    scores_by_model[a], scores_by_model[b], qids, n_resamples
                )
                sig = "significant" if p < 0.05 else "not significant"
                print(
                    f"{metric}: {a} ({mean_a*100:.2f}) vs {b} ({mean_b*100:.2f}) "
                    f"diff={diff*100:+.2f} 95% CI=[{lo*100:+.2f}, {hi*100:+.2f}] "
                    f"p={p:.4f} ({sig} at alpha=0.05)",
                    flush=True,
                )


def parse_overrides(items):
    out = {}
    for item in items or []:
        name, sep, path = item.partition("=")
        if not sep:
            sys.exit(f"Expected MODEL=PATH, got {item!r}")
        out[name] = path
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="qa", help="Holds the dev files and the official scorer.")
    ap.add_argument("--pred_dir", default="qa", help="Where the default prediction files live.")
    ap.add_argument("--eval_script", default=None, help="Default: <data_dir>/evaluate-v2.0.py")
    ap.add_argument("--squad_pred", action="append", metavar="MODEL=PATH",
                    help="Override a SQuAD predictions file (repeatable). Models: " + ", ".join(SQUAD_PREDICTIONS))
    ap.add_argument("--newsqa_pred", action="append", metavar="MODEL=PATH",
                    help="Override a NewsQA predictions file (repeatable). Models: " + ", ".join(NEWSQA_PREDICTIONS))
    ap.add_argument("--n_resamples", type=int, default=N_RESAMPLES)
    args = ap.parse_args()

    squad_dev = os.path.join(args.data_dir, "dev-v1.1.json")
    newsqa_dev = os.path.join(args.data_dir, "dev-newsqa.json")
    eval_script = args.eval_script or os.path.join(args.data_dir, "evaluate-v2.0.py")

    squad_preds = {m: os.path.join(args.pred_dir, f) for m, f in SQUAD_PREDICTIONS.items()}
    newsqa_preds = {m: os.path.join(args.pred_dir, f) for m, f in NEWSQA_PREDICTIONS.items()}
    for table, overrides, flag in ((squad_preds, args.squad_pred, "--squad_pred"),
                                   (newsqa_preds, args.newsqa_pred, "--newsqa_pred")):
        for m, p in parse_overrides(overrides).items():
            if m not in table:
                sys.exit(f"Unknown model {m!r} for {flag}; choose from {list(table)}")
            table[m] = p

    missing = [p for p in [squad_dev, newsqa_dev, eval_script, *squad_preds.values(), *newsqa_preds.values()]
               if not os.path.exists(p)]
    if missing:
        sys.exit("Missing files:\n  " + "\n  ".join(missing))

    squad_eval = load_scorer(eval_script)
    compare_all_pairs(squad_eval, squad_dev, squad_preds, "SQuAD dev (retention)", args.n_resamples)
    compare_all_pairs(squad_eval, newsqa_dev, newsqa_preds, "NewsQA dev (adaptation)", args.n_resamples)


if __name__ == "__main__":
    main()
