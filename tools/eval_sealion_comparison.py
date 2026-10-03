"""Adds a SEA-LION row to models/comparison_results.json, evaluated the same
way as the six sklearn models in bantay/ml/train.py: same held-out test split
(data/raw/bantay_corpus_2100.csv's _split=='test', 340 rows: 325 synthetic +
15 real), same masked input (mask_names() only - clean_text() is TF-IDF
tokenization-specific and would degrade an LLM prompt, not help it), same
metric functions (_tier_metrics/_group_metrics/_subset_metrics reused
directly from train.py), so the row lands in the table on the same basis as
the other six instead of a separately-invented methodology.

SEA-LION is zero-shot per call, not a fit()/predict() estimator - there is no
training step, and one classify() call happens per test row. That is also why
a full 340-row run is slow (a live local model call, not sklearn's
near-instant predict()): each row costs several seconds to tens of seconds,
so this defaults to a smaller sample rather than the full split. Pass
--n 340 for the true apples-to-apples number once the time budget allows it -
the "best_model" field is only updated from a full-340 run, never a sample,
so a quick run cannot accidentally promote itself over models measured on
the complete test set.

Real rows are prioritized in the sample over synthetic ones: with only 15
real rows in the whole test split, a proportional random sample at n=50 would
land 1-2 real rows, and that is the subset train.py's own CLI output calls
"the number to report" (see its main()). All 15 real rows are always
included; synthetic rows fill the rest of n.

    python tools/eval_sealion_comparison.py                    n=50 sample (default)
    python tools/eval_sealion_comparison.py --n 340             the full test split
    python tools/eval_sealion_comparison.py --model aisingapore/Gemma-SEA-LION-v4-4B-VL
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bantay.ml.mask import mask_names
from bantay.ml.train import _group_metrics, _subset_metrics, _tier_metrics, load_dataset


def _sample_test_rows(df, n, seed=42):
    """All real test rows, plus synthetic rows filling the rest of n."""
    test = df[df["_split"] == "test"]
    real = test[test["_provenance"] == "real"]
    synth = test[test["_provenance"] == "synthetic"]
    n_synth = max(0, n - len(real))
    if n_synth < len(synth):
        synth = synth.sample(n=n_synth, random_state=seed)
    return pd.concat([real, synth]).sample(frac=1, random_state=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(ROOT / "data/raw/bantay_corpus_2100.csv"))
    ap.add_argument("--n", type=int, default=50,
                    help="test rows to sample (340 = the full test split, directly comparable)")
    ap.add_argument("--model", default=None, help="Ollama model tag (default: sealion_classify's own default)")
    ap.add_argument("--results", default=str(ROOT / "models/comparison_results.json"))
    args = ap.parse_args()

    from bantay.ml.sealion_classify import TYPE_NAMES, available, classify

    if not available():
        sys.exit("BANTAY_OLLAMA is not set to 1 - set it before running this comparison.")

    df = load_dataset(args.input)
    if "_split" not in df.columns or not (df["_split"] == "test").any():
        sys.exit(f"{args.input} has no usable _split=='test' rows.")

    unknown = set(df["incident_type_primary"].unique()) - set(TYPE_NAMES)
    if unknown:
        sys.exit(f"Corpus has labels SEA-LION's TYPE_NAMES cannot produce: {unknown} - "
                 f"bridge them first, same reasoning as eval_gemini_classify.py's LABEL_BRIDGE.")

    total_test = int((df["_split"] == "test").sum())
    sample = _sample_test_rows(df, args.n)
    n_real = int((sample["_provenance"] == "real").sum())
    n_synth = len(sample) - n_real
    # Resolve the actual model name for the table label, not just "whatever
    # the default happens to be" - classify() itself does this same
    # os.environ.get(...) fallback resolution, replicated here only for
    # display so the label matches what the calls actually ran against.
    import os as _os
    model = args.model or _os.environ.get("BANTAY_OLLAMA_MODEL")
    if not model:
        from bantay.ml.sealion_classify import _DEFAULT_MODEL
        model = _DEFAULT_MODEL
    print(f"{len(sample)}/{total_test} test rows ({n_real} real, {n_synth} synthetic), model={model}")

    # Full-corpus label space, exactly as train_and_compare() builds it, so
    # this LabelEncoder assigns the identical integer<->label mapping the
    # other six models' tier_f1/group_f1 were computed against.
    label_encoder = LabelEncoder()
    label_encoder.fit(df["incident_type_primary"].tolist())

    y_true_labels, y_pred_labels, statuses = [], [], Counter()
    started = time.perf_counter()
    for i, (_, r) in enumerate(sample.iterrows(), 1):
        masked = mask_names(r["narrative"], r.get("remarks", ""))
        out = classify(masked, model=args.model)
        statuses[out["status"].split(":")[0].split("(")[0].strip()] += 1
        # A dead/no-answer call still has to score as SOMETHING for sklearn's
        # metrics to run - FALLBACK_CATEGORY is "no usable prediction", not a
        # guess dressed up as one, and it is almost never the gold label.
        pred = out["incident_type"] or TYPE_NAMES[-1]
        y_true_labels.append(r["incident_type_primary"])
        y_pred_labels.append(pred)
        print(f"  [{i:>3}/{len(sample)}] gold={r['incident_type_primary']:<30} pred={pred:<30} {out['status'][:40]}")
    elapsed = time.perf_counter() - started

    y_true = label_encoder.transform(y_true_labels)
    y_pred = label_encoder.transform(y_pred_labels)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)
    label = f"SEA-LION ({model})" if len(sample) >= total_test else f"SEA-LION ({model}, n={len(sample)} sample)"
    entry = {
        "model": label,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "cv_folds": None,
        "tier_f1": _tier_metrics(y_true, y_pred, label_encoder),
        "group_f1": _group_metrics(y_true, y_pred, label_encoder),
    }
    real_mask = (sample["_provenance"] == "real").to_numpy()
    if real_mask.any():
        entry["real_only"] = _subset_metrics(y_true, y_pred, real_mask, label_encoder)
        entry["synthetic_only"] = _subset_metrics(y_true, y_pred, ~real_mask, label_encoder)

    print(f"\nstatuses: {dict(statuses)}  ({elapsed:.0f}s total, {elapsed / max(len(sample), 1):.1f}s/row)")
    print(json.dumps(entry, indent=2))

    results_path = Path(args.results)
    data = {"best_model": None, "results": []}
    if results_path.exists():
        data = json.loads(results_path.read_text())

    data["results"] = [r for r in data["results"] if not str(r.get("model", "")).startswith("SEA-LION")]
    data["results"].append(entry)

    # Only a full-340 run is on the same footing as the other six (they were
    # all evaluated on the complete test split) - a sample cannot promote
    # itself to "best_model" over numbers measured on more data.
    if len(sample) >= total_test:
        current_best = next((r for r in data["results"] if r["model"] == data.get("best_model")), None)
        if current_best is None or entry["f1"] > current_best.get("f1", -1):
            data["best_model"] = entry["model"]

    results_path.write_text(json.dumps(data, indent=2))
    print(f"\n[write] {results_path}")


if __name__ == "__main__":
    main()
