"""Accuracy comparison of local Ollama SEA-LION models at 35-way incident-type
classification, same methodology as eval_gemini_classify.py (dev split of
data/gold_split.csv, LABEL_BRIDGE for the encoded->canonical taxonomy gap) -
reused directly from that module rather than re-derived, so the numbers land
on the same scale as Gemini's already-measured 69.0% and the trained
classifier's 6.7% (models/eval_report.json).

Runs one or more Ollama model tags back-to-back over the same records and
reports each one's accuracy side by side - the point is comparing candidates
against each other and against the existing measurements, not a verdict on
any single model.

    python tools/eval_sealion_classify.py                 all 3 pulled models, dev split (58 records each)
    python tools/eval_sealion_classify.py --limit 5        smoke test first
    python tools/eval_sealion_classify.py --models aisingapore/llama-sea-lion-v3.5-8b-r
"""
import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.eval_gemini_classify import LABEL_BRIDGE

# Exact tags as pulled (`ollama list`) - Ollama preserves the case a model was
# pulled with, so this has to match, not just resolve case-insensitively.
DEFAULT_MODELS = [
    "aisingapore/llama-sea-lion-v3.5-8b-r",
    "aisingapore/Gemma-SEA-LION-v4-4B-VL",
    "aisingapore/Gemma-SEA-LION-v4.5-E2B-IT",
]


def _load_rows(split, limit):
    gold = pd.read_csv(ROOT / "data/raw/bantay_ml_ready_gold.csv")
    gold = gold.rename(columns={"Source File/Page": "source_file"})
    split_df = pd.read_csv(ROOT / "data/gold_split.csv")
    keep = set(split_df["source_file"] if split == "all"
               else split_df.loc[split_df.split == split, "source_file"])
    rows = gold[gold.source_file.isin(keep)].sort_values("source_file")
    return rows.head(limit) if limit else rows


def _run_one(model, rows):
    from bantay.ml.sealion_classify import classify

    preds, statuses = [], Counter()
    started = time.perf_counter()
    for i, (_, r) in enumerate(rows.iterrows(), 1):
        narrative = str(r.get("Narrative/Summary") or "")
        out = classify(narrative, model=model)
        statuses[out["status"].split(":")[0].split("(")[0].strip()] += 1
        preds.append({"source_file": r["source_file"], "gold_type": r.get("Incident Type"),
                      "gold_group": r.get("Category Group"),
                      "pred_type": out["incident_type"], "pred_group": out["category_group"],
                      "status": out["status"]})
        print(f"  [{i:>2}/{len(rows)}] {r['source_file']:<24} "
              f"gold={preds[-1]['gold_type']!s:<30} pred={out['incident_type']!s:<30} {out['status'][:40]}")
    elapsed = time.perf_counter() - started

    preds_df = pd.DataFrame(preds)
    safe_name = model.replace("/", "_").replace(":", "_").replace(".", "_").lower()
    out_path = ROOT / f"models/preds_sealion_{safe_name}.csv"
    preds_df.to_csv(out_path, index=False, encoding="utf-8")

    scored = preds_df[preds_df["pred_type"].notna()].copy()
    n_err = sum(v for k, v in statuses.items() if k == "error")
    if len(scored):
        scored["gold_bridged"] = scored["gold_type"].map(lambda g: LABEL_BRIDGE.get(g, g))
        acc = (scored["pred_type"] == scored["gold_bridged"]).mean()
    else:
        acc = 0.0

    return {"model": model, "n": len(rows), "n_scored": len(scored), "n_err": n_err,
            "accuracy": acc, "seconds": elapsed, "statuses": dict(statuses),
            "csv": str(out_path)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "holdout", "all"])
    ap.add_argument("--limit", type=int, help="only the first N records per model (smoke test)")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                     help="Ollama model tags to compare (must already be pulled)")
    args = ap.parse_args()

    from bantay.ml.sealion_classify import available
    if not available():
        sys.exit("BANTAY_OLLAMA is not set to 1 - set it before running this comparison.")

    rows = _load_rows(args.split, args.limit)
    print(f"{len(rows)} records, split={args.split}, {len(args.models)} model(s) to compare\n")

    results = []
    for model in args.models:
        print(f"=== {model} ===")
        results.append(_run_one(model, rows))
        print()

    print("=" * 78)
    print(f"{'model':<42} {'accuracy':>9} {'scored/n':>10} {'errors':>7} {'time':>8}")
    for r in results:
        print(f"{r['model']:<42} {r['accuracy']:>8.1%} "
              f"{r['n_scored']:>4}/{r['n']:<5} {r['n_err']:>7} {r['seconds']:>7.0f}s")
    print("\ncompare against Gemini's 69.0% (tools/eval_gemini_classify.py) and the trained")
    print("classifier's 6.7% real-test accuracy (models/eval_report.json).")


if __name__ == "__main__":
    main()
