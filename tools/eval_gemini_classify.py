"""Standalone accuracy of the Gemini few-shot classifier against gold labels.

Answers one question: does a model with NO fine-tuning on this corpus do
reasonably at 35-way incident-type classification, given the trained
classifier (since removed) measured 6.7% real-test / 10.2% synthetic-test
accuracy (models/eval_report.json, models/clf_ablation.json)? This does not
diagnose or fix that training result - it is a separate, cheaper question,
and the two are reported side by side, not combined into one number.

Uses the same 73-real-record gold and dev/holdout split as the OCR evaluation
(data/gold_split.csv) for methodological consistency - dev for this first
measurement, holdout reserved for later. Narratives are already PII-masked in
bantay_ml_ready_2023.csv ([PERSON_n] tokens), same input the trained
classifier reads.

    python tools/eval_gemini_classify.py                 dev split
    python tools/eval_gemini_classify.py --limit 5        smoke test first
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# bantay_ml_ready_2023.csv's Incident Type column holds the 23 free-form
# ENCODED labels from the original logbook transcription, not the 35-item
# canonical taxonomy bantay/normalize.py and this module's TYPE_NAMES answer in.
# PIPELINE.md sec 2 documents this exact bridge (with per-row rationale, in
# out/label_bridge_audit.csv from the semisynth pipeline) - reproduced here
# because comparing a canonical-taxonomy prediction against the un-bridged
# column undercounts correct answers as wrong. Two rows (Consumer / Fare
# Dispute, Property Deposit / Safekeeping) were LOW-confidence bridge calls
# per that table, kept here for consistency, not because they are certain.
LABEL_BRIDGE = {
    "Molestation": "Peeping/Voyeurism",
    "Grave Threats": "Grave Threats / Light Threats",
    "Vehicular Accident": "Vehicular Incident",
    "Trespassing / Suspicious Persons": "Trespassing",
    "Neighbor / Nuisance Complaint": "Neighbor Dispute",
    "Contract Dispute / Estafa": "Estafa (Swindling)",
    "Threats / Coercion (Financial)": "Coercion",
    "Other / Unclear": "Others / Miscellaneous",
    "Verbal Altercation / Disturbance": "Public Disturbance",
    "Rental / Lease Dispute": "Breach of Contract",
    "Consumer / Fare Dispute": "Breach of Contract",
    "Property Deposit / Safekeeping": "Property Claim / Damage",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "holdout", "all"])
    ap.add_argument("--limit", type=int, help="only the first N pages (smoke test)")
    args = ap.parse_args()

    from bantay.ml.gemini_classify import available, classify

    if not available():
        sys.exit("No Gemini backend configured. Run python -m bantay.ml.gemini_classify first.")

    gold = pd.read_csv(ROOT / "data/raw/bantay_ml_ready_gold.csv")
    gold = gold.rename(columns={"Source File/Page": "source_file"})

    split = pd.read_csv(ROOT / "data/gold_split.csv")
    keep = set(split["source_file"] if args.split == "all"
               else split.loc[split.split == args.split, "source_file"])
    rows = gold[gold.source_file.isin(keep)].sort_values("source_file")
    if args.limit:
        rows = rows.head(args.limit)

    print(f"{len(rows)} records = {len(rows)} API calls, split={args.split}")

    preds, statuses = [], Counter()
    for i, (idx, r) in enumerate(rows.iterrows(), 1):
        narrative = str(r.get("Narrative/Summary") or "")
        out = classify(narrative)
        statuses[out["status"].split(":")[0].split("(")[0].strip()] += 1
        preds.append({"source_file": r["source_file"], "gold_type": r.get("Incident Type"),
                      "gold_group": r.get("Category Group"),
                      "pred_type": out["incident_type"], "pred_group": out["category_group"],
                      "status": out["status"]})
        print(f"  [{i:>2}/{len(rows)}] {r['source_file']:<24} "
              f"gold={preds[-1]['gold_type']!s:<30} pred={out['incident_type']!s:<30} {out['status'][:40]}")

    preds_df = pd.DataFrame(preds)
    out_path = ROOT / "models/preds_gemini_classify.csv"
    preds_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n[write] {out_path}")

    n_err = sum(v for k, v in statuses.items() if k == "error")
    scored = preds_df[preds_df["pred_type"].notna()]
    print(f"\nstatuses: {dict(statuses)}")
    if n_err == len(rows):
        sys.exit("EVERY page errored - broken config, not a result. "
                 "Run python -m bantay.ml.gemini_classify to check.")

    scored = scored.copy()
    scored["gold_bridged"] = scored["gold_type"].map(lambda g: LABEL_BRIDGE.get(g, g))
    n_bridged = scored["gold_type"].isin(LABEL_BRIDGE).sum()

    raw_acc = (scored["pred_type"] == scored["gold_type"]).mean() if len(scored) else 0.0
    bridged_acc = (scored["pred_type"] == scored["gold_bridged"]).mean() if len(scored) else 0.0
    print(f"\n{n_bridged}/{len(scored)} gold labels needed the encoded->canonical bridge rewrite.")
    print(f"type accuracy, raw column        (n={len(scored)}): {raw_acc:.1%}  <- undercounts, do not report")
    print(f"type accuracy, canonical bridged  (n={len(scored)}): {bridged_acc:.1%}  <- the honest number")

    if scored["gold_group"].notna().any():
        g = scored[scored["gold_group"].notna()]
        group_acc = (g["pred_group"] == g["gold_group"]).mean() if len(g) else None
        if group_acc is not None:
            print(f"group accuracy (derived from predicted type, n={len(g)}): {group_acc:.1%}")

    mism = scored[scored["pred_type"] != scored["gold_bridged"]][
        ["source_file", "gold_type", "gold_bridged", "pred_type"]]
    if len(mism):
        print(f"\nremaining mismatches after bridging (n={len(mism)}):")
        print(mism.to_string(index=False))

    print("\ncompare against models/eval_report.json's real_test.type_accuracy (0.0667, n=15)")
    print("and models/clf_ablation.json's best val macro F1 (0.1054-0.1537) for the trained classifier.")


if __name__ == "__main__":
    main()
