"""Day 3 AM: the reconciliation headline number.

OCR_ACCURACY_PLAN.md Phase 3's gate is "field accuracy conditional on
agreement" - the number that justifies auto-accepting the pages where the two
readers agree, exactly as dual_axis_loocv.json does for the incident-type
classifier (docs/REIMPLEMENTATION.md 3, "four training decisions").

No new API calls: this scores the predictions already collected by
tools/run_gemini_arms.py (models/preds_gemini_{text,image}.csv) against
models/preds_baseline.csv (the regex reader) using bantay/ocr/reconcile.py,
against the same human-verified gold tools/eval_fields.py uses.

    python tools/eval_reconcile.py                 both arms, dev split
    python tools/eval_reconcile.py --arm text
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bantay.ocr.reconcile import reconcile  # noqa: E402
from tools.eval_fields import GOLD_COLUMNS, NARRATIVE_GOLD, NARRATIVE_SLOT, load_gold, norm_value  # noqa: E402

SLOTS = list(GOLD_COLUMNS) + [NARRATIVE_SLOT]
GOLD_COL = dict(GOLD_COLUMNS, **{NARRATIVE_SLOT: NARRATIVE_GOLD})


def score_arm(arm, split="dev"):
    split_df = pd.read_csv(ROOT / "data/gold_split.csv")
    keep = set(split_df.loc[split_df.split == split, "source_file"] if split != "all"
               else split_df["source_file"])

    gold, n_verified, n_fixed = load_gold(apply_fixes=True)
    gold = gold[gold.source_file.isin(keep)].set_index("source_file")

    regex_df = pd.read_csv(ROOT / "models/preds_baseline.csv").fillna("").set_index("source_file")
    model_df = pd.read_csv(ROOT / f"models/preds_gemini_{arm}.csv").fillna("").set_index("source_file")

    pages = [p for p in keep if p in regex_df.index and p in model_df.index and p in gold.index]
    print(f"[gold ] {len(gold)} pages   {n_verified} human-verified, {n_fixed} corrections applied")
    print(f"[pages] {len(pages)} pages have both a regex and a {arm} reading")

    rows = {slot: {"agree_correct": 0, "agree_wrong": 0, "agree_n": 0,
                   "disagree_correct": 0, "disagree_wrong": 0, "disagree_n": 0}
            for slot in SLOTS}

    for src in pages:
        model_fields = {s: model_df.loc[src, s] for s in SLOTS if s in model_df.columns}
        regex_fields = {s: regex_df.loc[src, s] for s in SLOTS if s in regex_df.columns}
        recon = reconcile(model_fields, regex_fields)

        for slot in SLOTS:
            row = recon.get(slot, {"value": "", "agree": True})
            want = norm_value(gold.loc[src, GOLD_COL[slot]])
            if not want:
                continue                      # no gold for this page/field
            got = norm_value(row["value"])
            correct = got == want
            bucket = "agree" if row["agree"] else "disagree"
            rows[slot][f"{bucket}_n"] += 1
            rows[slot][f"{bucket}_{'correct' if correct else 'wrong'}"] += 1

    print(f"\n{'=' * 88}\nreconciled: regex (baseline) x gemini_{arm}   split={split}\n{'=' * 88}")
    print(f"{'field':<20}{'agree n':>10}{'agree acc':>12}{'disagree n':>12}{'disagree acc':>14}")
    print("-" * 88)
    tot_agree_c = tot_agree_n = tot_dis_c = tot_dis_n = 0
    for slot in SLOTS:
        r = rows[slot]
        agree_acc = r["agree_correct"] / r["agree_n"] if r["agree_n"] else None
        dis_acc = r["disagree_correct"] / r["disagree_n"] if r["disagree_n"] else None
        print(f"{slot:<20}{r['agree_n']:>10}{'' if agree_acc is None else f'{agree_acc:.1%}':>12}"
              f"{r['disagree_n']:>12}{'' if dis_acc is None else f'{dis_acc:.1%}':>14}")
        tot_agree_c += r["agree_correct"]; tot_agree_n += r["agree_n"]
        tot_dis_c += r["disagree_correct"]; tot_dis_n += r["disagree_n"]

    total_n = tot_agree_n + tot_dis_n
    print("-" * 88)
    print(f"agreement rate (scored fields): {tot_agree_n}/{total_n} = "
          f"{tot_agree_n / total_n:.1%}" if total_n else "no scored fields")
    if tot_agree_n:
        print(f"accuracy WHEN THE READERS AGREE:    {tot_agree_c / tot_agree_n:.1%}  "
              f"(n={tot_agree_n})  <- the auto-accept case")
    if tot_dis_n:
        print(f"accuracy when they disagree:        {tot_dis_c / tot_dis_n:.1%}  "
              f"(n={tot_dis_n})  <- the review-queue case")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["text", "image", "both"], default="both")
    ap.add_argument("--split", default="dev", choices=["dev", "holdout", "all"])
    args = ap.parse_args()
    for arm in (["text", "image"] if args.arm == "both" else [args.arm]):
        score_arm(arm, args.split)
        print()


if __name__ == "__main__":
    main()
