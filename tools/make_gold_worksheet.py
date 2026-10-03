"""Freeze the dev/holdout split and emit the Day 1 AM verification worksheet.

Why this exists: `Encoded By` on all 73 rows of data/raw/bantay_ml_ready_2023.csv
reads "Claude (automated transcription from source photograph)". Nothing in the
project records a human ever checking it. So the gold every accuracy number will
be measured against was produced by a model reading the same images the pipeline
is about to be tested on. That is circular, and it is the single cheapest thing
to fix before any measurement is taken - see OCR_ACCURACY_PLAN.md 7.2.

Run once. It writes two files:

  data/gold_split.csv       source_file -> dev | holdout. FROZEN. Seeded shuffle,
                            so re-running reproduces it exactly. Never regenerate
                            it after you have started measuring: a split that
                            moves under you turns a held-out number into a tuned
                            one, quietly.

  data/gold_worksheet.csv   the pages to verify by hand, one row per page,
                            gold_* columns pre-filled from the automated
                            transcription and fix_* columns blank. --split
                            chooses which side; --out keeps the two sheets apart.

VERIFY THE HOLDOUT FIRST. The dev sheet was written first because dev is where
the work happens, but the holdout is where the THESIS NUMBER comes from, and an
unverified holdout means that number is Claude grading Claude:

    python tools/make_gold_worksheet.py --split holdout         --n-verify 15 --out data/gold_worksheet_holdout.csv

Filling it in: open the worksheet next to data/raw/Blotter Pics, and for each row
look at the photo. Leave fix_* EMPTY when the automated value is right - blank
means "checked and correct", which is why `verified` is a separate column. Put
the correct value in fix_* when it is wrong. Type NONE in fix_* when the page
does not state that field at all and the transcriber invented one.

The corrections are the point twice over: eval_fields.py applies them, and their
RATE is a Chapter 4 finding in its own right - it is the measured error rate of
LLM transcription on handwritten Filipino blotter pages, which is a number the
prior literature does not report.

PII: this worksheet holds no names. The gold narrative is already mask-scrubbed
([PERSON_7]) and no reporting_party / respondent columns exist to fill, so
nothing here creates a new personal-data surface. Keep it that way - if you add
name columns, that file stops being safe to commit.

Usage:
    python tools/make_gold_worksheet.py
    python tools/make_gold_worksheet.py --n-verify 10   # if handwriting is slow
"""
import argparse
import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "data/raw/bantay_ml_ready_gold.csv"
IMAGES = ROOT / "data/raw/Blotter Pics"

# Frozen. Changing this re-rolls the split and invalidates every number measured
# against the old one, so it is a constant here rather than a flag.
SEED = 20260822
N_HOLDOUT = 15

# The gold columns that correspond to the record template's slots. Two slots -
# reporting_party and respondent - have no gold column at all; see eval_fields.py
# for why they are reported as out-of-scope rather than scored badly.
FIELD_MAP = {
    "date": "Date",
    "time": "Time",
    "location_purok": "Location/Purok",
    "incident_summary": "Narrative/Summary",
    "action_taken": "Action Taken",
    "status": "Status",
    "incident_type": "Incident Type",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-verify", type=int, default=20,
                    help="how many pages to put on the worksheet (default 20)")
    ap.add_argument("--split", default="dev", choices=["dev", "holdout"],
                    help="which side of the split to verify (default dev)")
    ap.add_argument("--out", default="data/gold_worksheet.csv",
                    help="worksheet path, so a holdout sheet does not collide with the dev one")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing gold_split.csv (destroys held-out discipline)")
    ap.add_argument("--gold", default=str(GOLD),
                    help="annotation CSV to split; a newly transcribed batch needs its own "
                         "split, since gold_split.csv only lists the pages it was built from")
    ap.add_argument("--n-holdout", type=int, default=N_HOLDOUT,
                    help=f"pages held out (default {N_HOLDOUT}); scale it with the batch")
    args = ap.parse_args()

    df = pd.read_csv(args.gold)
    split_path = ROOT / "data/gold_split.csv"

    if split_path.exists() and not args.force:
        split = pd.read_csv(split_path)
        print(f"[keep] {split_path.name} already exists - reusing it.")
        print("       Re-rolling the split after measuring has started would turn a")
        print("       held-out number into a tuned one. Pass --force only if nothing")
        print("       has been measured yet.")
    else:
        files = sorted(df["Source File/Page"])
        random.Random(SEED).shuffle(files)
        split = pd.DataFrame({
            "source_file": files,
            "split": ["holdout"] * args.n_holdout + ["dev"] * (len(files) - args.n_holdout),
        }).sort_values("source_file")
        split.to_csv(split_path, index=False, encoding="utf-8")
        print(f"[write] {split_path}  seed={SEED}  "
              f"{(split.split == 'dev').sum()} dev / {(split.split == 'holdout').sum()} holdout")

    wanted = set(split.loc[split.split == args.split, "source_file"])

    # Verify the pages in file order rather than a second random draw: the person
    # doing this is working through a stack of photos, and jumping around the
    # logbook to satisfy a shuffle costs real time for no statistical gain.
    rows = df[df["Source File/Page"].isin(wanted)].sort_values("Source File/Page")
    rows = rows.head(args.n_verify)

    missing_img = [f for f in rows["Source File/Page"] if not (IMAGES / f).exists()]
    if missing_img:
        print(f"[warn] {len(missing_img)} worksheet pages have no image on disk: "
              f"{missing_img[:3]}")

    out = pd.DataFrame({"source_file": rows["Source File/Page"].values})
    for slot, col in FIELD_MAP.items():
        out[f"gold_{slot}"] = rows[col].fillna("").astype(str).values
        out[f"fix_{slot}"] = ""
    out["verified"] = ""          # put your initials here once the row is checked
    out["notes"] = ""

    ws_path = ROOT / args.out
    if ws_path.exists():
        print(f"[stop] {ws_path.name} exists. Refusing to overwrite - it may hold "
              f"corrections you have already typed.")
        print("       Delete or rename it first if you really want a fresh sheet.")
    else:
        out.to_csv(ws_path, index=False, encoding="utf-8-sig")   # BOM so Excel opens UTF-8
        print(f"[write] {ws_path}  {len(out)} pages x {len(FIELD_MAP)} fields")

    print()
    print(f"Next: open {args.out} beside data/raw/Blotter Pics.")
    print("      fix_* blank = checked and correct.  fix_* = NONE = page does not state it.")
    print("      Then: python tools/eval_fields.py")


if __name__ == "__main__":
    main()
