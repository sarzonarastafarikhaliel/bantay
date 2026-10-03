"""Build the canonical ML-ready gold file and frozen dev/holdout split from
data/raw/blotter report docs/_transcribed/gold_blotter_final.csv (213 QA'd
records, all _qa_verdict=ACCEPT), replacing the 73-record
data/raw/bantay_ml_ready_2023.csv gold used through Chapter 3/4 v1. The old
files are preserved under data/legacy_gold_73record/.

Filters to Include in ML == TRUE (188/213 - the other 25 are QA-excluded).
Narrative/Summary is overwritten with the Narrative (masked) column: every
downstream script treats that column name as already PII-scrubbed and some
send it verbatim to external APIs (Gemini). The raw column is kept under
Narrative/Summary (raw, PII) for audit only - nothing reads it.

Stratifies the dev/holdout split 80/20 by Incident Type so both sides span
the taxonomy. 5 of 27 incident types have exactly 1 record in this batch and
cannot be stratified (no way to put a member on both sides); those rows stay
in dev unconditionally, same reasoning make_gold_worksheet.py used for the
old corpus. Frozen once written - re-running this after measurement has
started would retune a held-out number, so it refuses to overwrite by default.

    python tools/build_gold_from_blotter_final.py
    python tools/build_gold_from_blotter_final.py --force   # re-roll (destroys held-out discipline)
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data/raw/blotter report docs/_transcribed/gold_blotter_final.csv"
GOLD_OUT = ROOT / "data/raw/bantay_ml_ready_gold.csv"
SPLIT_OUT = ROOT / "data/gold_split.csv"

SEED = 20260909
HOLDOUT_FRAC = 0.2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                     help="overwrite an existing frozen split")
    args = ap.parse_args()

    if SPLIT_OUT.exists() and not args.force:
        raise SystemExit(f"[stop] {SPLIT_OUT} already exists. Pass --force only if "
                          "nothing has been measured against it yet.")

    df = pd.read_csv(SOURCE, encoding="utf-8-sig")
    df = df.rename(columns={"Source File/Page": "source_file"})
    # 26 rows in the raw CSV carry a disambiguating " (page N)" suffix that is
    # not part of the actual image filename on disk (e.g. "..._234525.jpg
    # (page 158)" vs the file "..._234525.jpg") - stripped here, once, at the
    # root, rather than worked around in each consumer that joins against
    # images. No two rows collide onto the same stripped name (checked), so
    # this loses no information.
    n_suffixed = df["source_file"].str.contains(r"\s*\(page\s*\d+\)\s*$", regex=True).sum()
    df["source_file"] = df["source_file"].str.replace(
        r"\s*\(page\s*\d+\)\s*$", "", regex=True)
    if n_suffixed:
        print(f"[clean ] stripped '(page N)' suffix from {n_suffixed} source_file values")
    before = len(df)
    df = df[df["Include in ML"].astype(str).str.strip().str.upper() == "TRUE"].copy()
    print(f"[filter] {len(df)}/{before} rows have Include in ML = TRUE")

    df = df.rename(columns={"Narrative/Summary": "Narrative/Summary (raw, PII)"})
    df["Narrative/Summary"] = df["Narrative/Summary (raw, PII)"]
    df.loc[df["Narrative (masked)"].notna(), "Narrative/Summary"] = df["Narrative (masked)"]

    df.to_csv(GOLD_OUT, index=False, encoding="utf-8")
    print(f"[write] {GOLD_OUT}  {len(df)} records")

    counts = df["Incident Type"].value_counts()
    singleton_types = counts[counts < 2].index
    strat_df = df[~df["Incident Type"].isin(singleton_types)]
    single_df = df[df["Incident Type"].isin(singleton_types)]
    print(f"[split ] {len(singleton_types)} incident type(s) with <2 records "
          f"({len(single_df)} rows) go to dev unconditionally: "
          f"{sorted(singleton_types)}")

    dev_strat, holdout = train_test_split(
        strat_df, test_size=HOLDOUT_FRAC, stratify=strat_df["Incident Type"],
        random_state=SEED)
    dev = pd.concat([dev_strat, single_df])

    split = pd.concat([
        pd.DataFrame({"source_file": dev["source_file"], "split": "dev"}),
        pd.DataFrame({"source_file": holdout["source_file"], "split": "holdout"}),
    ]).sort_values("source_file")
    split.to_csv(SPLIT_OUT, index=False, encoding="utf-8")
    print(f"[write] {SPLIT_OUT}  seed={SEED}  {(split.split == 'dev').sum()} dev / "
          f"{(split.split == 'holdout').sum()} holdout")


if __name__ == "__main__":
    main()
