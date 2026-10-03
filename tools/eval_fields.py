"""Field-level accuracy against the record gold. This is the headline metric.

Why field accuracy and not page CER: the system's job is to fill a blotter
record, not to reproduce a photograph of a page. A page CER punishes correctly
read form furniture as insertion error, punishes Vision for reproducing the
page's reading order when the gold narrative was written in the encoder's, and
punishes every name token because gold is mask-scrubbed and the scanner output is
not. OCR_ACCURACY_PLAN.md 0.3 measures all three: they are why the raw 0.87 CER
in this project cannot be interpreted. None of them touch "did the date land in
the date slot".

The comparator your literature offers is Abubo et al. (2024), who report 96%
accuracy measured by WORD COUNT - tokens produced, not tokens correct. A page of
confident nonsense scores 100% on that metric. Field exact-match is the honest
version of the same claim, and reporting it is the contribution.

  python tools/eval_fields.py                      generate + score the baseline
  python tools/eval_fields.py preds.csv            score any predictions CSV
  python tools/eval_fields.py --split holdout      the Day 3 number. Once. See below.

A predictions CSV needs `source_file` plus any of narrative.SLOTS as columns.
Missing columns score as blank, which is the correct outcome - a pipeline that
never fills `status` should be shown scoring zero on it, not excused.

BASELINE PATH: with no predictions file, this replays the offline pipeline -
cached Vision text -> OCRCorrector -> guess_fields -> narrative.backfill - which
is exactly what routes/scan.py runs when no Gemini backend is configured. It
reads Vision output from data/ocr_calibration.csv rather than re-scanning, so it
costs no API calls and is byte-reproducible.

HELD-OUT DISCIPLINE: --split holdout is for Day 3, once, at the end. If you run
it, look at the number, then change something and run it again, it is no longer
held out and the honest thing is to say so in Limitations. The default is dev.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.audit_ocr_gold import (  # noqa: E402
    FUZZY_FLOOR, MIN_TOKEN_LEN, _MASK_RE, normalise, quality_bucket,
)

# Slots with a gold column. reporting_party and respondent are deliberately
# absent: bantay_ml_ready_2023.csv has no column for either, so there is nothing
# to score them against. Scoring them as zero would be a lie about the pipeline;
# quietly dropping them would be a lie about the coverage. They are reported
# separately as out-of-scope, and that goes in Limitations.
GOLD_COLUMNS = {
    "date": "Date",
    "time": "Time",
    "location_purok": "Location/Purok",
    "action_taken": "Action Taken",
    "status": "Status",
}
NO_GOLD = ("reporting_party", "respondent")
NARRATIVE_SLOT = "incident_summary"
NARRATIVE_GOLD = "Narrative/Summary"


def norm_value(text):
    """Case- and punctuation-insensitive comparison key for a field value.

    Reuses audit_ocr_gold.normalise so a field and a narrative are normalised the
    same way - two normalisers drifting apart is how a metric quietly starts
    measuring its own preprocessing.

    pd.isna first, because a float NaN is TRUTHY: `str(nan or "")` is "nan",
    which normalises to the non-empty token "NAN" and makes an empty gold cell
    look like a gold value. Caught on action_taken, which is filled on 22 of 73
    rows and was scoring n=58.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    return normalise(str(text))


def field_key(slot, value):
    """Comparison key for one field, so a correct answer scores as correct.

    Exact text match on raw cells scored most right answers wrong: the gold
    writes dates "1/2/2026" and the pipeline "2026-01-02", and the gold's
    Location/Purok is a full address ("Purok 3, Anunas") where the pipeline
    stores "Purok 3". Dates and times go through the same normalisers the
    import path uses; a location that names a purok is compared on that purok
    alone. Anything that does not parse falls back to norm_value, so an
    unparseable gold cell still has to be matched verbatim.
    ponytail: a gold location naming two puroks ("Purok 1 / Purok 2") keys on
    the first; score both if multi-purok gold rows grow past a handful.
    """
    from bantay.normalize import find_purok_number, normalize_date, normalize_time

    text = norm_value(value)
    if not text:
        return ""
    raw = str(value)
    parsed = {"date": normalize_date, "time": normalize_time,
              "location_purok": find_purok_number}.get(slot, lambda _: None)(raw)
    return parsed or text


def token_recall(pred, gold):
    """Share of gold content tokens present in the prediction, exact and fuzzy.

    This replaces CER for the narrative, and the reason is PII, not preference.
    Gold reads `[PERSON_7]`; the scanner read the actual surname. Under CER every
    name is a guaranteed miss on both sides, so the number moves with how many
    people a page mentions. Dropping mask tokens from the denominator removes the
    asymmetry entirely instead of pretending it is small.

    Returns (exact, exact_or_fuzzy, n_gold_tokens). n is returned because a page
    whose gold narrative is four words long produces a recall of 0.75 or 1.00 and
    nothing between, and averaging that with a 60-token page without weighting is
    how a mean starts lying.
    """
    import difflib

    gold_tokens = [t for t in normalise(_MASK_RE.sub(" ", str(gold or ""))).split()
                   if len(t) >= MIN_TOKEN_LEN]
    if not gold_tokens:
        return None, None, 0
    pred_tokens = set(normalise(str(pred or "")).split())
    exact = sum(t in pred_tokens for t in gold_tokens)
    fuzzy = exact + sum(
        bool(difflib.get_close_matches(t, pred_tokens, n=1, cutoff=FUZZY_FLOOR))
        for t in gold_tokens if t not in pred_tokens)
    return exact / len(gold_tokens), fuzzy / len(gold_tokens), len(gold_tokens)


def load_gold(apply_fixes=True, path=None):
    """Gold records, with the Day 1 human corrections applied on top.

    A fix_* cell of NONE means the page does not state that field and the
    automated transcriber invented a value - so it clears the gold rather than
    replacing it. That distinction matters: an invented gold value makes a
    correctly-blank prediction score as wrong.
    """
    gold = pd.read_csv(path or ROOT / "data/raw/bantay_ml_ready_gold.csv")
    gold = gold.rename(columns={"Source File/Page": "source_file"})
    # Handwriting stratum, so --by-quality can split any report without the
    # caller re-joining to the annotation CSV.
    gold["quality"] = (gold["Readability"] if "Readability" in gold.columns
                       else "").map(quality_bucket)
    # gold_worksheet.csv is keyed to the retired 73-record corpus's source
    # filenames (Blotter_Pic (N).jpg) and cannot match this gold's filenames
    # (YYYYMMDD_HHMMSS.jpg), so it is scoped to a path that does not exist for
    # the new corpus rather than silently matching nothing under the old name.
    ws_path = ROOT / "data/gold_worksheet_2023_legacy.csv"
    n_fixed = n_verified = 0

    if apply_fixes and ws_path.exists():
        ws = pd.read_csv(ws_path).fillna("")
        n_verified = int((ws.get("verified", pd.Series(dtype=str)).astype(str).str.strip() != "").sum())
        by_file = gold.set_index("source_file")
        for _, row in ws.iterrows():
            src = row["source_file"]
            if src not in by_file.index:
                continue
            for slot, col in list(GOLD_COLUMNS.items()) + [(NARRATIVE_SLOT, NARRATIVE_GOLD)]:
                fix = str(row.get(f"fix_{slot}", "")).strip()
                if not fix:
                    continue
                by_file.loc[src, col] = "" if fix.upper() == "NONE" else fix
                n_fixed += 1
        gold = by_file.reset_index()
    elif "_qa_verdict" in gold.columns:
        # gold_blotter_final.csv carries per-record QA sign-off (see
        # PIPELINE.md) instead of a hand-worksheet fix pass, so "verified"
        # is read off that column rather than data/gold_worksheet.csv.
        n_verified = int((gold["_qa_verdict"].astype(str).str.strip().str.upper() == "ACCEPT").sum())
    return gold, n_verified, n_fixed


def baseline_predictions(sources):
    """Replay the offline scan path on cached Vision output. No API calls.

    Mirrors routes/scan.py's no-Gemini branch exactly - OCRCorrector.correct()
    then guess_fields() then narrative.backfill() - because the point of a
    baseline is that it is the thing currently running, not a reconstruction of
    it that happens to score better.
    """
    from bantay import create_app, narrative
    from bantay.ocr import OCRCorrector
    from bantay.routes.scan import guess_fields

    # guess_fields reads the purok list off current_app, so this runs inside a
    # real app context rather than a stub: the baseline has to be the code that
    # actually serves scans, config included, or it is not a baseline.
    app = create_app()
    calib = pd.read_csv(ROOT / "data/ocr_calibration.csv").set_index("source_file")
    rows = []
    with app.app_context():
        # from_dir with its stock defaults - lm_weight 0.6 and all - because that
        # is what routes/scan.py constructs. Passing lm_weight=0 here would score
        # a corrector nobody is running.
        corrector = OCRCorrector.from_dir(app.config["MODEL_DIR"])
        for src in sources:
            if src not in calib.index:
                continue
            text = str(calib.loc[src, "ocr_text"] or "")
            corrected = corrector.correct(text)["text"]
            fields = narrative.normalize_fields({})
            fields = narrative.backfill(fields, guess_fields(corrected))
            if not fields[NARRATIVE_SLOT]:
                # Same fallback scan.py uses: with no model extraction, the page
                # IS the summary. Better a full page in the slot than an empty
                # record.
                fields[NARRATIVE_SLOT] = corrected.strip()
            rows.append({"source_file": src, **fields})
    return pd.DataFrame(rows)


def score(preds, gold):
    """Per-field match, narrative recall, and the zero-edit record rate.

    date and location_purok are scored as the RECORD stores them
    (narrative.record_columns), because that is what the gold encodes - the
    entry date and the parties' purok, not D. Petsa / D. Lugar alone.
    """
    from bantay import narrative

    preds = preds.copy()
    columns = [narrative.record_columns(r) for r in preds.fillna("").to_dict("records")]
    for col in ("date", "location_purok"):
        preds[col] = [c[col] for c in columns]
    merged = gold.merge(preds, on="source_file", how="inner", suffixes=("", "_pred"))
    report, per_record_ok = {}, []

    for slot, col in GOLD_COLUMNS.items():
        present = slot in preds.columns
        hits, scored = [], 0
        for _, row in merged.iterrows():
            want = field_key(slot, row.get(col, ""))
            if not want:
                continue                      # no gold for this page/field - not scoreable
            got = field_key(slot, row.get(slot, "") if present else "")
            hits.append(got == want)
            scored += 1
        report[slot] = {
            "accuracy": (sum(hits) / scored) if scored else None,
            "n": scored,
            "present": present,
        }

    ex, fz, weights = [], [], []
    for _, row in merged.iterrows():
        e, f, n = token_recall(row.get(NARRATIVE_SLOT, ""), row.get(NARRATIVE_GOLD, ""))
        if n:
            ex.append(e * n)
            fz.append(f * n)
            weights.append(n)
    total = sum(weights)
    report[NARRATIVE_SLOT] = {
        # Token-weighted, not page-averaged: see token_recall's docstring.
        "exact": (sum(ex) / total) if total else None,
        "fuzzy": (sum(fz) / total) if total else None,
        "n_tokens": total,
        "n_pages": len(weights),
    }

    for _, row in merged.iterrows():
        checks = [field_key(slot, row.get(slot, "")) == field_key(slot, row.get(col, ""))
                  for slot, col in GOLD_COLUMNS.items() if field_key(slot, row.get(col, ""))]
        per_record_ok.append(bool(checks) and all(checks))
    report["_records"] = {"n": len(merged),
                          "zero_edit": (sum(per_record_ok) / len(merged)) if len(merged) else None}
    return report


def show(report, label):
    print(f"\n{'=' * 72}\n{label}\n{'=' * 72}")
    print(f"{'field':<20}{'accuracy':>10}{'n':>6}   note")
    print("-" * 72)
    for slot in GOLD_COLUMNS:
        r = report[slot]
        acc = "n/a" if r["accuracy"] is None else f"{r['accuracy']:.1%}"
        note = "" if r["present"] else "pipeline never fills this slot"
        print(f"{slot:<20}{acc:>10}{r['n']:>6}   {note}")
    for slot in NO_GOLD:
        print(f"{slot:<20}{'--':>10}{0:>6}   no gold column - out of scope, state in Limitations")

    nar = report[NARRATIVE_SLOT]
    print()
    if nar["exact"] is None:
        print(f"{NARRATIVE_SLOT}: no scoreable pages")
    else:
        print(f"{NARRATIVE_SLOT}: gold-token recall "
              f"{nar['exact']:.1%} exact, {nar['fuzzy']:.1%} incl. fuzzy(>={FUZZY_FLOOR}) "
              f"over {nar['n_tokens']} tokens / {nar['n_pages']} pages")
        print(f"{'':<20}masks excluded from the denominator; CER is not reported here "
              f"(see token_recall)")

    rec = report["_records"]
    print()
    zero = "n/a" if rec["zero_edit"] is None else f"{rec['zero_edit']:.1%}"
    print(f"records needing zero edits: {zero}  ({rec['n']} pages)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions", nargs="?", help="predictions CSV; omit to run the baseline")
    ap.add_argument("--split", default="dev", choices=["dev", "holdout", "all"])
    ap.add_argument("--no-fixes", action="store_true",
                    help="ignore data/gold_worksheet.csv (shows the circular, unverified number)")
    ap.add_argument("--gold", default=None,
                    help="annotation CSV to score against; default is the 188-record ML-ready gold")
    ap.add_argument("--by-quality", action="store_true",
                    help="report clean / degraded handwriting strata separately as well as pooled")
    args = ap.parse_args()

    split_path = ROOT / "data/gold_split.csv"
    if not split_path.exists():
        sys.exit("data/gold_split.csv missing - run tools/make_gold_worksheet.py first.")
    split = pd.read_csv(split_path)
    keep = set(split["source_file"] if args.split == "all"
               else split.loc[split.split == args.split, "source_file"])

    gold, n_verified, n_fixed = load_gold(apply_fixes=not args.no_fixes, path=args.gold)
    covered = gold.source_file.isin(keep)
    if args.gold and not covered.any():
        print(f"data/gold_split.csv lists none of the {len(gold)} pages in {args.gold}.")
        print("A new annotation batch needs its own frozen split first:")
        print(f'  python tools/make_gold_worksheet.py --gold "{args.gold}" --force')
        sys.exit(1)
    gold = gold[covered]

    if args.predictions:
        preds = pd.read_csv(args.predictions).fillna("")
        label = f"{Path(args.predictions).name}   split={args.split}"
    else:
        preds = baseline_predictions(sorted(keep))
        out = ROOT / "models/preds_baseline.csv"
        out.parent.mkdir(exist_ok=True)
        preds.to_csv(out, index=False, encoding="utf-8")
        label = f"BASELINE  Vision(cached) -> OCRCorrector -> guess_fields   split={args.split}"
        print(f"[write] {out}  {len(preds)} pages")

    print(f"[gold ] {len(gold)} pages"
          + (f"   {n_verified} human-verified, {n_fixed} corrections applied" if not args.no_fixes
             else "   FIXES IGNORED - this is the unverified, circular number"))
    if n_verified == 0 and not args.no_fixes:
        print("[warn ] no rows marked verified in data/gold_worksheet.csv.")
        print("        Every number below is measured against gold a model wrote from")
        print("        the same images. Do the Day 1 AM pass before quoting any of it.")

    show(score(preds, gold), label)

    if args.by_quality:
        for bucket in ("clean", "degraded", "unknown"):
            sub = gold[gold.quality == bucket]
            if sub.empty:
                continue
            show(score(preds, sub), f"{label}   handwriting={bucket}  ({len(sub)} pages)")
        if (gold.quality == "unknown").all():
            print("[warn ] every page bucketed `unknown` - the gold CSV has no "
                  "Readability column, so no clean/degraded split is possible "
                  "from it.")


if __name__ == "__main__":
    main()
