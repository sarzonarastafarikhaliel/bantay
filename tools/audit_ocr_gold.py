"""Audit what Google Vision actually reads off the scanned blotter pages.

Reproduces every number in OCR_ACCURACY_PLAN.md sections 0.2 and 1.1, so a
panelist can re-run the audit instead of taking it on trust.

Why this exists: notebooks/3_corrector_eval.ipynb evaluates the corrector on
data/stage/ocr_pairs.csv, and every row in that file is noise.corrupt() output -
including the 146 rows marked provenance="real", where "real" means the CLEAN
side came from a real record, not that the noisy side came from a scanner. The
only file in the project holding actual Vision output paired with gold is
data/ocr_calibration.csv, and nothing currently measures it. This does.

Two things it reports, and they answer different questions:

  CER / WER          how far Vision's page is from the gold string. Inflated
                     here by three ruler problems the plan documents (gold is a
                     FIELD not a page, gold is PII-masked and the hypothesis is
                     not, gold is re-ordered). Read it as an upper bound on
                     error, not as Vision's error rate.

  token accounting   where each gold content token IS in Vision's output:
                     exact / garbled-but-present / absent. This one is robust to
                     all three ruler problems, because it never compares
                     positions - only membership. It is the number the
                     architecture decision rests on: the "absent" share is what
                     no post-OCR text corrector can ever recover.

Usage:
    python tools/audit_ocr_gold.py
    python tools/audit_ocr_gold.py --calibration data/ocr_calibration_easyocr.csv
    python tools/audit_ocr_gold.py --per-page      # one row per page
"""
import argparse
import difflib
import re
import statistics
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Everything the encoder wrote into gold in square brackets that no OCR engine
# can ever emit: PII masks ([PERSON_1]) AND transcriber annotations the
# annotation guide asks for ([?], [?struck-through word], [illegible]). Left in
# the string they score as a guaranteed miss on every page, which is a property
# of the transcription policy, not of the scanner. One bracket rule rather than
# two lists, because every caller - CER, WER, token accounting, field recall -
# needs the same denominator and two regexes drifting apart is how a metric
# starts measuring its own preprocessing.
_MASK_RE = re.compile(r"\[[^\]]{0,120}\]")
_KEEP_RE = re.compile(r"[^A-Z0-9ÑÜ ]")

# Below this difflib ratio a gold token is counted absent rather than garbled.
# 0.75 is deliberately generous: it errs toward calling a token RECOVERABLE, so
# the "absent" share it reports is a floor, not a flattering estimate.
FUZZY_FLOOR = 0.75
MIN_TOKEN_LEN = 3       # 1-2 char tokens are punctuation noise on both sides


# The annotation schema's Readability column is free text: "good", "Clear",
# "Low - central paragraph genuinely illegible...". Bucket on the leading word,
# which is the only part annotators write consistently.
_CLEAN_WORDS = {"good", "clear", "readable", "high", "excellent", "legible"}
_DEGRADED_WORDS = {"fair", "moderate", "partial", "poor", "low", "unreadable",
                   "illegible", "bad", "faint"}


def quality_bucket(readability):
    """Readability free text -> 'clean' | 'degraded' | 'unknown'.

    Unknown is kept as its own bucket rather than folded into either side: a
    page nobody rated is not evidence that the handwriting was fine, and
    silently counting it as clean would flatter the clean stratum with exactly
    the pages an annotator skipped.
    """
    head = re.split(r"[^a-z]+", str(readability).strip().lower(), maxsplit=1)[0]
    if head in _CLEAN_WORDS:
        return "clean"
    if head in _DEGRADED_WORDS:
        return "degraded"
    return "unknown"


def normalise(text):
    """Upper, strip accents and punctuation, collapse whitespace.

    Applied to BOTH sides. Case and punctuation differences between a
    hand-transcription and an OCR engine are not recognition errors and should
    not be charged as such.
    """
    text = unicodedata.normalize("NFKC", str(text)).upper()
    return re.sub(r"\s+", " ", _KEEP_RE.sub(" ", text)).strip()


def cer(hyp, ref):
    """Character error rate: Levenshtein / len(ref)."""
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for j, rc in enumerate(ref, 1):
        cur = [j]
        for i, hc in enumerate(hyp, 1):
            cur.append(min(prev[i] + 1, cur[i - 1] + 1, prev[i - 1] + (hc != rc)))
        prev = cur
    return prev[-1] / len(ref)


def wer(hyp, ref):
    """Word error rate. Same distance, token granularity."""
    h, r = hyp.split(), ref.split()
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for j, rw in enumerate(r, 1):
        cur = [j]
        for i, hw in enumerate(h, 1):
            cur.append(min(prev[i] + 1, cur[i - 1] + 1, prev[i - 1] + (hw != rw)))
        prev = cur
    return prev[-1] / len(r)


def account_tokens(ocr_text, gold_text):
    """Where is each gold content token in the OCR output?

    Returns (n_gold, exact, garbled, absent) as counts. Membership only - no
    positional alignment - so re-ordered gold does not distort it.
    """
    ocr_set = {w for w in normalise(ocr_text).split() if len(w) >= MIN_TOKEN_LEN}
    ocr_list = list(ocr_set)
    gold = [w for w in normalise(_MASK_RE.sub(" ", gold_text)).split()
            if len(w) >= MIN_TOKEN_LEN]
    exact = garbled = absent = 0
    for word in gold:
        if word in ocr_set:
            exact += 1
        elif difflib.get_close_matches(word, ocr_list, n=1, cutoff=FUZZY_FLOOR):
            garbled += 1
        else:
            absent += 1
    return len(gold), exact, garbled, absent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration", default="data/ocr_calibration.csv",
                    help="CSV with source_file, ocr_text, gold_text")
    ap.add_argument("--per-page", action="store_true", help="print one row per page")
    ap.add_argument("--by-quality", action="store_true",
                    help="split every figure by clean vs degraded handwriting "
                         "(needs a `quality` column in the calibration CSV)")
    ap.add_argument("--markdown", action="store_true",
                    help="emit the Chapter 4 table rows instead of the console report")
    args = ap.parse_args()

    import pandas as pd

    path = Path(args.calibration)
    if not path.exists():
        print(f"{path} not found. Build it first:\n"
              f"  python tools/build_ocr_calibration.py --images data/raw/'Blotter Pics'")
        return 1
    df = pd.read_csv(path)

    rows = []
    for r in df.itertuples():
        gold_nomask = _MASK_RE.sub(" ", str(r.gold_text))
        hyp, ref = normalise(r.ocr_text), normalise(gold_nomask)
        n_gold, exact, garbled, absent = account_tokens(r.ocr_text, str(r.gold_text))
        if not n_gold:
            continue
        rows.append({
            "file": r.source_file,
            "quality": quality_bucket(getattr(r, "quality", "")),
            "cer_raw": cer(str(r.ocr_text), str(r.gold_text)),
            "cer_norm": cer(hyp, ref),
            "wer_norm": wer(hyp, ref),
            "len_ratio": len(hyp) / max(len(ref), 1),
            "n_gold": n_gold,
            "exact": exact / n_gold,
            "garbled": garbled / n_gold,
            "absent": absent / n_gold,
        })

    if not rows:
        print("no usable pages - every gold_text was empty after masking")
        return 1

    def mean(key):
        return statistics.fmean(r[key] for r in rows)

    n = len(rows)
    print(f"\n{path}  —  {n} pages\n")

    print("=" * 66)
    print("STRING DISTANCE  (upper bound on error — see the module docstring)")
    print("=" * 66)
    print(f"  CER, raw strings                     {mean('cer_raw'):.3f}")
    print(f"  CER, normalised                      {mean('cer_norm'):.3f}")
    print(f"  WER, normalised                      {mean('wer_norm'):.3f}")
    print(f"  OCR length / gold length             {mean('len_ratio'):.2f}x")
    if mean("len_ratio") > 1.15:
        print("    ^ OCR reads more than gold contains: gold is a FIELD, the")
        print("      engine returns the whole PAGE. Every correctly-read form")
        print("      header scores as an insertion error. Fix the ruler before")
        print("      reading anything into this CER.")

    print()
    print("=" * 66)
    print("TOKEN ACCOUNTING  (robust to ordering, masking and page furniture)")
    print("=" * 66)
    e, g, a = mean("exact"), mean("garbled"), mean("absent")
    print(f"  A. read exactly                      {e:6.1%}   nothing to fix")
    print(f"  B. read but garbled                  {g:6.1%}   <- all a text corrector can win")
    print(f"  C. absent from the OCR output        {a:6.1%}   <- unreachable from text")
    print(f"\n  mean gold content-tokens per page    {mean('n_gold'):.0f}")
    print(f"  ceiling for ANY post-OCR corrector   {e + g:.1%} token recall")
    print(f"  permanently lost to a text-only fix  {a:.1%}")

    bad = [r for r in rows if r["absent"] > 0.30]
    print(f"  pages losing >30% of gold tokens     {len(bad)} / {n}")
    if bad:
        worst = sorted(bad, key=lambda r: -r["absent"])[:5]
        print("    worst:", ", ".join(f"{r['file']} ({r['absent']:.0%})" for r in worst))

    print()
    print("Read this as: a perfect corrector — lexicon, Gemini-on-text, or a")
    print(f"human proofreading the transcription — takes you from {e:.1%} to {e + g:.1%}.")
    print(f"Reaching the remaining {a:.1%} requires re-reading the IMAGE, not the text.")

    if args.by_quality or args.markdown:
        print()
        print("=" * 66)
        print("BY HANDWRITING QUALITY")
        print("=" * 66)
        print(f"| {'Stratum':<10} | Pages | CER | WER | Exact | Garbled | Absent | Ceiling |")
        print("| :--- | --: | --: | --: | --: | --: | --: | --: |")
        for bucket in ("clean", "degraded", "unknown", "ALL"):
            sub = rows if bucket == "ALL" else [r for r in rows if r["quality"] == bucket]
            if not sub:
                continue
            m = lambda k: statistics.fmean(r[k] for r in sub)
            print(f"| {bucket:<10} | {len(sub)} | {m('cer_norm'):.3f} | {m('wer_norm'):.3f} "
                  f"| {m('exact'):.1%} | {m('garbled'):.1%} | {m('absent'):.1%} "
                  f"| {m('exact') + m('garbled'):.1%} |")
        if not any(r["quality"] != "unknown" for r in rows):
            print()
            print("  every page bucketed `unknown`: the calibration CSV has no")
            print("  `quality` column. Rebuild it with tools/build_ocr_calibration.py")
            print("  against an annotation CSV that carries Readability.")

    if args.per_page:
        print("\n" + "=" * 66)
        print("PER PAGE")
        print("=" * 66)
        print(f"{'file':<26}{'cer_n':>7}{'wer_n':>7}{'exact':>8}{'garb':>7}{'absent':>8}")
        for r in sorted(rows, key=lambda r: -r["absent"]):
            print(f"{r['file'][:25]:<26}{r['cer_norm']:>7.3f}{r['wer_norm']:>7.3f}"
                  f"{r['exact']:>8.1%}{r['garbled']:>7.1%}{r['absent']:>8.1%}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
