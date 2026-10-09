"""Score the offline lexicon corrector (bantay/ocr/correct.py) on real scanner output.

The old notebooks/3_corrector_eval.ipynb scored it on noise.corrupt() output,
so models/ocr_correction_report.json says how it does on synthetic noise. This
scores it on what Google Vision actually returned for the real pages in
data/ocr_calibration.csv, split by data/gold_split.csv. correct._CONFUSIONS is
fitted on the dev pages only (--fit), never the holdout pages.

Gold is the narrative field only and PII-masked, while the OCR is the whole
page, so each page's OCR tokens are aligned to its gold tokens (difflib, token
level) and every edit the corrector makes lands in exactly one bucket:

  fixed       a garbled token changed INTO the gold token it aligns with
  wrong       a garbled token changed into some other word
  damage      a token the scanner read correctly, changed anyway
  unverified  an edit outside the gold-aligned narrative - form labels, names
              (masked out of gold), stray marks. No gold says what those should
              be, so they are counted, not scored. Fewer is safer.

precision = fixed / (fixed + wrong + damage). Token accuracy is over the
aligned narrative tokens, before -> after.

Usage:
    python tools/eval_corrector.py                 # dev / holdout / all
    python tools/eval_corrector.py --examples 15   # plus the commonest bad edits
    python tools/eval_corrector.py --fit           # re-derive correct._CONFUSIONS
"""
import argparse
import collections
import csv
import difflib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bantay.ocr.correct import OCRCorrector, _WORD_RE  # noqa: E402

# Masks and transcriber notes ([PERSON_1], [?], [illegible]): same rule as
# tools/audit_ocr_gold.py, so both tools score against the same gold.
_MASK_RE = re.compile(r"\[[^\]]{0,120}\]")


def load_pages(calibration, split_csv):
    with open(split_csv, encoding="utf-8") as fh:
        split = {r["source_file"]: r["split"] for r in csv.DictReader(fh)}
    pages = []
    with open(calibration, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            ocr = [m.group() for m in _WORD_RE.finditer(r["ocr_text"])]
            gold = [m.group().upper() for m in _WORD_RE.finditer(_MASK_RE.sub(" ", r["gold_text"]))]
            aligned, loose = [], []     # (token, gold token | set of gold tokens)
            sm = difflib.SequenceMatcher(None, [w.upper() for w in ocr], gold, autojunk=False)
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag == "equal" or (tag == "replace" and i2 - i1 == j2 - j1):
                    aligned += zip(ocr[i1:i2], gold[j1:j2])
                elif tag == "replace":
                    loose += [(w, frozenset(gold[j1:j2])) for w in ocr[i1:i2]]
                elif tag == "delete":
                    loose += [(w, frozenset()) for w in ocr[i1:i2]]
            pages.append({"split": split.get(r["source_file"], "unsplit"),
                          "aligned": aligned, "loose": loose})
    return pages


def score(corrector, pages):
    memo = {}

    def fix(word):
        if word not in memo:
            memo[word] = corrector.correct(word)["text"].upper()
        return memo[word]

    c, bad = collections.Counter(), collections.Counter()
    for p in pages:
        for word, gold in p["aligned"]:
            after, before = fix(word), word.upper()
            c["tokens"] += 1
            c["right_before"] += before == gold
            c["right_after"] += after == gold
            if after == before:
                continue
            kind = "fixed" if after == gold else "damage" if before == gold else "wrong"
            c[kind] += 1
            if kind != "fixed":
                bad[(before, after, gold)] += 1
        for word, golds in p["loose"]:
            after, before = fix(word), word.upper()
            if after == before:
                continue
            kind = "fixed" if after in golds else "damage" if before in golds else "unverified"
            c[kind] += 1
            if kind == "damage":
                bad[(before, after, "?")] += 1
    judged = c["fixed"] + c["wrong"] + c["damage"]
    c["precision"] = c["fixed"] / judged if judged else 0.0
    return c, bad


def fit_confusions(pages, min_count=3, min_sim=0.6):
    """(read, written) glyph pairs from dev tokens aligned 1:1 with their gold.

    min_sim drops alignments that pair two unrelated words - their letters say
    nothing about how a glyph gets misread.
    """
    subs = collections.Counter()
    for p in pages:
        for word, gold in p["aligned"]:
            word = word.upper()
            if word == gold or difflib.SequenceMatcher(None, word, gold).ratio() < min_sim:
                continue
            for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
                    None, word, gold, autojunk=False).get_opcodes():
                if tag == "replace" and i2 - i1 == j2 - j1:
                    subs.update(zip(word[i1:i2], gold[j1:j2]))
    return [(pair, n) for pair, n in subs.most_common() if n >= min_count]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration", default=str(ROOT / "data/ocr_calibration.csv"))
    ap.add_argument("--split", default=str(ROOT / "data/gold_split.csv"))
    ap.add_argument("--models", default=str(ROOT / "models"))
    ap.add_argument("--examples", type=int, default=0, help="show the N commonest bad edits")
    ap.add_argument("--fit", action="store_true", help="print confusion pairs fitted on dev pages")
    args = ap.parse_args()

    pages = load_pages(args.calibration, args.split)
    if args.fit:
        pairs = fit_confusions([p for p in pages if p["split"] == "dev"])
        print(f"{len(pairs)} pairs (read, written) seen >= 3 times on dev pages:")
        print(" ".join(a + b for (a, b), _ in pairs))
        return 0

    corrector = OCRCorrector.from_dir(args.models)
    print(f"{'split':<8}{'pages':>6}{'fixed':>7}{'wrong':>7}{'damage':>8}{'unverif':>9}"
          f"{'precision':>11}   token accuracy")
    for name in ("dev", "holdout", "all"):
        sub = [p for p in pages if name == "all" or p["split"] == name]
        c, bad = score(corrector, sub)
        print(f"{name:<8}{len(sub):>6}{c['fixed']:>7}{c['wrong']:>7}{c['damage']:>8}"
              f"{c['unverified']:>9}{c['precision']:>11.3f}   "
              f"{c['right_before'] / c['tokens']:.4f} -> {c['right_after'] / c['tokens']:.4f}")
        if args.examples and name == "all":
            print("\ncommonest bad edits (read -> corrected, gold):")
            for (before, after, gold), n in bad.most_common(args.examples):
                print(f"  {n:>3}x  {before} -> {after}   (gold {gold})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
