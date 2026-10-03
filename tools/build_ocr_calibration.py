"""Build data/ocr_calibration.csv from real scanned pages + the corpus's own
gold transcriptions.

The gold half already exists: bantay_corpus_2100.csv has 73 real rows with a
Source File/Page filename ("Blotter_Pic (3).jpg") and the transcribed
narrative. All this script adds is the noisy half - OCR the actual image and
pair it with that gold text by filename.

Usage:
    python tools/build_ocr_calibration.py --images path/to/folder/of/scans

Matches files in --images against the corpus's Source File/Page column
case-insensitively. Ten matched pages is enough to calibrate; you do not need
all 73. Unmatched images and unmatched corpus rows are both reported so you can
see what to rename or skip.
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="folder containing the scanned page images")
    ap.add_argument("--corpus", default="data/raw/bantay_corpus_2100.csv")
    ap.add_argument("--out", default="data/ocr_calibration.csv")
    ap.add_argument("--backend", default=None, help="OCR backend; default = first installed")
    args = ap.parse_args()

    import pandas as pd
    from bantay.ocr.engine import QuotaExceeded, extract

    from tools.audit_ocr_gold import quality_bucket

    df = pd.read_csv(args.corpus)
    # Both the old corpus's "Source File/Page" and the new gold's
    # already-renamed "source_file" are accepted, since build_gold_from_
    # blotter_final.py writes the latter.
    df = df.rename(columns={"Source File/Page": "source_file"})
    # bantay_corpus_2100.csv mixes 73 real rows with 2027 synthetic ones and
    # marks them; a hand-written CSV of newly transcribed pages has no such
    # column and every row in it is real by construction. Requiring the column
    # made the obvious way to add pages fail with a KeyError.
    real = df[df["_provenance"] == "real"] if "_provenance" in df.columns else df
    gold_by_name = {str(f).strip().lower(): text
                    for f, text in zip(real["source_file"], real["Narrative/Summary"])}
    # Handwriting quality rides along with the gold so every downstream metric
    # can be split clean vs degraded without re-joining to the annotation CSV.
    # Absent column -> every page reads "unknown", which the audit reports as
    # its own stratum rather than silently pooling.
    readability = (real["Readability"] if "Readability" in real.columns
                   else pd.Series([""] * len(real), index=real.index))
    quality_by_name = {str(f).strip().lower(): quality_bucket(q)
                       for f, q in zip(real["source_file"], readability)}

    img_dir = Path(args.images)
    images = {p.name.lower(): p for p in img_dir.iterdir() if p.is_file()}

    matched = sorted(set(gold_by_name) & set(images))
    print(f"corpus has {len(gold_by_name)} real pages; folder has {len(images)} images; "
          f"{len(matched)} matched by filename")
    if not matched:
        print("no filename matches - check that the images keep their original "
              "'Blotter_Pic (N).jpg' style names from the corpus's Source File/Page column")
        return 1

    rows = []
    for name in matched:
        path = images[name]
        try:
            ocr = extract(path, backend=args.backend)
        except QuotaExceeded as exc:
            # Stop, do not skip: skipping would write a calibration file fitted
            # on a handful of pages and look like a successful run.
            print(f"  {exc}")
            print(f"  stopping with {len(rows)} of {len(matched)} pages done")
            break
        except RuntimeError as exc:
            print(f"  skip {path.name}: {exc}")
            continue
        rows.append({"source_file": path.name, "ocr_text": ocr["text"],
                     "gold_text": gold_by_name[name],
                     "quality": quality_by_name.get(name, "unknown")})
        print(f"  {path.name}: OCR {len(ocr['text'])} chars, mean_conf {ocr['mean_conf']}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["source_file", "ocr_text", "gold_text", "quality"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nwrote {len(rows)} pairs -> {out_path}")
    if len(rows) < 10:
        print("fewer than 10 pairs - stage 1's fit_confusions() still runs, but with less "
              "signal; add more pages if you have them")
    unmatched = sorted(set(gold_by_name) - set(images))
    if unmatched:
        print(f"\n{len(unmatched)} corpus pages had no matching image file, e.g.: "
              f"{unmatched[:5]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
