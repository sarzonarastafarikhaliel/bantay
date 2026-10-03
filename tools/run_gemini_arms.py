"""Run the two Gemini arms over the dev split and write predictions CSVs.

Day 2 of OCR_ACCURACY_PLAN.md section 7.3. Produces the two columns that sit
next to models/preds_baseline.csv:

  gemini_text    Vision transcription -> gemini.restore()
                 A text corrector. Cannot recover a word Vision never emitted,
                 so it is capped at 85.7% gold-token recall on this corpus no
                 matter how good the model is (plan 1.1).

  gemini_image   page photo + Vision transcription -> gemini.restore_from_image()
                 Reads the pixels. Not capped. This is the arm the whole
                 architecture argument rests on.

Both then run through narrative.normalize_fields + backfill(guess_fields), which
is what routes/scan.py does, so all three columns are scored on identical
downstream handling and the only difference is the reader.

WHY THIS IS NOT THE NOTEBOOK: notebooks/3_corrector_eval.ipynb stage3b appends a
CER on every iteration including the ones that errored, and verify() returns its
input unchanged on error - so the 21-of-30 error run in
models/ocr_correction_report.json averaged 9 real observations with 21 copies of
the corrector's own output and reported it as an improvement. Here every page
carries its status into the CSV, errors are counted per arm, and the run refuses
to look successful when it is not.

Cost: two calls per page. On the 58-page dev split that is 116 calls, roughly
25-45 minutes wall. Vision is not called at all - transcriptions come from
data/ocr_calibration.csv.

    python tools/run_gemini_arms.py                  both arms, dev split
    python tools/run_gemini_arms.py --arm image      just the image arm
    python tools/run_gemini_arms.py --limit 5        smoke test before the full run

PROMPT TUNING. Google's "minimally lossy text simplification with Gemini" did
not fine-tune a model: it looped - generate, auto-rate the output for
readability and fidelity, hand the ratings to a second model and ask it to
rewrite THE PROMPT - 824 times, until the scores stopped moving. On 73 gold
pages that loop is affordable and supervised tuning is not (Vertex SFT wants
100 examples minimum, and spending the gold on training leaves nothing to
measure with). --prompt-file and --tag are what make it runnable:

    python tools/run_gemini_arms.py --arm text --prompt-file prompts/v2.txt --tag v2
    python tools/eval_fields.py models/preds_gemini_text_v2.csv

Each run writes _repaired / _added / _lost per page - the fidelity half of the
rating. eval_fields.py is the accuracy half. A variant that raises field
accuracy while raising _added is not an improvement; it is a model that learned
to guess more confidently.
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

IMAGES = ROOT / "data/raw/blotter report docs"
MIMES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}


def run_arm(arm, sources, calib, corrector, limit=None):
    """One arm over the pages. Returns (predictions DataFrame, status Counter)."""
    from bantay import narrative
    from bantay.ocr import gemini
    from bantay.routes.scan import guess_fields

    rows, statuses = [], Counter()
    pages = sources[:limit] if limit else sources
    for i, src in enumerate(pages, 1):
        if src not in calib.index:
            continue
        vision_text = str(calib.loc[src, "ocr_text"] or "")
        t0 = time.perf_counter()

        if arm == "text":
            out = gemini.restore(vision_text, lexicon=corrector.lexicon)
        else:
            path = IMAGES / src
            if not path.exists():
                statuses["missing image"] += 1
                continue
            out = gemini.restore_from_image(
                path.read_bytes(), vision_text, lexicon=corrector.lexicon,
                mime=MIMES.get(path.suffix.lower(), "image/jpeg"))

        dt = time.perf_counter() - t0
        # Bucket by the leading word so "error: ServerError: 504 ..." and
        # "error: ServerError: 429 ..." collapse into one countable outcome.
        statuses[out["status"].split(":")[0].split("(")[0].strip()] += 1

        fields = narrative.normalize_fields(out["fields"])
        fields = narrative.backfill(fields, guess_fields(out["text"]))
        if not fields["incident_summary"]:
            fields["incident_summary"] = out["text"].strip()

        # loss/gain/distortion per page, so a prompt variant can be judged on
        # whether it PRESERVED the page and not only on whether the fields
        # landed. eval_fields.py scores the slots; these three score the
        # narrative, which is the half a field metric cannot see.
        fid = out.get("fidelity") or {}
        rows.append({"source_file": src, **fields,
                     "_status": out["status"], "_n_edits": len(out["edits"]),
                     "_repaired": fid.get("distortion", 0), "_added": fid.get("gain", 0),
                     "_lost": fid.get("loss", 0), "_page_tokens": fid.get("n_tokens", 0),
                     "_seconds": round(dt, 1)})
        print(f"  [{i:>3}/{len(pages)}] {src:<24} {dt:>5.1f}s  {out['status'][:56]}")
    return pd.DataFrame(rows), statuses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["text", "image", "both"], default="both")
    ap.add_argument("--split", default="dev", choices=["dev", "holdout", "all"])
    ap.add_argument("--limit", type=int, help="only the first N pages (smoke test)")
    ap.add_argument("--prompt-file", help="prompt variant to score instead of the built-in "
                                          "(sets BANTAY_RESTORE_PROMPT_FILE)")
    ap.add_argument("--tag", default="", help="suffix for the predictions CSV, so variants "
                                              "do not overwrite each other")
    args = ap.parse_args()

    # The loop from Google's minimally-lossy work, in the form this project can
    # afford: score a prompt, read the fidelity columns and the field accuracy,
    # rewrite the prompt, score again. Supervised tuning is the thing that is
    # NOT affordable here - see the note in tools/eval_fields.py and the module
    # docstring in bantay/ocr/gemini.py.
    if args.prompt_file:
        if not Path(args.prompt_file).exists():
            sys.exit(f"no such prompt file: {args.prompt_file}")
        os.environ["BANTAY_RESTORE_PROMPT_FILE"] = args.prompt_file
        print(f"[prompt] {args.prompt_file}")

    from bantay import create_app
    from bantay.ocr import OCRCorrector, gemini

    if not gemini.available():
        sys.exit("No Gemini backend configured. Run python tools/day0_check.py first.")

    split = pd.read_csv(ROOT / "data/gold_split.csv")
    sources = sorted(split["source_file"] if args.split == "all"
                     else split.loc[split.split == args.split, "source_file"])
    calib = pd.read_csv(ROOT / "data/ocr_calibration.csv").set_index("source_file")

    app = create_app()
    summary = {}
    with app.app_context():
        corrector = OCRCorrector.from_dir(app.config["MODEL_DIR"])
        for arm in (["text", "image"] if args.arm == "both" else [args.arm]):
            print(f"\n{'=' * 72}\nARM: gemini_{arm}   {len(sources)} pages   split={args.split}"
                  f"\n{'=' * 72}")
            started = time.perf_counter()
            preds, statuses = run_arm(arm, sources, calib, corrector, args.limit)
            wall = time.perf_counter() - started

            out = ROOT / f"models/preds_gemini_{arm}{args.tag and '_' + args.tag}.csv"
            out.parent.mkdir(exist_ok=True)
            preds.to_csv(out, index=False, encoding="utf-8")

            n_err = sum(v for k, v in statuses.items() if k.startswith(("error", "rejected")))
            summary[arm] = {"pages": len(preds), "wall_seconds": round(wall, 1),
                            "sec_per_page": round(wall / max(len(preds), 1), 1),
                            "statuses": dict(statuses), "errors": n_err,
                            "prompt": args.prompt_file or "built-in"}
            if len(preds):
                tok = max(preds["_page_tokens"].sum(), 1)
                summary[arm]["fidelity"] = {
                    "repaired_rate": round(preds["_repaired"].sum() / tok, 4),
                    "added_rate": round(preds["_added"].sum() / tok, 4),
                    "lost_rate": round(preds["_lost"].sum() / tok, 4)}
                print("        fidelity: {repaired_rate:.1%} words repaired, "
                      "{added_rate:.1%} added, {lost_rate:.1%} lost"
                      .format(**summary[arm]["fidelity"]))
            print(f"\n[write] {out}  {len(preds)} pages  {wall / 60:.1f} min "
                  f"({wall / max(len(preds), 1):.1f}s/page)")
            for status, n in statuses.most_common():
                print(f"        {n:>3}  {status}")
            if n_err:
                # The failure that produced the unusable 21/30 number. Say it loudly
                # rather than letting a mean quietly absorb it.
                print(f"\n[WARN ] {n_err}/{len(preds)} pages did not produce a model reading.")
                print("        Those rows carry the UNCHANGED Vision text, so scoring them")
                print("        measures the baseline, not this arm. Fix before quoting.")

    path = ROOT / f"models/gemini_arms_run{args.tag and '_' + args.tag}.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[write] {path}")
    print("\nNext:")
    for arm in summary:
        print(f"  python tools/eval_fields.py models/preds_gemini_{arm}"
              f"{args.tag and '_' + args.tag}.csv")


if __name__ == "__main__":
    main()
