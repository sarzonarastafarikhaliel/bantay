"""Side-by-side OCR-correction test: local Ollama SEA-LION vs Gemini, same
Vision transcriptions, same guards, same downstream field pipeline.

Mirrors run_gemini_arms.py's text arm (Vision transcription -> restore()) but
runs both correctors per page in one pass and prints them next to each other,
plus writes models/preds_sealion_text.csv and models/preds_gemini_text.csv so
tools/eval_fields.py can score each against gold on identical footing:

    python tools/compare_sealion_gemini_ocr.py --limit 5
    python tools/eval_fields.py models/preds_sealion_text.csv
    python tools/eval_fields.py models/preds_gemini_text.csv

No vision arm here - text-only, matching gemini.restore()'s text arm and its
85.7% gold-token-recall ceiling (OCR_ACCURACY_PLAN.md 1.1). Vision is not
called either; transcriptions come from data/ocr_calibration.csv.
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def run_one(backend, src, vision_text, corrector):
    from bantay import narrative
    from bantay.routes.scan import guess_fields

    t0 = time.perf_counter()
    out = backend.restore(vision_text, lexicon=corrector.lexicon)
    dt = time.perf_counter() - t0

    fields = narrative.normalize_fields(out["fields"])
    fields = narrative.backfill(fields, guess_fields(out["text"]))
    if not fields["incident_summary"]:
        fields["incident_summary"] = out["text"].strip()

    row = {"source_file": src, **fields, "_status": out["status"],
           "_n_edits": len(out["edits"]), "_seconds": round(dt, 1)}
    return row, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "holdout", "all"])
    ap.add_argument("--limit", type=int, help="only the first N pages (smoke test)")
    args = ap.parse_args()

    from bantay import create_app
    from bantay.ocr import OCRCorrector, gemini, sealion

    missing = [name for name, mod in (("SEA-LION/Ollama", sealion), ("Gemini", gemini))
               if not mod.available()]
    if missing:
        sys.exit(f"Not configured: {', '.join(missing)}. "
                 "Set BANTAY_OLLAMA=1 (Ollama running the model) and a Gemini "
                 "backend (see python -m bantay.ocr.gemini) before comparing.")

    split = pd.read_csv(ROOT / "data/gold_split.csv")
    sources = sorted(split["source_file"] if args.split == "all"
                     else split.loc[split.split == args.split, "source_file"])
    if args.limit:
        sources = sources[:args.limit]
    calib = pd.read_csv(ROOT / "data/ocr_calibration.csv").set_index("source_file")

    app = create_app()
    sealion_rows, gemini_rows = [], []
    with app.app_context():
        corrector = OCRCorrector.from_dir(app.config["MODEL_DIR"])
        for i, src in enumerate(sources, 1):
            if src not in calib.index:
                continue
            vision_text = str(calib.loc[src, "ocr_text"] or "")

            s_row, s_out = run_one(sealion, src, vision_text, corrector)
            g_row, g_out = run_one(gemini, src, vision_text, corrector)
            sealion_rows.append(s_row)
            gemini_rows.append(g_row)

            print(f"\n[{i}/{len(sources)}] {src}")
            print(f"  vision  : {vision_text[:100]}{'...' if len(vision_text) > 100 else ''}")
            print(f"  sealion : {s_out['text'][:100]}{'...' if len(s_out['text']) > 100 else ''}")
            print(f"            {s_row['_seconds']:.1f}s  {s_out['status'][:70]}")
            print(f"  gemini  : {g_out['text'][:100]}{'...' if len(g_out['text']) > 100 else ''}")
            print(f"            {g_row['_seconds']:.1f}s  {g_out['status'][:70]}")

    for name, rows in (("sealion", sealion_rows), ("gemini", gemini_rows)):
        out = ROOT / f"models/preds_{name}_text.csv"
        out.parent.mkdir(exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8")
        print(f"\n[write] {out}  {len(rows)} pages")

    print("\nNext, score both against gold on the same basis:")
    print("  python tools/eval_fields.py models/preds_sealion_text.csv")
    print("  python tools/eval_fields.py models/preds_gemini_text.csv")


if __name__ == "__main__":
    main()
