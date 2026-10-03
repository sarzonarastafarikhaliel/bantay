"""Batch intake: blotter page images -> one encoding-schema CSV row per page.

The Flask scan route (bantay/routes/scan.py) already runs this pipeline for one
uploaded page at a time, but it needs a request, a session and a database. This
is the same pipeline with no web layer, for the case the study actually has: a
folder of scans that need to become encodable rows.

Nothing here re-implements OCR, repair or classification - every step calls the
module that owns it, so the guards documented in Chapter 3 sec 3.2.6 apply here
exactly as they do in the app:

    ocr.extract()        Google Vision (or a local engine) reads the page
    gemini.restore()     guarded repair + field extraction, one call
    reconcile()          model fields vs. regex fields, disagreements surfaced
    sealion_classify()   incident-type suggestion (never auto-accepted)

NARRATIVE PRESERVATION. The `Narrative/Summary` column is the full corrected
page, not the model's `incident_summary` slot. gemini.restore() splices only
gated edits into the ORIGINAL transcription and rejects the whole repair if it
would change more than BANTAY_RESTORE_MAX_CHANGE of the page, so what lands in
that column is what the scanner saw with minor corrections applied - never a
paraphrase and never a summary. The extracted summary slot is written to its own
column for reference, and the loss/gain rates are written beside it so an encoder
can see how much of the page survived rather than trust that it did.

    python tools/scan_blotter.py data/raw/"Blotter Pics"           # a folder
    python tools/scan_blotter.py page1.jpg page2.jpg -o rows.csv   # named pages
    python tools/scan_blotter.py pics/ --backend paddle             # offline engine
"""
import argparse
import csv
import datetime as dt
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

# The encoding schema. The first block is what a human encoder fills or confirms;
# the _qa block is provenance the encoder reads to decide how hard to look, and is
# prefixed so it sorts away from the record itself.
COLUMNS = [
    "Record ID", "Batch Number", "Source File/Page",
    "Date", "Date (as written)", "Time", "Location/Purok",
    "Reporting Party", "Respondent",
    "Narrative/Summary", "Narrative (masked)", "Extracted Summary",
    "Action Taken", "Status",
    "Incident Type", "Incident Type (Secondary)",
    "Readability", "Include in ML", "Fold",
    "Encoded By", "Date Encoded", "Remarks",
    "Label Confidence", "Label Source", "Second Encoder Label",
    "_qa_ocr_engine", "_qa_mean_conf", "_qa_repair_status", "_qa_n_edits",
    "_qa_loss_rate", "_qa_gain_rate", "_qa_distortion_rate",
    "_qa_blank_slots", "_qa_disagreements", "_qa_needs_review", "_qa_seconds",
    "_pii_spotcheck_needed",
]


def _corrector(app):
    """The lexicon corrector: supplies restore()'s vocabulary hint, and is the
    whole repair step when no Gemini backend is configured."""
    from bantay.ocr import OCRCorrector
    try:
        return OCRCorrector.from_dir(app.config["MODEL_DIR"])
    except Exception:                                # noqa: BLE001 - no lexicon built yet
        return OCRCorrector(lexicon=set())


def _mask(text, reporting, respondent):
    """Narrative with the named parties replaced by [PERSON_n] tokens."""
    from bantay.ml.mask import mask_names
    if reporting:
        text = mask_names(text, "Complainant: {}".format(reporting), "[PERSON_1]")
    if respondent:
        text = mask_names(text, "Respondent: {}".format(respondent), "[PERSON_2]")
    return text


def _images(paths):
    out = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            out += sorted(q for q in path.rglob("*") if q.suffix.lower() in IMAGE_SUFFIXES)
        elif path.suffix.lower() in IMAGE_SUFFIXES:
            out.append(path)
        else:
            print("  skipping (not an image): {}".format(path), file=sys.stderr)
    return out


def scan_one(path, corrector, backend, batch, classify):
    """One page -> one row dict. Never raises: a page that fails OCR still gets a
    row, with the failure in _qa_repair_status, because a silently dropped page is
    the one outcome an encoder cannot notice."""
    from bantay import narrative
    from bantay.ocr import extract, gemini
    from bantay.ocr.reconcile import reconcile, fields_only, disagreements
    from bantay.routes.scan import guess_fields

    t0 = time.perf_counter()
    row = {c: "" for c in COLUMNS}
    row.update({"Source File/Page": path.name, "Batch Number": batch,
                "Readability": "readable", "Include in ML": "TRUE",
                "Label Source": "model-suggested",
                "Date Encoded": dt.date.today().isoformat()})

    try:
        ocr = extract(str(path), backend=backend)
    except Exception as exc:                         # noqa: BLE001 - engine/quota/IO
        row["_qa_repair_status"] = "ocr failed: {}: {}".format(type(exc).__name__, exc)[:200]
        row["_qa_needs_review"] = "YES"
        row["Readability"] = "unreadable"
        row["Include in ML"] = "FALSE"
        row["_qa_seconds"] = round(time.perf_counter() - t0, 1)
        return row
    row["_qa_ocr_engine"] = ocr["engine"]
    row["_qa_mean_conf"] = ocr["mean_conf"]

    # Guarded repair. Offline this degrades to the lexicon corrector, which can
    # only swap one out-of-lexicon token for one in-lexicon token and so cannot
    # introduce a sentence the scanner never saw.
    if gemini.available():
        gem = gemini.restore(ocr["text"], lexicon=corrector.lexicon)
    else:
        local = corrector.correct(ocr["text"])
        gem = {"text": local["text"], "edits": local["edits"], "fields": {},
               "fidelity": gemini.fidelity(ocr["text"], local["text"]),
               "status": "gemini off - lexicon corrector only"}
    corrected = gem["text"]

    fid = gem.get("fidelity") or {}
    row.update({"_qa_repair_status": str(gem.get("status", ""))[:200],
                "_qa_n_edits": len(gem.get("edits") or []),
                "_qa_loss_rate": fid.get("loss_rate", ""),
                "_qa_gain_rate": fid.get("gain_rate", ""),
                "_qa_distortion_rate": fid.get("distortion_rate", "")})

    # Two independent readers per slot; where they disagree the encoder is told
    # rather than one silently winning (bantay/ocr/reconcile.py).
    recon = reconcile(narrative.normalize_fields(gem.get("fields") or {}),
                      guess_fields(corrected))
    fields = fields_only(recon)
    disagree = disagreements(recon)

    # The narrative column is the whole corrected page. The model's summary slot
    # is kept separately - useful to classify on, but it is not the record.
    row["Narrative/Summary"] = corrected.strip()
    row["Extracted Summary"] = str(fields.get("incident_summary") or "").strip()
    columns = narrative.record_columns(fields)   # the corpus convention - see narrative.py
    row["Date"] = columns["date"]
    row["Date (as written)"] = columns["date"]
    row["Time"] = fields.get("time", "")
    row["Location/Purok"] = columns["location_purok"]
    row["Action Taken"] = fields.get("action_taken", "")
    row["Status"] = fields.get("status", "")
    row["Reporting Party"] = fields.get("reporting_party", "")
    row["Respondent"] = fields.get("respondent", "")

    # Two narrative columns, because the record and the ML feature have different
    # privacy rules (Chapter 3 sec 3.2.1): the operational record keeps real names
    # for barangay case handling, while anything the classifier reads is masked
    # first. mask_names() is reused as-is, fed the party slots in the
    # "Complainant: X" shape it already parses, once per party so the two get
    # distinguishable tokens matching the 2023 corpus convention.
    row["Narrative (masked)"] = _mask(corrected.strip(),
                                      row["Reporting Party"], row["Respondent"])

    blanks = narrative.missing(fields)
    row["_qa_blank_slots"] = "; ".join(blanks)
    # disagreements() returns the slot names; the winning value and which reader
    # won live in recon, so the encoder sees what was chosen and over what.
    row["_qa_disagreements"] = "; ".join(
        "{}={!r} (kept {})".format(slot, recon[slot]["value"], recon[slot]["source"])
        for slot in (disagree or []))

    if classify:
        from bantay.ml import sealion_classify
        if sealion_classify.available():
            pred = sealion_classify.classify(
                row["Narrative (masked)"] or row["Extracted Summary"])
            row["Incident Type"] = pred.get("incident_type") or ""
            row["Label Confidence"] = pred.get("confidence_label", "")
            row["Remarks"] = (pred.get("reason") or "")[:120]
            if not pred.get("incident_type"):
                row["_qa_repair_status"] += " | classify: " + str(pred.get("status", ""))[:80]

    # A page is flagged whenever anything downstream would be guessing: a rejected
    # or errored repair, a slot no reader filled, the two readers disagreeing, or
    # a repair that dropped more than a tenth of the page's tokens.
    # mask_names() can only mask the names the field extraction actually found,
    # so a blank party slot means that party is provably still in the narrative
    # in the clear. This is a fact about the run, not a guess about the text -
    # a blank slot cannot have been masked. Same column name the 2023 corpus uses.
    row["_pii_spotcheck_needed"] = "" if (row["Reporting Party"]
                                          and row["Respondent"]) else "YES"

    loss = fid.get("loss_rate") or 0
    status_l = row["_qa_repair_status"].lower()
    row["_qa_needs_review"] = "YES" if (
        blanks or disagree or loss > 0.10
        or "reject" in status_l or "error" in status_l
        or not row["Incident Type"]) else ""
    row["_qa_seconds"] = round(time.perf_counter() - t0, 1)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="image files and/or folders of images")
    ap.add_argument("-o", "--out", default="data/raw/scanned_rows.csv")
    ap.add_argument("--backend", help="OCR engine (default $BANTAY_OCR_BACKEND)")
    ap.add_argument("--batch", default=dt.date.today().isoformat(),
                    help="batch number stamped on every row")
    ap.add_argument("--start-id", type=int, default=1)
    ap.add_argument("--no-classify", action="store_true",
                    help="skip the incident-type suggestion (OCR + fields only)")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    images = _images(args.paths)
    if args.limit:
        images = images[:args.limit]
    if not images:
        sys.exit("no images found in: " + ", ".join(args.paths))

    # The app is created for its config, not to serve: guess_fields() reads the
    # barangay's purok list off current_app, and the corrector loads from
    # MODEL_DIR. No DB write happens here - the CSV is the output.
    from bantay import create_app
    from bantay.ocr import gemini
    from bantay.ml import sealion_classify
    app = create_app()
    print("{} page(s)".format(len(images)))
    print("  repair   : {}".format("gemini (guarded)" if gemini.available()
                                   else "lexicon corrector only (no Gemini backend)"))
    print("  classify : {}\n".format(
        "off (--no-classify)" if args.no_classify
        else ("ollama" if sealion_classify.available() else "off (set BANTAY_OLLAMA=1)")))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows, flagged = [], 0
    with app.app_context():
        corrector = _corrector(app)
        for i, path in enumerate(images):
            row = scan_one(path, corrector, args.backend, args.batch, not args.no_classify)
            row["Record ID"] = args.start_id + i
            rows.append(row)
            flagged += row["_qa_needs_review"] == "YES"
            print("  [{:>3}/{}] {:<34} {:<26} {}".format(
                i + 1, len(images), path.name[:34],
                (row["Incident Type"] or "-")[:26],
                "REVIEW" if row["_qa_needs_review"] else "ok"), flush=True)

    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    print("\nwrote {} ({} row(s), {} flagged for review)".format(out, len(rows), flagged))
    print("Every row needs a human pass: confirm or correct Incident Type, fill the")
    print("blank slots in _qa_blank_slots, and resolve _qa_disagreements before import.")


if __name__ == "__main__":
    main()
