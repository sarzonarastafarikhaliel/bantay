"""Scan intake: upload a blotter page, get a pre-filled record draft.

    image -> Google Vision OCR -> Gemini repair -> template -> SEA-LION classify

Three models, one job each, and the assignment is deliberate:
  Google Vision (ocr/engine.py)     reads the handwriting off the photo. The
                                    only backend that reads a real logbook page
                                    well enough for the repair pass to have
                                    something to fix rather than reconstruct.
  Gemini 3.7 Flash (ocr/gemini.py)  repairs Vision's misreads and fills the
                                    record template, in one call. Cloud, 3-11s
                                    per page measured.
  SEA-LION 9B q4 (ml/sealion_classify.py)
                                    picks the incident type from the repaired
                                    narrative. Local via Ollama, ~5.5s warm,
                                    no per-call billing, no data egress, and
                                    tuned for Tagalog/Cebuano rather than
                                    general-purpose.

Two Google hops, one credential: the Vision call and the Gemini call both
authenticate with the same ADC service account (see ocr/engine.py and
ocr/gemini.py), so deployment is one key, not two. SEA-LION needs no key at
all - just BANTAY_OLLAMA=1 and the model pulled.

Nothing is written to the database here. The encoder always sees the raw Vision
transcription, the repaired text, every edit Gemini made, and every template
slot the page did not fill, then presses Save on the records form themselves. An
automated pipeline that silently commits a misread legal record is exactly the
failure this design refuses to allow.

Gemini returns whole text, but it does not get to hand that text through: every
changed token is diffed, guarded and spliced back into the Vision output by
ocr/gemini.py, so the stored page is still built from what the scanner saw. Its
edits are tagged `source: gemini` so the encoder can see which repairs came from
a model capable of inventing.

Degradation, because a barangay hall's network is not a datacentre's:
  no Gemini backend  -> the lexicon corrector (ocr/correct.py) does the repair
                        and the regex reader fills the template. Offline-safe,
                        but expect more blank slots - the lexicon cannot fill
                        template fields, only fix words.
                        (ocr/sealion.py can also do this repair pass and is
                        kept for the offline research tooling, but it is NOT in
                        the request path: measured 65-107s per real page here
                        against Gemini's 3-11s, and SEA-LION's job in this
                        pipeline is classification.)
  no SEA-LION        -> the legacy TF-IDF classifier. Measured far worse
                        (6.7% real-test accuracy vs the few-shot LLM's 76.8%)
                        and shown for reference only, never as the
                        authoritative answer.
  no Vision          -> whichever OCR engine is installed; the dropdown stays.
Every one of those is surfaced in the UI. A degraded scan is never a silent one.
"""
import json
import os
import re
import time

from flask import Blueprint, current_app, redirect, url_for
from flask_login import login_required

from .. import narrative
from ..normalize import find_purok_number, get_category_group, get_kp_status, get_pnp_classification
from ..ocr.reconcile import disagreements
from ..ocr.reconcile import fields_only as reconcile_fields
from ..ocr.reconcile import reconcile
from ..ml import sealion_classify

scan_bp = Blueprint("scan", __name__, url_prefix="/scan")

ALLOWED = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

# Vision is the designed intake path: it is the only backend that reads the
# handwritten narrative well enough for the Gemini pass to have something to
# repair rather than reconstruct. The dropdown still works - a page that must
# not leave the barangay's machine (RA 10173) can be run on a local engine, at
# a documented cost in accuracy.
DEFAULT_BACKEND = os.environ.get("BANTAY_OCR_BACKEND", "gvision")

# DD/MM/YY, D-M-YYYY, and "JAN. 3, 2026" all appear in the real logbook.
_DATE_NUM_RE = re.compile(r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})\b")
_DATE_TXT_RE = re.compile(
    r"\b(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*\.?\s+(\d{1,2}),?\s+(\d{4})\b", re.I)
# Loosened day/year separator ([\s,-]+ instead of ",? "), for the schedule-date
# search only (below): real pages write "NOVEMBER 23-2023" for an appointment
# date. Widening the main _DATE_TXT_RE instead regressed the primary date
# field - on pages with two dates, the loose pattern let a schedule date win
# a global .search() that used to correctly land on the incident date.
_SCHEDULE_DATE_RE = re.compile(
    r"\b(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*\.?\s+(\d{1,2})[\s,\-]+(\d{4})\b", re.I)
_TIME_RE = re.compile(r"\b(\d{1,2})[:.](\d{2})\s*([AP]\.?M\.?)?\b", re.I)
_MONTHS = {m: i for i, m in enumerate(
    "JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split(), 1)}

# Closed-set status has exactly 3 values in the corpus (Filed 60 / Settled 8 /
# For Hearing 5) - see OCR_ACCURACY_PLAN.md Lever 4. Detected from explicit
# resolution language on the page, never guessed: a page with neither signal
# stays blank rather than defaulting to "Filed", per the project's no-inference
# rule (docs/REIMPLEMENTATION.md sec 3/4).
_SETTLED_RE = re.compile(r"KASUNDUAN|NAGKASUNDO|SETTLEMENT|SETTLED", re.I)
_HEARING_RE = re.compile(r"SCHEDULE|TAKDA", re.I)
# "Filed" is the corpus majority (60/73) and pages state it explicitly: they
# carry a "FOR BLOTTER" / "IPA-BLOTTER" / "PARA MABLOTTER" marker naming what
# the walk-in was for. Matching that marker is reading the page, not inferring
# from its absence, so it keeps the no-inference rule (docs/REIMPLEMENTATION.md
# sec 3/4) - a page with NO status language at all still ends up blank below.
# O/0 and TT/TI/II are Vision's usual confusions on this word, hence the
# character classes rather than a plain literal.
# Measured on the 20 gold-labelled pages (data/gold_worksheet.csv): exact
# status went 15% -> 75%, blanks 80% -> 15%.
_FILED_RE = re.compile(
    r"\bF[O0]R[\s.:-]*BL[O0][TI1]{1,2}[EO]R|(?:IPA|PA|MA)[\s-]*BL[O0][TI1]{1,2}[EO]R"
    r"|\bBL[O0][TI1]{1,2}[EO]R\b", re.I)


def _corrector():
    """Lazy per-process singleton. Lexicon load is not free; requests are hot."""
    ext = current_app.extensions.setdefault("bantay_scan", {})
    if "corrector" not in ext:
        from ..ocr import OCRCorrector
        ext["corrector"] = OCRCorrector.from_dir(current_app.config["MODEL_DIR"])
    return ext["corrector"]


def _puroks():
    path = os.path.join(os.path.dirname(current_app.root_path), "data", "purok_coordinates.json")
    try:
        with open(path, encoding="utf-8") as fh:
            return list(json.load(fh).get("puroks", {}))
    except (OSError, ValueError):
        return []


def guess_fields(text):
    """Cheap deterministic field extraction, used to backfill slots Gemini left
    empty and to fill the template outright when no Gemini backend is configured.

    Regex, not a model, and deliberately so: a regex miss leaves the slot blank
    for the encoder to fill, while a model miss fills it with a confident wrong
    date. See docs/REIMPLEMENTATION.md section 4 for where that line sits.

    The first date on a logbook page is the entry's own date, so it is read as
    date_reported (Petsa ng Pagblotter), not as D. Petsa: a regex cannot tell
    an incident date from a filing date, and reconciling it against the model's
    incident date flagged a "disagreement" on every page where the two differ.
    """
    fields = {"date_reported": "", "time": "", "location_purok": "", "status": "", "action_taken": ""}

    m = _DATE_TXT_RE.search(text)
    if m:
        fields["date_reported"] = "%04d-%02d-%02d" % (
            int(m.group(3)), _MONTHS[m.group(1)[:3].upper()], int(m.group(2)))
    else:
        m = _DATE_NUM_RE.search(text)
        if m:
            a, b, y = (int(g) for g in m.groups())
            y += 2000 if y < 100 else 0
            # Logbook is written M/D/Y; a value >12 in the first slot means the
            # encoder wrote D/M/Y that day, so swap instead of emitting month 25.
            month, day = (a, b) if a <= 12 else (b, a)
            if 1 <= month <= 12 and 1 <= day <= 31:
                fields["date_reported"] = "%04d-%02d-%02d" % (y, month, day)

    m = _TIME_RE.search(text)
    if m:
        hour, minute, mer = int(m.group(1)), int(m.group(2)), (m.group(3) or "").upper()
        if mer.startswith("P") and hour < 12:
            hour += 12
        if mer.startswith("A") and hour == 12:
            hour = 0
        if hour < 24 and minute < 60:
            fields["time"] = "%02d:%02d" % (hour, minute)

    found = find_purok_number(text)
    if found:
        fields["location_purok"] = found
    else:
        upper = text.upper()
        for purok in _puroks():
            if purok.upper() in upper:
                fields["location_purok"] = purok
                break

    # Status/action taken: closed-set, detected from explicit language on the
    # page only. Resolution language (settled/hearing) outranks the "FOR
    # BLOTTER" marker because a page that was later settled still carries the
    # marker from when it was first walked in - the resolution is the newer
    # fact. A page stating none of the three is still left blank.
    # ponytail: flat precedence over the whole page, no positional weighting.
    # Costs 2 of 20 gold pages where a schedule note and a blotter marker
    # co-occur and the wrong one wins; upgrade to "last marker on the page
    # wins" if that pair gets more common.
    settled = bool(_SETTLED_RE.search(text))
    hearing_m = _HEARING_RE.search(text)
    action_parts = []
    if settled:
        action_parts.append("Settlement / confrontation held at barangay")
    if hearing_m:
        # Look just past the keyword for the appointment's own date/time -
        # not the incident date/time already captured above.
        window = text[hearing_m.end():hearing_m.end() + 60]
        dm = _SCHEDULE_DATE_RE.search(window) or _DATE_NUM_RE.search(window)
        tm = _TIME_RE.search(window)
        if dm and tm:
            action_parts.append(f"Scheduled: {dm.group(0).strip()} TIME {tm.group(0).strip()}")
    if settled:
        fields["status"] = "Settled"
    elif hearing_m:
        fields["status"] = "For Hearing"
    elif _FILED_RE.search(text):
        fields["status"] = "Filed"
    fields["action_taken"] = " ".join(action_parts)
    return fields


@scan_bp.route("/", methods=["GET"])
@login_required
def scan_page():
    """Scanning now lives on the New Record page (records.new_record) as an
    optional upload step alongside manual entry - this URL just forwards an
    old bookmark or link there instead of 404ing."""
    return redirect(url_for("records.new_record"))


def run_scan_pipeline(file, backend=None):
    """Run OCR -> Gemini repair -> template fill -> SEA-LION classify on an
    uploaded page image.

    Returns (result, error): if the upload can't even be read, error is a
    user-facing message and result is None; otherwise error is None and
    result is the full draft dict - the same shape whichever page embeds the
    scan-upload step, so records_new.html renders it exactly as this module
    used to render it on its own page (see module docstring for the pipeline).
    """
    from ..ocr import gemini

    if not file or os.path.splitext(file.filename or "")[1].lower() not in ALLOWED:
        return None, "Upload a page image (.jpg, .png, .tif)."

    # --- 1. Google Vision reads the page -----------------------------------
    from ..ocr import extract
    started = time.perf_counter()
    try:
        ocr = extract(file.stream, backend=backend or DEFAULT_BACKEND)
    except RuntimeError as exc:
        return None, str(exc)
    ocr_ms = (time.perf_counter() - started) * 1000

    corrector = _corrector()

    # --- 2. Gemini repairs it and fills the template ------------------------
    # One call does both: the field extraction needs the repaired text to read
    # from, and a second call would re-send the page for no benefit.
    started = time.perf_counter()
    if gemini.available():
        gem = gemini.restore(ocr["text"], lexicon=corrector.lexicon)
        repair_engine = ("gemini (vertex)" if os.environ.get("GOOGLE_CLOUD_PROJECT")
                         else "gemini (api key)")
    else:
        # Offline path: the lexicon corrector cannot fill template slots, so the
        # regex reader below is the only field source. Same shape, so nothing
        # downstream branches.
        local = corrector.correct(ocr["text"])
        gem = {"text": local["text"], "edits": local["edits"], "fields": {},
               "status": "off - lexicon corrector used instead"}
        repair_engine = "lexicon only"
    repair_ms = (time.perf_counter() - started) * 1000

    corrected_text = gem["text"]

    # --- 3. Fill the template ----------------------------------------------
    # Two independent readers per field: the model (Gemini, or {} offline) and
    # the deterministic regex reader below. Where they agree, that agreement is
    # itself a confidence signal (see ocr/reconcile.py); where they disagree,
    # the encoder sees both and is warned before saving rather than either
    # reader silently winning.
    model_fields = narrative.normalize_fields(gem["fields"])
    regex_fields = guess_fields(corrected_text)
    recon = reconcile(model_fields, regex_fields)
    fields = reconcile_fields(recon)
    if not fields["incident_summary"]:
        # No model extraction (or it declined): the page itself is the summary.
        # Better a full page in the slot than an empty record.
        fields["incident_summary"] = corrected_text.strip()
    blank_slots = narrative.missing(fields)
    field_disagreements = disagreements(recon)
    templated = narrative.render(fields)

    # --- 4. Classify --------------------------------------------------------
    # On the incident_summary slot, NOT the rendered template: the template's
    # labelled boilerplate is a constant prefix that carries no incident signal
    # and only dilutes the narrative the model reads. See narrative.py.
    #
    # SEA-LION (bantay/ml/sealion_classify.py) is THE classifier. Gemini's build
    # of this same few-shot approach measured via tools/eval_gemini_classify.py
    # at 76.8% type accuracy (n=151, dev split) against a fine-tuned encoder's
    # 6.7% (real-test, n=15) - a large gap from a model with ZERO fine-tuning on
    # this corpus, because 188 real records cannot usefully train a 35-way head.
    # The legacy TF-IDF classifier is kept only as an offline degradation path
    # and is shown as reference, never as the authoritative answer.
    to_classify = fields["incident_summary"] or corrected_text
    started = time.perf_counter()

    # classify() never raises - a dead Ollama, an unset BANTAY_OLLAMA, a timeout
    # or a malformed reply all come back as incident_type=None with the reason
    # in ["status"]. That status is carried into the result below and rendered:
    # a silent fallback is what made a misconfigured Ollama look like a working
    # one.
    sealion_pred = (sealion_classify.classify_with_agreement(to_classify) if sealion_classify.available()
                    else {"incident_type": None, "status": "off (set BANTAY_OLLAMA=1)"})
    if sealion_pred and sealion_pred.get("incident_type"):
        # confidence_label now comes from whether sealion_classify.CHECK_MODEL
        # agrees with the primary, the same validated-agreement trick
        # ocr/reconcile.py already uses between the regex and Gemini field
        # readers - see classify_with_agreement's docstring for the measured
        # split (85.2% accurate when they agree vs 47.1% when they don't,
        # n=151 dev split) that replaced the old self-reported-confidence
        # mapping. "high" is the only tier that clears the default 0.6
        # threshold; "medium" and "low" both raise a check-this warning.
        confidence = {"high": 0.8, "medium": 0.5, "low": 0.3}.get(
            sealion_pred["confidence_label"].lower(), 0.5)
        pred = {"incident_type": sealion_pred["incident_type"], "type_confidence": confidence,
               "category_group": sealion_pred["category_group"], "group_confidence": None,
               "axes_agree": True, "confidence": confidence, "top_k": [],
               "source": "sealion", "reason": sealion_pred.get("reason", "")}
        fallback_pred = {}
    else:
        # SEA-LION is off or gave no usable answer. The TF-IDF classifier answers
        # so the draft is not left empty, and the UI says so - see records_new.html.
        fallback_pred = {}
        if to_classify.strip():
            conf = current_app.classifier.classify(to_classify)
            label = max(conf, key=conf.get) if conf else None
            if label:
                fallback_pred = {"incident_type": label, "type_confidence": conf[label],
                                 "category_group": get_category_group(label),
                                 "group_confidence": None, "axes_agree": True,
                                 "confidence": conf[label], "top_k": []}
        pred = dict(fallback_pred, source="tf-idf" if fallback_pred else "none")
    predict_ms = (time.perf_counter() - started) * 1000

    # --- 5. Checks the encoder should look at before saving --------------
    # These used to set review_status="needs_review" and drop the record into
    # /review. They no longer do: the encoder is looking at this draft RIGHT
    # NOW, with every one of these warnings rendered above the Save button, and
    # queueing a page a human just read for a second human to read again is
    # review theatre - it inflates the queue with records nobody will change
    # and trains encoders to clear it without looking. The signals are still
    # computed and still shown; what changed is that acting on them is this
    # encoder's job at this screen, not a later reviewer's.
    #
    # /review stays for the intake paths where NO human saw the model's answer:
    # CSV import and the plain /records/new form (see routes/records.py).
    #
    # Low OCR confidence is the one the classifier cannot see for itself: a page
    # the scanner barely read still yields a confidently-wrong label. Blank slots
    # is a record missing its date, its location or its account of what happened,
    # however confident the classifier is about the part that did come through.
    # Field disagreement is the model and the regex reader independently
    # disagreeing - the same signal axis disagreement is for the classifier, see
    # ocr/reconcile.py.
    threshold = current_app.config["CONFIDENCE_THRESHOLD"]
    low_ocr = 0 < ocr["mean_conf"] < 0.60
    low_pred = pred.get("confidence", 0.0) < threshold
    axes_disagree = not pred.get("axes_agree", True)
    llm_rejected = gem["status"].startswith(("rejected", "error"))
    field_disagree = bool(field_disagreements)

    # The raw Vision text, the repaired text and Gemini's before/after edit
    # list are deliberately NOT returned: the encoder sees only the filled
    # record template, never what the repair step replaced. Leaving them out
    # here (not just unrendered) keeps them out of the page source too. The
    # edit count alone stays, for the remarks provenance line.
    result = {
        "narrative": templated,
        "fields": fields,
        "reconciliation": recon,
        "blank_slots": blank_slots,
        "n_edits": len(gem["edits"]),
        "repair_status": gem["status"],
        "gemini_on": gemini.available(),
        "repair_engine": repair_engine,
        "classify_status": sealion_pred.get("status", ""),
        "engine": ocr["engine"],
        "mean_conf": ocr["mean_conf"],
        "model": ("sealion few-shot" if pred.get("source") == "sealion"
                  else "tf-idf (fallback)"),
        "prediction": pred,
        "pnp_tier": get_pnp_classification(pred["incident_type"])[0] if pred.get("incident_type") else "",
        # KP referability travels with the draft so the encoder sees, at intake,
        # whether the Lupon can take the case at all - see normalize.KP_STATUS.
        "kp_status": get_kp_status(pred["incident_type"])[0] if pred.get("incident_type") else "",
        "kp_note": get_kp_status(pred["incident_type"])[1] if pred.get("incident_type") else "",
        # The encoder reviews this draft on screen, so what they save is
        # already reviewed - see step 5. Posted as a hidden field so
        # routes/records.py does not re-decide it from a recomputed confidence.
        "review_status": "accepted",
        "warnings": [w for w, hit in (
            ("The scanner read this page poorly - check every field against the image", low_ocr),
            ("The classifier is unsure of the incident type - confirm it yourself", low_pred),
            ("The classifier's two axes disagree - confirm the incident type", axes_disagree),
            ("Gemini's repair was rejected, so this is raw scanner output", llm_rejected),
            ("The page never stated: " + ", ".join(blank_slots), bool(blank_slots)),
            ("The two readers disagree on: " + ", ".join(field_disagreements),
             field_disagree)) if hit],
        "confidence_threshold": threshold,
        "source_file_page": file.filename,
        "timings_ms": {"ocr": round(ocr_ms), "repair": round(repair_ms),
                       "classify": round(predict_ms),
                       "total": round(ocr_ms + repair_ms + predict_ms)},
    }
    return result, None
