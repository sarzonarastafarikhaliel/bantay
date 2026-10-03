import csv
import io
import re
from datetime import datetime, timedelta

from flask import Blueprint, Response, render_template, request, redirect, url_for, flash, current_app, jsonify
from flask_login import login_required, current_user

from .. import db
from .. import export_docx
from .. import narrative as narrative_tpl
from ..models import IncidentRecord
from ..crypto import narrative_digest
from ..ml import sealion_classify
from ..ml.infer import decide_review_status
from ..normalize import (
    CANONICAL_CATEGORIES, CATEGORY_GROUP_NAMES, KP_STATUSES, PNP_TIERS,
    clean_row, get_category_group, get_kp_status, get_pnp_classification,
    normalize_batch_number,
)
from ..rbac import role_required
from . import scan as scan_route

records_bp = Blueprint("records", __name__, url_prefix="/records")

# Canonical category names for dropdowns
CATEGORY_NAMES = [name for name, _ in CANONICAL_CATEGORIES]


def _classify_narrative(text, date="", time="", location_purok="", model=None):
    """SEA-LION first, legacy TF-IDF fallback - the same priority order as the
    scan intake pipeline (routes/scan.py step 4), so a manually-typed record
    gets the same classifier a scanned one does. Shared by predict() (the
    new_record form's live preview) and new_record()'s final save, so the two
    never disagree on how a label was chosen.
    """
    text = (text or "").strip()
    model = sealion_classify.resolve_model(model)
    sealion_pred = (sealion_classify.classify_with_agreement(text, model=model) if text and sealion_classify.available()
                    else {"incident_type": None, "status": "off (set BANTAY_OLLAMA=1)"})
    if sealion_pred and sealion_pred.get("incident_type"):
        # confidence_label here comes from whether a second, independent
        # model agrees, not from the primary's own self-report - see
        # sealion_classify.classify_with_agreement's docstring.
        confidence = {"high": 0.8, "medium": 0.5, "low": 0.3}.get(
            sealion_pred["confidence_label"].lower(), 0.5)
        return {"incident_type": sealion_pred["incident_type"], "confidence": confidence,
                "category_group": sealion_pred["category_group"], "source": "sealion",
                "reason": sealion_pred.get("reason", ""), "status": sealion_pred.get("status", ""),
                "model": model}

    label, conf_value = None, None
    if text:
        confidences = current_app.classifier.classify(
            text, date=date, time=time, location_purok=location_purok)
        label = max(confidences, key=confidences.get) if confidences else None
        conf_value = confidences.get(label) if label else None
    return {"incident_type": label, "confidence": conf_value,
            "category_group": get_category_group(label) if label else None,
            "source": "tf-idf" if label else "none", "reason": "",
            "status": sealion_pred.get("status", ""), "model": model}


# Columns of the import template, in the New Blotter Record form's order and
# under its field names - the narrative slots are narrative.SLOTS, the paper
# BLOTTER FORM's fields. Only incident_summary (the form's required E. Salaysay)
# is required; everything else may be left blank.
IMPORT_COLUMNS = (
    "batch_number", "encoded_by", "source_file_page",
    *narrative_tpl.SLOTS,
    "incident_type_primary", "incident_type_secondary",
    "pnp_classification", "category_group", "kp_status",
    "readability", "remarks",
)

# Other spellings of the same fields, after _header_key() lower/underscores a
# header - the encoding-schema CSVs (data/raw/gold_blotter_final.csv,
# tools/scan_blotter.py) say "Location/Purok", "Narrative/Summary" and so on.
_HEADER_ALIASES = {
    "complainant": "reporting_party", "place_of_incident": "location_purok",
    "date_of_incident": "date", "time_of_incident": "time",
    "source_file": "source_file_page",
}
# Where the narration can come from, best first: the form's own field, the scan
# tool's extracted slot, then a whole-page or legacy free-text narrative.
_SUMMARY_KEYS = ("incident_summary", "extracted_summary", "narration_of_facts",
                 "narrative_summary", "narrative_summary_raw_pii", "narrative")


def _header_key(name):
    key = re.sub(r"[^a-z0-9]+", "_", (name or "").lower()).strip("_")
    return _HEADER_ALIASES.get(key, key)


def _import_row(raw):
    """Map one CSV row or parsed PDF entry onto the New Blotter Record fields."""
    row = {}
    for name, value in raw.items():
        value = (value or "").strip() if isinstance(value, str) else value
        key = _header_key(name)
        if name and value and not row.get(key):
            row[key] = value
    summary = next((row[k] for k in _SUMMARY_KEYS if row.get(k)), "")
    if re.search(r"^\s*(BLOTTER FORM|BARANGAY BLOTTER ENTRY)\s*$", summary, re.M | re.I):
        # A narrative exported from BANTAY is already the rendered template:
        # read its slots back instead of nesting a template inside a template.
        parsed = (narrative_tpl.parse(summary) or [{}])[0]
        summary = parsed.pop("incident_summary", "")
        for key, value in parsed.items():
            if value and not row.get(key):
                row[key] = value
    row["incident_summary"] = summary
    return row


def import_csv_rows(rows, batch_number=None, encoded_by=None, created_by=None):
    """Save imported rows exactly the way new_record() saves the form.

    Each row's slots are rendered through the same BLOTTER FORM template, a
    row with no incident type is classified by the same SEA-LION-first _classify_narrative, and the PNP tier / category group / KP
    status follow from the type unless the row overrides them with a valid
    value. A row with no narration is skipped (the form requires it), and so is
    one whose rendered narrative is already stored - re-importing a file must
    not double every record. batch_number/encoded_by fill rows that leave them
    blank.
    """
    threshold = current_app.config["CONFIDENCE_THRESHOLD"]
    stats = {"saved": 0, "excluded": 0, "skipped_blank": 0, "skipped_existing": 0}
    seen = set()
    for raw in rows:
        row = _import_row(raw)
        given_type = row.get("incident_type_primary") or row.get("incident_type")
        overrides = {k: row.get(k) for k in ("pnp_classification", "category_group", "kp_status")}
        had_location = bool(row.get("location_purok"))
        row = clean_row(row)
        if not row.get("incident_summary"):
            stats["skipped_blank"] += 1
            continue

        slots = {k: row.get(k) or "" for k in narrative_tpl.SLOTS}
        if not had_location:
            slots["location_purok"] = ""       # clean_row's "Unspecified" would hide the blank
        narrative = narrative_tpl.render(slots)
        columns = narrative_tpl.record_columns(slots)
        digest = narrative_digest(narrative)
        if digest in seen or IncidentRecord.query.filter_by(narrative_digest=digest).first():
            stats["skipped_existing"] += 1
            continue
        seen.add(digest)

        confidence = None
        if not given_type:
            # ponytail: one classifier call per untyped row, inside the request;
            # a few hundred untyped rows on SEA-LION need a background job.
            clf = _classify_narrative(slots["incident_summary"], date=columns["date"],
                                      time=slots["time"], location_purok=columns["location_purok"])
            primary, confidence = clf["incident_type"], clf["confidence"]
        elif given_type in CATEGORY_NAMES:
            primary = given_type                # a dropdown value - keep it verbatim
        else:
            primary = row.get("incident_type_primary")
        secondary = row.get("incident_type_secondary") or None

        readability = row.get("readability") or "readable"
        include_in_ml = readability != "unreadable" and str(
            row.get("include_in_ml") or "").upper() not in ("FALSE", "0", "NO")
        if readability in ("unreadable", "partial"):
            review_status = "needs_review"
        elif confidence is not None:
            review_status = decide_review_status(confidence, threshold)
        else:
            review_status = "accepted"

        db.session.add(IncidentRecord(
            batch_number=row.get("batch_number") or normalize_batch_number(batch_number),
            source_file_page=row.get("source_file_page") or None,
            date=columns["date"] or None,
            time=slots["time"] or None,
            location_purok=columns["location_purok"] or None,
            incident_type_primary=primary,
            incident_type_secondary=secondary,
            pnp_classification=overrides["pnp_classification"] if overrides["pnp_classification"] in PNP_TIERS
            else (get_pnp_classification(primary)[0] if primary else None),
            category_group=overrides["category_group"] if overrides["category_group"] in CATEGORY_GROUP_NAMES
            else (get_category_group(primary) if primary else None),
            kp_status=overrides["kp_status"] if overrides["kp_status"] in KP_STATUSES
            else (get_kp_status(primary)[0] if primary else None),
            narrative=narrative,
            action_taken=slots["action_taken"] or None,
            status=slots["status"] or None,
            readability=readability,
            include_in_ml=include_in_ml,
            model_confidence=confidence,
            review_status=review_status,
            encoded_by=row.get("encoded_by") or encoded_by or None,
            date_encoded=row.get("date_encoded") or None,
            remarks=row.get("remarks") or None,
            created_by=created_by,
        ))
        stats["saved"] += 1
        if not include_in_ml:
            stats["excluded"] += 1
    db.session.commit()
    return stats


def sync_review_queue():
    """Re-evaluate every unresolved record against the current confidence
    threshold and readability rules, flagging anything that now qualifies
    as needs_review so it shows up in the Review Queue (which is just a
    filtered view over this same field). Records already resolved by a
    human (review_status == "corrected") are left alone - this is a triage
    sweep, not a way to re-open a decision already made in the queue.

    Runs once at app startup (a scan or import may have written rows under
    an older CONFIDENCE_THRESHOLD) and again on demand from the All
    Records button.
    """
    threshold = current_app.config["CONFIDENCE_THRESHOLD"]
    flagged = 0
    for record in IncidentRecord.query.filter(IncidentRecord.review_status != "corrected"):
        needs_review = (
            record.readability in ("unreadable", "partial")
            or not (record.narrative or "").strip()
            or (record.model_confidence is not None and record.model_confidence < threshold)
        )
        new_status = "needs_review" if needs_review else "accepted"
        if new_status != record.review_status:
            record.review_status = new_status
            flagged += 1
    if flagged:
        db.session.commit()
    return flagged


@records_bp.route("/check-review", methods=["POST"])
@login_required
def check_review():
    flagged = sync_review_queue()
    flash(f"Review check complete: {flagged} record(s) flagged for review."
          if flagged else "Review check complete: no records need review.")
    return redirect(url_for("review.queue"))


def _filtered_query():
    """The records list's filters, read from the query string. Shared with the
    .docx export so "Download" gives exactly the records the list shows."""
    query = IncidentRecord.query

    # Search & filter (§Dashboard Features)
    category = request.args.get("category", "").strip()
    location = request.args.get("location", "").strip()
    date_from = request.args.get("date_from", "").strip()
    date_to = request.args.get("date_to", "").strip()
    q = request.args.get("q", "").strip()
    status_filter = request.args.get("status", "").strip()
    kp_status_filter = request.args.get("kp_status", "").strip()

    if category:
        query = query.filter(IncidentRecord.incident_type_primary == category)
    if location:
        query = query.filter(IncidentRecord.location_purok == location)
    if date_from:
        query = query.filter(IncidentRecord.date >= date_from)
    if date_to:
        query = query.filter(IncidentRecord.date <= date_to)
    if q:
        # narrative is AES-GCM at rest, so SQL LIKE cannot see inside it -
        # search_narrative decrypts the already-filtered set in Python and hands
        # SQL back the surviving ids, so ordering and paging stay in SQL.
        query = IncidentRecord.search_narrative(query, q)
    if status_filter:
        query = query.filter(IncidentRecord.review_status == status_filter)
    if kp_status_filter:
        # Lets the Lupon Secretary pull up exactly the KP-Mediable caseload
        # (or Conditional/Excluded) instead of only seeing aggregate counts -
        # see the dashboard's Katarungang Pambarangay Referability panel,
        # whose stat cards deep-link here with this same param.
        query = query.filter(IncidentRecord.kp_status == kp_status_filter)
    return query.order_by(IncidentRecord.created_at.desc())


@records_bp.route("/")
@login_required
def list_records():
    page = request.args.get("page", 1, type=int)
    records = _filtered_query().paginate(page=page, per_page=25)

    categories = sorted({r.incident_type_primary for r in IncidentRecord.query.all() if r.incident_type_primary})
    locations = sorted({r.location_purok for r in IncidentRecord.query.all() if r.location_purok})

    # Remove 'page' from args so we can pass **filters to url_for without conflict
    filters = request.args.to_dict()
    filters.pop("page", None)

    return render_template(
        "records_list.html",
        records=records,
        categories=categories,
        locations=locations,
        filters=filters,
    )


@records_bp.route("/<int:record_id>")
@login_required
def view_record(record_id):
    record = IncidentRecord.query.get_or_404(record_id)
    pnp_tier, legal_basis = get_pnp_classification(record.incident_type_primary)
    category_group = record.category_group or get_category_group(record.incident_type_primary)
    kp_status, kp_note = get_kp_status(record.incident_type_primary)
    return render_template(
        "records_view.html", record=record, pnp_tier=pnp_tier, legal_basis=legal_basis,
        category_group=category_group, kp_status=kp_status, kp_note=kp_note,
    )


def _docx_response(records, filename):
    """The records as BLOTTER FORM pages. Non-admins get the same redacted
    narrative the visible_narrative filter shows them on screen."""
    show_names = current_user.role == "admin"
    fields = [export_docx.record_fields(
        r, r.narrative if show_names else narrative_tpl.redact_names(r.narrative))
        for r in records]
    return Response(
        export_docx.build(fields),
        mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f"attachment; filename={filename}"})


@records_bp.route("/<int:record_id>/export.docx")
@login_required
def export_record_docx(record_id):
    record = IncidentRecord.query.get_or_404(record_id)
    return _docx_response([record], f"blotter_{record.id}.docx")


@records_bp.route("/export.docx")
@login_required
def export_records_docx():
    # ponytail: whole filtered set in one request; page it if exports reach thousands.
    records = _filtered_query().all()
    if not records:
        flash("No records match those filters.")
        return redirect(url_for("records.list_records", **request.args))
    return _docx_response(records, "blotter_records.docx")


@records_bp.route("/import", methods=["GET", "POST"])
@login_required
@role_required("reviewer", "admin")
def import_csv():
    if request.method == "POST":
        file = request.files.get("file")
        name = (file.filename or "").lower() if file else ""
        if name.endswith(".csv"):
            try:
                rows = list(csv.DictReader(io.StringIO(file.stream.read().decode("utf-8-sig"))))
            except UnicodeDecodeError:
                flash("That CSV is not UTF-8 text. In Excel, use Save As > CSV UTF-8.")
                return redirect(url_for("records.import_csv"))
        elif name.endswith(".pdf"):
            from pypdf import PdfReader
            from pypdf.errors import PyPdfError
            try:
                text = "\n".join(page.extract_text() or "" for page in PdfReader(file.stream).pages)
            except (PyPdfError, ValueError):
                flash("Could not read that PDF.")
                return redirect(url_for("records.import_csv"))
            rows = narrative_tpl.parse(text)
            if not rows:
                flash("No BLOTTER FORM entry found in that PDF. Type the entries in the "
                      "PDF template's layout - a scanned or photographed page has no text to "
                      "read, so use Scan a Blotter Page on New Blotter Record instead.")
                return redirect(url_for("records.import_csv"))
        else:
            flash("Upload a .csv or .pdf file.")
            return redirect(url_for("records.import_csv"))

        stats = import_csv_rows(rows, batch_number=request.form.get("batch_number"),
                                encoded_by=request.form.get("encoded_by"),
                                created_by=current_user.id)
        message = f"Saved {stats['saved']} blotter record(s)."
        if stats["excluded"]:
            message += f" {stats['excluded']} kept out of ML training (unreadable or marked)."
        if stats["skipped_blank"]:
            message += f" Skipped {stats['skipped_blank']} with no narration of facts."
        if stats["skipped_existing"]:
            message += f" Skipped {stats['skipped_existing']} already in the system."
        flash(message)
        return redirect(url_for("records.list_records"))
    return render_template("records_import.html", pnp_tiers=PNP_TIERS, kp_statuses=KP_STATUSES)


@records_bp.route("/import/template.csv")
@login_required
def import_template_csv():
    """The CSV import template: the New Blotter Record fields as columns, with
    one example row to overwrite."""
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(IMPORT_COLUMNS)
    example = {
        "batch_number": "BATCH-2026-001", "encoded_by": "Encoder A", "source_file_page": "logbook p.12",
        "blotter_no": "2026-001", "date_reported": "2026-01-03",
        "reporting_party": "Juan Dela Cruz", "respondent": "Pedro Santos",
        "complaint": "Pagnanakaw ng manok",
        "date": "2026-01-03", "time": "14:30", "location_purok": "Purok 3",
        "incident_summary": "Nagsadya si Juan upang ireklamo ang pagkawala ng kanyang manok.",
        "action_taken": "Both parties summoned for mediation", "status": "For Hearing",
        "readability": "readable",
    }
    writer.writerow([example.get(c, "") for c in IMPORT_COLUMNS])
    return Response("﻿" + out.getvalue(), mimetype="text/csv", headers={
        "Content-Disposition": "attachment; filename=blotter_import_template.csv"})


def _render_new_record(result=None):
    """New Record's own render_template call, factored out so the plain GET
    and the post-scan draft (see new_record() below) show the exact same
    page - result=None is the blank manual-entry form, result=<scan output>
    is the pre-filled draft, matching what routes/scan.py used to render on
    its own page before the two were merged into one.
    """
    from ..ocr import available_engines, gemini
    return render_template(
        "records_new.html", categories=CATEGORY_NAMES, record=None,
        blank_marker=narrative_tpl.BLANK, result=result,
        narrative_template=narrative_tpl._TEMPLATE, narrative_slots=narrative_tpl.SLOTS,
        engines=available_engines(), default_backend=scan_route.DEFAULT_BACKEND,
        gemini_on=gemini.available(), sealion_on=sealion_classify.available(),
        models=sealion_classify.EVALUATED_MODELS,
        active_model=sealion_classify.resolve_model(None),
        category_groups=CATEGORY_GROUP_NAMES, pnp_tiers=PNP_TIERS, kp_statuses=KP_STATUSES,
    )


@records_bp.route("/new", methods=["GET", "POST"])
@login_required
def new_record():
    if request.method == "POST":
        # An uploaded page image is the scan-intake step, now embedded on this
        # same page instead of living at a separate /scan URL (see
        # routes/scan.py's run_scan_pipeline). It produces a draft to review,
        # not a saved record - the encoder still presses Save below.
        image = request.files.get("image")
        if image and image.filename:
            result, error = scan_route.run_scan_pipeline(
                image, backend=request.form.get("backend"))
            if error:
                flash(error)
                return redirect(url_for("records.new_record"))
            return _render_new_record(result=result)

        date = request.form.get("date")
        time = request.form.get("time")
        location_purok = request.form.get("location_purok")

        # Two intake shapes post here. The scan draft posts the template slots,
        # in which case the stored narrative is RE-RENDERED server-side from
        # those slots - the browser renders it live too, but a readonly textarea
        # is not a guarantee, and the stored narrative must have the same shape
        # whether or not the encoder had JavaScript. The plain /records/new form
        # posts a free-text narrative and nothing changes for it.
        summary = (request.form.get("incident_summary") or "").strip()
        if summary:
            slots = {k: request.form.get(k, "") for k in narrative_tpl.SLOTS}
            narrative = narrative_tpl.render(slots)
            # The columns follow the corpus convention, not D. alone - see
            # narrative.record_columns.
            columns = narrative_tpl.record_columns(slots)
            date, location_purok = columns["date"] or None, columns["location_purok"] or None
            # Classify the free prose, not the templated block: the template's
            # labelled boilerplate is a constant prefix that carries no incident
            # signal and only dilutes what the classifier reads.
            to_classify = summary
        else:
            narrative = request.form.get("narrative", "").strip()
            to_classify = narrative

        if not narrative:
            flash("Narrative is required.")
            return redirect(url_for("records.new_record"))

        # Saving is not a repeatable action, but this endpoint used to treat it
        # as one. There is no CSRF token and no submission id on either form
        # that posts here, so a POST delivered twice inserted two byte-identical
        # rows - observed in the pilot database as records 6 and 7, same page,
        # same encoder, same narrative, 6 ms apart. Six milliseconds is one
        # click arriving twice, not a person clicking twice.
        #
        # Guarding here rather than at the scan page covers every caller: the
        # scan draft, the plain /records/new form, and anything that posts this
        # form later. The client-side button lock in the templates stops the
        # second request being sent at all; this stops it counting if it is.
        #
        # Same narrative + same encoder + inside a minute is a replay, not a
        # second incident: two genuine incidents do not share a verbatim
        # narrative, and the same one re-entered deliberately an hour later
        # still saves.
        #
        # Compared by keyed digest, not by the narrative itself: the column is
        # AES-GCM encrypted with a fresh nonce per write, so the same text
        # produces a different ciphertext every time and an equality test on it
        # silently never matches - which is this guard quietly switched off.
        # models.IncidentRecord keeps the digest in step with the narrative.
        replay = IncidentRecord.query.filter(
            IncidentRecord.narrative_digest == narrative_digest(narrative),
            IncidentRecord.created_by == current_user.id,
            IncidentRecord.created_at >= datetime.utcnow() - timedelta(seconds=60),
        ).first()
        if replay:
            flash(f"Already saved as record #{replay.id} - ignored a duplicate submission.")
            return redirect(url_for("records.view_record", record_id=replay.id))

        threshold = current_app.config["CONFIDENCE_THRESHOLD"]
        # SEA-LION first, legacy TF-IDF fallback - same priority as the scan
        # intake pipeline, so a manually-entered record is classified the same
        # way a scanned one is (see _classify_narrative and routes/scan.py step 4).
        clf = _classify_narrative(to_classify, date=date, time=time, location_purok=location_purok)
        primary = clf["incident_type"]
        secondary = request.form.get("incident_type_secondary") or None

        # Allow manual override from form
        form_primary = request.form.get("incident_type_primary", "").strip()
        if form_primary:
            primary = form_primary
        form_secondary = request.form.get("incident_type_secondary", "").strip()
        if form_secondary:
            secondary = form_secondary

        # A hand-edited or absent value must not become a 0.0 confidence.
        try:
            posted_conf = float(request.form["model_confidence"])
        except (KeyError, TypeError, ValueError):
            posted_conf = None

        record = IncidentRecord(
            batch_number=request.form.get("batch_number") or None,
            date=date,
            time=time,
            location_purok=location_purok,
            incident_type_primary=primary,
            incident_type_secondary=secondary or None,
            # Encoder's dropdown choice wins over the derived tables below -
            # SEA-LION picks the incident type, but the PNP tier / category
            # group / KP status that follow from it are a legal judgment call
            # the encoder must be able to correct (e.g. a value threshold that
            # flips "Theft" from Conditional to Excluded KP-mediability).
            pnp_classification=request.form.get("pnp_classification") or (
                get_pnp_classification(primary)[0] if primary else None),
            category_group=request.form.get("category_group") or (
                get_category_group(primary) if primary else None),
            kp_status=request.form.get("kp_status") or (
                get_kp_status(primary)[0] if primary else None),
            narrative=narrative,
            action_taken=request.form.get("action_taken"),
            status=request.form.get("status"),
            readability="readable",
            include_in_ml=True,
            # Prefer what the scan already computed. The scan ran SEA-LION and a
            # six-reason gate (OCR confidence, axis disagreement, a rejected
            # LLM repair, unfilled template slots, reader disagreement) that this
            # route cannot see and the TF-IDF classifier below cannot reproduce.
            # Recomputing here
            # would quietly downgrade a flagged page to "accepted". Falls back
            # to the local decision for the plain /records/new form, which posts
            # neither field.
            model_confidence=posted_conf if posted_conf is not None else clf["confidence"],
            review_status=request.form.get("review_status") or decide_review_status(
                clf["confidence"] or 0.0, threshold
            ),
            source_file_page=request.form.get("source_file_page") or None,
            encoded_by=request.form.get("encoded_by") or None,
            remarks=request.form.get("remarks"),
            created_by=current_user.id,
        )
        db.session.add(record)
        db.session.commit()
        flash("Record saved.")
        return redirect(url_for("records.list_records"))
    return _render_new_record()


@records_bp.route("/<int:record_id>/edit", methods=["GET", "POST"])
@login_required
@role_required("reviewer", "admin")
def edit_record(record_id):
    record = IncidentRecord.query.get_or_404(record_id)
    if request.method == "POST":
        narrative = request.form.get("narrative", "").strip()
        if not narrative:
            flash("Narrative is required.")
            return redirect(url_for("records.edit_record", record_id=record_id))

        record.batch_number = request.form.get("batch_number") or None
        record.encoded_by = request.form.get("encoded_by") or None
        record.date = request.form.get("date")
        record.time = request.form.get("time")
        record.location_purok = request.form.get("location_purok")
        record.status = request.form.get("status")
        record.narrative = narrative
        record.action_taken = request.form.get("action_taken")
        record.incident_type_primary = request.form.get("incident_type_primary", "").strip() or None
        record.incident_type_secondary = request.form.get("incident_type_secondary", "").strip() or None
        record.pnp_classification = request.form.get("pnp_classification") or (
            get_pnp_classification(record.incident_type_primary)[0]
            if record.incident_type_primary else None
        )
        record.category_group = request.form.get("category_group") or (
            get_category_group(record.incident_type_primary)
            if record.incident_type_primary else None
        )
        record.kp_status = request.form.get("kp_status") or (
            get_kp_status(record.incident_type_primary)[0]
            if record.incident_type_primary else None
        )
        record.remarks = request.form.get("remarks")

        # Rerun Classifier (records_new.html) only ever showed its fresh
        # confidence in the browser - nothing wrote it back here, so a
        # record's model_confidence stayed pinned to whatever it scored at
        # creation/import even after a reviewer reran it against a corrected
        # narrative (see the Jan 2026 field-evaluation batch: records #2, #6,
        # #8, #12, #15, #17). The rerun JS now stashes its last confidence in
        # a hidden #model_confidence field so a subsequent Save actually
        # persists it.
        raw_confidence = request.form.get("model_confidence", "").strip()
        if raw_confidence:
            try:
                new_confidence = float(raw_confidence)
            except ValueError:
                new_confidence = None
            if new_confidence is not None and 0.0 <= new_confidence <= 1.0:
                record.model_confidence = new_confidence
                # Keep the review queue in step with the new score, same rule
                # sync_review_queue() uses: never reopen a status a human
                # already resolved by hand in the Review Queue.
                if record.review_status != "corrected":
                    threshold = current_app.config["CONFIDENCE_THRESHOLD"]
                    record.review_status = decide_review_status(new_confidence, threshold)

        db.session.commit()
        flash(f"Record #{record.id} updated.")
        return redirect(url_for("records.view_record", record_id=record.id))
    return render_template("records_new.html", categories=CATEGORY_NAMES, record=record,
                           models=sealion_classify.EVALUATED_MODELS,
                           active_model=sealion_classify.resolve_model(None),
                           category_groups=CATEGORY_GROUP_NAMES, pnp_tiers=PNP_TIERS, kp_statuses=KP_STATUSES)


@records_bp.route("/predict", methods=["POST"])
@login_required
def predict():
    """AJAX endpoint: live SEA-LION (or TF-IDF fallback) preview for the
    new_record form, plus the same PNP/KP/category-group screening the scan
    intake path shows (routes/scan.py step 4) - so typing a narrative here
    surfaces the same decision-support signals uploading a page would.
    """
    text = request.json.get("narrative", "")
    date = request.json.get("date", "")
    time = request.json.get("time", "")
    location_purok = request.json.get("location_purok", "")
    # Untrusted: resolve_model() drops anything not in the evaluated allowlist
    # rather than forwarding it to Ollama.
    model = request.json.get("model", "")
    threshold = current_app.config["CONFIDENCE_THRESHOLD"]

    clf = _classify_narrative(text, date=date, time=time,
                              location_purok=location_purok, model=model)
    label = clf["incident_type"]
    conf = clf["confidence"] or 0.0
    pnp_tier, legal_basis = get_pnp_classification(label) if label else ("", "")
    kp_status, kp_note = get_kp_status(label) if label else ("", "")
    return jsonify({
        "label": label,
        "confidence": round(conf, 4),
        "review": decide_review_status(conf, threshold),
        "category_group": clf["category_group"],
        "pnp_tier": pnp_tier,
        "legal_basis": legal_basis,
        "kp_status": kp_status,
        "kp_note": kp_note,
        "source": clf["source"],
        "reason": clf["reason"],
        "model": clf.get("model"),
    })


@records_bp.route("/delete-all", methods=["POST"])
@login_required
def delete_all():
    if current_user.role != "admin":
        flash("Only admins can delete all records.")
        return redirect(url_for("records.list_records"))
    # Require explicit confirmation token to prevent CSRF / accidental triggers
    if request.form.get("confirm") != "DELETE ALL":
        flash("Confirmation text did not match. No records were deleted.")
        return redirect(url_for("records.list_records"))
    count = IncidentRecord.query.count()
    IncidentRecord.query.delete()
    db.session.commit()
    flash(f"Deleted all {count} records.")
    return redirect(url_for("records.list_records"))


@records_bp.route("/delete/<int:record_id>", methods=["POST"])
@login_required
def delete_record(record_id):
    if current_user.role != "admin":
        flash("Only admins can delete records.")
        return redirect(url_for("records.list_records"))
    record = db.session.get(IncidentRecord, record_id)
    if record is None:
        flash("Record not found.")
        return redirect(url_for("records.list_records"))
    db.session.delete(record)
    db.session.commit()
    flash(f"Record #{record_id} deleted.")
    return redirect(request.referrer or url_for("records.list_records"))
