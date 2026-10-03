from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_required

from .. import db
from ..models import IncidentRecord
from ..normalize import CANONICAL_CATEGORIES, get_category_group, get_kp_status, get_pnp_classification
from ..rbac import role_required

review_bp = Blueprint("review", __name__, url_prefix="/review")

CATEGORY_NAMES = [name for name, _ in CANONICAL_CATEGORIES]


@review_bp.route("/")
@login_required
def queue():
    records = (
        IncidentRecord.query.filter_by(review_status="needs_review")
        .order_by(IncidentRecord.created_at.desc())
        .all()
    )
    return render_template("review_queue.html", records=records)


@review_bp.route("/<int:record_id>", methods=["GET", "POST"])
@login_required
@role_required("reviewer", "admin")
def resolve(record_id):
    record = db.session.get(IncidentRecord, record_id)
    if record is None:
        flash("Record not found.")
        return redirect(url_for("review.queue"))

    if request.method == "POST":
        new_primary = request.form.get("incident_type_primary", "").strip()
        new_secondary = request.form.get("incident_type_secondary", "").strip() or None

        if not new_primary:
            flash("Primary incident type is required.")
            return redirect(url_for("review.resolve", record_id=record_id))

        changed = (
            new_primary != (record.incident_type_primary or "")
            or new_secondary != (record.incident_type_secondary or "")
        )
        record.incident_type_primary = new_primary
        record.incident_type_secondary = new_secondary
        record.pnp_classification = get_pnp_classification(new_primary)[0]
        record.category_group = get_category_group(new_primary)
        record.kp_status = get_kp_status(new_primary)[0]
        record.review_status = "corrected" if changed else "accepted"

        # Same gap as records.py's edit_record(): Rerun Classifier here only
        # ever showed a fresh confidence in the browser, nothing persisted it.
        # The template stashes the last rerun's confidence in a hidden
        # #model_confidence field so Save Review can write it through - the
        # reviewer's own corrected/accepted verdict above still wins either way.
        raw_confidence = request.form.get("model_confidence", "").strip()
        if raw_confidence:
            try:
                new_confidence = float(raw_confidence)
            except ValueError:
                new_confidence = None
            if new_confidence is not None and 0.0 <= new_confidence <= 1.0:
                record.model_confidence = new_confidence

        db.session.commit()
        flash("Record reviewed.")
        return redirect(url_for("review.queue"))

    return render_template(
        "review_resolve.html",
        record=record,
        categories=CATEGORY_NAMES,
    )
