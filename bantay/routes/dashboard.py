"""
Dashboard route — revised for Chapter 3 alignment.

Enhancements:
  - Incident summary cards: total, needs_review, newly added, categorized counts.
  - Cross-tabulated summaries: category × purok, category × month.
  - Trend-based risk indicators: month-over-month change per category.
  - Model comparison results from last training run.
  - Confidence threshold display.
  - Yearly trend toggle (monthly or yearly aggregation).
"""
import os
from collections import Counter, defaultdict
from datetime import datetime

from flask import Blueprint, current_app, render_template, request, redirect, url_for, flash
from flask_login import login_required, current_user

from ..models import IncidentRecord
from ..ml import sealion_classify
from ..ml.registry import MODEL_FILENAME
from ..normalize import get_category_group, get_kp_status

dashboard_bp = Blueprint("dashboard", __name__)


def _filtered_query():
    query = IncidentRecord.query
    category = request.args.get("category")
    location = request.args.get("location")
    date_from = request.args.get("date_from")
    date_to = request.args.get("date_to")
    q = request.args.get("q")

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
    return query


def _compute_risk_indicators(records):
    """Month-over-month change per category; flag categories with rising trends."""
    # Group incidents by (month, category)
    monthly_cat = defaultdict(lambda: defaultdict(int))
    for r in records:
        if r.date and r.incident_type_primary:
            month = r.date[:7]
            monthly_cat[month][r.incident_type_primary] += 1

    if not monthly_cat:
        return {}

    months = sorted(monthly_cat.keys())
    if len(months) < 2:
        return {cat: "stable" for cat in {r.incident_type_primary for r in records if r.incident_type_primary}}

    last_month = months[-1]
    prev_month = months[-2]

    categories = {r.incident_type_primary for r in records if r.incident_type_primary}
    indicators = {}
    for cat in categories:
        last = monthly_cat[last_month].get(cat, 0)
        prev = monthly_cat[prev_month].get(cat, 0)
        if last > prev:
            indicators[cat] = "rising"
        elif last < prev:
            indicators[cat] = "declining"
        else:
            indicators[cat] = "stable"
    return indicators


def _compute_cross_tabs(records):
    """Build category × purok and category × month cross-tabulation data."""
    cat_purok = defaultdict(lambda: defaultdict(int))
    cat_month = defaultdict(lambda: defaultdict(int))
    puroks = set()
    months = set()

    for r in records:
        cat = r.incident_type_primary or "Uncategorized"
        purok = r.location_purok or "Unknown"
        month = (r.date or "")[:7]
        cat_purok[cat][purok] += 1
        puroks.add(purok)
        if month:
            cat_month[cat][month] += 1
            months.add(month)

    return {
        "cat_purok": dict(cat_purok),
        "cat_month": dict(cat_month),
        "puroks": sorted(puroks),
        "months": sorted(months),
    }


@dashboard_bp.route("/")
@login_required
def index():
    records = _filtered_query().order_by(IncidentRecord.created_at.desc()).all()
    trend_mode = request.args.get("trend", "monthly")

    total = len(records)
    needs_review = sum(1 for r in records if r.review_status == "needs_review")
    accepted = sum(1 for r in records if r.review_status == "accepted")
    corrected = sum(1 for r in records if r.review_status == "corrected")
    categorized = sum(1 for r in records if r.incident_type_primary)

    category_counts = Counter(
        r.incident_type_primary or "Uncategorized" for r in records
    )
    purok_counts = Counter(r.location_purok or "Unknown" for r in records)
    group_counts = Counter(
        r.category_group or get_category_group(r.incident_type_primary)
        for r in records if r.incident_type_primary
    )
    kp_counts = Counter(
        r.kp_status or get_kp_status(r.incident_type_primary)[0]
        for r in records if r.incident_type_primary
    )

    if trend_mode == "yearly":
        trend_counts = Counter((r.date or "")[:4] for r in records if r.date)
    else:
        trend_counts = Counter((r.date or "")[:7] for r in records if r.date)

    categories = sorted({r.incident_type_primary for r in IncidentRecord.query.all() if r.incident_type_primary})
    locations = sorted({r.location_purok for r in IncidentRecord.query.all() if r.location_purok})

    risk_indicators = _compute_risk_indicators(records)
    cross_tabs = _compute_cross_tabs(records)

    model_loaded = current_app.classifier.model is not None
    model_trained_at = None
    model_path = os.path.join(current_app.config["MODEL_DIR"], MODEL_FILENAME)
    if os.path.exists(model_path):
        model_trained_at = datetime.fromtimestamp(os.path.getmtime(model_path)).strftime("%Y-%m-%d %H:%M")

    # Recent classifier comparison (Chapter 4, §4.3.2) - see
    # sealion_classify.load_model_comparison's docstring for why this reads
    # classifier_metrics.json rather than a retrained model's results.
    comparison_results = sealion_classify.load_model_comparison(current_app.config["MODEL_DIR"])

    confidence_threshold = current_app.config["CONFIDENCE_THRESHOLD"]

    return render_template(
        "dashboard.html",
        total=total,
        needs_review=needs_review,
        accepted=accepted,
        corrected=corrected,
        categorized=categorized,
        category_counts=dict(sorted(category_counts.items(), key=lambda x: -x[1])),
        purok_counts=dict(sorted(purok_counts.items(), key=lambda x: -x[1])),
        group_counts=dict(sorted(group_counts.items(), key=lambda x: -x[1])),
        kp_counts=dict(sorted(kp_counts.items(), key=lambda x: -x[1])),
        trend_counts=dict(sorted(trend_counts.items())),
        trend_mode=trend_mode,
        recent_records=records[:20],
        categories=categories,
        locations=locations,
        filters=request.args,
        model_loaded=model_loaded,
        model_trained_at=model_trained_at,
        comparison_results=comparison_results,
        confidence_threshold=confidence_threshold,
        risk_indicators=risk_indicators,
        cross_tabs=cross_tabs,
    )


@dashboard_bp.route("/reload-model", methods=["POST"])
@login_required
def reload_model():
    if current_user.role != "admin":
        flash("Only admins can reload the model.")
        return redirect(url_for("dashboard.index"))
    loaded = current_app.classifier.reload()
    if loaded:
        flash("Model reloaded successfully.")
    else:
        flash("No trained model found — train one first.")
    return redirect(url_for("dashboard.index"))

